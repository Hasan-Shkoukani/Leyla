"""State + plumbing: config, sqlite, graph lookups, orders.csv, stats csv, outbound messages and admin alerts."""
import csv, hashlib, json, os, re, shutil, sqlite3, threading, time, unicodedata
from contextlib import closing
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from build_graph import fold, removed

ROOT = Path(__file__).parent


def cfg(k, d=""):
    return os.environ.get(k, d)


_MADE = set()


def state_dir():
    """Where the shop's state lives (orders, chats, customers, prices, outbox, backups...): STATE_DIR if set, else <project>/state once ./migrate_state.sh has moved it there, else (an older layout) the project folder.
    The folder is never chosen by guessing: an empty state/ cannot appear by itself, it only exists after the migration."""
    env = cfg("STATE_DIR", "")
    return Path(env) if env else (ROOT / "state" if (ROOT / "state").is_dir() else ROOT)


def P(name):  # all state files live in state_dir(); tests point STATE_DIR at tmp
    d = state_dir()
    if d not in _MADE or not d.is_dir():  # one mkdir per directory, not one per call (P is called thousands of times a minute)
        d.mkdir(parents=True, exist_ok=True)
        _MADE.add(d)
    return d / name


def now():
    """The shop's own clock (SHOP_TZ, default Asia/Nicosia = GMT+2 in winter), as a naive datetime: delivery times, departures and dates are always the shop's, whatever time zone the computer or the customer is in."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(os.environ.get("SHOP_TZ") or "Asia/Nicosia")).replace(tzinfo=None)


def norm(n):
    """Digits of a phone number. A Telegram customer is the id "tg:<chat id>", which stays as it is."""
    s = str(n or "").strip()
    if s.startswith("ext:"):  # a person who ordered in person / by phone with the owner and has no number on file: "ext:<name>"
        return "ext:" + re.sub(r"[^a-z0-9]+", "-", fold(s[4:])).strip("-")
    return "tg:" + re.sub(r"\D", "", s) if s.startswith("tg:") else re.sub(r"\D", "", s)


def admins():
    """The owners' WhatsApp numbers (ADMIN_NUMBERS)."""
    return {norm(x) for x in cfg("ADMIN_NUMBERS").split(",") if norm(x)}


# ---------------------------------------------------------------- sqlite
SCHEMA = """
DROP TABLE IF EXISTS msgs;  -- chats are no longer written to disk: anything an older version stored is erased the first time the database is opened
CREATE TABLE IF NOT EXISTS paused(phone TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS msgref(id TEXT PRIMARY KEY, text TEXT);
CREATE TABLE IF NOT EXISTS contact(phone TEXT PRIMARY KEY, last TEXT, n INT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS stat(kind TEXT, period TEXT, key TEXT, n INT, PRIMARY KEY(kind,period,key));
CREATE TABLE IF NOT EXISTS active(kind TEXT, period TEXT, phone TEXT, PRIMARY KEY(kind,period,phone));
CREATE TABLE IF NOT EXISTS customers(phone TEXT PRIMARY KEY, first_ts TEXT);
CREATE TABLE IF NOT EXISTS vision(phone TEXT, day TEXT, n INT, PRIMARY KEY(phone,day));
"""


_SCHEMA_DONE = set()


def q(sql, *a):
    path = P("chat.db")
    fresh = path not in _SCHEMA_DONE or not path.exists()  # the schema script runs once per database file, not on every query
    with closing(sqlite3.connect(path)) as c:
        if fresh:
            c.executescript(SCHEMA)
            _SCHEMA_DONE.add(path)
        rows = c.execute(sql, a).fetchall()
        c.commit()
        return rows


def kget(k, d=""):
    r = q("SELECT v FROM kv WHERE k=?", k)
    return r[0][0] if r else d


def atomic_write(path, text):
    """Write a state file so that a crash or a power cut can never leave it half written: temp file, flush to disk, then one atomic rename."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")  # one temp file per writer: two threads never write the same one
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def kset(k, v):
    q("INSERT INTO kv VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", k, str(v))


INBOX_LOCK = threading.Lock()
INFLIGHT = set()  # journal ids being answered right now (in memory): a second flush never answers the same message twice


def inbox():
    """The unanswered-messages journal (inbox.json): every customer message that arrived and has not been answered yet, of any kind (text, voice note, photo).
    A message is written here the moment it arrives and removed when its answer is sent, so a crash or a restart in between loses nothing. Media bytes are kept as files in inbox_media/."""
    p = P("inbox.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def inbox_add(item):
    """Journal one message. Returns its id."""
    import uuid
    item = dict(item)
    item["jid"] = item.get("jid") or uuid.uuid4().hex[:12]
    if item.get("data"):
        P("inbox_media").mkdir(exist_ok=True)
        (P("inbox_media") / item["jid"]).write_bytes(item.pop("data"))
        item["file"] = item["jid"]
    item.setdefault("ts", now().isoformat(timespec="seconds"))
    with INBOX_LOCK:
        items = inbox()
        existing = next((x for x in items if x.get("jid") == item["jid"]), None)
        if existing is not None:
            existing.update(item)
        else:
            items.append(item)
        atomic_write(P("inbox.json"), json.dumps(items, ensure_ascii=False))
    return item["jid"]


def inbox_load(it):
    """A journal entry with its media bytes read back."""
    it = dict(it)
    f = P("inbox_media") / it.pop("file", "-")
    if f.is_file():
        it["data"] = f.read_bytes()
    return it


def inbox_done(jids):
    jids = {j for j in jids if j}
    if not jids:
        return
    with INBOX_LOCK:
        keep = [x for x in inbox() if x.get("jid") not in jids]
        atomic_write(P("inbox.json"), json.dumps(keep, ensure_ascii=False))
        for j in jids:
            (P("inbox_media") / j).unlink(missing_ok=True)
            INFLIGHT.discard(j)


def inbox_take(live=True):
    """Everything not yet being answered, loaded (media included), and marked as in flight: the caller answers it and the batch then calls inbox_done."""
    with INBOX_LOCK:
        out = [inbox_load(x) for x in inbox() if x.get("jid") not in INFLIGHT and (live or not x.get("live"))]
        INFLIGHT.update(x["jid"] for x in out)
        return out


def tgq_add(u):
    """A Telegram update is written down the moment it arrives (before Telegram is told "received"), and removed when it has been handed to the agent: a restart in between loses nothing."""
    with INBOX_LOCK:
        p = P("tg_updates.json")
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        atomic_write(p, json.dumps(d + [u], ensure_ascii=False))


def tgq_done(uid):
    with INBOX_LOCK:
        p = P("tg_updates.json")
        if p.exists():
            atomic_write(p, json.dumps([u for u in json.loads(p.read_text(encoding="utf-8")) if u.get("update_id") != uid], ensure_ascii=False))


def tgq_all():
    p = P("tg_updates.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def inbox_drop_held():
    """The owner chose not to answer the messages kept while the agent was off."""
    inbox_done([x["jid"] for x in inbox() if not x.get("live")])


CHAT = {}  # phone -> [(iso time, role, text)]: a cache of the latest messages. The chats themselves live in chats/<customer>.jsonl, one JSON document per message, appended and never deleted.
CHAT_LOCK = threading.Lock()


def _chat_file(phone):
    return P("chats") / (re.sub(r"[^\w.-]", "_", str(phone)) + ".jsonl")


def chat_log(phone):
    """EVERY message ever exchanged with a customer, oldest first: [(iso time, role, text)]. Read from the chat store, so it is complete after any restart."""
    f = _chat_file(phone)
    if not f.exists():
        return []
    out = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(ln)
            out.append((d["ts"], d["role"], d["content"]))
        except (ValueError, KeyError):  # a line half written by a crash is skipped, the rest stays readable
            continue
    return out


def _rows(phone):
    """The latest HISTORY_MAX messages of the customer (what the agent remembers), loaded from the chat store the first time after a start. Call with CHAT_LOCK held."""
    if phone not in CHAT:
        CHAT[phone] = chat_log(phone)[-int(cfg("HISTORY_MAX", "200")):]
    return CHAT[phone]


def chat_add(phone, role, content, ts=None):
    ts = ts or now().isoformat(timespec="seconds")
    if role == "user":  # the customer list keeps the count and the time of the last message
        q("INSERT INTO contact VALUES(?,?,1) ON CONFLICT(phone) DO UPDATE SET last=excluded.last, n=n+1", phone, ts)
    with CHAT_LOCK:
        rows = _rows(phone)
        f = _chat_file(phone)
        f.parent.mkdir(exist_ok=True)
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": ts, "role": role, "content": content}, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        rows.append((ts, role, content))
        del rows[:-int(cfg("HISTORY_MAX", "200"))]


def save(phone, role, content):
    chat_add(phone, role, content)


def history(phone):
    """What the agent remembers of this customer: the latest HISTORY_MAX (default 200) messages. The older ones stay in the chat store (chat_log, search_chats)."""
    with CHAT_LOCK:
        return [(r, c) for ts, r, c in _rows(phone)]


def last_user_ts(phone):
    with CHAT_LOCK:
        return next((ts for ts, r, c in reversed(_rows(phone)) if r == "user"), None)


def search_chats(query, phone="", limit=30):
    """Words (all must appear, accents and case ignored) searched in every stored chat, newest first: [(customer, time, role, text)]."""
    toks = [t for t in re.split(r"\s+", fold(query).strip()) if t]
    files = [_chat_file(phone)] if phone else list(P("chats").glob("*.jsonl")) if P("chats").exists() else []
    hits = []
    for f in files:
        who = phone or f.stem
        for ts, role, text in (chat_log(phone) if phone else chat_log(f.stem)):
            if all(t in fold(text) for t in toks):
                hits.append((who, ts, role, text))
    return sorted(hits, key=lambda h: h[1], reverse=True)[:limit]


def chat_stats():
    """{phone: (messages the customer sent, time of the last one)}: kept on disk, so it survives a restart."""
    return {p: (n, last) for p, n, last in q("SELECT phone, n, last FROM contact")}


def chat_prune():
    """Chats are never deleted; only the in-memory cache is trimmed."""
    with CHAT_LOCK:
        for p in CHAT:
            del CHAT[p][:-int(cfg("HISTORY_MAX", "200"))]


def vision_allowed(phone):  # counts and reserves one call; False when over daily cap
    day = now().strftime("%Y-%m-%d")
    r = q("SELECT n FROM vision WHERE phone=? AND day=?", phone, day)
    if r and r[0][0] >= int(cfg("VISION_DAILY_CAP", "10")):
        return False
    q("INSERT INTO vision VALUES(?,?,1) ON CONFLICT(phone,day) DO UPDATE SET n=n+1", phone, day)
    return True


# ---------------------------------------------------------------- knowledge graph
FIELDS = ("name", "description", "colors", "available", "price", "category", "same_day")  # what the admin may override. A scrape never touches these; a price set here is PINNED (otherwise prices are read from the product pages)


def overrides():
    p = P("overrides.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def set_override(pid, field, value):
    """Admin-only. Changes live on top of data/graph.json, so a rebuild keeps them and /reset restores the default."""
    if field not in FIELDS or not product(pid):
        return False
    v = str(value).strip()
    if field == "colors":
        v = [fold(c) for c in re.split(r"[,/]", v) if c.strip()]
    elif field in ("available", "same_day"):
        v = v.lower() not in ("no", "false", "0", "off", "out", "unavailable")
    elif field == "price":
        try:
            if v.startswith("-"):  # parse_amount drops the sign: "-5" would silently become 5
                return False
            v = float(parse_amount(v))
        except ValueError:
            return False
        if v <= 0:
            return False
    elif field == "category":
        from build_graph import categorize
        v = categorize(v, product(pid)["name"])
        if v == "other":
            v = fold(str(value)).strip()
        if not v:
            return False
    with STATE_LOCK:
        o = overrides()
        o.setdefault(str(pid), {})[field] = v
        atomic_write(P("overrides.json"), json.dumps(o, ensure_ascii=False, indent=1))
    return True


def unpin(pid, field):
    """Drop ONE hand edit (e.g. the pinned price): that field follows the shop again."""
    with STATE_LOCK:
        o = overrides()
        if o.get(str(pid), {}).pop(field, None) is not None:
            atomic_write(P("overrides.json"), json.dumps(o, ensure_ascii=False, indent=1))


def remove_product(pid):
    """Remove a scraped item for good: it disappears from the catalog now and no re-scrape or rebuild ever adds it back (matched by id AND page link). restore_product undoes it."""
    n = product(pid)
    if not n or n.get("custom"):
        return False
    with STATE_LOCK:
        r = removed()
        r[str(pid)] = {"name": n.get("name"), "url": n.get("url"), "ts": time.time()}
        atomic_write(P("removed.json"), json.dumps(r, ensure_ascii=False, indent=1))
    return True


def restore_product(pid):
    r = removed()
    if str(pid) not in r:
        return False
    del r[str(pid)]
    atomic_write(P("removed.json"), json.dumps(r, ensure_ascii=False, indent=1))
    return True


def clear_override(pid="all"):
    o = {} if pid == "all" else {k: v for k, v in overrides().items() if k != str(pid)}
    atomic_write(P("overrides.json"), json.dumps(o, ensure_ascii=False, indent=1))


def customs():
    """Items the owner added himself (state file, never touched by a re-scrape): {id 'c-N': {name, price, category, description, colors, url, aliases}}."""
    p = P("custom.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_customs(d):
    atomic_write(P("custom.json"), json.dumps(d, ensure_ascii=False, indent=1))


def set_custom_photo(cid, jpeg_bytes):
    """The photo of one of the owner's own items: dl/data/catalog/<id>/0.jpg (what the dashboard shows and the photo matcher learns from). Returns the path; raises if it was not written."""
    d = ROOT / "dl/data/catalog" / cid
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.*"):
        old.unlink()
    f = d / "0.jpg"
    f.write_bytes(jpeg_bytes)
    if not f.is_file() or f.stat().st_size == 0:
        raise OSError("the photo was not saved")
    return f


def add_custom(name, price, category, description="", colors="", url="", aliases=(), names=None):
    from build_graph import categorize, colors_of
    d = customs()
    cid = "c-%d" % (max([int(k[2:]) for k in d] or [0]) + 1)
    cat = categorize(category, name)
    if cat == "other" and fold(category).strip():
        cat = fold(category).strip()
    cols = [fold(c) for c in re.split(r"[,/]", colors) if c.strip()] or colors_of(name)
    d[cid] = {"name": name.strip(), "price": float(price), "category": cat, "description": description.strip(), "colors": cols, "url": url.strip(), "aliases": list(aliases)}
    for lg in ("en", "tr", "ru"):  # the official name in each language (LLM-translated once when the owner creates the item), so every record and every admin view uses the same one
        if (names or {}).get(lg):
            d[cid]["name_" + lg] = str(names[lg]).strip()
    save_customs(d)
    return cid


_PRICE_LOCK = threading.Lock()
_PRICE_LAST = {}


def price_store():
    """{product id: {"price", "in_stock", "ts"}}: the prices the last update read from the shop's pages. This is what every customer, order and report uses: no page is read while someone waits.
    A damaged file never means "no prices": the last good copy of THAT file is kept."""
    p = P("prices.json")
    if not p.exists():
        return {}
    try:
        _PRICE_LAST[str(p)] = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        pass
    return dict(_PRICE_LAST.get(str(p), {}))


def save_prices(updates):
    """Merge fresh reads {pid: {"price", "in_stock", "ts"}} into the stored prices (an item that could not be read keeps its old price). One writer at a time: simultaneous updates all land."""
    with _PRICE_LOCK:
        s = price_store()
        for pid, e in updates.items():
            if e:
                rec = {"price": float(e["price"]), "in_stock": bool(e["in_stock"]), "ts": float(e.get("ts") or time.time())}
                if e.get("old_price") and float(e["old_price"]) > rec["price"]:  # a discounted item: `price` is what is paid (after), old_price the crossed-out one (before)
                    rec["old_price"] = float(e["old_price"])
                if e.get("same_day") is not None or e.get("old_price"):
                    rec["flags"] = True  # this read parsed the page's price block: its before price (or none) and same-day flag are the truth
                if e.get("same_day") is not None:
                    rec["same_day"] = bool(e["same_day"])
                elif "same_day" in s.get(str(pid), {}):  # the page did not say this time: what we knew stays
                    rec["same_day"] = s[str(pid)]["same_day"]
                s[str(pid)] = rec
        atomic_write(P("prices.json"), json.dumps(s))
        _PRICE_LAST[str(P("prices.json"))] = s
        return len(s)


@lru_cache(maxsize=4)
def _load(path, mtime, ovr_mtime=0, cus_mtime=0, rem_mtime=0, price_mtime=0):
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    gone = removed()  # the owner's removals hold even if graph.json (rebuilt by a scrape) still lists the item
    urls = {r["url"] for r in gone.values() if r.get("url")}
    nodes = {n["id"]: n for n in g["nodes"] if not (n["type"] == "product" and (n["id"] in gone or n.get("url") in urls))}
    g["edges"] = [e for e in g["edges"] if e["from"] in nodes or e["from"].split(":")[0] in ("cat", "color")]
    for n_ in nodes.values():
        if n_["type"] == "product" and "old_price" not in n_:
            n_["old_price"] = n_.get("before")  # the last full scrape's crossed-out price; a price read that parsed the page replaces it below
    for cid, c in customs().items():  # same shape as a scraped product; the price is stored (no page to read it from)
        nodes[cid] = {"id": cid, "type": "product", "custom": True, "name": c["name"], "category": c["category"], "colors": c["colors"], "aliases": c.get("aliases") or [],
                      "name_en": c.get("name_en"), "name_tr": c.get("name_tr"), "name_ru": c.get("name_ru"), "urls": [c["url"]] if c.get("url") else [], "url": c.get("url") or None, "image": None,
                      "description": c.get("description", ""), "fixed_price": c["price"], "old_price": c.get("old_price") or None, "same_day": c.get("same_day", True), "badges": []}
        nodes.setdefault("cat:" + c["category"], {"id": "cat:" + c["category"], "type": "category", "name": c["category"]})
        g["edges"].append({"from": cid, "to": "cat:" + c["category"], "rel": "in_category"})
        for col in c["colors"]:
            nodes.setdefault("color:" + col, {"id": "color:" + col, "type": "color", "name": col})
            g["edges"].append({"from": cid, "to": "color:" + col, "rel": "has_color"})
    for pid, fields in overrides().items():
        if pid in nodes:
            nodes[pid].update({k: v for k, v in fields.items() if k in FIELDS and k != "price"})
            if "price" in fields:
                nodes[pid]["pinned_price"] = fields["price"]  # read by live_prices: this item's page is never asked
            if "category" in fields:
                nodes.setdefault("cat:" + fields["category"], {"id": "cat:" + fields["category"], "type": "category", "name": fields["category"]})
                g["edges"] = [e for e in g["edges"] if not (e["from"] == pid and e["rel"] == "in_category")] + [{"from": pid, "to": "cat:" + fields["category"], "rel": "in_category"}]
            if "colors" in fields:  # keep colour edges in sync
                g["edges"] = [e for e in g["edges"] if not (e["from"] == pid and e["rel"] == "has_color")]
                g["edges"] += [{"from": pid, "to": "color:" + c, "rel": "has_color"} for c in fields["colors"]]
    for pid, e in price_store().items():  # the stored prices are part of the graph: every read is a dictionary lookup
        if pid in nodes and nodes[pid]["type"] == "product":
            nodes[pid]["stored_price"], nodes[pid]["stored_in_stock"], nodes[pid]["price_ts"] = e["price"], e.get("in_stock", True), e.get("ts", 0)
            if e.get("flags") or "same_day" in e:  # this read parsed the page's price block, so the before price and the same-day flag belong to it (a finished promotion loses its before price at the next read)
                nodes[pid]["old_price"] = e.get("old_price")
                if "same_day" in e and "same_day" not in overrides().get(pid, {}):  # the owner's own setting wins over what the page said
                    nodes[pid]["same_day"] = e["same_day"]
            # a record without them (an older reader, or a page that could not be parsed) keeps what the last full scrape found
    by = {}  # (rel, target id) -> set(product ids)
    for e in g["edges"]:
        by.setdefault((e["rel"], e["to"]), set()).add(e["from"])
    for n_ in nodes.values():
        if n_.get("type") == "product":
            n_["url"], n_["urls"] = None, []
        _no_shop_name(n_)
    return nodes, by


SHOP_NAME = re.compile(re.escape(cfg("SHOP_PRIVATE_NAME", "__no_private_shop_name__")), re.I)


def _no_shop_name(o):
    """The agent does not know the name of the shop it works at: the trade names are taken out of every text it can read (descriptions, the about / blog texts, the business card). Links (which carry the website's domain) stay."""
    for k, v in (o.items() if isinstance(o, dict) else enumerate(o)):
        if isinstance(v, str):
            if SHOP_NAME.search(v):
                o[k] = "" if k in ("name", "company") else "".join(x if x.startswith("http") else SHOP_NAME.sub("the shop", x) for x in re.split(r"(https?://\S+)", v))  # text around a link is cleaned, the link itself is not touched
        elif isinstance(v, (dict, list)):
            _no_shop_name(v)


def graph():
    p = cfg("GRAPH_PATH", str(ROOT / "data/graph.json"))
    o, c, r, pr = P("overrides.json"), P("custom.json"), P("removed.json"), P("prices.json")
    return _load(p, os.path.getmtime(p), *(f.stat().st_mtime_ns if f.exists() else 0 for f in (o, c, r, pr)))


_NAMES = {}


def product(pid):
    """A product by id, tolerant of how a model writes it: with or without the "k-" prefix, or by its exact name (English or Turkish) when that is unambiguous."""
    g = graph()[0]
    n = g.get(str(pid)) or g.get("k-" + str(pid))
    if n is None and pid:
        key = id(g)
        if key not in _NAMES:
            _NAMES.clear()
            d = {}
            for x in g.values():
                if x.get("type") == "product":
                    for nm in {x.get("name"), x.get("name_en"), x.get("name_tr")}:
                        if nm:
                            d.setdefault(fold(nm), set()).add(x["id"])
            _NAMES[key] = {k: next(iter(v)) for k, v in d.items() if len(v) == 1}
        n = g.get(_NAMES[key].get(fold(str(pid))))
    return n if n and n["type"] == "product" else None


class PriceUnavailable(Exception):
    pass


def unit_price(p):
    """Price of a PRICED product copy (see priced / priced_lines); graph nodes carry none."""
    if p.get("price") is None:
        raise PriceUnavailable(p.get("name", "?"))
    return p["price"]


def live_prices(products, fresh=False, fallback=False, show=False, age=None):
    """{id: {price, in_stock, ts} | None} from the STORED prices (nothing is fetched; the arguments only remain so older callers keep working). The owner's pinned prices and his own items win;
    an item with no stored price gives None and the caller says the team confirms the price."""
    out = {}
    for p in products:
        if p.get("pinned_price") is not None:  # a price the owner set by hand wins, also on his own items (change_product pins it)
            out[p["id"]] = {"price": Decimal(str(p["pinned_price"])), "in_stock": p.get("available") is not False, "ts": time.time()}
        elif p.get("custom"):
            out[p["id"]] = {"price": Decimal(str(p["fixed_price"])), "in_stock": p.get("available") is not False, "ts": time.time()}
        elif p.get("stored_price") is not None:
            out[p["id"]] = {"price": Decimal(str(p["stored_price"])), "in_stock": bool(p.get("stored_in_stock", True)), "ts": p.get("price_ts") or 0}
        else:
            out[p["id"]] = None
    return out


_EXT = {}  # {graph id: (graph, lists)}: replaced as a whole, never edited in place


def extremes():
    """Precomputed once per graph load: product ids sorted by price (cheapest first / priciest first) for the whole shop and for each category, in stock and available only.
    {"": {"asc": [...], "desc": [...]}, "<category>": {...}}: 'the cheapest teddy bear' or 'the priciest thing' is a list lookup, not a search."""
    nodes = graph()[0]
    global _EXT
    k = id(nodes)
    hit = _EXT.get(k)
    if hit is None:
        base = sorted((n for n in nodes.values() if n["type"] == "product" and n.get("available") is not False), key=lambda n: n["name"])
        es = live_prices(base)
        ok = [n for n in base if es.get(n["id"]) and es[n["id"]]["in_stock"]]
        key = lambda n: es[n["id"]]["price"]
        out = {"": {"asc": [n["id"] for n in sorted(ok, key=key)], "desc": [n["id"] for n in sorted(ok, key=key, reverse=True)]}}
        for cat in {n["category"] for n in ok}:
            sub = [n for n in ok if n["category"] == cat]
            out[cat] = {"asc": [n["id"] for n in sorted(sub, key=key)], "desc": [n["id"] for n in sorted(sub, key=key, reverse=True)]}
        _EXT = {k: (nodes, out)}  # the graph itself is kept here, so its id cannot be reused by a different graph while this entry exists
        return out
    return hit[1]


def customers():
    """CRM, owner-written only: {phone: {name, relationship, notes, stage, prio, pos}}. Delivery is free or paid by what the customer chose in the chat; the owner can change one order's fee when he confirms it."""
    p = P("customers.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def canon_phone(n):
    """Digits in WhatsApp form (country code, no +). A number written without a country code (10 digits, with or without a leading 0: mobile or landline) is North Cyprus / Turkey, +90."""
    d = norm(n)
    if d.startswith(("tg:", "ext:")):
        return d
    d = d[2:] if d.startswith("00") else d[1:] if d.startswith("0") else d
    return "90" + d if len(d) == 10 else d


def thumb(src, w=120):
    """A small cached JPEG of an image (the originals are 50 KB and more each, the dashboard shows them at 48-84 px): built once per image version, kept in the state folder."""
    src = Path(src)
    st = src.stat()
    d = P("thumbs")
    d.mkdir(parents=True, exist_ok=True)
    out = d / (hashlib.md5(f"{src}|{st.st_mtime_ns}|{st.st_size}|{w}".encode()).hexdigest() + ".jpg")
    if not out.exists():
        from PIL import Image, ImageOps
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((w, w))
            if im.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
                im = bg
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            im.convert("RGB").save(tmp, "JPEG", quality=80, optimize=True)
            os.replace(tmp, out)
    return out


def plain(s):
    """A name as plain lower-case letters: fancy Telegram fonts ('𝓢𝓪𝓻𝓪') and accents are flattened, so 'sara' finds it."""
    return fold("".join(c for c in unicodedata.normalize("NFKC", str(s or "")) if not unicodedata.combining(c)))


def known_names():
    """{customer id: name} from the chat names and the owner's notebook (his own entry wins)."""
    d = {k[5:]: v for k, v in q("SELECT k, v FROM kv WHERE k LIKE 'name:%'")}
    d.update({p: c["name"] for p, c in customers().items() if c.get("name")})
    return d


RECENT_HOURS = 24


def last_chat_age(phone):
    """Hours since the last message with this customer (either way); None if there is no chat."""
    log = chat_log(phone)
    if not log:
        return None
    try:
        t = datetime.fromisoformat(log[-1][0])
        return max(0.0, (now().replace(tzinfo=None) - t.replace(tzinfo=None)).total_seconds() / 3600)
    except ValueError:
        return None


def find_people(text, recent=True):
    """The customers (never the owners) whose first or last name (3+ letters) is written in `text`. By default only those who chatted in the last RECENT_HOURS hours: a name means the person the owner just
    dealt with, never an old customer who happens to share it. recent=False lists every match (used to explain why someone was not found)."""
    words = {w for w in re.split(r"\W+", plain(text)) if len(w) >= 3}
    hits = [p for p, n in known_names().items() if p not in admins() and words & {w for w in re.split(r"\W+", plain(n)) if len(w) >= 3}]
    return [p for p in hits if (a := last_chat_age(p)) is not None and a <= RECENT_HOURS] if recent else hits


def pick_person(hits, focus=""):
    """Out of several customers with the same name: the one the owner is on right now (focus), else the one who chatted most recently. Only when the two freshest were active within 15 minutes of
    each other (and neither is in focus) is it a real tie: returns (None, [those ids]). Returns (id, [])."""
    if focus in hits:
        return focus, []
    by_age = sorted((last_chat_age(h), h) for h in hits)
    close = [h for a, h in by_age if a - by_age[0][0] <= 0.25]
    return (by_age[0][1], []) if len(close) == 1 else (None, close)


PHONE_WHY = {"empty": ("empty", "boş", "пусто"), "too short, or a country code is missing": ("too short, or a country code is missing", "çok kısa ya da ülke kodu eksik", "слишком короткий или нет кода страны"),
             "the length does not fit the country code": ("the length does not fit the country code", "uzunluk ülke koduna uymuyor", "длина не подходит к коду страны")}


def phone_why(why, lang=0):
    """The reason check_phone gave, in the dashboard's language (0 en, 1 tr, 2 ru)."""
    return PHONE_WHY.get(why, (why,) * 3)[lang]


def check_phone(raw):
    """A phone number a person typed -> (digits WITH country code, "") or ("", why). No country code (10 digits, or 11 with the leading 0) means Turkey / North Cyprus, +90 is added;
    a number written with its code (+, 00, or 11+ digits) is kept. The length must fit the country (+90: 12 digits, +357: 11, any other: 10-15); anything else is not guessed: the caller asks for the code."""
    s = str(raw or "").strip()
    d = re.sub(r"\D", "", s)
    if not d:
        return "", "empty"
    if s.startswith("+"):
        pass
    elif d.startswith("00"):
        d = d[2:]
    elif len(d) == 10:
        d = "90" + d
    elif len(d) == 11 and d[0] == "0":
        d = "90" + d[1:]
    elif len(d) < 11:
        return "", "too short, or a country code is missing"
    if d.startswith("90") and len(d) == 13 and d[2] == "0":  # "+90 0533...": the trunk 0 is dropped
        d = "90" + d[3:]
    ok = len(d) == 12 if d.startswith("90") else len(d) == 11 if d.startswith("357") else 10 <= len(d) <= 15
    return (d, "") if ok else ("", "the length does not fit the country code")


def set_contact_phone(ph, raw, c=None):
    """Set the number to CALL / reach a person on (their chat id stays as it is: someone may chat on WhatsApp and prefer another number for calls, or have changed number). The number is checked
    (+90 added when there is no country code); a note says what changed; the id is linked to the number (the same person on another app is found by it); the sender phone of their orders that are
    not shipped or closed follows. `c` = the profile dict when the caller saves it itself. Returns ("+digits", what was adjusted) or ("", why not)."""
    ok_, why = check_phone(raw)
    if not ok_:
        return "", f"the phone '{raw}' does not look right ({why}): write it with the country code, e.g. +90 5xx xxx xx xx"
    new = "+" + ok_
    with STATE_LOCK:
        own = c is None
        d = customers() if own else None
        c = d.setdefault(ph, {}) if own else c
        old = c.get("phone") or next((r["sender_phone"] for r in reversed(orders()) if r["customer_number"] == ph and r["sender_phone"]), "")
        if canon_phone(old) == canon_phone(new):
            return new, "unchanged"
        c["phone"] = new
        chat = "" if ph.startswith("ext:") else f" (chats from {plus(ph)})"
        line = f"contact phone: {new}{chat}" + (f", was {old}" if old else "")
        c["notes"] = (c.get("notes", "") + "\n" if c.get("notes") else "") + line
        if own:
            atomic_write(P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))
    link(ph, _tok_phone(ok_))
    n = 0
    with _lock:
        rows = orders()
        for r in rows:
            if r["customer_number"] == ph and r["status"] not in ("completed", "rejected", "cancelled") and r["sender_phone"] != new:
                r["sender_phone"] = new
                n += 1
        if n:
            _write(rows)
    return new, f"saved as the contact phone, noted, linked to the person" + (f", and the sender phone of {n} open order(s) updated" if n else "")


def move_customer(old, new, lang=0):
    """The customer's WHATSAPP NUMBER changed, or the one we have is wrong (a number saved without its country code can never be reached): everything that belongs to the old id moves to the new one: orders,
    the CRM entry, the chat history and every setting kept per customer. If the new id already exists the two are merged (the existing one wins on a clash). Returns what moved, or "" if it cannot be done."""
    old, new = norm(old), norm(new)
    if not old or not new or old == new or old.startswith(("tg:", "ext:")) or new.startswith(("tg:", "ext:")):
        return ""
    with _lock:
        rows = orders()
        n_orders = 0
        for r in rows:
            if r["customer_number"] == old:
                r["customer_number"] = new
                n_orders += 1
            if r["sender_phone"] and canon_phone(r["sender_phone"]) == old:
                r["sender_phone"] = "+" + new
        if n_orders:
            _write(rows)
    with STATE_LOCK:
        d = customers()
        if old in d:
            mine = d.pop(old)
            if new in d:
                cur = d[new]
                notes = "\n".join(x for x in (cur.get("notes"), mine.get("notes")) if x)
                for k, v in mine.items():
                    cur.setdefault(k, v)
                if notes:
                    cur["notes"] = notes
            else:
                d[new] = mine
            atomic_write(P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))
    with CHAT_LOCK:
        fo, fn = _chat_file(old), _chat_file(new)
        if fo.exists():
            lines = fo.read_text(encoding="utf-8").splitlines() + (fn.read_text(encoding="utf-8").splitlines() if fn.exists() else [])
            def ts_(l):
                try:
                    return json.loads(l)["ts"]
                except (ValueError, KeyError):
                    return ""
            fn.write_text("\n".join(sorted(lines, key=ts_)) + "\n", encoding="utf-8")
            fo.unlink()
        CHAT.pop(old, None)
        CHAT.pop(new, None)
    q("UPDATE OR IGNORE contact SET phone=? WHERE phone=?", new, old); q("DELETE FROM contact WHERE phone=?", old)
    for tbl in ("customers", "paused", "active", "vision"):
        q(f"UPDATE OR IGNORE {tbl} SET phone=? WHERE phone=?", new, old); q(f"DELETE FROM {tbl} WHERE phone=?", old)
    for k, v in q("SELECT k, v FROM kv"):
        nk, nv = (new.join(k.rsplit(old, 1)) if k.endswith(":" + old) else k), v.replace(f'"{old}"', f'"{new}"')  # per-customer settings (name:, declined:, tokens:...) and ids inside stored lists / records
        if nk != k:
            q("INSERT OR IGNORE INTO kv VALUES(?,?)", nk, nv); q("DELETE FROM kv WHERE k=?", k)
        elif nv != v:
            q("UPDATE kv SET v=? WHERE k=?", nv, k)
    link(new, _tok_phone(new))
    return (f"{n_orders} order(s), the chat history and the customer's settings moved to {plus(new)}", f"{n_orders} sipariş, sohbet geçmişi ve müşteri ayarları {plus(new)} numarasına taşındı",
            f"{n_orders} заказ(ов), история чата и настройки клиента перенесены на {plus(new)}")[lang]


def plus(n):
    """A phone number as the owner should see it: +<country code><number>, which WhatsApp turns into a tappable link. Anything that is not a phone number is returned as it was."""
    if str(n or "").startswith("tg:"):
        return "Telegram " + str(n)[3:]
    if str(n or "").startswith("ext:"):
        return "In person: " + str(n)[4:].replace("-", " ").title()
    d = canon_phone(n) if re.fullmatch(r"[\d\s+()-]{7,20}", str(n or "")) else ""
    return "+" + d if d else str(n or "")


STATE_LOCK = threading.RLock()  # one change at a time for the small JSON state files that are read, changed and written back


def set_profile(phone, **f):
    with STATE_LOCK:  # two customers writing at once must not overwrite each other's profile
        d, ph = customers(), canon_phone(phone)
        c = d.setdefault(ph, {})
        for k, v in f.items():
            if k == "hidden":
                c["hidden"] = bool(v)
                continue
            if v is None or v == "":
                continue
            if k == "notes":
                c["notes"] = (c.get("notes", "") + "\n" if c.get("notes") else "") + str(v)
            else:
                c[k] = v
        atomic_write(P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))
        return ph, c


def priced(p, e, phone=""):
    """A product copy with the price read from its page (no discounts exist; `phone` is kept so callers need no change)."""
    return {**p, "price": e["price"], "in_stock": e["in_stock"], **({"stale_age": e["age"]} if e.get("age") else {})}


def stale_note(lines):
    """'X (price read 12 min ago, page unreadable right now)' for order lines that had to use a last-known price, '' if all were read fresh."""
    return "; ".join(f"{p['name']} (price read {int(p['stale_age'] // 60)} min ago, page unreadable right now)" for p, _ in lines if p.get("stale_age"))


def priced_lines(lines, phone=""):
    """[(product, qty)] -> [(product with its LIVE price, qty)] read now; PriceUnavailable (naming the items) if any page cannot be read or is out of stock."""
    es = live_prices([p for p, _ in lines], fresh=True, fallback=True)
    bad = [p["name"] for p, _ in lines if not es.get(p["id"])]
    if bad:
        raise PriceUnavailable(", ".join(bad))
    out = [p["name"] for p, _ in lines if not es[p["id"]]["in_stock"]]
    if out:
        raise PriceUnavailable("OUT OF STOCK: " + ", ".join(out))
    return [(priced(p, es[p["id"]], phone), n) for p, n in lines]


def display(p, lang=""):
    """Product name in the customer's language (en/tr, both stored in the graph); any other language gets the English name."""
    n = (p.get("name_" + lang) if lang in ("en", "tr", "ru") else None) or p.get("name_en") or p["name"]
    return re.sub(r"(?<!\.)\.\s*$", "", n)  # the website's own names are sometimes written as a sentence ("Nine red roses in a Paşabahçe vase."): shown without the final full stop


def brief(p, lang=""):
    d = {k: p.get(k) for k in ("id", "name", "aliases", "category", "colors", "url") if p.get(k) not in (None, [])}
    if p.get("same_day") is False:
        d["same_day"] = False  # a few items are not delivered the same day (absent = same-day as usual)
    if p.get("badges"):
        d["badges"] = p["badges"]
    if p.get("price") is not None:  # only a freshly read price is ever shown
        d["price"] = float(p["price"])
        d["currency"] = "TRY"
        if p.get("in_stock") is False:
            d["in_stock"] = False
    if lang == "both":  # the model picks/translates; no language guessing in code
        d.pop("name", None); d.pop("aliases", None)
        d["name_en"], d["name_tr"] = p.get("name_en") or p["name"], p.get("name_tr") or p["name"]
        if p.get("name_ru"):
            d["name_ru"] = p["name_ru"]
    elif lang:
        d["name"] = display(p, lang)
        d.pop("aliases", None)
    return d


SYN = {  # Turkish <-> English product words (folded), so "301 gül" finds "301 Red Roses" and "papatya" finds "daisy" items
    "gul": ("rose", "roses"), "rose": ("gul",), "roses": ("gul",), "papatya": ("daisy", "daisies", "chamomile"), "daisy": ("papatya",), "daisies": ("papatya",),
    "lilyum": ("lily", "lilies"), "lily": ("lilyum",), "lilies": ("lilyum",), "orkide": ("orchid", "orchids"), "orchid": ("orkide",), "orchids": ("orkide",),
    "cikolata": ("chocolate",), "chocolate": ("cikolata",), "ayicik": ("bear", "bears", "teddy"), "bear": ("ayicik",), "teddy": ("ayicik",), "bears": ("ayicik",),
    "meyve": ("fruit",), "fruit": ("meyve",), "celenk": ("wreath",), "wreath": ("celenk",), "buket": ("bouquet",), "bouquet": ("buket",),
    "kirmizi": ("red",), "red": ("kirmizi",), "beyaz": ("white",), "white": ("beyaz",), "pembe": ("pink",), "pink": ("pembe",), "mor": ("purple",), "purple": ("mor",),
}


PRICE_CAP = 10**6  # prices are stored, so every candidate can be ranked  # most candidate pages read in one ranking / budget question


def _hits(nodes, ids, query):
    toks = [t for t in re.split(r"\W+", fold(query)) if len(t) > 1 and not t.isdigit()]
    return any(any(t in fold(" ".join([nodes[i]["name"], nodes[i].get("name_en") or "", nodes[i].get("name_tr") or "", nodes[i].get("name_ru") or "", *(nodes[i].get("aliases") or [])])) for t in toks) for i in ids if nodes[i].get("available") is not False) if toks else True


def find_flowers(query="", category="", color="", max_price=0, limit=8, sort="", min_price=0, offset=0, lang="", with_prices=True):
    """Typed catalog query. Text/category/color/availability come from the graph; PRICES never do: they are read from the product pages.
    Budget filters and price sorting rank with recently read prices, then everything returned is read FRESH and re-checked against the budget."""
    nodes, by = graph()
    query = normalize_query(query) if query else query  # keywords, in any order, typos and "5li" understood
    if sort in ("price_asc", "price_desc", "most_expensive") and not (query or color or max_price or min_price):  # "the cheapest / priciest (in a category)": a lookup in the precomputed lists
        cat = fold(category).strip()
        lst = extremes().get(cat if cat in extremes() else "", {}).get("desc" if sort != "price_asc" else "asc", [])
        es = live_prices([nodes[i] for i in lst[offset:offset + limit]], show=True)
        return [brief(priced(nodes[i], es[i]), lang) for i in lst[offset:offset + limit] if es.get(i)]
    allids = {i for i, n in nodes.items() if n["type"] == "product"}
    ids = set(allids)
    if category and ("in_category", "cat:" + fold(category).strip()) in by:  # a category the shop does not have (e.g. "plants" before they existed) is no filter
        ids &= by[("in_category", "cat:" + fold(category).strip())]
    if color and ("has_color", "color:" + fold(color).strip()) in by:
        ids &= by[("has_color", "color:" + fold(color).strip())]
    if query and (category or color) and not _hits(nodes, ids, query):  # the words match something outside the filter the model guessed: the words win
        if _hits(nodes, allids, query):
            ids = allids
        else:
            query = ""  # the words name nothing in the catalog (e.g. just "plants"): the category alone is the answer
    toks = [t for t in re.split(r"\W+", fold(query)) if len(t) > 1 or t.isdigit()]  # a single digit is a quantity ("5 roses"), never noise
    cands = []
    for i in ids:
        p = nodes[i]
        if p.get("available") is False:
            continue
        hay = fold(" ".join([p["name"], p.get("name_en") or "", p.get("name_tr") or "", p.get("name_ru") or "", *(p.get("aliases") or []), p.get("description") or ""]))  # English + Turkish names, and the short description ("15 gül 23 çikolata")
        score = 0
        nm_hay = fold(" ".join([p["name"], p.get("name_en") or "", p.get("name_tr") or "", p.get("name_ru") or "", *(p.get("aliases") or [])]))  # a word in the NAME counts more than one in the description
        for t in toks:  # numbers must match as whole numbers and weigh most ("301" is not "1301"); words also match their TR/EN synonym; order of the words never matters
            if t.isdigit():
                rx = rf"(?<!\d){t}(?!\d)"
                score += 10 * bool(re.search(rx, nm_hay)) or 3 * bool(re.search(rx, hay))
            else:
                ws = (t, *SYN.get(t, ()))
                score += 2 * any(w in nm_hay for w in ws) or any(w in hay for w in ws)
        if toks and not score:
            continue
        extra = len(set(re.split(r"\W+", fold(p["name"]))) - set(toks) - {""}) if toks else 0  # equal score: the name with fewer words the customer did not write is the closer one
        cands.append((-score, extra, p["name"], p))
    cands.sort(key=lambda x: x[:3])
    ps = [p for *_, p in cands]
    desc = sort in ("price_desc", "most_expensive")
    if max_price or min_price or sort in ("price_asc", "price_desc", "most_expensive"):
        es = live_prices(ps[:PRICE_CAP])
        ps = [priced(p, es[p["id"]]) for p in ps[:PRICE_CAP] if es.get(p["id"]) and es[p["id"]]["in_stock"] and (min_price or 0) <= es[p["id"]]["price"] <= (max_price or 10**12)]
        ps.sort(key=lambda p: p["price"], reverse=desc) if sort else None
        page = ps[offset:offset + limit]
        es2 = live_prices(page, show=True)  # what the customer sees is read now, and must still fit the budget
        page = [priced(p, es2[p["id"]]) for p in page if es2.get(p["id"]) and (min_price or 0) <= es2[p["id"]]["price"] <= (max_price or 10**12)]
    else:
        page = ps[offset:offset + limit]
        if not with_prices:  # the dashboard picker lists first and reads a price only for what the admin clicks
            return [brief(p, lang) for p in page]
        es2 = live_prices(page, show=True)
        page = [priced(p, es2[p["id"]]) if es2.get(p["id"]) else p for p in page]  # unreadable price: no price key, the agent must say the team confirms
    return [brief(p, lang) for p in page]


def count_products(query="", category="", color="", max_price=0, min_price=0):
    if not (max_price or min_price):  # no prices involved: count straight from the graph, nothing is fetched
        nodes, by = graph()
        ids = {i for i, n in nodes.items() if n["type"] == "product" and n.get("available") is not False}
        if category:
            ids &= by.get(("in_category", "cat:" + fold(category).strip()), set())
        if color:
            ids &= by.get(("has_color", "color:" + fold(color).strip()), set())
        toks = [t for t in re.split(r"\W+", fold(query)) if len(t) > 1 or t.isdigit()]
        return sum(1 for i in ids if not toks or any(t in fold(" ".join([nodes[i]["name"], nodes[i].get("name_en") or "", nodes[i].get("name_tr") or "", nodes[i].get("name_ru") or ""])) for t in toks))
    return len(find_flowers(query, category, color, max_price, PRICE_CAP, "", min_price))


def overview():
    """What the shop sells: per category the number of items and its cheapest and priciest item (stored prices, TL), plus colors."""
    nodes, _ = graph()
    cats, cols = {}, {}
    for p in nodes.values():
        if p["type"] != "product" or p.get("available") is False:
            continue
        cats[p["category"]] = cats.get(p["category"], 0) + 1
        for col in p.get("colors") or []:
            cols[col] = cols.get(col, 0) + 1
    ext, out = extremes(), {k: {"count": v} for k, v in sorted(cats.items())}
    for k in out:  # cheapest and priciest of each category (stored prices, TL)
        e = ext.get(k)
        if e and e["asc"]:
            lo, hi = nodes[e["asc"][0]], nodes[e["desc"][0]]
            es = live_prices([lo, hi])
            out[k].update(cheapest={"id": lo["id"], "name": lo["name"], "price": float(es[lo["id"]]["price"])}, priciest={"id": hi["id"], "name": hi["name"], "price": float(es[hi["id"]]["price"])})
    return {"products": sum(cats.values()), "categories": out, "colors": dict(sorted(cols.items(), key=lambda x: -x[1]))}


def product_detail(pid, lang=""):
    p = product(pid)
    e = live_prices([p], show=True).get(p["id"]) if p else None
    p = priced(p, e) if e else p
    return None if not p else {**brief(p, lang), "description": p.get("description"), "available": p.get("available") is not False}


def business_info():
    b = {k: v for k, v in graph()[0]["business"].items() if v and k not in ("id", "type", "info")}  # `info` (the shop's pages) is read page by page with shop_page
    b["delivery_policy"] = delivery_policy_text()  # from data/delivery.json
    b["policies"] = {k: v for k, v in _json("policies.json").items() if k in ("cancellation", "same_day", "blog")}
    return b


LANG3 = ("en", "tr", "ru")


def policy(key, lang="en", **kw):
    """A fixed policy sentence of data/policies.json in en / tr / ru (written once, never rewritten by a model)."""
    d = _json("policies.json").get(key) or {}
    return (d.get(lang) or d.get("en") or "").format(**kw)


def shop_page(topic, lang="en"):
    """The shop's BLOG (the list, or a post found by its words) in `lang`, word for word as scraped; '' if there is none. No other page of the shop is stored."""
    info = graph()[0]["business"].get("info") or {}
    t = fold(topic).strip()
    key = next((k for k in info if k == t), None)
    if key is None:
        words = [w for w in re.split(r"\W+", t) if len(w) > 2]
        posts = [k for k in info if k.startswith("blog/")]
        scored = sorted(((sum(w in fold(k + " " + (info[k].get(lang) or info[k].get("en") or {}).get("title", "")) for w in words), k) for k in posts), reverse=True)
        key = scored[0][1] if scored and scored[0][0] else next((k for k in info if t and t in k), None)
    if key is None:
        return ""
    if key == "blog":  # the list of posts
        website = cfg("SHOP_BASE_URL", "https://example.com").rstrip("/")
        return "\n".join(f"{(info[k].get(lang) or info[k].get('en') or {}).get('title', k)} - {website}/{lang}/{k}" for k in info if k.startswith("blog/"))
    pg = info[key].get(lang) or info[key].get("en") or next(iter(info[key].values()))
    return f"{pg.get('title', '')}\n{pg.get('text', '')}"[:3500]


BADGES = ("promotional", "featured", "heartfelt", "bestSeller", "new")  # the shop's own collections (its /badges/<slug> pages)


def badge_products(badge):
    """The catalog products that carry this shop badge, from the last scrape (the shop's own list), hidden ones left out."""
    return [n for n in graph()[0].values() if n["type"] == "product" and badge in (n.get("badges") or []) and n.get("available") is not False]


def delivered_age(oid):
    """Hours since the order was marked shipped / delivered (status completed); None if it never was (or the time is unknown: an order completed before the time was kept counts as long ago)."""
    t = (jget("doneat") or {}).get(str(oid))
    try:
        return max(0.0, (now() - datetime.fromisoformat(t)).total_seconds() / 3600) if t else None
    except ValueError:
        return None


# ---------------------------------------------------------------- outbound + feed
FEED = []  # in-memory log for the test UI (also fed in real mode)
TRANSPORT = None  # real WhatsApp sender (app.py) or None = mock
TG = None  # reserved for legacy saved chat ids; no Telegram transport is bundled


def _tx(to):
    """The channel a customer / owner is on: Telegram ids start with tg:, everything else is a WhatsApp number."""
    if str(to).startswith("ext:") or str(to).startswith("dashboard"):  # "dashboard" / "dashboard:<user>" = the web page, never a chat number
        return None  # a person without a number on file: nothing can be sent, the owner hands things over himself
    return None if str(to).startswith("tg:") else TRANSPORT


def _alog_file(name):
    return P("admin_chats") / (re.sub(r"[^\w.-]", "_", str(name)) + ".jsonl")


def alog_add(name, role, body, kind="text", file=None, by=""):
    """Append one admin chat event: "live" is the shared permanent chat; "assistant_<user>" is that dashboard user's private daily thread. role = user, assistant or alert."""
    f = _alog_file(name)
    f.parent.mkdir(exist_ok=True)
    with CHAT_LOCK, open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": now().isoformat(timespec="seconds"), "role": role, "by": by, "kind": kind, "body": body or "", "file": file}, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def alog_read(name):
    """Every message of an admin chat, oldest first (a half-written line of a crash is skipped)."""
    f = _alog_file(name)
    out = []
    for ln in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []):
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def alog_slice(name, start, stop):
    """(total, [(position, message)]) for the messages start <= position < stop (0-based), parsing only those lines: a chat page never parses the whole history."""
    f = _alog_file(name)
    lines = [ln for ln in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []) if ln.strip()]
    out = []
    for i in range(max(0, start), min(stop, len(lines))):
        try:
            out.append((i, json.loads(lines[i])))
        except ValueError:
            continue
    return len(lines), out


def alog_clear(name):
    """Clear one private assistant thread at the daily boundary; the shared live admin log is never cleared."""
    f = _alog_file(name)
    with CHAT_LOCK:
        if f.exists():
            atomic_write(f, "")


_FEED_BLANKED = 0


def feed(to, kind, body, **kw):
    global _FEED_BLANKED
    cut = len(FEED) - 20000  # the test page's log lives in memory too: keep it bounded (ids are positions, so old rows are blanked, not removed)
    while _FEED_BLANKED < cut:
        FEED[_FEED_BLANKED]["body"], FEED[_FEED_BLANKED]["file"] = "", None
        _FEED_BLANKED += 1
    FEED.append({"id": len(FEED) + 1, "ts": now().strftime("%H:%M:%S"), "to": to, "kind": kind, "body": body, **kw})
    if kind in ("text", "document", "image") and (body or kw.get("file")):  # live history is permanent; private dashboard threads are reset daily
        try:
            if to == "dashboard":
                alog_add("live", "assistant", body, kind, kw.get("file"))
            elif str(to).startswith("dashboard:"):
                alog_add("assistant_" + to[10:], "assistant", body, kind, kw.get("file"))
            elif to in admins():
                alog_add("live", "alert", body, kind, kw.get("file"))
        except OSError:
            pass


def log(msg):
    feed("", "log", msg)


MSGS = {}  # message id -> text, the last 5000 messages in either direction: lets a WhatsApp "reply" be tied to the message it quotes. ponytail: lost on restart (an old quote is then simply not shown)


def remember_msg(mid, text):
    if mid and text:
        MSGS[mid] = str(text)[:600]
        if len(MSGS) > 5000:
            MSGS.pop(next(iter(MSGS)))
        try:
            q("INSERT OR REPLACE INTO msgref VALUES(?,?)", str(mid), MSGS[mid])  # on disk too: a customer can reply to a message from before a restart
        except Exception as e:  # remembering a message must never stop a message from being sent or understood
            print(f"remember_msg failed: {e!r}")


def quoted_text(mid):
    """The text of the message a customer replied to ('' if unknown)."""
    if not mid:
        return ""
    if MSGS.get(mid):
        return MSGS[mid]
    try:
        return next((r[0] for r in q("SELECT text FROM msgref WHERE id=?", str(mid))), "")
    except Exception:
        return ""


_ADMIN_NAMES = {}
ADMIN_TRANSLATE = None  # set by agent.py: (text, code) -> text, an LLM translation of what the code wrote in English


def admin_names(to, body):
    """An admin's language mode (/english /turkish /russian, default English): every product name in what he receives is replaced by the OFFICIAL catalog name of that language (stored names, never a translation), so an order of "5 red roses" reads the same as the record."""
    if to not in admins() and not str(to).startswith("dashboard"):
        return body
    code = kget("adminlang:" + to) or "en"
    prods = [p for p in graph()[0].values() if p.get("type") == "product"]
    key = (code, len(prods))
    if key not in _ADMIN_NAMES:
        m = {}
        for p in prods:
            off = p.get("name_" + code) or p.get("name_en") or p["name"]
            for v in {p.get("name"), p.get("name_en"), p.get("name_tr"), p.get("name_ru")}:
                if v and len(v) > 5:
                    m[v] = off
        _ADMIN_NAMES.clear()
        _ADMIN_NAMES[key] = (re.compile("|".join(re.escape(v) for v in sorted(m, key=len, reverse=True))), m) if m else (None, m)
    rx, m = _ADMIN_NAMES[key]
    return rx.sub(lambda x: m[x.group(0)], body) if rx else body


def without_links(text):
    """Local demo messages never send external URLs, including stale model output."""
    text = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", str(text), flags=re.I)
    text = re.sub(r"(?:https?://|www\.)[^\s<>\"']+", "", text, flags=re.I)
    text = re.sub(r"(?<![@\w])(?:[a-z0-9-]+\.)+(?:com|net|org|dev|io|app|co|shop|store)\b(?:/[^\s<>\"']*)?", "", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def send_text(to, body):
    """Returns a message ref (WhatsApp message id, or 'f<feed id>' in mock mode) so replies can be tied back to it."""
    if ADMIN_TRANSLATE and (to in admins() or str(to).startswith("dashboard")) and (kget("adminlang:" + to) or "en") != "en":
        try:
            body = ADMIN_TRANSLATE(body, kget("adminlang:" + to))
        except Exception as e:
            log(f"admin translation failed, sent as written: {e!r}")
    body = without_links(admin_names(to, body))
    if not body:
        return ""
    feed(to, "text", body)
    ref = f"f{FEED[-1]['id']}"
    remember_msg(ref, body)
    tx = _tx(to)
    if not tx or to == "dashboard":  # the web dashboard is not a chat number: its answers only go to the page
        return ref
    name = "telegram" if tx is TG else "whatsapp"
    try:
        r = tx.send_text(to, body) or ref
        remember_msg(r, body)
    except Exception as e:
        problem(name, e)
        raise
    ok(name)
    return r


def send_doc(to, path, caption=""):
    if caption and (to in admins() or str(to).startswith("dashboard")):  # the caption follows the admin's language mode like every other message to him
        try:
            caption = admin_names(to, ADMIN_TRANSLATE(caption, kget("adminlang:" + to)) if ADMIN_TRANSLATE and (kget("adminlang:" + to) or "en") != "en" else caption)
        except Exception as e:
            log(f"admin caption translation failed: {e!r}")
    caption = without_links(caption)
    out = P("outbox") / f"{now().strftime('%H%M%S')}_{Path(path).name}"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(path, out)
    feed(to, "document", caption, file=out.name)
    if _tx(to):
        remember_msg(_tx(to).send_doc(to, str(out), caption), caption)


def send_image(to, jpeg_bytes, caption=""):
    caption = without_links(caption)
    out = P("outbox") / f"{now().strftime('%H%M%S%f')}_photo.jpg"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(jpeg_bytes)
    feed(to, "image", caption, file=out.name)
    remember_msg(f"f{FEED[-1]['id']}", caption)
    if _tx(to):
        remember_msg(_tx(to).send_image(to, str(out), caption), caption)


def send_template(to, name, params, preview):
    params = [without_links(value) for value in params]
    preview = without_links(preview)
    feed(to, "template", preview, tpl=name)
    ref = f"f{FEED[-1]['id']}"
    if str(to).startswith("tg:"):  # legacy ids are retained only for existing records
        return send_text(to, preview)
    return (TRANSPORT.send_template(to, name, params) or ref) if TRANSPORT else ref


TPL = {"order": "order_alert", "quote": "order_alert", "issue": "issue_alert", "limit": "limit_warning", "info": "admin_notice"}


CHEERS = tuple(f"{a} {b}" for a in (  # hand-written, formal, no emojis: one goes to the owners each time an order is completed (payment confirmed, receipt sent)
    "The order is complete.", "Payment has been verified.", "The receipt has been issued.", "The order is now closed.", "The customer was notified.",
) for b in (
    "Thank you for your work.", "Thank you for your diligence.", "We appreciate your attention.", "Well done to the whole team.", "Your efforts are appreciated.",
    "Thank you for your care.", "Excellent service, thank you.", "Thank you for acting promptly.", "The team is appreciated.", "Thank you for your dedication.",
))


def tell_admins(text):
    """A plain message to the owners (a WhatsApp owner only inside the 24-hour window; a Telegram owner always)."""
    for a in admins():
        try:
            if a.startswith("tg:") or time.time() - float(kget("seen:" + a, 0)) < 86400:
                send_text(a, text)
        except Exception as e:
            log(f"message to {a} failed: {e!r}")


def whatsapp_check(get=None):
    """'' when WhatsApp works (or is not set up, or there is no internet right now); otherwise the problem in plain words. An expired token tells the owners (at most once every 12 hours)."""
    tok, num = cfg("WHATSAPP_TOKEN"), cfg("PHONE_NUMBER_ID")
    if not (tok and num):
        return ""
    try:
        import httpx
        r = (get or httpx.get)(f"https://graph.facebook.com/v20.0/{num}", params={"fields": "display_phone_number"}, headers={"Authorization": "Bearer " + tok}, timeout=15)
    except Exception:
        return ""
    if r.status_code < 400:
        kset("wa_state", "")
        return ""
    try:
        err = r.json().get("error") or {}
    except ValueError:
        err = {}
    if r.status_code != 401 and err.get("code") != 190:
        return ""
    what, todo = friendly_error(RuntimeError(f"graph.facebook.com {r.status_code} code {err.get('code')} {err.get('message', '')}"))
    kset("wa_state", what)
    if time.time() - float(kget("wa_alert_ts", 0) or 0) > 12 * 3600:
        kset("wa_alert_ts", time.time())
        tell_admins(f"⚠️ WhatsApp is disconnected.\n\n*What happened:* {what}\n*What to do:* {todo}")
    return what


def cheer():
    """A short encouraging message to the owners, a different one from the last time."""
    import random
    last = kget("last_cheer")
    msg = random.choice([m for m in CHEERS if m != last])
    kset("last_cheer", msg)
    for a in admins():
        try:
            if a.startswith("tg:") or time.time() - float(kget("seen:" + a, 0)) < 86400:  # a WhatsApp owner outside the 24-hour window cannot be written to freely
                send_text(a, msg)
        except Exception as e:
            log(f"cheer to {a} failed: {e!r}")


def alert(kind, text, params, phone=None, oid=None, form=None):
    """Free-form text if the admin's 24h window is open, else the approved UTILITY template (docs/templates.md).
    phone/oid = who/what the alert is about: remembered so the admin can answer it by replying to it, or by writing right after it."""
    meta = json.dumps({"phone": phone, "oid": oid, "ts": time.time(), "title": re.sub(r"[*_]", "", ((text("dashboard") if callable(text) else text) or "").split("\n")[0])[:90]})
    delivered = 0
    for a in admins() or {"dashboard"}:
        try:  # one owner's channel being down (an expired WhatsApp token) must not stop the alert reaching the other owner
            body = (text(a) if callable(text) else text)
            body = body + "\n\n" + form(a) if form else body
            if a == "dashboard" or a.startswith("tg:") or time.time() - float(kget("seen:" + a, 0)) < 86400:
                ref = send_text(a, body)
            else:
                ref = send_template(a, TPL[kind], [re.sub(r"\s+", " ", str(p))[:200] for p in params], body)
        except Exception as e:
            log(f"alert to {a} failed: {e!r}")
            continue
        delivered += 1
        if phone or oid:
            for rf in (getattr(_tx(a), "refs", lambda: [])() or [ref]):  # every part of a long message can be replied to
                kset("ref:" + str(rf), meta)
            kset("lastalert", meta)
    return delivered  # how many owners really received it (0 = nobody: the caller can retry later)


def jget(k):
    try:
        return json.loads(kget(k) or "null") or {}
    except ValueError:
        return {}


def remember_focus(oid=None, phone=None):
    """The admin's 'current order': sticky until the admin names/replies about another one."""
    kset("afocus", json.dumps({"oid": oid, "phone": phone, "ts": time.time()}))


def order_row(oid):
    return next((r for r in orders() if str(r["order_id"]) == str(oid)), None)


def latest_order_of(phone):
    rows = [r for r in orders() if r["customer_number"] == phone]
    return rows[-1] if rows else None


# ---------------------------------------------------------------- currency
FX_TTL = 600  # seconds a fetched rate is reused (10 min)


def _live_rate(cur):
    """Free sources, no key. Live: Yahoo Finance chart (per-minute, unofficial). Daily fallbacks: ECB via frankfurter, open.er-api, Turkish central bank.
    The live value is only trusted if it is within 3% of a daily source (a broken or fat-fingered quote can never reach a customer)."""
    import httpx, re
    h = {"User-Agent": "Mozilla/5.0"}
    daily = None
    for name, get in (
        ("ECB/frankfurter", lambda: httpx.get(f"https://api.frankfurter.dev/v1/latest?base={cur}&symbols=TRY", timeout=6).json()["rates"]["TRY"]),
        ("open.er-api", lambda: httpx.get(f"https://open.er-api.com/v6/latest/{cur}", timeout=6).json()["rates"]["TRY"]),
        ("TCMB", lambda: float(re.search(rf'CurrencyCode="{cur}".*?<ForexSelling>([\d.]+)</ForexSelling>', httpx.get("https://www.tcmb.gov.tr/kurlar/today.xml", timeout=6).text, re.S).group(1))),
    ):
        try:
            daily = (float(get()), name)
            break
        except Exception:
            continue
    try:
        live = float(httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{cur}TRY=X?interval=1m&range=1d", headers=h, timeout=6).json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
        if not daily or abs(live / daily[0] - 1) < 0.03:
            return live, "live (Yahoo Finance)"
    except Exception:
        pass
    return daily if daily else (None, "")


def fx_info(cur):
    """(TL per 1 unit of cur, source). The owner's /rate wins but only for 24 h (a forgotten test rate can't linger); else a free live rate cached 10 min; else (None, '') and nothing is guessed."""
    cur = cur.upper()
    if cur in ("TRY", "TL"):
        return 1.0, "TL"
    o = jget("rate:" + cur)
    if isinstance(o, dict) and time.time() - o.get("ts", 0) < 86400:  # an old plain-number override counts as expired
        return float(o["v"]), "your /rate (valid 24 h)"
    c = jget("fx:" + cur)
    c = c if isinstance(c, dict) else {}
    if c and time.time() - c.get("ts", 0) < FX_TTL:
        return c["v"], c["src"]
    v, src = _live_rate(cur)
    if v:
        kset("fx:" + cur, json.dumps({"v": v, "src": src, "ts": time.time()}))
        return v, src
    return (c["v"], c["src"] + ", stale") if c else (None, "")  # last known beats nothing; with no rate at all the caller shows TL only


def fx(cur):
    return fx_info(cur)[0]


def catalog_photo(pid):
    """Resolve an existing local catalog image; product ids cannot escape the catalog."""
    base = (ROOT / "dl/data/catalog").resolve()
    folder = (base / str(pid)).resolve()
    if folder.parent != base:
        raise ValueError("invalid catalog product id")
    return next((p for p in sorted(folder.glob("*"))
                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".avif"}
                 and p.resolve().is_relative_to(base)), None)


def product_jpeg(p):
    """JPEG bytes from a local photo only. Missing photos never trigger a download."""
    import io
    from PIL import Image
    photo = catalog_photo(p["id"])
    if photo is None:
        raise FileNotFoundError(f"No local photo for product {p['id']}")
    with Image.open(photo) as source:
        im = source.convert("RGB")
    im.thumbnail((1280, 1280))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def html_unescape(s):
    import html
    return html.unescape(s)


# ---------------------------------------------------------------- orders.csv
OCOLS = ["order_id", "timestamp", "customer_number", "customer_name", "items", "quantities", "unit_prices", "subtotal", "custom_items",
         "custom_price", "total", "delivery_type", "delivery_area", "delivery_time", "delivery_fee", "payable", "paid", "sender_name", "sender_phone",
         "recipient_name", "recipient_phone", "address", "delivery_date", "notes", "change_request", "status", "custom_lines"]
# custom_lines: the custom items as JSON [{"name": "1 white rose", "price": "500"}], each with its own name and price ('' = the team has not priced it yet), like any catalog line.
#   custom_items (the names) and custom_price (their sum, '' while one is unpriced) are derived from it by sync_custom(), so everything that reads those two keeps working.
# items / quantities / unit_prices: the catalog lines, always with their price. subtotal = sum of those lines.
# custom_items: what the catalog cannot price (text); custom_price: the owner's price for it. total = subtotal + custom_price (the GOODS; '' while the custom part waits for its price).
# delivery_fee: 0 for normal delivery / pick-up, the jet fee for jet delivery ('' while unknown). payable = total + delivery_fee: what the customer pays.
# paid: what the owner confirmed as received. change_request: a customer request to change / cancel a locked order, JSON, applied only after the owner approves.
# status: quote -> quoted -> pending (priced, waiting for the owner's first confirmation) -> awaiting_payment (payment details sent) -> payment_review (proof received) -> confirmed.
_lock = threading.Lock()


def parse_amount(s):
    """'77.85' -> 77.85, '77.850' / '77,850' -> 77850, '1.234,50' -> 1234.5, '214200' -> 214200. Money is never guessed by a model."""
    s = re.sub(r"[^\d.,]", "", str(s))
    if not s:
        raise ValueError("no amount")
    if "." in s and "," in s:
        dec = max(s.rfind("."), s.rfind(","))
        return float(re.sub(r"[.,]", "", s[:dec]) + "." + s[dec + 1:])
    for sep in ".,":
        if sep in s:
            parts = s.split(sep)
            if len(parts) > 2 or len(parts[-1]) == 3:  # thousands separator
                return float("".join(parts))
            return float(parts[0] + "." + parts[1])
    return float(s)


from decimal import Decimal, ROUND_HALF_UP


def D(x):
    """Money is Decimal, never float: str() of a float is its shortest exact decimal form."""
    return x if isinstance(x, Decimal) else Decimal(str(x))


def mstr(x):
    """Plain decimal string for the orders file: '214200' or '214200.50', never '2.142e+05' (what f"{x:g}" would give above 6 digits)."""
    s = format(D(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
    return s[:-3] if s.endswith(".00") else s


def fmt_tl(x):
    d = D(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return (f"{d:,.0f}" if d == d.to_integral_value() else f"{d:,.2f}") + " TL"


def ssum(lines):
    """Exact total of [(product, qty)] at current catalog prices."""
    return sum((D(unit_price(p)) * n for p, n in lines), Decimal(0))


def _nm(name):
    return str(name).replace("; ", ", ")


def row_lines(row):
    """[(name, qty, unit price)] of an order's catalog lines; [] if none; None if the columns do not line up."""
    try:
        if not row["items"] and not row["quantities"]:
            return []
        names, qs, us = row["items"].split("; "), row["quantities"].split("; "), row["unit_prices"].split("; ")
        return list(zip(names, [int(q) for q in qs], [D(u) for u in us])) if len(names) == len(qs) == len(us) else None
    except Exception:
        return None


def row_problems(row):
    """Every money invariant of one order; [] when the maths is right: subtotal == sum of the lines; total == subtotal + custom price (empty while the custom part is unpriced)."""
    L = row_lines(row)
    if L is None:
        return ["items / quantities / unit_prices do not line up"]
    out = []
    if any(q < 1 for _, q, _ in L):
        out.append("quantity below 1")
    sub = sum((q * u for _, q, u in L), Decimal(0))
    if D(row["subtotal"] or 0) != sub:
        out.append(f"subtotal {row['subtotal']} != sum of the lines {mstr(sub)}")
    if row["custom_items"]:
        want = mstr(sub + D(row["custom_price"])) if row["custom_price"] else ""
    else:
        want = mstr(sub)
        if row["custom_price"]:
            out.append("custom price without custom items")
    if row["total"] != want:
        out.append(f"total {row['total']!r} should be {want!r}")
    if row["status"] in ("confirmed", "awaiting_payment", "payment_review") and not row["total"]:
        out.append(f"{row['status']} without a total")
    want_pay = mstr(D(row["total"]) + D(row["delivery_fee"])) if row["total"] and row["delivery_fee"] != "" else ""
    if row["payable"] != want_pay:
        out.append(f"payable {row['payable']!r} should be {want_pay!r}")
    return out


ADJ_RE = re.compile(r" \| owner adjustment: (-?[\d.]+)")


def adjustment(r):
    """What the owner added (service fee, +) or took off (discount, -) the amount to pay when he confirmed the order; Decimal(0) if nothing."""
    m = ADJ_RE.search(r["notes"] or "")
    return D(m.group(1)) if m else Decimal(0)


def set_payable(oid, amount):
    """At the first confirmation the owner may name the TOTAL TO PAY himself (higher = a service fee line, lower = a discount line in the customer's message).
    The difference to items + delivery is stored in the notes; returns the row, or None if the order is not waiting for its first confirmation or the delivery fee is unknown."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or r["status"] not in OPEN or not r["total"] or r["delivery_fee"] == "":
            return None
        base = D(r["total"]) + D(r["delivery_fee"])
        adj = (D(str(amount)) - base).quantize(Decimal("0.01"))
        if D(str(amount)) <= 0:
            return None
        r["notes"] = ADJ_RE.sub("", r["notes"]) + (f" | owner adjustment: {mstr(adj)}" if adj else "")
        recompute_row(r)
        _write(rows)
        return dict(r)


def custom_lines(r):
    try:
        L = json.loads(r.get("custom_lines") or "[]")
    except ValueError:
        L = []
    return [{"name": str(x.get("name", "")), "price": str(x.get("price", ""))} for x in L if isinstance(x, dict) and x.get("name")]


def sync_custom(r):
    """Keep custom_lines (name + price per item), custom_items and custom_price in step. Whoever changed the text or the aggregate price last wins:
    new names become unpriced lines, a changed aggregate price is spread over the unpriced lines (the whole price of a single item goes to that item)."""
    L = custom_lines(r)
    text = str(r["custom_items"] or "")
    if text != "; ".join(x["name"] for x in L):
        old = {fold(x["name"]): x["price"] for x in L}
        names = [p.strip() for p in text.split("; ") if p.strip()]
        if len(names) > 1 and not L and r["custom_price"] and not any(old.values()):
            names = [text]  # an older order: several items with ONE price stay one line (the price is not split)
        L = [{"name": n, "price": old.get(fold(n), "")} for n in names]
    agg = str(r["custom_price"] or "")
    if L and agg and not (all(x["price"] for x in L) and sum(D(x["price"]) for x in L) == D(agg)):
        pend = [x for x in L if not x["price"]] or L
        rest = D(agg) - sum((D(x["price"]) for x in L if x["price"] and x not in pend), Decimal(0))
        share = (rest / len(pend)).quantize(Decimal("0.01"))
        for i, x in enumerate(pend):
            x["price"] = mstr(share if i < len(pend) - 1 else rest - share * (len(pend) - 1))
    if not L:
        r["custom_lines"] = r["custom_items"] = r["custom_price"] = ""
        return r
    r["custom_lines"] = json.dumps(L, ensure_ascii=False)
    r["custom_items"] = "; ".join(x["name"] for x in L)
    r["custom_price"] = mstr(sum((D(x["price"]) for x in L), Decimal(0))) if all(x["price"] for x in L) else ""
    return r


def custom_rows(r):
    """[(name, price as Decimal or None while unpriced)] of an order's custom items (works on older rows too)."""
    t = sync_custom(dict(r))
    return [(x["name"], D(x["price"]) if x["price"] else None) for x in custom_lines(t)]


def recompute_row(row):
    """After any change: subtotal is exactly the sum of the lines and the total follows it (no discounts: a catalog order's total IS its subtotal)."""
    sync_custom(row)
    L = row_lines(row)
    if L is not None:
        sub = sum((q * u for _, q, u in L), Decimal(0))
        row["subtotal"] = mstr(sub)
        if row["custom_items"]:
            row["total"] = mstr(sub + D(row["custom_price"])) if row["custom_price"] else ""
        else:
            row["custom_price"] = ""
            row["total"] = mstr(sub)
        row["payable"] = mstr(D(row["total"]) + D(row["delivery_fee"]) + adjustment(row)) if row["total"] and row["delivery_fee"] != "" else ""
    return row


def order_summary(r):
    """One readable block for alerts: every item with its price, the custom part, the total."""
    L = row_lines(r) or []
    parts = [f"{q} x {nm} @ {fmt_tl(u)} = {fmt_tl(q * u)}" for nm, q, u in L]
    for nm, pr in custom_rows(r):
        parts.append(f"{nm}" + (f" = {fmt_tl(pr)}" if pr is not None else " (needs your price)"))
    tail = (f" | Subtotal {fmt_tl(r['subtotal'] or 0)}" if L else "") + (f" | TOTAL {fmt_tl(r['total'])}" if r["total"] else " | TOTAL: waiting for your price for the custom part")
    if r["delivery_type"] or r["delivery_fee"] != "":
        tail += f" | Delivery: {r['delivery_type'] or 'normal'}" + (f" {dtime(r)}" if r["delivery_time"] else "") + (f" to {_json('delivery.json')['areas'].get(r['delivery_area'], {}).get('label', {}).get('en', custom_area(r) or r['delivery_area'])}" if r["delivery_area"] else "") \
                + (f" = {fmt_tl(r['delivery_fee'])}" if r["delivery_fee"] != "" else " (fee not known yet)") \
                + (f" | {'Service fee' if adjustment(r) > 0 else 'Discount'} {fmt_tl(abs(adjustment(r)))}" if adjustment(r) else "") + (f" | TO PAY {fmt_tl(r['payable'])}" if r["payable"] else "")
    return "; ".join(parts) + tail


def order_facts(r):
    """EVERYTHING about one order as plain lines for the customer agent: each item with unit price and line total, custom part, subtotal, delivery (type, area, time, fee), service fee / discount, total, amount to pay, paid so far and the balance, and the form details. So any question about it is answered from facts, never from memory."""
    out = [order_block(r, note=False)]
    paid, pay = D(r["paid"] or 0), D(r["payable"] or 0)
    if paid > 0:
        out.append(f"Paid so far: {fmt_tl(paid)}" + (f" · balance {fmt_tl(pay - paid)}" if pay else ""))
    note = re.findall(r"\| note: ([^|]*)", r["notes"] or "")
    det = [f"sender {r['sender_name'] or '-'} {r['sender_phone'] or ''}".rstrip(), f"recipient {r['recipient_name'] or '-'} {r['recipient_phone'] or ''}".rstrip(), f"address {r['address'] or '-'}",
           f"date {r['delivery_date'] or '-'} {dtime(r)}".rstrip(), f"card note {note[-1].strip()}" if note else "no card note"]
    out.append("Form: " + " · ".join(det))
    if r["delivery_type"] == "jet" and r["delivery_time"]:  # whether the requested jet time still waits for the team (the recap said "team approval required")
        out.append(f"Jet time {dtime(r)}: " + ("ACCEPTED: the team confirmed this order with that time." if r["status"] in LOCKED else "only REQUESTED: the team approves it when it confirms the order."))
    miss = form_missing(r) + delivery_missing(r)
    prob = delivery_calc(r)["problems"] if r["status"] in OPEN else []
    if r["status"] in OPEN and (miss or prob or r["custom_items"] and not r["custom_price"]):
        out.append("FINAL PRICE NOT READY: " + ("; ".join(["missing " + ", ".join(miss)] * bool(miss) + prob) or "the team must price the custom part") + ". Give ONLY the goods total of the items; never a 'total to pay', never a delivery fee, never say the team will confirm it.")
    g = kget(f"gift:{r['order_id']}")
    if g:
        out.append(f"FREE GIFT offered by the team: {g} (complimentary, no charge, not part of the total; say nothing more about it than this).")
    return "\n".join(out)


def fx_cached(cur):
    """The rate we already hold (the owner's, valid 24 h, or the last live one); never a network call."""
    cur = cur.upper()
    o = jget("rate:" + cur)
    if isinstance(o, dict) and time.time() - o.get("ts", 0) < 86400:
        return float(o["v"])
    c = jget("fx:" + cur)
    return float(c["v"]) if isinstance(c, dict) and c.get("v") else None


def cur_note(r):
    """The customer asked for prices in another currency (dollars...): the owner sees the same amount in it next to the lira, so what the customer was told and what the owner sees agree."""
    cur = (kget("cur:" + str(r.get("customer_number"))) or "").upper()
    if cur in ("", "TRY", "TL") or time.time() - float(kget("curts:" + str(r.get("customer_number")), 0) or 0) > 12 * 3600:  # the currency a customer used in the LAST 12 hours, not forever
        return ""
    try:
        amt, rate = float(D(r.get("payable") or r.get("total") or 0)), fx_cached(cur)
    except Exception:
        return ""
    return f"💱 The customer asked for {cur}: about {amt / rate:,.0f} {cur} (1 {cur} = {rate:,.2f} TL, approximate)" if amt and rate else ""


def hours_note(r):
    """For the owner only: the asked delivery time lies outside the working hours (WORK_FROM / WORK_TO, default 08:00-23:00). The owner decides, so this is a note, never a refusal."""
    m = re.search(r"(\d{1,2})[:.](\d{2})", r["delivery_time"] or "")
    if not m:
        return ""
    lo, hi = cfg("WORK_FROM", "08:00"), cfg("WORK_TO", "23:00")
    mins = int(m.group(1)) * 60 + int(m.group(2))
    f = lambda hhmm: int(hhmm[:2]) * 60 + int(hhmm[3:5])
    if f(lo) <= mins <= f(hi):
        return ""
    return f"🕒 *Delivery time {int(m.group(1)):02d}:{m.group(2)} is OUTSIDE the working hours ({lo}–{hi}).* Your call."


def approval_notes(r):
    """What only the owner can decide, written into his order confirmation (the card, the order request, the order block): a JET time is a REQUEST he approves by confirming (never asked separately); an area that is not in the price list needs his fee."""
    out = []
    if (r["delivery_type"] or "") == "jet":
        out.append(f"⏱ JET delivery requested for {r['delivery_date'] or '?'} {dtime(r) or '(time not given)'}: confirm the order ONLY if that time is possible.")
    if custom_area(r):
        out.append(f"📍 Area '{custom_area(r)}' is not in the price list: set the delivery fee yourself.")
    return out


def order_block(r, note=True):
    """order_summary as a readable multi-line block for the owner's chat (templates keep the one-line order_summary: they take no line breaks)."""
    L = row_lines(r) or []
    out = [f"• {q} x {nm} @ {fmt_tl(u)} = {fmt_tl(q * u)}" for nm, q, u in L]
    for nm, pr in custom_rows(r):
        out.append(f"• {nm}" + (f" - {fmt_tl(pr)}" if pr is not None else " (needs your price)"))
    for s in order_summary(r).split(" | ")[1:]:
        if s.startswith("TOTAL: waiting"):
            out.append("💰 " + s.replace("TOTAL: waiting", "Total: waiting"))
        elif s.startswith("TOTAL "):
            out.append("💰 *" + s.replace("TOTAL", "Total", 1) + "*")
        elif s.startswith("TO PAY"):
            out.append("💳 *" + s.replace("TO PAY", "To pay", 1) + "*")
        else:
            out.append(("🚚 " if s.startswith("Delivery") else "⚖️ " if s.startswith(("Service fee", "Discount")) else "") + s)
    out += approval_notes(r)
    n = cur_note(r)
    if n:
        out.append(n)
    for m_ in re.findall(r"\| memo: ([^|]*)", r["notes"] or ""):
        out.append(f"📝 Customer note: {m_.strip()}")
    if note and hours_note(r):
        out.append(hours_note(r))
    return "\n".join(out)


def card(emoji, title, who="", *blocks, todo=""):
    """Every message the owner gets looks the same: a bold title, who it is about, short blocks with blank lines between them, and the ONE thing to do in bold after a divider."""
    bar = "━━━━━━━━━━━━"
    parts = [f"{emoji} *{title}*" + (f"\n👤 {who}" if who else ""), bar, *[b.strip() for b in blocks if b and b.strip()]]
    if todo:
        parts += [bar, f"👉 *{todo}*"]
    return "\n\n".join(parts)


def _items_text(r):
    return "; ".join([f"{q} x {nm}" for nm, q, _ in (row_lines(r) or [])] + [nm for nm, _ in custom_rows(r)])


def active_line(r):
    return f"#{r['order_id']} [{r['status']}] {_items_text(r)} - " + (fmt_tl(r["total"]) if r["total"] else "price pending")


def _json(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _place_index():
    """Every spelling that points to a delivery area: the landmarks of data/landmarks.json (universities, hotels, hospitals... with their short forms like NEU / YDU / EMU) and the city names of data/delivery.json: [(folded name, area key, is_landmark)]."""
    rows = [(fold(n), o["area"], True) for o in _json("landmarks.json")["landmarks"] for n in o["names"] if len(fold(n)) >= 3]
    rows += [(fold(n), k, False) for k, a in _json("delivery.json")["areas"].items() for n in a["names"]]
    return sorted(set(rows), key=lambda x: (-len(x[0]), x[0], x[1]))  # same length: by name, so the order never depends on the process


def _fuzzy_area(t):
    """The area of the closest place name in the text, tolerating typos ("near est university", "kyrenya"): windows of 1-6 words against names of 6+ letters, ratio >= 0.86. Short forms (NEU, YDU) never match fuzzily."""
    import difflib
    toks = re.findall(r"[^\W_]+", t)
    best = (0.0, None)
    for n in range(1, 7):
        for i in range(len(toks) - n + 1):
            w = " ".join(toks[i:i + n])
            if len(w) < 5:
                continue
            for nm, key, _ in _place_index():
                if len(nm) < 6 or abs(len(nm) - len(w)) > 4:
                    continue
                sm = difflib.SequenceMatcher(None, w, nm)
                if sm.real_quick_ratio() >= 0.86 and sm.quick_ratio() >= 0.86:
                    r_ = sm.ratio()
                    if r_ >= 0.86 and (r_, len(nm)) > (best[0], 0):
                        best = (r_, key)
    return best[1]


GENERIC_WORDS = {"university", "universitesi", "hotel", "hospital", "hastanesi", "resort", "college", "school", "street", "road", "royal", "beach", "palace", "the", "and", "dorm", "yurt"}


def partial_landmarks(text):
    """The landmarks the customer named only in part ("near east" for "Near East University", "merit royal" for "Merit Royal Hotel"): the longest run of 2 or more words of the text that is a run of words of a saved landmark name.
    Compared with the shop's own names only. Several landmarks may share the run (the Near East hospital and university) but only when the plainest of them (the shortest name holding those words) is in ONE delivery area: otherwise [] (nobody guesses)."""
    toks = re.findall(r"[^\W_]+", fold(text or ""))
    lms = [(o, [re.findall(r"[^\W_]+", fold(n)) for n in o["names"]]) for o in _json("landmarks.json")["landmarks"]]
    for n in range(min(6, len(toks)), 1, -1):
        for i in range(len(toks) - n + 1):
            run = toks[i:i + n]
            if all(w in GENERIC_WORDS or len(w) < 3 for w in run):
                continue
            hit = [(min(len(nm) for nm in names if any(nm[j:j + n] == run for j in range(len(nm) - n + 1))), o) for o, names in lms if any(any(nm[j:j + n] == run for j in range(len(nm) - n + 1)) for nm in names)]
            if hit:
                short = min(k for k, _ in hit)  # "near east" is the University (3 words), not the Near East Yeniboğaziçi College (4 words, another town): the plainest name that contains the words wins
                best = [o for k, o in hit if k == short]
                return best if len({o["area"] for o in best}) == 1 else []
    return []


def place_info(text):
    """Where a place is, ONLY as far as the shop's own tables say (data/landmarks.json: universities, hotels, hospitals... with their short forms, and the cities of data/delivery.json): the matching places
    with the delivery area they belong to. [] = the shop's tables do not know it: nobody may guess."""
    t = fold(text or "")
    areas = _json("delivery.json")["areas"]
    out = []
    for o in _json("landmarks.json")["landmarks"]:
        for n in o["names"]:
            if len(fold(n)) >= 3 and re.search(rf"(?<![^\W_]){re.escape(fold(n))}(?![^\W_])", t):
                out.append({"place": o["names"][0], "type": o.get("type", "place"), "area": areas.get(o["area"], {}).get("label", {}).get("en", o["area"])})
                break
    if not out:
        areas_ = areas
        out += [{"place": o["names"][0], "type": o.get("type", "place"), "area": areas_.get(o["area"], {}).get("label", {}).get("en", o["area"])} for o in partial_landmarks(text)[:4]]  # "near east" = Near East University
    if not out:  # "merit hotel" for "Merit Royal Hotel": every word (2 or more, 4+ letters) of the question is a word of ONE landmark's name
        words = {w for w in re.findall(r"[^\W_]+", t) if len(w) >= 4 and w not in ("where", "located", "location", "nerede", "what", "which")}
        cands = [o for o in _json("landmarks.json")["landmarks"] if len(words) >= 2 and any(words <= set(re.findall(r"[^\W_]+", fold(n))) for n in o["names"])]
        if 1 <= len(cands) <= 4:  # several places share those words (the Merit hotels): all of them are returned, never one picked at random
            out += [{"place": o["names"][0], "type": o.get("type", "place"), "area": areas.get(o["area"], {}).get("label", {}).get("en", o["area"])} for o in cands]
    if not out:
        for k, a in areas.items():
            if any(re.search(rf"(?<![^\W_]){re.escape(fold(n))}(?![^\W_])", t) for n in a["names"]):
                out.append({"place": a["label"]["en"], "type": "town / delivery area", "area": a["label"]["en"]})
    return out


def detect_area(address):
    """The delivery area key for an address. 1) a CITY written by the customer (outside any landmark name) wins: no landmark lookup. 2) no city: a landmark / short form in the text (NEU, YDU, Merit Royal...). 3) the closest spelling of either (typos). Else None.
    "Girne American University" is a landmark of ITS zone, so the "Girne" inside its name does not count as a written city. Tables: data/delivery.json, data/landmarks.json; the fee and minimum come from the area, never from the text."""
    t = fold(address or "")
    lm, city = [], []
    for nm, key, is_lm in _place_index():
        for m in re.finditer(rf"(?<![^\W_]){re.escape(nm)}(?![^\W_])", t):
            (lm if is_lm else city).append((len(nm), key, m.span()))
    city = [c for c in city if not any(l[2][0] <= c[2][0] and c[2][1] <= l[2][1] for l in lm)]  # a city word that is part of a landmark's name is not a written city
    spec = lambda k: 1 if k.endswith("_outer") or k == "remote" else 0  # a village named next to its district is the more specific place ("Lapta, Kyrenia" is the outer area)
    for group in (city, lm):
        if group:
            return max(group, key=lambda x: (spec(x[1]), x[0], x[1]))[1]
    part = partial_landmarks(t)
    if part:
        return part[0]["area"]
    return _fuzzy_area(t) if t.strip() else None


def place_text_left(address):
    """The words of an address that are NOT a known place name (city or landmark): what the customer added to point to the exact spot."""
    t = fold(address or "")
    for nm, _, _ in _place_index():
        t = re.sub(rf"(?<![^\W_]){re.escape(nm)}(?![^\W_])", " ", t)
    return t


def _hhmm(mins):
    return f"{mins // 60:02d}:{mins % 60:02d}"


FOREIGN = re.compile(r"\b(?:sharjah|dubai|abu dhabi|ajman|fujairah|ras al khaimah|uae|emirates|saudi|riyadh|jeddah|mecca|qatar|doha|kuwait|bahrain|manama|oman|muscat|iraq|baghdad|iran|tehran|israel|tel aviv|jerusalem|jordan|amman|lebanon|beirut|syria|damascus|egypt|cairo|libya|tunisia|morocco|algeria|london|england|britain|manchester|birmingham|scotland|ireland|dublin|germany|berlin|munich|france|paris|spain|madrid|barcelona|italy|rome|milan|netherlands|amsterdam|belgium|brussels|switzerland|zurich|austria|vienna|sweden|stockholm|norway|oslo|denmark|poland|warsaw|czech|prague|hungary|budapest|romania|bucharest|bulgaria|sofia|serbia|belgrade|ukraine|kyiv|kiev|russia|moscow|st petersburg|belarus|minsk|georgia|tbilisi|armenia|azerbaijan|baku|kazakhstan|uzbekistan|pakistan|india|delhi|mumbai|china|beijing|shanghai|japan|tokyo|korea|seoul|thailand|bangkok|singapore|australia|sydney|canada|toronto|usa|america|new york|los angeles|miami|chicago|brazil|mexico|argentina|nigeria|lagos|south africa|kenya|istanbul|ankara|izmir|antalya|bursa|mersin|adana|konya|turkey|turkiye|limassol|larnaca|paphos|south cyprus|southern cyprus|greek cyprus|nicosia south|greece|athens|crete|another country|abroad|overseas|international)\b")  # places a North Cyprus shop does not deliver to
CUSTOM_AREA = "other:"  # a place the customer named that is not among the saved ones: what they said IS the area, the team sets its fee


def custom_area(r):
    a = str(r["delivery_area"] or "")
    return a[len(CUSTOM_AREA):] if a.startswith(CUSTOM_AREA) else ""


def non_same_day(r):
    """The catalog lines of this order whose item is NOT delivered the same day (same_day False on the shop page, or set so by the owner)."""
    by = {_nm(n["name"]): n for n in graph()[0].values() if n["type"] == "product"}
    return [nm for nm, _, _ in (row_lines(r) or []) if by.get(nm) and by[nm].get("same_day") is False]


def same_day_problems(r):
    """Items that are NOT delivered the same day cannot be ordered for today: the order must be for tomorrow or later (a catalog rule, on even when the time rules are switched off)."""
    try:
        d = datetime.fromisoformat(str(r["delivery_date"])[:10]).date()
    except ValueError:
        return []
    late = non_same_day(r) if d == now().date() else []
    return [policy("not_same_day", "en", item=", ".join(late)) + f" TELL THE CUSTOMER the order must be for tomorrow ({d + timedelta(days=1):%d.%m.%Y}) or later, or offer another item."] if late else []


def today_problems(r, a):
    """Delivery for TODAY: not-same-day items need tomorrow; the standard run of the area has already left (then tomorrow's standard delivery, or a jet REQUEST that only the team can approve: no jet time is ever checked or promised by code)."""
    try:
        d = datetime.fromisoformat(str(r["delivery_date"])[:10]).date()
    except ValueError:
        return []
    n = now()
    if d != n.date():
        return []
    if same_day_problems(r):
        return same_day_problems(r)
    if os.environ.get("DATE_RULES", "1") == "0":
        return []
    if (r["delivery_type"] or "normal") == "normal":
        h, m = (int(x) for x in a["departure"].split(":"))
        if n.hour * 60 + n.minute >= h * 60 + m:
            return [f"today's standard delivery for {a['label']['en']} already left at {a['departure']}: SAY WHEN NORMAL DELIVERY GOES for {a['label']['en'].split(' (')[0]} (every day it leaves at {a['departure']}, free) and offer tomorrow's standard delivery ({(d + timedelta(days=1)):%d.%m.%Y}, leaving at {a['departure']}); " + ("it is after 22:00: ALL DELIVERIES ARE DONE FOR TODAY (no jet either): say that plainly and offer tomorrow, normal at its usual time or a jet request for tomorrow (08:00-22:00), never a jet for today" if n.hour >= 22 else "if the customer still wants it today, jet delivery is only a REQUEST the team approves (never say it is possible, never name times or windows)")]
    return []


def date_problems(date_s):
    """Orders are taken for today and at most MAX_DAYS_AHEAD (2) days ahead: ordered Tuesday morning, the latest delivery is Thursday evening. An unreadable date is left to the owner."""
    if os.environ.get("DATE_RULES", "1") == "0":
        return []
    try:
        d = datetime.fromisoformat(str(date_s)[:10]).date()
    except ValueError:
        return []
    today, mx = now().date(), int(cfg("MAX_DAYS_AHEAD", "2"))
    if d < today:
        return [f"the delivery date {d:%d.%m.%Y} is in the past: ask for a new date"]
    if d > today + timedelta(days=mx):
        return [f"we take orders for at most {mx} days ahead: the latest delivery date is {today + timedelta(days=mx):%A %d.%m.%Y}. Ask for a date up to then, or to order again closer to the day"]
    return []


def delivery_options(r, lang):
    """The follow-up to the personal form: delivery type, date, and (for jet/pick-up) time."""
    if r["delivery_type"] == "pickup":
        return {"tr": "Mağazadan teslim alma için hangi tarih ve saat uygundur?",
                "ru": "На какую дату и время оформить самовывоз из магазина?",
                "en": "What date and time would you like for pick-up at the shop?"}[lang]
    areas = _json("delivery.json")["areas"]
    key = detect_area(r["address"]) or r["delivery_area"]
    a = areas.get(key)
    if not a and custom_area(r):  # a place that is not in the price list: the same three choices, the team sets the fee and the schedule
        pl = custom_area(r)
        return {"tr": f"{pl} için teslimat\n1) Normal teslimat: ücreti ve çıkış saatini ekip belirler.\n2) Jet teslimat: tarihi ve istediğiniz saati yazın; ücreti ve talebi ekip onaylar.\n3) Mağazadan teslim alma.\nLütfen teslimat türünü ve tarihi seçin; jet veya mağazadan teslim alma için saati de belirtin.",
                "ru": f"Доставка: {pl}\n1) Обычная доставка: стоимость и время выезда подтвердит команда.\n2) Экспресс-доставка: укажите дату и желаемое время; стоимость и запрос подтвердит команда.\n3) Самовывоз из магазина.\nВыберите тип доставки и дату; для экспресс-доставки или самовывоза укажите время.",
                "en": f"Delivery to {pl}\n1) Normal delivery: the team confirms the fee and the departure time.\n2) Jet delivery: choose a date and tell me the time you would like; the team confirms the fee and the request.\n3) Pick-up at the shop.\nPlease choose the delivery type and date; include a time for jet or pick-up."}[lang]
    if not a:
        return {"tr": "Teslimat mı, mağazadan teslim alma mı? Teslimat için şehir / ilçeyi ve istediğiniz tarihi yazın. Jet teslimat veya mağazadan teslim alma için saati de belirtin.",
                "ru": "Доставка или самовывоз? Для доставки укажите город и желаемую дату. Для экспресс-доставки или самовывоза укажите также время.",
                "en": "Would you like delivery or pick-up? For delivery, tell me the town and date. For jet delivery or pick-up, include the time."}[lang]
    lab = a["label"].get(lang) or a["label"]["en"]
    jf = f"{fmt_tl(a['jet_fee'])}" if a["jet_fee"] is not None else None
    gone = bool(today_problems(dict(r, delivery_type="normal"), a)) and "left at" in (today_problems(dict(r, delivery_type="normal"), a) or [""])[0]
    fee_en, fee_tr, fee_ru = (f", +{jf}", f", +{jf}", f", +{jf}") if jf else (", the team confirms the fee", ", ücreti ekip belirler", ", стоимость подтвердит команда")
    if gone:  # today's standard run has left: say so and offer tomorrow, or a jet request
        return (f"Delivery to {lab} 🌸\nToday's standard delivery already left at {a['departure']}.\n1) Tomorrow: standard delivery (free, leaves at {a['departure']}).\n2) Jet delivery: choose your date and tell me the time you would like; the team must approve the request{fee_en}.\n3) Pick-up at the shop.\nPlease choose the delivery type and date; include a time for jet or pick-up.",
                f"{lab} için teslimat 🌸\nBugünkü standart teslimat saat {a['departure']}'te çıktı.\n1) Yarın standart teslimat: ücretsiz, {a['departure']}'te çıkar.\n2) Jet teslimat: tarihi ve istediğiniz saati yazın; talebi ekibin onaylaması gerekir{fee_tr}.\n3) Mağazadan teslim alma.\nLütfen teslimat türünü ve tarihi seçin; jet veya mağazadan teslim alma için saati de belirtin.",
                f"Доставка: {lab} 🌸\nСегодняшний стандартный выезд уже был в {a['departure']}.\n1) Стандартная доставка завтра: бесплатно, выезд в {a['departure']}.\n2) Экспресс-доставка: укажите дату и желаемое время; запрос должна подтвердить команда{fee_ru}.\n3) Самовывоз из магазина.\nВыберите тип доставки и дату; для экспресс-доставки или самовывоза укажите время.")[2 if lang == "ru" else 1 if lang == "tr" else 0]
    en = (f"Delivery to {lab} 🌸\n1) Normal delivery: free, our standard schedule for your area leaves at {a['departure']}.\n2) Jet delivery: choose a date and tell me the time you would like; the team must approve the request{fee_en}.\n3) Pick-up at the shop.\nPlease choose the delivery type and date; include a time for jet or pick-up.")
    tr = (f"{lab} için teslimat 🌸\n1) Normal teslimat: ücretsiz, bölgeniz için standart çıkış saati {a['departure']}.\n2) Jet teslimat: bir tarih ve istediğiniz saati yazın; talebi ekibin onaylaması gerekir{fee_tr}.\n3) Mağazadan teslim alma.\nLütfen teslimat türünü ve tarihi seçin; jet veya mağazadan teslim alma için saati de belirtin.")
    ru = (f"Доставка: {lab} 🌸\n1) Обычная доставка: бесплатно, стандартный выезд для вашего района в {a['departure']}.\n2) Экспресс-доставка: укажите дату и желаемое время; запрос должна подтвердить команда{fee_ru}.\n3) Самовывоз из магазина.\nВыберите тип доставки и дату; для экспресс-доставки или самовывоза укажите время.")
    return (en, tr, ru)[2 if lang == "ru" else 1 if lang == "tr" else 0]


def today_status(key=None, r=None):
    """What delivering TODAY looks like right now (the shop's clock), for one area or for all of them: has the standard run left, otherwise tomorrow. A jet time is never promised: it is a request the team approves. Written by code: the model never decides it."""
    areas = _json("delivery.json")["areas"]
    n = now()
    def one(k, a):
        h, m = (int(x) for x in a["departure"].split(":"))
        return f"{a['label']['en'].split(' (')[0]}: standard run " + ("ALREADY LEFT" if n.hour * 60 + n.minute >= h * 60 + m else "still to leave") + f" at {a['departure']}"
    rows = [one(key, areas[key])] if key in areas else [one(k, a) for k, a in areas.items() if a.get("departure") and k in ("girne", "lefkosa", "gazimagusa", "guzelyurt", "iskele_bafra")]
    late = non_same_day(r) if r else []
    if late:
        return f"Shop time now: {n:%A %d.%m.%Y %H:%M}. " + policy("not_same_day", "en", item=", ".join(late)) + " THE ORDER MUST BE FOR TOMORROW or later: tell the customer so."
    return (f"Shop time now: {n:%A %d.%m.%Y %H:%M}. " + "; ".join(rows) + ". RULE: before the standard run leaves, today is possible with standard delivery. AFTER it left, today is possible ONLY if the team approves a JET request: the customer asks for a jet time, you never say it is possible and never name times or windows; "
            "otherwise tomorrow (standard delivery at its usual time, or a jet request). Orders are taken for today and at most 2 days ahead. "
            + ("AFTER 22:00 NOTHING IS POSSIBLE TODAY, not even jet: say all deliveries are done for today and offer TOMORROW (standard delivery at its usual time, free; or a jet request between " + cfg("WORK_FROM", "08:00") + " and 22:00 with the area's jet fee) and recheck the choices for tomorrow. " if n.hour >= 22 else ""))


def today_answer(lang, key=None, r=None):
    """The direct answer to "can you deliver today / right now?", from the shop's clock and the area's departure time: before the run leaves = yes; after it left = a jet delivery can be REQUESTED (the team approves the time) or tomorrow's standard delivery.
    An order with a not-same-day item: it must be for tomorrow. Written by code in en / tr / ru (0 / 1 / 2)."""
    areas = _json("delivery.json")["areas"]
    a = areas.get(key)
    deps = sorted(x["departure"] for x in areas.values() if x.get("departure"))
    late = non_same_day(r) if r else []
    if late:
        return ("No, ", "Hayır, ", "Нет, ")[lang] + policy("not_same_day", ("en", "tr", "ru")[lang], item=", ".join(late))
    n = now()
    t = n.hour * 60 + n.minute
    mins = lambda hhmm: int(hhmm[:2]) * 60 + int(hhmm[3:5])
    dep = a["departure"] if a else (deps[0] if deps[0] == deps[-1] else f"{deps[0]}-{deps[-1]}")
    lo, hi = (mins(a["departure"]),) * 2 if a else (mins(deps[0]), mins(deps[-1]))
    dtr = f"{dep}'te" if "-" not in dep else f"{dep} arasında"  # Turkish: "at 13:00" / "between 13:00-15:00"
    if t < lo:  # the standard run has not left (in any area): today is possible
        return ("Yes 🌸 standard delivery today leaves at {d}, once the team confirms your order.", "Evet 🌸 bugünkü standart teslimat saat {dt} çıkıyor, siparişinizi ekibimiz onayladıktan sonra.", "Да 🌸 стандартная доставка сегодня выезжает в {d}, после подтверждения заказа командой.")[lang].format(d=dep, dt=dtr)
    if t < hi:  # some areas have left, others not: the city decides
        return ("It depends on your city: the standard run leaves between {d} (area by area). Which city is it for?", "Şehrinize göre değişir: standart çıkış bölgeye göre {d} arasında. Hangi şehir için?", "Зависит от города: стандартный выезд {d} (по районам). Для какого города?")[lang].format(d=dep)
    fee = fmt_tl(a["jet_fee"]) if a and a.get("jet_fee") is not None else None
    fee_text = ((f" Jet delivery costs {fee}.", f" Jet teslimat ücreti {fee}.", f" Стоимость экспресс-доставки: {fee}.")[lang] if fee else "")
    if t >= 22 * 60:  # even a jet request can no longer be made for today: all deliveries are done, tomorrow is checked (standard run + jet window)
        wf = cfg("WORK_FROM", "08:00")
        jet = ((f", or a jet delivery at a time you choose between {wf} and 22:00" + (f" ({fee})" if fee else " (the fee depends on the area)") + ", which the team approves",
                f", ya da {wf} ile 22:00 arasında seçeceğiniz bir saatte jet teslimat" + (f" ({fee})" if fee else " (ücret bölgeye göre değişir)") + "; ekip onaylar",
                f", либо экспресс-доставка в выбранное вами время с {wf} до 22:00" + (f" ({fee})" if fee else " (стоимость зависит от района)") + ", которую подтверждает команда")[lang])
        return (f"No, all deliveries are done for today 🌸 I can plan a delivery for tomorrow: standard delivery leaves at {dep} (free){jet}. Would you like to order for tomorrow?",
                f"Hayır, bugünkü tüm teslimatlar tamamlandı 🌸 Yarın için teslimat planlayabilirim: standart teslimat saat {dep}'de çıkar (ücretsiz){jet}. Yarın için sipariş vermek ister misiniz?",
                f"Нет, на сегодня все доставки завершены 🌸 Могу запланировать доставку на завтра: стандартная доставка выезжает в {dep} (бесплатно){jet}. Хотите оформить заказ на завтра?")[lang]
    return ("No, today's standard delivery already left. Tomorrow standard delivery leaves at {d}. If you want it today, tell me the jet time you would like: it is a request that the team approves when confirming your order.",
            "Hayır, bugünkü standart teslimat çıktı. Yarın standart teslimat saat {dt} çıkar. Bugün istiyorsanız istediğiniz jet saatini yazın: bu bir taleptir, siparişinizi onaylarken ekip onaylar.",
            "Нет, сегодняшний стандартный выезд уже был. Завтра стандартная доставка выезжает в {d}. Если нужно сегодня, напишите желаемое время экспресс-доставки: это запрос, который команда подтверждает при подтверждении заказа.")[lang].format(d=dep, dt=dtr) + fee_text


def delivery_calc(r):
    """Fee, minimum and schedule for an order from its address and delivery choice, all from data/delivery.json: {key, fee (Decimal|None), minimum, departure, below_minimum, problems}."""
    dtype = r["delivery_type"] or "normal"
    if dtype == "pickup":
        return {"key": "", "fee": Decimal(0), "minimum": Decimal(0), "departure": "", "below_minimum": False, "problems": []}
    areas = _json("delivery.json")["areas"]
    key = detect_area(r["address"]) or (r["delivery_area"] if r["delivery_area"] in areas else None)
    if not key and custom_area(r):  # a place that is not among the saved ones, with the city the customer named: that city IS the area, and the team sets the fee
        return {"key": r["delivery_area"], "fee": None, "minimum": Decimal(0), "departure": "", "below_minimum": False, "custom": True,
                "problems": [f"'{custom_area(r)}' is not among the shop's saved delivery places, so it is taken as the area the customer named: the team sets the delivery fee when confirming. Say so in one short line and give NO delivery fee"] + date_problems(r["delivery_date"]) + same_day_problems(r)}
    if not key:
        return {"key": "", "fee": None, "minimum": None, "departure": "", "below_minimum": False, "problems": [f"the place '{split_address(r['address'])[0][:60]}' is not among the shop's saved places: ASK the customer which city / town it is in (one short question) and pass the answer as `city`; whatever city they name is used as the area"]}
    a, problems = areas[key], []
    requested = re.search(r"(\d{1,2})[:.](\d{2})", r["delivery_time"] or "") if dtype == "jet" else None
    requested_time = f"{int(requested.group(1)):02d}:{requested.group(2)}" if requested else ""
    standard_time = dtype == "jet" and requested_time == a["departure"]
    fee = Decimal(0) if dtype == "normal" or standard_time else (D(a["jet_fee"]) if a["jet_fee"] is not None else None)
    own = re.search(r"owner delivery fee: ([\d.]+)", r["notes"])
    if own:  # the owner set this order's delivery fee himself
        fee = D(own.group(1))
    shared, shared_goods = None, Decimal(0)
    if fee is not None and not own and r["address"] and r["delivery_date"]:  # one shipment = one fee: another order of this customer going to the same address at the same date and time already pays it (different products don't matter)
        mine = (re.sub(r"\W", "", fold(r["address"])), r["delivery_date"], requested_time if dtype == "jet" and not standard_time else a["departure"])
        for x in orders():
            x_time = re.search(r"(\d{1,2})[:.](\d{2})", x["delivery_time"] or "") if x["delivery_type"] == "jet" else None
            x_jet_time = f"{int(x_time.group(1)):02d}:{x_time.group(2)}" if x_time else ""
            x_standard = x["delivery_type"] == "jet" and x_jet_time == a["departure"]
            if x["customer_number"] == r["customer_number"] and int(x["order_id"]) < int(r["order_id"]) and x["status"] not in ("rejected", "cancelled", "completed") and x["delivery_type"] != "pickup" \
                    and (re.sub(r"\W", "", fold(x["address"])), x["delivery_date"], x_jet_time if x["delivery_type"] == "jet" and not x_standard else a["departure"]) == mine:
                fee, shared = Decimal(0), int(x["order_id"])
                shared_goods = D(x["total"] or x["subtotal"] or 0)
                break
    if fee is None:
        problems.append(f"the jet fee for {a['label']['en']} is not in the price list: the team must set it")
    problems += date_problems(r["delivery_date"]) + today_problems(r, a)
    issue = jet_time_issue(r) if dtype == "jet" and r["delivery_time"] else ""
    if issue == "cutoff":
        problems.append("jet delivery requests must be no later than 22:00 Cyprus time")
    elif issue == "past":
        problems.append(f"the requested jet time has already passed in Cyprus ({now():%H:%M}); ask for a later time")
    elif issue == "invalid":
        problems.append("jet delivery time is invalid; ask for a time in HH:MM format")
    goods = D(r["total"]) if r["total"] else D(r["subtotal"] or 0)
    shipment_goods = goods + shared_goods
    below = bool(r["total"]) and shipment_goods < D(a["minimum"])
    if below:
        problems.append(f"minimum order for {a['label']['en']} is {fmt_tl(a['minimum'])} of products, this order has {fmt_tl(goods)}")
    return {"key": key, "fee": fee, "minimum": D(a["minimum"]), "departure": a["departure"], "below_minimum": below, "problems": problems, "shared": shared}


ADDR_MAP = re.compile(r"maps\.app\.goo\.gl|goo\.gl/maps|google\.[a-z.]+/maps|maps\.google|\[shared location\]|geo:", re.I)
ADDR_PLACE = re.compile(r"\b(apartment|apartments|apt|flat|villa|office|ofis|daire|bina|blok|site|plaza|hotel|otel|building|residence|rezidans|floor|kat|house|ev)\b", re.I)
SPECIFIC_ADDRESS = "specific address (street and house/building number, apartment or office; or a Google Maps link or a shared WhatsApp location)"


def address_ok(a):
    """A delivery address must point to an exact spot: a house / building number, an apartment or office, a map link / shared pin, or a described place. A city, a district or a bare landmark (a university, a hotel name) alone is not enough: the building, gate, room or a number is still asked."""
    a = a or ""
    if ADDR_MAP.search(a) or MAP_URL.search(a):
        return True
    rest = place_text_left(a)
    if re.search(r"\d", rest) or ADDR_PLACE.search(rest) or re.search(r"\b(yurt|dorm|gate|entrance|reception|room|oda|kapi|blok)\b", rest):
        return True
    return len(re.findall(r"[^\W\d_]{3,}", rest)) >= 3  # a described place ("near the blue mosque, opposite the bakery") is a location too


def form_missing(r):
    """Required personal/address fields collected in the first form; schedule is collected in the following delivery question."""
    req = [("sender_name", "sender name")] + ([] if str(r["customer_number"]).startswith("ext:") else [("sender_phone", "sender phone")])
    if r["delivery_type"] != "pickup":
        req += ([] if "recipient = sender" in r["notes"] else ([] if "recipient name unknown" in r["notes"] else [("recipient_name", "recipient name")]) + ([] if "recipient phone = sender" in r["notes"] else [("recipient_phone", "recipient phone")])) + [("address", "address" if not r["address"] or address_ok(r["address"]) else SPECIFIC_ADDRESS)]
    out = [label for k, label in req if not r[k] or (k == "address" and not address_ok(r[k]) and r["delivery_type"] != "pickup")]
    areas = _json("delivery.json")["areas"]
    if r["delivery_type"] != "pickup" and not (detect_area(r["address"]) or r["delivery_area"] in areas or custom_area(r)):
        out.append("city / town")
    return out


def delivery_missing(row):
    """The delivery question collects type, date and any requested time."""
    missing = []
    if not row["delivery_type"]:
        missing.append("delivery choice (normal, jet or pick-up)")
    if not row["delivery_date"]:
        missing.append("delivery date")
    if row["delivery_type"] in ("jet", "pickup") and not row["delivery_time"]:
        missing.append("delivery time" if row["delivery_type"] == "jet" else "pickup time")
    elif row["delivery_type"] == "jet" and jet_time_issue(row):
        missing.append("delivery time")
    return missing


def time_window(kind):
    """(from, to) the shop takes a JET request or a PICK-UP in: jet 08:00 - 22:00, pick-up = the opening hours (WORK_FROM - WORK_TO, 08:00 - 23:00)."""
    return cfg("WORK_FROM", "08:00"), "22:00" if kind == "jet" else cfg("WORK_TO", "23:00")


def _hm(s):
    m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", str(s or ""))
    return (f"{int(m[1]):02d}:{m[2]}" if int(m[1]) < 24 and int(m[2]) < 60 else "") if m else ""


def check_window(start, end, kind, date_s=""):
    """The time the LLM read from the customer (start = 24h HH:MM, end = the optional "before" deadline) -> (start, end, problem). A PICK-UP must lie inside the opening hours and not in the past; a range must
    end inside the allowed hours and after its start; a jet START keeps being judged by jet_time_issue as before. The start is the stored time, the end only a deadline kept in the notes."""
    lo, hi = time_window(kind)
    s_, e_ = _hm(start), _hm(end)
    if (start and not s_) or (end and not e_):
        return "", "", f"the time '{start}{'-' + end if end else ''}' is not valid: give HH:MM in 24h."
    mins = lambda x: int(x[:2]) * 60 + int(x[3:5])
    try:
        today = datetime.fromisoformat(str(date_s)[:10]).date() == now().date()
    except ValueError:
        today = False
    nowm = now().hour * 60 + now().minute
    bad = f"{'jet delivery' if kind == 'jet' else 'pick-up'} is possible between {lo} and {hi}: tell the customer these hours and ask for a time inside them (a range such as 14:00-18:00, or any time, is fine)."
    if kind != "jet" and s_ and not e_ and not (lo <= s_ <= hi):
        return "", "", f"{s_} is outside the opening hours: " + bad
    if e_:
        if s_ < lo and e_ >= lo:
            s_ = lo  # "between 7 and 11" opens at the opening time
        if not (lo <= e_ <= hi) or e_ <= s_ or (kind != "jet" and not (lo <= s_ <= hi)):
            return "", "", f"the time {start}-{end} is outside the allowed hours: " + bad
    if today and s_ and kind != "jet" and not e_ and mins(s_) <= nowm:
        return "", "", f"{s_} has already passed in Cyprus ({now():%H:%M}): ask for a later time today (up to {hi}) or for another day."
    if today and e_:
        if mins(e_) <= nowm:
            return "", "", f"that time has already passed in Cyprus ({now():%H:%M}): ask for a later time today (up to {hi}) or for another day."
        if mins(s_) <= nowm:  # the window has started: from the next quarter hour, deadline unchanged
            q = min((nowm // 15 + 1) * 15, 23 * 60 + 59)
            s_ = f"{q // 60:02d}:{q % 60:02d}"
    return s_, e_, ""


def before_of(r):
    """The deadline the customer gave with a time range ("must be there before"), or ''."""
    m = re.search(r" \| before: (\d\d:\d\d)", r["notes"] or "")
    return m[1] if m else ""


def dtime(r, lg=0):
    """The delivery / pick-up time as people read it: '09:00' or '09:00 (must arrive before 11:00)' in en / tr / ru (0 / 1 / 2)."""
    s, e = r["delivery_time"] or "", before_of(r)
    if not s or not e:
        return s
    pick = r["delivery_type"] == "pickup"
    return s + (("(collect before {e})", "(en geç {e})", "(забрать до {e})") if pick else ("(must arrive before {e})", "(en geç {e})", "(доставить до {e})"))[lg].replace("(", " (", 1).format(e=e)




def jet_time_issue(row):
    """A jet request must be in the future for today and no later than 22:00 Cyprus time."""
    m = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", row.get("delivery_time") or "")
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        return "invalid"
    minutes = int(m.group(1)) * 60 + int(m.group(2))
    if minutes > 22 * 60:
        return "cutoff"
    try:
        is_today = datetime.fromisoformat(str(row.get("delivery_date") or "")[:10]).date() == now().date()
    except ValueError:
        is_today = False
    if is_today and minutes <= now().hour * 60 + now().minute:
        return "past"
    return ""


def submission_missing(row):
    """Check stored form, basket and prices before sending an order request."""
    missing = form_missing(row) + delivery_missing(row)
    lines = row_lines(row)
    if not lines and not row["custom_items"]:
        missing.append("order items")
    if lines is None or any(qty < 1 or price <= 0 for _, qty, price in lines):
        missing.append("valid item quantities and prices")
    if row["custom_items"] and not row["custom_price"]:
        missing.append("custom item price")
    if not row["total"] or not row["payable"]:
        missing.append("order total")
    if row["delivery_fee"] == "":
        missing.append("delivery fee")
    missing.extend(delivery_calc(row)["problems"])
    return list(dict.fromkeys(missing))


REL_ONLY = re.compile(r"^(my|the|our|his|her|benim|bizim)?\s*(mom|mum|mommy|mother|dad|father|sister|brother|wife|husband|friend|girlfriend|boyfriend|fianc[eé]e?|boss|colleague|neighbou?r|grandma|grandmother|grandpa|aunt|uncle|cousin|daughter|son|anne|annem|baba|babam|kardeş|kardeşim|abla|ablam|ağabey|abi|eşim|karım|kocam|arkadaş|arkadaşım|sevgilim|kız arkadaşım|teyze|teyzem|hala|halam|amca|dayı|nine|babaanne|anneanne|kızım|oğlum|someone|somebody|biri|birisi)\.?$", re.I)


def received_orders(phone):
    """Orders that were SENT TO this number (they are the recipient, somebody else ordered) and are paid / on the way / delivered, newest last."""
    return [r for r in orders() if r["recipient_phone"] and canon_phone(r["recipient_phone"]) == phone and r["customer_number"] != phone and r["status"] in ("confirmed", "completed")]


def is_external(r):
    """An order the owner entered himself for someone who ordered in person / by phone."""
    return "external:" in (r["notes"] or "")


def is_paid(r):
    """True when the order is paid: confirmed / shipped, or money already received."""
    return r["status"] in ("confirmed", "completed") or D(r["paid"] or 0) > 0


def _tok_phone(value):
    d = re.sub(r"\D", "", str(value or ""))
    return "p:" + d[-9:] if len(d) >= 9 else ""


def link(ident, token):
    """Remember that this id (a WhatsApp number, a Telegram id, a hidden-number id...) carries this identifier (a user tag, a phone number). Ids that share ANY identifier are the same person."""
    if not token:
        return
    ids = jget("ident:" + token) or []
    if ident not in ids:
        kset("ident:" + token, json.dumps(ids + [ident]))
    toks = jget("tokens:" + ident) or []
    if token not in toks:
        kset("tokens:" + ident, json.dumps(toks + [token]))


def set_handle(phone, handle):
    """The customer's user tag (@name): on WhatsApp some people have a tag and no visible number, some a number and no tag, some both; on Telegram there is always an id. Each identifier we ever see is linked."""
    h = "@" + str(handle or "").strip().lstrip("@").lower()
    if len(h) < 3:
        return
    if kget("handle:" + phone) != h:
        kset("handle:" + phone, h)
    link(phone, "h:" + h)


def handle_of(phone):
    return kget("handle:" + str(phone)) or ""


def linked_ids(phone):
    """Every id the same person writes from: ids that share a user tag or a phone number with this one, transitively (a WhatsApp number, later a Telegram id that gave the same number). The given id first; just itself when nothing is known."""
    seen, todo = [phone], [phone]
    while todo:
        cur = todo.pop()
        toks = list(jget("tokens:" + cur) or [])
        if cur.isdigit() and _tok_phone(cur):
            toks.append(_tok_phone(cur))  # a WhatsApp number is its own phone identifier
        for t in toks:
            for i in jget("ident:" + t) or []:
                if i not in seen:
                    seen.append(i); todo.append(i)
    return seen


def last_form(phone):
    """Latest saved details, including cancelled orders, for contact reuse. Shipment reuse must pass shipment_details validation."""
    ids = linked_ids(phone)
    for r in reversed([x for x in orders() if x["customer_number"] in ids]):
        if r["sender_name"] and (r["address"] or r["delivery_type"] == "pickup"):
            return r
    return None


def pkey(name, phone=""):
    """One person = one entry: the name (folded); an anonymous recipient is told apart by the last digits of the phone."""
    nm_ = fold(name or "").strip()
    return nm_ if nm_ not in ("anonymous", "") else f"{nm_}|{re.sub(chr(92) + 'D', '', phone or '')[-9:]}"


def _people_from_orders(phone):
    """The address book as the ORDERS show it (kept permanently): each recipient with phone, address, the date of the last time and how often. Newest wins."""
    out = {}
    ids = linked_ids(phone)
    for r in reversed([x for x in orders() if x["customer_number"] in ids and x["recipient_name"]]):
        k = pkey(r["recipient_name"], r["recipient_phone"])
        c = out.setdefault(k, {"key": k, "name": r["recipient_name"], "phone": r["recipient_phone"], "address": r["address"] if not r["address"].startswith("PICKUP") else "", "last": r["timestamp"][:10], "times": 0, "order": r["order_id"], "relation": "", "note": ""})
        c["times"] += 1
        c["address"] = c["address"] or (r["address"] if not r["address"].startswith("PICKUP") else "")
    return out


def people(phone, n=200):
    """EVERYONE this customer sends to (their permanent memory): what the owner / the customer saved (customers.json, `people`) laid over what the orders show. An entry the owner removed stays removed.
    Newest first. Each: key, name, phone, address, relation, note, last, times."""
    out = _people_from_orders(phone)
    for pid in linked_ids(phone):
        for k, e in ((customers().get(pid) or {}).get("people") or {}).items():
            if e.get("deleted"):
                out.pop(k, None)
                continue
            c = out.setdefault(k, {"key": k, "name": e.get("name", ""), "phone": "", "address": "", "last": "", "times": 0, "order": None, "relation": "", "note": ""})
            for f in ("name", "phone", "address", "relation", "note"):
                if e.get(f):
                    c[f] = e[f]
            c["last"] = max(c["last"], e.get("last", ""))
    return sorted(out.values(), key=lambda x: x["last"], reverse=True)[:n]


def _save_people(ph, fn):
    with STATE_LOCK:
        d = customers()
        c = d.setdefault(ph, {})
        ppl = c.setdefault("people", {})
        before = json.dumps(ppl, sort_keys=True, ensure_ascii=False)
        fn(ppl)
        if json.dumps(ppl, sort_keys=True, ensure_ascii=False) != before:  # a form saved again with nothing new writes nothing
            atomic_write(P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))


def remember_person(phone, name, phone_="", address="", relation="", note="", last=None):
    """Save (or update) one person in the customer's memory. Empty fields change nothing; a person removed earlier comes back."""
    name = (name or "").strip()
    if not name:
        return ""
    k = pkey(name, phone_)
    def f(ppl):
        e = ppl.get(k) or {}
        e.pop("deleted", None)
        e["name"] = name
        for fld, v in (("phone", phone_), ("address", address), ("relation", relation), ("note", note)):
            if v and str(v).strip():
                e[fld] = str(v).strip()
        e["last"] = last or e.get("last") or now().strftime("%Y-%m-%d")
        ppl[k] = e
    _save_people(canon_phone(phone), f)
    return k


def edit_person(customer, key, **fields):
    """The owner changes a saved person (any field, also the name: the entry moves to its new key)."""
    ph = canon_phone(customer)
    cur = next((p for p in people(ph) if p["key"] == key), None)
    if not cur:
        return ""
    new = {**{f: cur[f] for f in ("name", "phone", "address", "relation", "note")}, **{f: str(v).strip() for f, v in fields.items() if f in ("name", "phone", "address", "relation", "note")}}
    if not new["name"]:
        return ""
    nk = pkey(new["name"], new["phone"])
    def f(ppl):
        if nk != key:
            ppl[key] = {"deleted": True}  # the old key stays hidden, the orders keep showing the old spelling otherwise
        ppl[nk] = {**new, "last": cur["last"] or now().strftime("%Y-%m-%d")}
    _save_people(ph, f)
    return nk


def forget_person(phone, key):
    ph = canon_phone(phone)
    def f(ppl):
        ppl[key] = {"deleted": True}
    _save_people(ph, f)


RELATIONS = {  # relation words in en / tr / ru -> one canonical word, to find "my mom" among the saved people
    "mother": ("mom", "mum", "mother", "mommy", "mama", "anne", "annem", "anneme", "annemi", "annemin", "annemle", "anneannem", "мама", "маме", "мамы", "мать"), "father": ("dad", "father", "daddy", "baba", "babam", "babama", "babamı", "babamin", "babamla", "папа", "папе", "отец"),
    "wife": ("wife", "eşim", "esim", "eşime", "esime", "karım", "karim", "karıma", "karima", "жена", "жене", "супруга"), "husband": ("husband", "kocam", "муж", "мужу", "супруг"), "sister": ("sister", "kız kardeş", "kiz kardes", "kız kardeşime", "ablam", "ablama", "abla", "сестра", "сестре"),
    "brother": ("brother", "kardeşim", "kardesim", "abim", "ağabey", "брат", "брату"), "girlfriend": ("girlfriend", "sevgilim", "kız arkadaş", "девушка", "девушке"), "boyfriend": ("boyfriend", "erkek arkadaş", "парень", "парню"),
    "friend": ("friend", "arkadaş", "arkadasim", "друг", "подруга", "подруге", "другу"), "daughter": ("daughter", "kızım", "дочь", "дочке"), "son": ("son", "oğlum", "oglum", "сын", "сыну"),
    "grandmother": ("grandma", "grandmother", "nine", "babaanne", "anneanne", "бабушка", "бабушке"), "boss": ("boss", "patron", "müdür", "начальник", "начальнице")}


def _rel_of(word):
    w = fold(word or "")
    return next((c for c, ws in RELATIONS.items() if any(fold(x) == w or fold(x) in w.split() for x in ws)), "")


def find_person(phone, text):
    """The saved people a message points at: their name (a whole word of 3+ letters, any language) or their relation ("my mom", "annem", "моей маме"). Newest first; [] = nobody saved matches."""
    t = fold(text or "")
    words = set(re.findall(r"[^\W\d_]{3,}", t))
    wanted = {c for c, ws in RELATIONS.items() if any(re.search(rf"(?<![^\W_]){re.escape(fold(x))}(?![^\W_])", t) for x in ws)}
    hits = []
    for p in people(phone):
        nm = {w for w in re.findall(r"[^\W\d_]{3,}", fold(p["name"]))}
        if (nm & words) or (p["relation"] and (_rel_of(p["relation"]) in wanted or fold(p["relation"]) in t)):
            hits.append(p)
    return hits


def _digits(x):
    return re.sub(r"\D", "", str(x or ""))


def search_orders(query="", status="", since="", limit=20):
    """Find orders among ALL orders ever made (kept permanently), newest first, by whatever the owner remembers: the order number ("#12" or 12), a PHONE NUMBER in any writing (+90 533..., 0533..., 533...: the last 9 digits
    are compared, with the customer's, the sender's, the recipient's and the saved profile's numbers), the customer's / sender's / recipient's name, the ITEMS in any language (English, Turkish, Russian names of the products),
    the LOCATION (the address, the town or delivery area, a landmark like NEU) and the notes. Every word of the query must match. Returns (matches, how many)."""
    q = str(query or "")
    phones = [_digits(m)[-9:] for m in re.findall(r"\+?\d[\d\s().-]{6,}\d", q) if len(_digits(m)) >= 7]
    rest = re.sub(r"\+?\d[\d\s().-]{6,}\d", " ", q)
    toks = [t for t in re.split(r"\s+", fold(rest).strip()) if t]
    ids = {t.lstrip("#") for t in toks if re.fullmatch(r"#?\d{1,5}", t)}
    words = [t for t in toks if not re.fullmatch(r"#\d{1,5}", t)]
    nodes = graph()[0]
    langs = {}
    for n in nodes.values():
        if n["type"] == "product":
            langs[_nm(n["name"])] = " ".join(x for x in (n.get("name"), n.get("name_en"), n.get("name_tr"), n.get("name_ru"), *(n.get("aliases") or [])) if x)
    areas = _json("delivery.json")["areas"]
    prof = customers()
    out = []
    for r in reversed(orders()):
        if status and r["status"] != status or since and r["timestamp"][:10] < since:
            continue
        p = prof.get(canon_phone(r["customer_number"])) or {}
        items = " ".join(langs.get(nm, nm) for nm, _, _ in (row_lines(r) or [])) or r["items"]
        area = areas.get(r["delivery_area"] or detect_area(r["address"]) or "", {})
        hay = fold(" ".join([f"#{r['order_id']}", r["customer_number"], r["customer_name"], p.get("name", ""), p.get("notes", ""), p.get("location", ""), r["sender_name"], r["recipient_name"], r["address"],
                             " ".join(area.get("names", [])), area.get("label", {}).get("en", ""), area.get("label", {}).get("tr", ""), " ".join(h["place"] for h in place_info(r["address"])),
                             items, r["custom_items"], r["notes"], r["delivery_date"], r["timestamp"][:10], r["status"]]))
        nums = {_digits(x)[-9:] for x in (r["customer_number"], r["sender_phone"], r["recipient_phone"], p.get("phone")) if _digits(x)}
        if phones and not all(ph in nums or any(ph[-8:] == n[-8:] for n in nums) for ph in phones):
            continue
        if ids and not any(i == str(r["order_id"]) or re.search(rf"(?<!\d){i}(?!\d)", hay) for i in ids):
            continue
        if all(w in hay for w in words):
            out.append(r)
    return out[:max(1, min(int(limit or 20), 50))], len(out)


def contacts(phone, n=8):
    """The address book of one customer (their permanent memory, see people()): each person they send to with phone, address, when and how often. Newest first. Someone they send flowers to again is found here, never asked from zero."""
    return people(phone, n)


def remember_from_order(r):
    """A form changed: the recipient goes into the customer's memory, and the sender's own name and phone into their profile when it has none yet (what the owner typed there is never overwritten)."""
    ph = r["customer_number"]
    if r["recipient_name"] and (r["recipient_phone"] or (r["address"] and not r["address"].startswith("PICKUP"))) and fold(r["recipient_name"]) != "anonymous" and "recipient = sender" not in r["notes"]:
        remember_person(ph, r["recipient_name"], r["recipient_phone"], r["address"] if not r["address"].startswith("PICKUP") else "", last=r["timestamp"][:10])
    c = customers().get(canon_phone(ph)) or {}
    fills = {k: v for k, v in (("name", r["sender_name"]), ("phone", r["sender_phone"])) if v and not c.get(k)}
    if fills:
        with STATE_LOCK:
            d = customers()
            e = d.setdefault(canon_phone(ph), {})
            for k, v in fills.items():
                e.setdefault(k, v)
            atomic_write(P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))


def saved_locations(phone, n=6):
    """The places this customer has used before, newest first, without repeats: the delivery address of each earlier order (a shared map pin is stored as its link) with who it was for,
    plus the map pins they shared in the chat. So "same location / same address / same place as before" can be understood."""
    out, seen = [], set()
    ids = linked_ids(phone)
    for r in reversed([x for x in orders() if x["customer_number"] in ids and x["address"] and not x["address"].startswith("PICKUP")]):
        k = fold(r["address"])
        if k not in seen:
            seen.add(k)
            out.append({"address": r["address"], "for": r["recipient_name"] if r["recipient_name"] != "Anonymous" else "", "phone": r["recipient_phone"], "order": r["order_id"], "date": r["timestamp"][:10]})
    for ts, role, text in reversed(chat_log(phone)[-300:]):
        m = re.search(r"\[shared location\]\s*(\S+)", text) if role == "user" else None
        if m and fold(m.group(1)) not in seen:
            seen.add(fold(m.group(1)))
            out.append({"address": m.group(1), "for": "", "phone": "", "order": "", "date": ts[:10]})
    return out[:n]


def shipment_details(phone, order_id):
    """Validated delivery details for an explicit request to ship with an existing order."""
    row = order_row(order_id)
    if not row or row["customer_number"] not in linked_ids(phone):
        raise ValueError("that shipment does not belong to this customer")
    if row["status"] not in OPEN + LOCKED or row["status"] == "completed":
        raise ValueError("that order is no longer available to ship together")
    if form_missing(row):
        raise ValueError("the original shipment details are incomplete")
    fields = {k: row[k] for k in ("sender_name", "sender_phone", "recipient_name", "recipient_phone", "address", "delivery_date", "delivery_type", "delivery_time")}
    fields["delivery_type"] = row["delivery_type"] or "normal"
    for flag, marker in (("recipient_is_sender", "recipient = sender"), ("recipient_unknown", "recipient name unknown"), ("recipient_phone_is_sender", "recipient phone = sender")):
        if marker in row["notes"]:
            fields[flag] = True
    notes = re.findall(r"\| note: ([^|]*)", row["notes"])
    if notes:
        fields["note"] = notes[-1].strip()
    return fields


def addon_of(r):
    m = re.search(r"add-on to #(\d+)", r["notes"] or "")
    return int(m.group(1)) if m else None


def pay_state(r):
    """paid | partial | unpaid for the dashboard (an order that has no amount to pay yet counts as unpaid)."""
    pay, paid = D(r["payable"] or 0), D(r["paid"] or 0)
    if r["status"] in ("confirmed", "completed") or (paid > 0 and paid >= pay):
        return "paid"
    return "partial" if paid > 0 else "unpaid"


MAP_URL = re.compile(r"https?://\S*(?:maps\.app\.goo\.gl|goo\.gl/maps|google\.[a-z.]+/maps|maps\.google)\S*", re.I)


def split_address(a):
    """(written place, map link) of an address string: a map link and a written address can both be given, and both are kept."""
    a = a or ""
    m = MAP_URL.search(a)
    link = m.group(0).rstrip(").,]") if m else ""
    text = re.sub(r"\[shared location\]", " ", MAP_URL.sub(" ", a), flags=re.I)
    return re.sub(r"\s*\|\s*$|^\s*\|\s*", "", re.sub(r"\s{2,}", " ", text)).strip(" ,|"), link


def merge_address(old, new):
    """The address after the customer gave `new` while the order had `old`: a map link and written details are BOTH kept. New written details replace the old written ones (and drop a pin that belonged to a different written place); a new pin replaces the old pin and keeps the written details."""
    o_text, o_link = split_address(old)
    n_text, n_link = split_address(new)
    text = n_text or o_text
    link = n_link or (o_link if (not n_text or not o_text or fold(n_text) == fold(o_text)) else "")  # a changed written place (a friend moved) must not keep the old pin
    return " | ".join(x for x in (text, link) if x)


def order_form_update(oid, **f):
    """Write form fields / delivery choices into an UNCONFIRMED order and re-derive area, fee and payable. Returns (row, delivery_calc)."""
    with _lock:
        rows = orders()
        r = next(x for x in rows if int(x["order_id"]) == int(oid))
        assert r["status"] in OPEN
        before, time_bad = None, []  # deadline of a time range (None = untouched)
        end_ = f.pop("deliver_before", "")
        if f.get("delivery_time") or end_:
            kind_ = f.get("delivery_type") or r["delivery_type"] or "jet"
            st_, en_, why_ = check_window(f.get("delivery_time") or r["delivery_time"], end_, "pickup" if kind_ == "pickup" else "jet", f.get("delivery_date") or r["delivery_date"])
            if st_:
                f["delivery_time"], before = st_, en_
            else:
                f.pop("delivery_time", None)
                if why_:
                    time_bad = [why_]  # reported like a refused date: nothing is stored, the customer is told
        if f.get("delivery_time") and not f.get("delivery_type") and r["delivery_type"] != "pickup":
            f["delivery_type"] = "jet"  # a specific requested time is priced as jet delivery
        if f.get("delivery_type") == "pickup" and r["delivery_type"] != "pickup":
            r["delivery_time"] = ""
        note = f.pop("note", "")
        memo = str(f.pop("memo", "") or "").strip()
        city = str(f.pop("city", "") or "").strip()
        same = f.pop("recipient_is_sender", False)
        unknown = f.pop("recipient_unknown", False)  # the customer does not want to give the recipient's NAME: "Anonymous"
        samep = f.pop("recipient_phone_is_sender", False)  # the customer SAID the recipient's number is their own
        bad_date, bad_phone = [], []
        far = FOREIGN.search(fold(f"{city} {f.get('address') or ''}"))
        if far:  # delivery is North Cyprus only: the place is NOT stored, and neither is the street that came with it (it must not be re-attached to another town)
            f.pop("address", None)
            kset(f"foreign:{oid}", far.group(0))
            bad_phone.append(f"'{far.group(0)}' is outside North Cyprus and we only deliver within North Cyprus: the address was NOT saved. Tell the customer this plainly, and ask for a delivery address in North Cyprus (street and building, a landmark, and the town). Never reuse the street or building of the foreign address.")
            city = ""
        if f.get("address") and not far:
            kset(f"foreign:{oid}", "")  # a usable address came in: the explanation is over
        for k, v in f.items():
            if k == "delivery_date" and v and (bad_date := date_problems(v)):  # a date outside the rules is NOT stored: it stays missing and the customer is told why
                continue
            if k == "recipient_name" and REL_ONLY.match(str(v or "").strip()):  # "my sister" / "annem" is who she is to the customer, not her name: the name is still asked for
                continue
            if k in ("sender_phone", "recipient_phone") and v not in ("", None):  # typed by a person: a country code is added (+90) or checked; a number that sounds off is NOT stored and the customer is asked for it with the country code
                ph_, why_ = check_phone(v)
                if ph_:
                    r[k] = "+" + ph_
                else:
                    bad_phone.append(f"the {'sender' if k == 'sender_phone' else 'recipient'} phone number '{v}' sounds wrong ({why_}): it was NOT saved. Ask them to send it WITH the country code, e.g. +90 5xx xxx xx xx (or +971 … for the UAE).")
                continue
            if v not in ("", None):
                r[k] = merge_address(r[k], str(v)) if k == "address" and not str(r[k]).startswith("PICKUP") else str(v)
        if city:
            key_ = detect_area(city)
            if key_:
                r["delivery_area"] = key_
            elif not detect_area(r["address"]):  # not among the saved places: what the customer said IS the area (the team sets its fee)
                r["delivery_area"] = CUSTOM_AREA + city[:60]
        new_rn, new_rp = str(f.get("recipient_name") or "").strip(), str(f.get("recipient_phone") or "").strip()
        if not same and "recipient = sender" in r["notes"] and ((new_rn and fold(new_rn) != fold(r["sender_name"] or "")) or (new_rp and check_phone(new_rp)[0] and canon_phone(new_rp) != canon_phone(r["sender_phone"] or ""))):
            r["notes"] = r["notes"].replace(" | recipient = sender", "")  # it was for themselves, now it goes to someone else: the recipient they named is kept (editable at any step)
            r["recipient_name"] = new_rn or ("" if fold(r["recipient_name"] or "") == fold(r["sender_name"] or "") else r["recipient_name"])
            if new_rp and check_phone(new_rp)[0]:
                r["recipient_phone"] = "+" + check_phone(new_rp)[0]
            elif canon_phone(r["recipient_phone"] or "") == canon_phone(r["sender_phone"] or ""):
                r["recipient_phone"] = ""
        if not same and r["sender_phone"] and r["recipient_phone"] and "recipient = sender" not in r["notes"] and canon_phone(r["recipient_phone"]) == canon_phone(r["sender_phone"]):
            same = True  # the recipient's number is the sender's own: it is for themselves (a friend may collect it), the recipient is never asked
        if same and "recipient = sender" not in r["notes"]:
            r["notes"] += " | recipient = sender"
        if "recipient = sender" in r["notes"]:  # the customer sends it to themselves: the recipient follows the sender, never asked again
            r["recipient_name"], r["recipient_phone"] = r["sender_name"], r["sender_phone"]
        if memo and memo.replace("|", "/") not in [x.strip() for x in re.findall(r"\| memo: ([^|]*)", r["notes"])]:  # something the customer asked the team to keep in mind (not the card text)
            r["notes"] += f" | memo: {memo.replace('|', '/')}"
        if note and fold((re.findall(r"\| note: ([^|]*)", r["notes"]) or [""])[-1].strip()) != fold(note.strip()):  # the same card message said twice is kept once
            r["notes"] += f" | note: {note}"
        if f.get("delivery_type") == "normal":
            r["delivery_time"] = ""
        if r["delivery_type"] == "pickup":  # collecting at the shop replaces any delivery address
            r["address"] = "PICKUP at the shop (Girne)"
        if unknown and "recipient name unknown" not in r["notes"]:
            r["notes"] += " | recipient name unknown"
        if samep and "recipient phone = sender" not in r["notes"]:
            r["notes"] += " | recipient phone = sender"
        if "recipient name unknown" in r["notes"]:
            r["recipient_name"] = r["recipient_name"] or "Anonymous"
        if "recipient phone = sender" in r["notes"]:  # only because the customer said so; a number given later replaces it
            r["recipient_phone"] = r["recipient_phone"] or r["sender_phone"]
        if r["sender_phone"]:
            link(r["customer_number"], _tok_phone(r["sender_phone"]))  # the number they gave links this id to the same person on another app
        if r["delivery_type"] == "jet":  # choosing the normal departure is normal delivery, with no jet fee
            area_key = detect_area(r["address"]) or r["delivery_area"]
            area = _json("delivery.json")["areas"].get(area_key)
            picked = re.fullmatch(r"\s*(\d{1,2})[:.](\d{2})\s*", r["delivery_time"] or "")
            if area and picked and f"{int(picked.group(1)):02d}:{picked.group(2)}" == area["departure"]:
                r["delivery_type"], r["delivery_time"] = "normal", ""
        if before is not None or r["delivery_type"] == "normal" or not r["delivery_time"]:  # the deadline follows the time it belongs to
            r["notes"] = re.sub(r" \| before: \d\d:\d\d", "", r["notes"])
            if before and r["delivery_time"]:
                r["notes"] += f" | before: {before}"
        calc = delivery_calc(r)
        calc["problems"] = calc["problems"] + [p_ for p_ in bad_date + bad_phone + time_bad if p_ not in calc["problems"]]
        r["delivery_area"] = calc["key"]
        if calc["key"] in _json("delivery.json")["areas"] and split_address(r["address"])[0] and not r["address"].startswith("PICKUP"):  # the written address always carries the city (from the city given, or found from the landmark)
            nm_ = _json("delivery.json")["areas"][calc["key"]]["label"]["en"].split(" (")[0]
            txt_, link_ = split_address(r["address"])
            if not any(fold(n_) in fold(txt_) for n_ in _json("delivery.json")["areas"][calc["key"]]["names"]) and fold(nm_) not in fold(txt_):
                r["address"] = " | ".join(x for x in (f"{nm_}, {txt_}", link_) if x)
        r["delivery_fee"] = mstr(calc["fee"]) if calc["fee"] is not None else ""
        recompute_row(r)
        _write(rows)
        out_ = (dict(r), calc)
    remember_from_order(out_[0])  # the customer's memory is updated whenever a form changes (never inside the orders lock)
    return out_


def set_delivery_fee(oid, amount):
    """The owner's own delivery fee for one unconfirmed order (e.g. a jet fee that is not in the price list)."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or r["status"] not in OPEN:
            return None
        r["notes"] = re.sub(r" \| owner delivery fee: [\d.]+", "", r["notes"]) + f" | owner delivery fee: {mstr(amount)}"
        r["delivery_fee"] = mstr(amount)
        recompute_row(r)
        _write(rows)
        return dict(r)


def render_form(r, lang, template=None):
    """Fill the fixed or translated template with known values; customer data is never translated."""
    note = re.findall(r"\| note: ([^|]*)", r["notes"])
    phone = r["sender_phone"] and ("+" + r["sender_phone"].lstrip("+") if r["sender_phone"].isdigit() else r["sender_phone"])
    if r["delivery_type"] == "pickup":  # no form: the shop's location and hours, and a request to WRITE the name, phone, day and time
        return pickup_block(r, lang if lang in PICKUP_WORDS else "en", ask=True)
    key_ = detect_area(r["address"]) or r["delivery_area"]
    area = _json("delivery.json")["areas"].get(key_) or {}
    city = area.get("label", {}).get("en", "").split(" (")[0] or custom_area(r)
    address = r["address"]
    if address.startswith("PICKUP") or fold(address).strip(" ,./") in {fold(n) for n in [city, *area.get("names", [])]}:
        address = ""
    vals = [r["sender_name"], phone, r["recipient_name"], r["recipient_phone"], city, address, note[-1].strip() if note else ""]
    lines, k = (template or _json("order_form.json").get(lang) or _json("order_form.json")["en"]).split("\n"), 0
    for i, ln in enumerate(lines):
        if k < len(vals) and ln.strip() and ln.rstrip().endswith((":", ";")):
            if vals[k]:
                lines[i] = ln.rstrip() + " " + str(vals[k])
            k += 1
    return "\n".join(lines)


PAY_WORDS = {  # the fixed words of the payment message (pure code, never written by a model)
    "en": dict(order="Your order", free="free", pickup="Pick-up at the shop: ", jet="Jet delivery ({label}, {time}): ", normal="Delivery ({label}, normal schedule, leaves at {dep}): ",
               fee="Service fee: ", discount="Discount: ", newtotal="NEW TOTAL: ", alreadypaid="ALREADY PAID: ", paynow="TO PAY NOW: ", total="TOTAL TO PAY: "),
    "tr": dict(order="Siparişiniz", free="ücretsiz", pickup="Mağazadan teslim alma: ", jet="Jet teslimat ({label}, {time}): ", normal="Teslimat ({label}, normal program, {dep}'te çıkış): ",
               fee="Hizmet bedeli: ", discount="İndirim: ", newtotal="TOPLAM: ", alreadypaid="ZATEN ÖDENEN: ", paynow="ŞİMDİ ÖDENECEK: ", total="ÖDENECEK TOPLAM: "),
    "ru": dict(order="Ваш заказ", free="бесплатно", pickup="Самовывоз из магазина: ", jet="Экспресс-доставка ({label}, {time}): ", normal="Доставка ({label}, обычный график, выезд в {dep}): ",
               fee="Сервисный сбор: ", discount="Скидка: ", newtotal="НОВАЯ СУММА: ", alreadypaid="УЖЕ ОПЛАЧЕНО: ", paynow="К ОПЛАТЕ СЕЙЧАС: ", total="ИТОГО К ОПЛАТЕ: "),
}


WHY_WORDS = {
    "en": dict(head="Why the total changed:", removed="removed: {x}", added="added: {x}", qty="{x}: quantity {a} → {b}", unit="{x}: price {a} → {b}", dlv="Delivery: {a} → {b}", adj="{x}: {a} → {b}", fee="Service fee / discount", total="Total: {a} → {b}",
               free="free", normal="normal delivery", jet="jet delivery", pickup="pick-up at the shop"),
    "tr": dict(head="Toplam neden değişti:", removed="çıkarıldı: {x}", added="eklendi: {x}", qty="{x}: adet {a} → {b}", unit="{x}: fiyat {a} → {b}", dlv="Teslimat: {a} → {b}", adj="{x}: {a} → {b}", fee="Hizmet bedeli / indirim", total="Toplam: {a} → {b}",
               free="ücretsiz", normal="normal teslimat", jet="jet teslimat", pickup="mağazadan teslim"),
    "ru": dict(head="Почему изменилась сумма:", removed="убрано: {x}", added="добавлено: {x}", qty="{x}: количество {a} → {b}", unit="{x}: цена {a} → {b}", dlv="Доставка: {a} → {b}", adj="{x}: {a} → {b}", fee="Сервисный сбор / скидка", total="Итого: {a} → {b}",
               free="бесплатно", normal="обычная доставка", jet="экспресс-доставка", pickup="самовывоз"),
}


def change_explain(old, new, lang="en"):
    """Why the amount changed between two versions of an order, in plain words, built by code from the two rows (items added / removed / re-quantified / re-priced, the delivery type, time and fee, a service fee or discount, then old total → new total). '' when nothing about the money changed."""
    L = WHY_WORDS.get(lang, WHY_WORDS["en"])
    a_, b_ = {n: (q, u) for n, q, u in row_lines(old) or []}, {n: (q, u) for n, q, u in row_lines(new) or []}
    out = []
    for n in a_.keys() - b_.keys():
        out.append(L["removed"].format(x=f"{a_[n][0]} x {n} ({fmt_tl(a_[n][0] * a_[n][1])})"))
    for n in b_.keys() - a_.keys():
        out.append(L["added"].format(x=f"{b_[n][0]} x {n} ({fmt_tl(b_[n][0] * b_[n][1])})"))
    for n in a_.keys() & b_.keys():
        (qa, ua), (qb, ub) = a_[n], b_[n]
        if qa != qb:
            out.append(L["qty"].format(x=n, a=qa, b=qb))
        if ua != ub:
            out.append(L["unit"].format(x=n, a=fmt_tl(ua), b=fmt_tl(ub)))
    if (old["custom_items"], old["custom_price"]) != (new["custom_items"], new["custom_price"]):
        out.append(L["added"].format(x=new["custom_items"] or old["custom_items"]) + (f" ({fmt_tl(new['custom_price'])})" if new["custom_price"] else ""))
    def dlv(r):
        t = r["delivery_type"] or "normal"
        fee = fmt_tl(r["delivery_fee"]) if r["delivery_fee"] != "" and D(r["delivery_fee"]) else L["free"]
        return f"{L.get(t, t)}{' ' + dtime(r) if t == 'jet' and r['delivery_time'] else ''} ({fee})"
    if (old["delivery_type"], old["delivery_time"], old["delivery_fee"], old["delivery_area"]) != (new["delivery_type"], new["delivery_time"], new["delivery_fee"], new["delivery_area"]):
        out.append(L["dlv"].format(a=dlv(old), b=dlv(new)))
    if adjustment(old) != adjustment(new):
        out.append(L["adj"].format(x=L["fee"], a=fmt_tl(adjustment(old)), b=fmt_tl(adjustment(new))))
    ta, tb = D(old["payable"] or old["total"] or 0), D(new["payable"] or new["total"] or 0)
    if ta == tb and not out:
        return ""
    out.append(L["total"].format(a=fmt_tl(ta), b=fmt_tl(tb)))
    return L["head"] + "\n" + "\n".join("• " + x for x in out)


PICKUP_WORDS = {
    "en": dict(title="📍 Pick-up at the shop", open="🕘 Open every day {hours}", ask="Please write your name, your phone number, and the day and time (or a time range, such as 14:00-18:00, or \"any time\") you will come.",
               when="Your pick-up: {when}", who="Order #{oid} · name on the order: {name}", friend="Anyone can collect it, such as a friend: they just need the order number or the name.", nofee="No pick-up fee."),
    "tr": dict(title="📍 Mağazadan teslim alma", open="🕘 Her gün açığız {hours}", ask="Lütfen adınızı, telefon numaranızı ve geleceğiniz gün ve saati (ya da 14:00-18:00 gibi bir saat aralığını, veya \"fark etmez\") yazın.",
               when="Teslim alma: {when}", who="Sipariş #{oid} · siparişteki isim: {name}", friend="Siparişi bir arkadaşınız da alabilir: sipariş numarasını ya da adı söylemesi yeterli.", nofee="Mağazadan teslim almada ücret yoktur."),
    "ru": dict(title="📍 Самовывоз из магазина", open="🕘 Работаем каждый день {hours}", ask="Напишите, пожалуйста, ваше имя, номер телефона, а также день и время (или промежуток, например 14:00-18:00, или «в любое время»), когда придёте.",
               when="Самовывоз: {when}", who="Заказ #{oid} · имя в заказе: {name}", friend="Заказ может забрать и друг: достаточно назвать номер заказа или имя.", nofee="Самовывоз бесплатный."),
}


def pickup_block(r, lang, ask=False):
    """The fixed (hard-coded) pick-up message: where, when we are open, and either what to write (ask) or the customer's pick-up window as a reminder."""
    L = PICKUP_WORDS.get(lang, PICKUP_WORDS["en"])
    b = business_info()
    hours = re.sub(r"^.*?\)\s*", "", b.get("hours") or "08:00 - 23:00")
    lines = [L["title"], b.get("address") or "", b.get("maps_url") or "", L["open"].format(hours=hours)]
    if ask:
        lines += ["", L["ask"]]
    else:
        when = " ".join(x for x in (r["delivery_date"], dtime(r, LANG3.index(lang) if lang in LANG3 else 0)) if x)
        lines += ["", L["when"].format(when=when), L["who"].format(oid=r["order_id"], name=r["sender_name"] or "-"), L["friend"], L["nofee"]]
    return "\n".join(x for x in lines if x is not None)


def payment_text(r, lang, intro=None, outro=None, why=""):
    """The message sent after the owner's first confirmation: every item with its price, delivery, TOTAL TO PAY, then the bank details. The prices, the bank details and the amounts are pure code; only the opening line and the closing line can be worded by the model (intro / outro)."""
    L = PAY_WORDS.get(lang, PAY_WORDS["en"])
    pay, areas = _json("payment.json"), _json("delivery.json")["areas"]
    out = [intro or L["order"] + f" #{r['order_id']} 🌸"]
    if why:  # an approved change: what changed and why the amount is different, before the new breakdown
        out += [why, ""]
    by_name = {_nm(p["name"]): p for p in graph()[0].values() if p.get("type") == "product"}
    out += [f"• {q} x {display(by_name[nm], lang) if nm in by_name else nm} - {fmt_tl(q * u)}" for nm, q, u in (row_lines(r) or [])]  # item names in the customer's language
    for nm, pr in custom_rows(r):
        out.append(f"• {nm} - {fmt_tl(pr or 0)}")
    a = areas.get(r["delivery_area"])
    label = a["label"].get(lang) or a["label"]["en"] if a else ""
    fee = fmt_tl(r["delivery_fee"]) if D(r["delivery_fee"]) else L["free"]
    if r["delivery_type"] == "pickup":
        out.append(L["pickup"] + fee)
    elif r["delivery_type"] == "jet":
        out.append(L["jet"].format(label=label, time=dtime(r, LANG3.index(lang) if lang in LANG3 else 0)) + fee)
    else:
        out.append(L["normal"].format(label=label, dep=a["departure"] if a else "") + fee)
    adj = adjustment(r)
    if adj:  # the owner set the amount himself at the confirmation: shown as its own line
        out.append((L["fee"] + "+" if adj > 0 else L["discount"] + "-") + fmt_tl(abs(adj)))
    paid = D(r["paid"] or 0)
    if paid > 0:  # an approved change that costs more: only the difference is asked
        out.append(L["newtotal"] + fmt_tl(r["payable"]))
        out.append(L["alreadypaid"] + fmt_tl(paid))
        out.append(L["paynow"] + fmt_tl(D(r["payable"]) - paid))
    else:
        out.append(L["total"] + fmt_tl(r["payable"]))
    out += ["", pay["holder"], pay["bank"], pay["iban"]] + ([pay["extra"]] if pay.get("extra") else []) + [outro or pay["receipt_line"].get(lang) or pay["receipt_line"]["en"]]
    if r["delivery_type"] == "pickup":  # where and when to come, again, under the bank details
        out += ["", pickup_block(r, lang)]
    return "\n".join(out)


def delivery_policy_text():
    """Plain-text delivery price list for the business info the agent answers from."""
    d = _json("delivery.json")
    rows = [f"- {a['label']['en']}: minimum order {fmt_tl(a['minimum'])}; normal delivery free (leaves {a['departure']}); jet (timed) delivery "
            + (f"+{fmt_tl(a['jet_fee'])}" if a["jet_fee"] is not None else "fee set by the team") for a in d["areas"].values()]
    return d["note"] + "\n" + "\n".join(rows)


def change_preview(r):
    """The order as it WOULD be after the customer's pending change request (nothing is written). Returns (proposed row, delivery calc | None)."""
    req = json.loads(r["change_request"])
    p = dict(r)
    if req["kind"] == "cancel":
        p["status"] = "rejected"  # an approved cancellation is booked as Lost / rejected
        return p, None
    if req.get("lines") is not None:
        p.update(_cols([(n, q, D(u)) for n, q, u in req["lines"] if q > 0]))
    det_ = dict(req.get("details") or {})
    end_ = det_.pop("deliver_before", "")
    for k, v in det_.items():
        p[k] = v
    if "delivery_time" in det_ or end_:  # a new time brings its own deadline (or none): the old "before" never stays behind
        p["notes"] = re.sub(r" \| before: \d\d:\d\d", "", p["notes"])
        if end_ and p["delivery_time"]:
            p["notes"] += f" | before: {end_}"
    if req.get("note"):
        p["notes"] += f" | note: {req['note']}"
    if p["delivery_type"] == "pickup":
        p["address"] = "PICKUP at the shop (Girne)"
    calc = delivery_calc(p)
    p["delivery_area"] = calc["key"]
    if calc["fee"] is not None:
        p["delivery_fee"] = mstr(calc["fee"])
    recompute_row(p)
    return p, calc


def request_change(oid, phone, req):
    """Store a customer's change / cancel request on a locked order. The order itself is NOT touched until the owner approves."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid) and x["customer_number"] == phone and (x["status"] in LOCKED or (req.get("kind") == "cancel" and x["status"] in OPEN))), None)
        if not r:
            return None
        if r["change_request"] and json.loads(r["change_request"]).get("kind") == req.get("kind") == "cancel":
            return dict(r)  # the same cancellation is already waiting for the owner: no second request
        r["change_request"] = json.dumps({**req, "ts": time.time()}, ensure_ascii=False)
        _write(rows)
        return dict(r)


def change_apply(oid, requote=False):
    """The owner approved: write the change into orders.csv. Returns (new row, balance, why-not). balance = new payable - paid (> 0: the customer owes the difference).
    requote=True: a direct admin edit before payment confirmation resets approval: the order goes back to the first-confirmation step (status pending), a new invoice follows after the customer and the owner confirm it again."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or not r["change_request"] or not (r["status"] in LOCKED or (r["status"] in OPEN and json.loads(r["change_request"])["kind"] == "cancel")):
            return None, None, "there is no pending change request on this order"
        p, calc = change_preview(r)
        if calc and [x for x in calc["problems"] if "minimum order" in x or "not recognised" in x]:
            return None, None, "; ".join(calc["problems"])
        if p["status"] != "rejected" and p["payable"] == "" and not requote:
            return None, None, "the new amount to pay cannot be computed yet (delivery fee unknown): set the delivery fee first"
        p["change_request"] = ""
        if requote and p["status"] != "rejected":
            p["status"] = "pending"
            r.update(p)
            _write(rows)
            return dict(r), D(p["payable"] or 0) - D(r["paid"] or 0), ""
        bal = D(p["payable"] or 0) - D(r["paid"] or 0) if p["status"] != "rejected" else -D(r["paid"] or 0)
        if p["status"] != "rejected" and bal > 0:
            p["status"] = "awaiting_payment"  # pays only the difference
        elif bal < 0:  # money to give back: written in the notes, and `paid` follows the new amount so later changes are measured against it
            p["notes"] += f" | refund due {mstr(-bal)}"
            if p["status"] != "rejected":
                p["paid"] = p["payable"]
        r.update(p)
        _write(rows)
        return dict(r), bal, ""


DIFF_STATES = ("awaiting_difference", "difference_review")  # an approved change that costs more: the paid order stays untouched until the difference is verified
LEDGER_LOCK = threading.Lock()


def ledger_add(oid, kind, amount, status="done", note=""):
    """Append-only money log (payments.jsonl, one JSON line per event, never edited or deleted): kind = payment | extra_payment | refund_due | refund | adjustment. Returns the entry id."""
    with LEDGER_LOCK:
        n = len(ledger_events()) + 1
        with open(P("payments.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": n, "ts": now().isoformat(timespec="seconds"), "order_id": int(oid), "kind": kind, "amount": mstr(D(amount)), "status": status, "note": note}, ensure_ascii=False) + "\n")
        return n


def ledger_events():
    try:
        return [json.loads(x) for x in P("payments.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []


def ledger(oid):
    """Every money event of an order, oldest first. An order paid before the log existed shows one `payment` entry made from its `paid` column (nothing is rewritten)."""
    ev = [e for e in ledger_events() if e["order_id"] == int(oid)]
    r = order_row(oid)
    if r and D(r["paid"] or 0) > 0 and not any(e["kind"] in ("payment", "extra_payment") for e in ev):
        ev.insert(0, {"id": 0, "ts": r["timestamp"], "order_id": int(oid), "kind": "payment", "amount": mstr(D(r["paid"])), "status": "legacy", "note": "from the paid column"})
    return ev


def change_state(r):
    return (json.loads(r["change_request"]).get("state") or "") if r.get("change_request") else ""


def change_hold(oid):
    """The owner approved a change that costs MORE on a paid order: the order is not touched; the request gets state awaiting_difference with the exact amount. Returns (preview row, difference) or (None, why)."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or not r["change_request"] or r["status"] not in LOCKED or change_state(r):
            return None, "there is no new change request on this order"
        p, calc = change_preview(r)
        if calc and [x for x in calc["problems"] if "minimum order" in x or "not recognised" in x]:
            return None, "; ".join(calc["problems"])
        diff = D(p["payable"] or 0) - D(r["paid"] or 0)
        if diff <= 0:
            return None, "no extra payment is needed"
        r["change_request"] = json.dumps({**json.loads(r["change_request"]), "state": "awaiting_difference", "diff": mstr(diff), "ts_hold": time.time()}, ensure_ascii=False)
        _write(rows)
        return p, diff


def change_complete(oid):
    """The difference was verified by the owner: NOW the paid order is updated (same id), paid follows the new total, the extra payment goes to the ledger. Returns (row, difference) or (None, why)."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or change_state(r) not in DIFF_STATES:
            return None, "no extra payment is pending on this order"
        diff = D(json.loads(r["change_request"])["diff"])
        p, _ = change_preview(r)
        p["change_request"], p["paid"] = "", mstr(D(r["paid"] or 0) + diff)
        r.update(p)
        _write(rows)
    ledger_add(oid, "extra_payment", diff, note="difference of an approved change, verified by the owner")
    return dict(r), diff


def waiting_rows(phone):
    """Orders of this customer that wait for money: normal ones, plus a paid order whose approved change waits for its difference (shown with payable = the difference, `_diff` set)."""
    out = []
    for r in orders():
        if r["customer_number"] != phone:
            continue
        if r["status"] in ("awaiting_payment", "payment_review"):
            out.append(r)
        elif r["status"] in LOCKED and change_state(r) in DIFF_STATES:
            out.append({**r, "payable": json.loads(r["change_request"])["diff"], "_diff": "1"})
    return out


def mark_review(row):
    """The customer sent proof: normal orders go to payment_review; for a pending difference only the request changes (the paid order keeps its status)."""
    if row.get("_diff"):
        with _lock:
            rows = orders()
            r = next((x for x in rows if int(x["order_id"]) == int(row["order_id"])), None)
            if r and change_state(r) == "awaiting_difference":
                r["change_request"] = json.dumps({**json.loads(r["change_request"]), "state": "difference_review"}, ensure_ascii=False)
                _write(rows)
        return
    order_update(row["order_id"], ("awaiting_payment", "payment_review"), status="payment_review")


def change_drop(oid):
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or not r["change_request"]:
            return None
        r["change_request"] = ""
        _write(rows)
        return dict(r)


def audit_orders():
    return {r["order_id"]: p for r in orders() if (p := row_problems(r))}


def recent_orders_brief(n=6):
    """'#12 - 2 x Red Roses; 1 x Teddy Bear' for the latest orders: number and what the order is, nothing else."""
    return "\n".join(f"#{r['order_id']} - {_items_text(r) or '(empty)'}" for r in orders()[-n:]) or "(no orders)"


def active_orders_brief(n=6):
    """Like recent_orders_brief, but only the orders that can still be acted on (not shipped, not rejected / cancelled), with the stage."""
    rows = [r for r in orders() if r["status"] not in ("completed", "rejected", "cancelled")][-n:]
    return "\n".join(f"#{r['order_id']} - {_items_text(r) or '(empty)'}" for r in rows) or "(no active orders)"


def dup_order(phone, items, quantities="", custom="", minutes=60):
    """Same customer, same items, still open, created in the last hour -> the existing row (stops double orders/quotes)."""
    for r in reversed(orders()):
        if r["customer_number"] == phone and r["status"] in ("pending", "quote", "quoted") and r["items"] == items and r["quantities"] == quantities \
                and r["custom_items"] == custom and (now() - datetime.fromisoformat(r["timestamp"])).total_seconds() < minutes * 60:
            return r
    return None


def orders():
    p = P("orders.csv")
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [{k: (r.get(k) or "") for k in OCOLS} for r in csv.DictReader(f)]


def _write(rows):
    import io
    try:
        track_lost(rows)  # first: it may adjust rows (a refunded sale)
    except Exception as e:  # bookkeeping for the dashboard must never block saving an order
        print(f"track_lost failed: {e!r}")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, OCOLS)
    w.writeheader()
    w.writerows(rows)
    if P("orders.csv").exists():
        shutil.copy(P("orders.csv"), P("orders.csv.prev"))  # the version before this write: a safety net if the file is ever damaged
    atomic_write(P("orders.csv"), buf.getvalue())


LOST_STATUSES = ("rejected", "cancelled")


def track_lost(rows):
    """When did each order become LOST? (kv `lostat`, {order id: iso time}): the dashboard board shows a lost order for one day, then only the History keeps it.
    An order seen as lost for the first time in the old file gets a time of 'long ago'."""
    prev = {r["order_id"]: r for r in orders()}
    old = {k: r["status"] for k, r in prev.items()}
    for r in rows:  # a PAID order (won, shipped, or money received) moved to lost / rejected: the sale is undone everywhere. The refund is made by hand outside the system, so it is only recorded here.
        o = prev.get(r["order_id"])
        if o and r["status"] in LOST_STATUSES and o["status"] not in LOST_STATUSES and (o["status"] in ("confirmed", "completed") or D(o["paid"] or 0) > 0):
            amt = D(o["paid"] or 0) or D(o["payable"] or o["total"] or 0)
            r["paid"] = ""
            r["notes"] = (r["notes"] or "") + f" | refunded outside the system: {mstr(amt)}"
            for key in ("orders_confirmed", "orders_shipped") if o["status"] == "completed" else ("orders_confirmed",):
                try:
                    stat(key, -1)  # the daily / weekly / monthly counters no longer count this sale
                except Exception as e:
                    print(f"stat undo failed: {e!r}")
    dn = jget("doneat") or {}
    for r in rows:  # when each order was completed (shipped): the weekly / monthly reports count it in that period
        oid = str(r["order_id"])
        if r["status"] == "completed" and oid not in dn:
            dn[oid] = now().isoformat(timespec="seconds") if old.get(oid) != "completed" else "2000-01-01T00:00:00"
            kset("doneat", json.dumps(dn))
        elif r["status"] != "completed" and oid in dn:
            del dn[oid]
            kset("doneat", json.dumps(dn))
    d = jget("lostat") or {}
    changed = False
    for r in rows:
        oid, lost = str(r["order_id"]), r["status"] in LOST_STATUSES
        if lost and (old.get(oid) not in LOST_STATUSES and oid in old or oid not in old):
            d[oid] = now().isoformat(timespec="seconds"); changed = True
        elif lost and oid not in d:
            d[oid] = "2000-01-01T00:00:00"; changed = True
        elif not lost and oid in d:
            del d[oid]; changed = True
    if changed:
        kset("lostat", json.dumps(d))


def due_at(r):
    """When the order was due to be delivered: the date and the time of the order (a normal delivery without a time leaves at its area's departure time), or None when there is no readable date."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", r["delivery_date"] or "")
    if not m:
        return None
    t = re.search(r"(\d{1,2})[:.](\d{2})", r["delivery_time"] or "") or re.search(r"(\d{1,2})[:.](\d{2})", (_json("delivery.json")["areas"].get(r["delivery_area"]) or {}).get("departure") or "")
    try:
        h, mi = (int(t.group(1)), int(t.group(2))) if t else (12, 0)
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), h, mi)
    except ValueError:
        return None


def is_late(r, now_=None, grace_hours=1):
    """A paid order (confirmed) that nobody marked as shipped one hour after its delivery time."""
    d = due_at(r)
    return r["status"] == "confirmed" and d is not None and (now_ or now()).replace(tzinfo=None) > d + timedelta(hours=grace_hours)


def late_check():
    """Tell the owners ONCE about each late order (every few minutes from the server loop)."""
    for r in orders():
        if is_late(r) and not kget(f"late_alerted:{r['order_id']}"):
            kset(f"late_alerted:{r['order_id']}", time.time())
            d = due_at(r)
            oid = int(r["order_id"])
            alert("order", card("⏰", f"Order #{oid} is LATE", f"{kget('name:' + r['customer_number']) or r['customer_name'] or '?'} ({plus(r['customer_number'])})",
                                f"*Due:* {d:%d %b %H:%M} · {(now().replace(tzinfo=None) - d).total_seconds() / 3600:.1f} h ago\n*Address:* {r['address']}", order_block(r),
                                todo=f"If it is delivered, click Shipped in the dashboard or tell me to mark #{oid} as shipped."), [oid, r["customer_number"], "late order", ""], phone=r["customer_number"], oid=oid)


def lost_recent(r, hours=24):
    """A lost order younger than `hours` still shows in the dashboard's Lost column."""
    if r["status"] not in LOST_STATUSES:
        return False
    ts = (jget("lostat") or {}).get(str(r["order_id"]))
    return bool(ts) and (now() - datetime.fromisoformat(ts)).total_seconds() < hours * 3600


def restore_orders_if_broken():
    """At start: if orders.csv is unreadable or empty while the previous version has orders, put the previous version back. Returns a message or ''."""
    p, prev = P("orders.csv"), P("orders.csv.prev")
    try:
        with open(p, newline="", encoding="utf-8") as f:
            ok = "order_id" in (csv.DictReader(f).fieldnames or [])
    except Exception:
        ok = False
    if p.exists() and ok:
        return ""
    if prev.exists():
        shutil.copy(prev, p)
        return "orders.csv was damaged and has been restored from the previous version"
    return "" if not p.exists() else "orders.csv is damaged and there is no previous version: restore it from backups/"


def _cols(cur):
    return {"items": "; ".join(n for n, _, _ in cur), "quantities": "; ".join(str(q) for _, q, _ in cur), "unit_prices": "; ".join(mstr(u) for _, _, u in cur)}


def order_create(phone, name, lines, address, date, notes, custom="", status=None, custom_price=""):
    """lines = [(priced product, qty)]: every item is stored with the price it had when read. A custom item may be priced by the owner at entry."""
    with _lock:
        rows = orders()
        oid = max([int(r["order_id"]) for r in rows] + [0]) + 1
        row = {**{k: "" for k in OCOLS}, "order_id": oid, "timestamp": now().isoformat(timespec="seconds"), "customer_number": phone, "customer_name": name,
               **_cols([(_nm(p["name"]), n, D(unit_price(p))) for p, n in lines]), "subtotal": "0", "custom_items": custom, "custom_price": mstr(custom_price) if custom_price not in (None, "") else "",
               "sender_phone": "" if str(phone).startswith(("tg:", "ext:")) else phone, "address": address, "delivery_date": date, "notes": notes,
               "status": status or ("quote" if custom and not custom_price else "pending")}
        rows.append(recompute_row(row))
        _write(rows)
        return oid


OPEN = ("pending", "quote", "quoted")


def open_order(phone):
    """The customer's newest order that is not confirmed/rejected yet (new items get merged into it; a confirmed order stays separate)."""
    return next((r for r in reversed(orders()) if r["customer_number"] == phone and r["status"] in OPEN), None)


def clean_custom(s):
    """A custom item's title: 'customer wants' words, brackets and end punctuation removed ('One single white rose (customer wants 1 white rose).' -> 'One single white rose')."""
    s = re.sub(r"\([^)]*\)", "", str(s or ""))
    s = re.sub(r"^\s*(?:the )?(?:customer|they|he|she)\s+(?:wants|would like|asks for)\s+", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip(" .;,-")[:80]


def same_custom(a, b):
    a, b = fold(clean_custom(a)), fold(clean_custom(b))
    return bool(a and b and (a in b or b in a))


def order_merge(oid, lines=(), other="", address="", date="", notes=""):
    """Add items to an unconfirmed order; every line keeps its price, totals recomputed by code. Returns (row, price_reset).
    Same-name lines add up (at the newest price). Unpriced additions go to custom_items and, like any change after the owner priced the custom part, clear that price."""
    with _lock:
        rows = orders()
        r = next(x for x in rows if int(x["order_id"]) == int(oid))
        assert r["status"] in OPEN
        was_priced = r["status"] == "quoted" or bool(r["custom_price"])
        cur = [list(x) for x in (row_lines(r) or [])]
        for p, n in lines:
            hit = next((x for x in cur if x[0] == _nm(p["name"])), None)
            if hit:
                hit[1], hit[2] = hit[1] + n, D(unit_price(p))
            else:
                cur.append([_nm(p["name"]), n, D(unit_price(p))])
        r.update(_cols(cur))
        other = "; ".join(o for o in (clean_custom(x) for x in str(other or "").split(";")) if o and not any(same_custom(o, h) for h in r["custom_items"].split(";")))  # the same wish is never added twice
        if other:
            r["custom_items"] = (r["custom_items"] + "; " + other) if r["custom_items"] else other
        if r["custom_items"] and other:
            r["custom_price"] = ""  # the new item has no price yet (the others keep theirs: see sync_custom)
            r["status"] = "quote"
        elif was_priced and not r["custom_items"]:
            r["status"] = "pending"
        if address and r["address"] and address.strip().lower() != r["address"].strip().lower():
            r["notes"] += f" | ALSO deliver to: {address} ({date or '-'})"
        else:
            r["address"] = r["address"] or address
            r["delivery_date"] = r["delivery_date"] or date
        if notes:
            r["notes"] += f" | added later: {notes}"
        recompute_row(r)
        _write(rows)
        return dict(r), was_priced and r["total"] == ""


def order_set_lines(oid, wanted, clear_custom=False, custom_items=None, custom_price=None):
    """Replace an unconfirmed order's catalog lines; optional custom fields replace its custom part at the owner's entered aggregate price."""
    with _lock:
        rows = orders()
        r = next(x for x in rows if int(x["order_id"]) == int(oid))
        assert r["status"] in OPEN
        before = row_lines(r) or []
        was_priced = r["status"] == "quoted" or bool(r["custom_price"])
        r.update(_cols([(_nm(p["name"]), n, D(unit_price(p))) for p, n in wanted if n > 0]))
        if custom_items is not None:
            r["custom_items"] = str(custom_items)
            r["custom_lines"] = ""
            r["custom_price"] = mstr(D(custom_price)) if custom_price not in (None, "") else ""
        elif clear_custom:
            r["custom_items"], r["custom_price"], r["custom_lines"] = "", "", ""
        if not r["items"] and not r["custom_items"]:
            r["status"] = "rejected"
            r["notes"] += " | cancelled by the customer (all items removed)"
        elif r["custom_items"]:
            if custom_items is None:
                r["custom_price"], r["custom_lines"], r["status"] = "", "", "quote"  # the basket changed: the custom items are priced again
            else:
                r["status"] = "pending" if r["custom_price"] else "quote"
        else:
            r["status"] = "pending"
        recompute_row(r)
        _write(rows)
        return dict(r), before, was_priced


def order_price_custom(oid, total, status="quoted"):
    """The owner's custom-inclusive goods total; pending is used when pricing and first confirmation happen together."""
    with _lock:
        rows = orders()
        r = next((x for x in rows if int(x["order_id"]) == int(oid)), None)
        if not r or r["status"] not in OPEN:
            return None, "order not found or already decided"
        if not r["custom_items"]:
            return None, "this order has no custom part to price: catalog items are priced by the shop's pages and no discounts are given"
        sub = D(r["subtotal"] or 0)
        if D(total) <= sub:
            return None, f"the total must be higher than the catalog items' subtotal {fmt_tl(sub)} (the difference is the price of: {r['custom_items']})"
        r["custom_price"], r["status"] = mstr(D(total) - sub), status
        recompute_row(r)
        _write(rows)
        return dict(r), ""


STATUSES = ("pending", "quote", "quoted", "awaiting_payment", "payment_review", "confirmed", "completed", "rejected", "cancelled")  # completed = shipped / delivered: it leaves the board and lives in the dashboard History
LOCKED = ("awaiting_payment", "payment_review", "confirmed", "completed")  # items are locked: changes / cancellation need the owner's approval


def apply_orders_edit(new, base=None):
    """The owner's python edit of the orders list: validate, back up the current file, write, report the diff. Nothing invalid is ever written.
    base = the list the caller read before editing: if the file changed since (a payment confirmed meanwhile), only the caller's own field changes are applied on top of the current rows, so nobody's write is lost."""
    if not isinstance(new, list) or not all(isinstance(r, dict) for r in new):
        return "REFUSED: `orders` must stay a list of row dicts. Nothing was changed."
    rows = []
    for r in new:
        bad = set(r) - set(OCOLS)
        if bad:
            return f"REFUSED: unknown columns {sorted(bad)}. Nothing was changed."
        row = {k: "" if r.get(k) is None else str(r.get(k)) for k in OCOLS}
        try:
            int(row["order_id"])
        except ValueError:
            return f"REFUSED: bad order_id {row['order_id']!r}. Nothing was changed."
        if row["status"] not in STATUSES:
            return f"REFUSED: status must be one of {STATUSES}, got {row['status']!r}. Nothing was changed."
        rows.append(recompute_row(row))
    ids = [int(r["order_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        return "REFUSED: duplicate order_id. Nothing was changed."
    with _lock:
        old = orders()
        if base is not None and base != old:
            b, mine = {r["order_id"]: r for r in base}, {r["order_id"]: r for r in rows}
            merged = []
            for c in old:
                oid = c["order_id"]
                if oid in b and oid not in mine:
                    continue  # the caller deleted it
                if oid in b and oid in mine:
                    c = {**c, **{k: mine[oid][k] for k in OCOLS if mine[oid][k] != b[oid][k]}}
                merged.append(c)
            have = {c["order_id"] for c in old}
            rows = [recompute_row(dict(r)) for r in merged] + [r for r in rows if r["order_id"] not in b and r["order_id"] not in have]
        if rows == old:
            return "No change."
        backup = order_backup_path()
        if P("orders.csv").exists():
            shutil.copy(P("orders.csv"), backup)
        _write(rows)
        prune_order_backups()
    o, n = {int(r["order_id"]): r for r in old}, {int(r["order_id"]): r for r in rows}
    lines = [f"removed order(s): {sorted(set(o) - set(n))}" if set(o) - set(n) else "", f"added order(s): {sorted(set(n) - set(o))}" if set(n) - set(o) else ""]
    for i in sorted(set(o) & set(n)):
        ch = [f"{k}: {o[i][k]!r} -> {n[i][k]!r}" for k in OCOLS if o[i][k] != n[i][k]]
        if ch:
            lines.append(f"#{i}: " + "; ".join(ch)[:300])
    return f"Applied (backup {backup.name}; ask to undo if wrong). " + " | ".join(x for x in lines[:14] if x)


ORDER_BACKUPS_KEEP = 10


def order_backup_path(tag=""):
    d = P("order_backups")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"orders_{now():%Y%m%d_%H%M%S_%f}{tag}.csv"


def prune_order_backups():
    """Only the newest ORDER_BACKUPS_KEEP copies of orders.csv taken before an edit stay: enough to undo, not a growing pile."""
    for old in sorted(P("order_backups").glob("orders_*.csv"))[:-ORDER_BACKUPS_KEEP]:
        old.unlink()


def undo_orders_edit():
    bs = sorted(P("order_backups").glob("orders_*.csv")) if P("order_backups").is_dir() else []
    if not bs:
        return "No backup to restore."
    last = bs[-1]
    with _lock:
        if P("orders.csv").exists():
            shutil.copy(P("orders.csv"), P("order_backups") / f"undone_{now():%Y%m%d_%H%M%S_%f}.csv")  # outside the orders_*.csv pattern, so a second undo goes one edit further back
        shutil.copy(last, P("orders.csv"))
    last.unlink()
    return f"Restored orders from {last.name} ({len(orders())} orders)."


def orders_by_phone_status(phone, oid, status):
    return next((r for r in orders() if int(r["order_id"]) == int(oid) and r["customer_number"] == phone and r["status"] == status), None)


def order_update(oid, frm, **fields):
    """Update fields if the order is currently in one of the statuses in frm; returns the row or None."""
    with _lock:
        rows = orders()
        for r in rows:
            if int(r["order_id"]) == int(oid) and r["status"] in frm:
                r.update(fields)
                _write(rows)
                return r
    return None


def order_set(oid, status):
    """Returns the row if it was pending (and updates it), else None."""
    with _lock:
        rows = orders()
        for r in rows:
            if int(r["order_id"]) == int(oid):
                if r["status"] != "pending":
                    return None
                r["status"] = status
                _write(rows)
                return r
    return None


# ---------------------------------------------------------------- stats csv (daily/weekly/monthly)
KEYS = ["unique_customers", "new_customers", "returning_customers", "messages", "voice_notes", "photos",
        "orders_pending", "orders_confirmed", "orders_rejected", "custom_requests", "blocked_items"]
FILES = {"day": "daily.csv", "week": "weekly.csv", "month": "monthly.csv"}


def _periods():
    d = now()
    iso = d.isocalendar()
    return {"day": d.strftime("%Y-%m-%d"), "week": f"{iso[0]}-W{iso[1]:02d}", "month": d.strftime("%Y-%m")}


def _bump(kind, per, key, n=1):
    q("INSERT INTO stat VALUES(?,?,?,?) ON CONFLICT(kind,period,key) DO UPDATE SET n=n+?", kind, per, key, n, n)


def dump_stats():
    for kind, fn in FILES.items():
        data = {}
        for per, key, n in q("SELECT period,key,n FROM stat WHERE kind=?", kind):
            data.setdefault(per, {})[key] = n
        with open(P(fn), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["period"] + KEYS)
            for per in sorted(data):
                w.writerow([per] + [data[per].get(k, 0) for k in KEYS])


def stat(key, n=1):
    for kind, per in _periods().items():
        _bump(kind, per, key, n)
    dump_stats()


def register(phone):
    """Count a customer once per period; new = never seen before, returning = seen in an earlier period."""
    known = q("SELECT 1 FROM customers WHERE phone=?", phone)
    if str(phone).isdigit():
        link(phone, _tok_phone(phone))
    for kind, per in _periods().items():
        if not q("SELECT 1 FROM active WHERE kind=? AND period=? AND phone=?", kind, per, phone):
            q("INSERT INTO active VALUES(?,?,?)", kind, per, phone)
            _bump(kind, per, "unique_customers")
            _bump(kind, per, "returning_customers" if known else "new_customers")
    if not known:
        q("INSERT INTO customers VALUES(?,?)", phone, now().isoformat())
    prof = customers().get(phone)
    if prof and prof.get("hidden"):  # removed from the CRM earlier: a new message brings them back
        set_profile(phone, hidden=False)
    dump_stats()


# ---------------------------------------------------------------- calendar: what is coming up in North Cyprus
def _nth_sunday(y, m, n):
    d = datetime(y, m, 1).date()
    return d + timedelta((6 - d.weekday()) % 7 + 7 * (n - 1))


# Bayram dates move with the moon: approximate, extend the table yearly (shortcut: 2026-2028 only).
BAYRAM = {2026: ((3, 20), (5, 27)), 2027: ((3, 9), (5, 16)), 2028: ((2, 26), (5, 5))}


def upcoming_events(today):
    """[(name, date, days_left)] of occasions whose gift season has started (each has its own lead time in days)."""
    out = []
    for y in (today.year, today.year + 1):
        ev = [("Valentine's Day", datetime(y, 2, 14).date(), 21), ("International Women's Day", datetime(y, 3, 8).date(), 10),
              ("Mother's Day", _nth_sunday(y, 5, 2), 14), ("Father's Day", _nth_sunday(y, 6, 3), 10),
              ("Teachers' Day", datetime(y, 11, 24).date(), 10), ("Christmas", datetime(y, 12, 25).date(), 14), ("New Year's Eve", datetime(y, 12, 31).date(), 10)]
        if y in BAYRAM:
            ev += [("Ramadan Bayram (Eid al-Fitr)", datetime(y, *BAYRAM[y][0]).date(), 10), ("Kurban Bayram (Eid al-Adha)", datetime(y, *BAYRAM[y][1]).date(), 10)]
        out += [(n, d, (d - today).days) for n, d, lead in ev if 0 <= (d - today).days <= lead]
    return sorted(out, key=lambda x: x[2])


# ---------------------------------------------------------------- health: failure counter + nightly backup
_fails = {}


_last_ok, _last_bad, BOOT = {}, {}, time.time()


def ok(kind):
    _fails.pop(kind, None)
    _last_ok[kind] = time.time()


KIND_LABEL = {"openai": "The AI assistant", "whatsapp": "WhatsApp", "telegram": "Telegram"}


def friendly_error(e):
    """An exception in plain words for the owner: (what happened, what to do). The raw error text never goes to the owner."""
    s = f"{type(e).__name__} {e}".lower()
    if "no credits" in s or "insufficient_quota" in s or "exceeded your current quota" in s or "billing" in s:
        return "The AI account has no credit left.", "Add credit at platform.openai.com > Billing. Until then customers get no AI answers."
    if "ratelimit" in s or "rate limit" in s or re.search(r"\b429\b", s):
        return "The AI service is too busy right now (too many requests at once).", "It retries by itself. If it keeps happening, wait a minute."
    if "timeout" in s or "timed out" in s:
        return "The AI took too long to answer.", "Usually a one-off. If it repeats, check the server's internet connection."
    if "graph.facebook.com" in s and (re.search(r"\b401\b|\b190\b", s) or "expired" in s or "access token" in s):
        return "The WhatsApp access token has expired or is invalid, so nothing can be sent or received through WhatsApp.", "In Meta (developers.facebook.com > your app > WhatsApp > API Setup, or a System User) create a new token, paste it as WHATSAPP_TOKEN in the .env file and restart the server. Telegram keeps working."
    if re.search(r"\b401\b", s) or "invalid api key" in s or "incorrect api key" in s or "authentication" in s:
        return "The AI key was not accepted.", "Check OPENAI_API_KEY in the .env file."
    if "recursion" in s:
        return "The assistant got stuck going in circles on this message.", "Answer the customer yourself; they can also just write again."
    if "connect" in s or "network" in s or "name resolution" in s or "unreachable" in s:
        return "Could not reach the service (connection problem).", "Check the server's internet connection."
    return "Something unexpected went wrong.", f"Technical detail for the developer: {type(e).__name__}."


def problem(kind, err):
    """Count consecutive failures of one kind (openai, whatsapp); from the 3rd on, tell the owner in plain words, at most once an hour."""
    n, t = _fails.get(kind, (0, 0))
    _fails[kind] = (n + 1, t)
    _last_bad[kind] = (time.time(), friendly_error(err)[0] if isinstance(err, BaseException) else str(err))
    if n + 1 >= 3 and time.time() - t > 3600:
        _fails[kind] = (n + 1, time.time())
        what, todo = friendly_error(err) if isinstance(err, BaseException) else (str(err), "")
        for a in admins():
            try:
                send_text(a, f"⚠️ {KIND_LABEL.get(kind, kind.title())} has failed {n + 1} times in a row.\n\n*What happened:* {what}\n*What to do:* {todo or 'Check it and try again.'}\n\nCustomers may not be getting answers until this is fixed.")
            except Exception:
                pass


def health_checks():
    """How the agent (Leyla) is doing, in plain words, for her card: [{"name", "status": ok | warn | bad | off, "note"}]. Nothing here calls a service: it reads what the agent already knows."""
    out = []
    ago = lambda t: (lambda m: f"{int(m)} min ago" if m < 120 else f"{m / 60:.0f} h ago")((time.time() - t) / 60)
    for kind, label, conf in (("openai", "AI brain", bool(cfg("OPENAI_API_KEY"))), ("whatsapp", "WhatsApp", bool(cfg("WHATSAPP_TOKEN") and cfg("PHONE_NUMBER_ID")))):
        n = _fails.get(kind, (0, 0))[0]
        if not conf:
            out.append({"name": label, "status": "bad" if kind == "openai" else "off", "note": "no key set" if kind == "openai" else "not connected"})
        elif kind == "whatsapp" and kget("wa_state"):
            out.append({"name": label, "status": "bad", "note": kget("wa_state")})
        elif n >= 3:
            out.append({"name": label, "status": "bad", "note": _last_bad.get(kind, (0, "failing"))[1]})
        elif n:
            out.append({"name": label, "status": "warn", "note": f"{n} failed attempt(s) just now"})
        else:
            out.append({"name": label, "status": "ok", "note": ("last worked " + ago(_last_ok[kind])) if kind in _last_ok else "ready"})
    last = kget("refresh_at")
    hrs = (time.time() - float(last)) / 3600 if last else None
    out.append({"name": "Prices", "status": "warn" if hrs is None or hrs > 36 else "ok", "note": "never updated" if hrs is None else f"updated {hrs:.0f} h ago"})
    up = (time.time() - BOOT) / 3600
    out.append({"name": "Running for", "status": "ok", "note": f"{up * 60:.0f} min" if up < 2 else f"{up / 24:.1f} days" if up > 48 else f"{up:.0f} h"})
    return out


BACKUP_FILES = ("orders.csv", "customers.json", "custom.json", "overrides.json", "removed.json", "prices.json", "payments.jsonl", "dash_users.json", "daily.csv", "weekly.csv", "monthly.csv")


def backup(keep=7):
    """Dated copy of the shop's state: chat.db (safe sqlite copy), orders, CRM, custom items, overrides, the stored prices, the payment log, the chats and the data/ files INCLUDING the product list
    (graph.json and the last scrape, data/raw/site.json). The last `keep` days stay. Returns the folder; backup_info(folder) says how it went."""
    d = P("backups") / now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    if P("chat.db").exists():
        with closing(sqlite3.connect(P("chat.db"))) as a, closing(sqlite3.connect(d / "chat.db")) as b:
            a.backup(b)
    for f in BACKUP_FILES:
        if P(f).exists():
            shutil.copy(P(f), d / f)
    for folder in ("chats", "admin_chats"):
        if P(folder).exists():
            shutil.copytree(P(folder), d / folder, dirs_exist_ok=True)
    shutil.copytree(ROOT / "data", d / "data", ignore=shutil.ignore_patterns("raw", "*.bak"), dirs_exist_ok=True)
    site = ROOT / "data/raw/site.json"  # the last scrape, the source the product list is rebuilt from
    if site.exists():
        (d / "data/raw").mkdir(parents=True, exist_ok=True)
        shutil.copy(site, d / "data/raw/site.json")
    for old in sorted(x for x in P("backups").iterdir() if x.is_dir())[:-keep]:
        shutil.rmtree(old)
    return d


def backup_info(d):
    """How a finished backup went: sizes and what the product list in it holds; also kept as `backup_last` for the dashboard."""
    d = Path(d)
    files = [f for f in d.rglob("*") if f.is_file()]
    try:
        g = json.loads((d / "data/graph.json").read_text(encoding="utf-8"))
        n_prod = sum(1 for n in g["nodes"] if n.get("type") == "product")
    except Exception:
        n_prod = 0
    try:
        scraped = json.loads((d / "data/raw/site.json").read_text(encoding="utf-8")).get("scraped_at", "")
    except Exception:
        scraped = ""
    count = lambda f: len(json.loads((d / f).read_text(encoding="utf-8"))) if (d / f).exists() else 0
    info = {"dir": str(d), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1), "files": len(files), "products": n_prod, "scraped_at": scraped, "prices": count("prices.json"),
            "orders": len(list(csv.DictReader(open(d / "orders.csv", newline="", encoding="utf-8")))) if (d / "orders.csv").exists() else 0, "customers": count("customers.json"),
            "chats": len(list((d / "chats").glob("*.jsonl"))) if (d / "chats").exists() else 0, "kept": len([x for x in P("backups").iterdir() if x.is_dir()])}
    kset("backup_last", json.dumps({**info, "ts": time.time(), "ok": True}))
    return info


def backup_due(hour, today, refresh_day, refresh_busy, backup_exists):
    """When the nightly backup runs: not before 03:00, never while the nightly product update is running, and only AFTER that update finished today (so it holds the new product list and prices);
    from 05:00 on it goes ahead whatever happened to the update. Once a day."""
    return not backup_exists and hour >= 3 and not refresh_busy and (refresh_day == today or hour >= 5)


def backup_report(info):
    """How the backup went, for the owners (one short message)."""
    if not info.get("ok", True):
        return f"⚠️ The backup FAILED: {info.get('error')}. Nothing was lost, the live data is untouched; I will try again."
    return (f"🗄 Backup done ({now():%Y-%m-%d %H:%M}): {info['mb']} MB, {info['files']} files. Product list: {info['products']} products" + (f" (last scrape {info['scraped_at']})" if info["scraped_at"] else "")
            + f", {info['prices']} prices. Orders {info['orders']}, customers {info['customers']}, chats {info['chats']}. The last {info['kept']} days are kept.")


_VOCAB = {}


def _vocab():
    """Every word that appears in a product name (any language) or alias, folded: what a customer's misspelt word is corrected against."""
    nodes = graph()[0]
    key = id(nodes)
    if key not in _VOCAB:
        _VOCAB.clear()
        words = set()
        for p in nodes.values():
            if p["type"] == "product":
                for t in re.split(r"\W+", fold(" ".join([p["name"], p.get("name_en") or "", p.get("name_tr") or "", p.get("name_ru") or "", *(p.get("aliases") or [])]))):
                    if len(t) >= 3 and not t.isdigit():
                        words.add(t)
        _VOCAB[key] = sorted(words)
    return _VOCAB[key]


def normalize_query(q):
    """The keywords of what the customer wrote, in no particular order: folded, "5li" / "5'li" / "5 li" = the number 5, and a misspelt word ("kirmizli", "bouqet") replaced by the closest word that exists in the catalog (similarity >= 0.8). Words that are fine stay as they are."""
    import difflib
    t = re.sub(r"(\d+)\s*['’`]?\s*(?:li|lu)\b", r"\1 ", fold(q or ""))
    vocab, out = _vocab(), []
    known = set(vocab)
    for w in [w for w in re.split(r"\W+", t) if w]:
        if len(w) >= 4 and not w.isdigit() and w not in known:
            close = difflib.get_close_matches(w, vocab, n=1, cutoff=0.8)
            w = close[0] if close else w
        out.append(w)
    return " ".join(out)


def best_matches(text, n=3):
    """Catalog products whose NAME holds the keywords of a customer's sentence, in any order, typos fixed: only the words that exist in the catalog count (names, 'please', 'for my wife' do not), a number must match as a whole number, at least 2 keywords and every one of them must be in the name. [(id, name, url)]"""
    toks = [t for t in re.split(r"\W+", normalize_query(text)) if t]
    known = set(_vocab())
    stop = set("i me my we you your the a an of to for with and or in on at is it this that these those can could would will do does have has need needs want wants wish like please thanks thank hi hello hey get give send order buy take show some any one two new wife husband mom mum dad friend bir istiyorum istiyoruz isterim lutfen icin ve ile bu su bana benim merhaba selam var mi ver gonder siparis".split())
    kw = [t for t in dict.fromkeys(toks) if (t.isdigit() or t in known) and t not in stop]
    if len(kw) < 2:
        return []
    out = []
    for p in graph()[0].values():
        if p["type"] != "product" or p.get("available") is False:
            continue
        hay = fold(" ".join([p["name"], p.get("name_en") or "", p.get("name_tr") or "", p.get("name_ru") or "", *(p.get("aliases") or [])]))
        if all((re.search(rf"(?<!\d){re.escape(t)}(?!\d)", hay) if t.isdigit() else t in hay) for t in kw):
            out.append((len(hay), p["id"], p["name"], p.get("url") or "(no link)"))
    return [(i, nm, u) for _, i, nm, u in sorted(out)[:n]]


def closest_item(query):
    """(id, name, url) when ONE catalog item is clearly the closest to the keywords (it has strictly fewer unmatched name words than the next), else None."""
    hits = find_item(query)
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    toks = set(re.split(r"\W+", normalize_query(query)))
    ex = lambda h: len(set(re.split(r"\W+", fold(h[1]))) - toks - {""})
    return hits[0] if ex(hits[0]) < ex(hits[1]) else None


def find_item(query):
    """Products matching a name or a link: [(id, name, url)]. A link matches any of the product's pages exactly (query-string and trailing slash ignored)."""
    nodes = graph()[0]
    prods = [n for n in nodes.values() if n["type"] == "product"]
    cut = lambda u: (u or "").split("#")[0].rstrip("/")
    if re.match(r"https?://", query.strip()):
        hit = [p for p in prods if cut(query.strip()) in {cut(u) for u in (p.get("urls") or [])}]
    else:
        toks = [t for t in re.split(r"\W+", normalize_query(query)) if len(t) > 1 or t.isdigit()]
        hay = lambda p: fold(" ".join([p["name"], p.get("name_en") or "", p.get("name_tr") or "", p.get("name_ru") or "", *(p.get("aliases") or [])]))
        has = lambda p, t: bool(re.search(rf"(?<!\d){re.escape(t)}(?!\d)", hay(p))) if t.isdigit() else t in hay(p)  # a number is a whole number: 5 is not 25
        hit = [p for p in prods if toks and all(has(p, t) for t in toks)]
        exact = [p for p in hit if fold(p["name"]) == fold(query)]
        qset = set(toks)
        extra = lambda p: len(set(re.split(r"\W+", fold(p["name"]))) - qset - {""})  # words of the product's name the customer did not write: fewer = closer
        hit = exact or sorted(hit, key=lambda p: (extra(p), len(p["name"])))
    return [(p["id"], p["name"], p.get("url") or "(no link)") for p in hit]


ONGOING = ("pending", "quote", "quoted", "awaiting_payment", "payment_review", "confirmed")


def _who(r):
    h = handle_of(r["customer_number"])
    return (customers().get(r["customer_number"], {}).get("name") or r["customer_name"] or "?") + (f" {h}" if h else "") + f" ({plus(r['customer_number'])})"


def ship_text(day=None):
    """The delivery plan of one day (default today): paid orders due that day (to ship), on today also the late ones from before, and orders due that day that are NOT paid yet (a warning). '' when there is nothing."""
    day = day or now().date()
    due = lambda r: (lambda d: d.date() if d else None)(due_at(r))
    mine = sorted([r for r in orders() if r["status"] == "confirmed" and due(r) == day], key=lambda r: due_at(r))
    late = [r for r in orders() if is_late(r) and due(r) != day] if day == now().date() else []
    unpaid = [r for r in orders() if r["status"] in ("pending", "quote", "quoted", "awaiting_payment", "payment_review") and due(r) == day]
    line = lambda r: f"• *{_who(r)}* · #{r['order_id']} · {_items_text(r)[:60]}\n   🕒 {due_at(r):%H:%M} · 📍 {(r['address'] or '-')[:50]}" + (" · ⏰ LATE" if is_late(r) else "")
    out = []
    if mine:
        out.append(f"📦 *To ship {'today' if day == now().date() else day.strftime('%a %d %b')}* ({len(mine)})\n" + "\n".join(line(r) for r in mine))
    if late:
        out.append(f"⏰ *Late from before* ({len(late)})\n" + "\n".join(line(r) + f" · due {due_at(r):%d %b}" for r in late))
    if unpaid:
        out.append(f"⚠️ *Due {'today' if day == now().date() else day.strftime('%a %d %b')} but NOT paid yet* ({len(unpaid)})\n" + "\n".join(f"• *{_who(r)}* · #{r['order_id']} · {r['status'].replace('_', ' ')} · {fmt_tl(r['payable'] or r['total'] or 0)}" for r in unpaid))
    return "\n\n".join(out)


def summary(kind):
    """The MORNING report (kind='opening'): ONLY the on-going orders (grouped by who must act, however old) and the shipments planned for today. kind='closing' is the night Z-report: what was done today."""
    if kind != "opening":
        return report("day")
    rows, today = orders(), now()
    groups = [("🔴 Needs you", [r for r in rows if r["status"] in ("pending", "quote", "payment_review")]),
              ("🟡 Waiting for the customer", [r for r in rows if r["status"] in ("quoted", "awaiting_payment")]),
              ("🟢 Paid, to deliver", [r for r in rows if r["status"] == "confirmed"])]
    n = sum(len(rs) for _, rs in groups)
    out = [f"☀️ *Morning report* · {today:%a %d %b}" + (f"\n⏰ *Reminder: {n} on-going order(s)*" if n else "")]
    ship = ship_text()
    if ship:  # the delivery plan of the day comes first in the morning message
        out.append(ship)
    for title, rs in groups:
        if rs:
            out.append(f"*{title}* ({len(rs)})\n" + "\n".join(f"• *{_who(r)}*\n   {active_line(r)}" + (f" · delivery {r['delivery_date']}" if r["delivery_date"] else "") for r in rs[:15]) + ("\n…" if len(rs) > 15 else ""))
    if len(out) == 1:
        out.append("No updates")
    return "\n\n".join(out)


def period_bounds(kind, today=None):
    """(start, end, title): the day, the ISO week (Monday to Sunday) or the calendar month that `today` is in."""
    t = today or now()
    d0 = t.replace(hour=0, minute=0, second=0, microsecond=0)
    if kind == "week":
        s = d0 - timedelta(days=d0.weekday())
        return s, s + timedelta(days=7), f"Weekly report · {s:%d %b} – {(s + timedelta(days=6)):%d %b}"
    if kind == "month":
        s = d0.replace(day=1)
        e = (s + timedelta(days=32)).replace(day=1)
        return s, e, f"Monthly report · {s:%B %Y}"
    return d0, d0 + timedelta(days=1), f"Z-report · {d0:%a %d %b}"


CHEERS_TR = (  # hand-written Turkish closing lines for the reports (not generated)
    "Emeğinize sağlık, harika bir dönem oldu! 🌸", "Her sipariş bir gülümseme; devam edin! 💐", "Bugünkü çalışmanız için teşekkürler, böyle devam! 💪",
    "Çiçekler sizinle daha güzel açıyor! 🌷", "Müşterileriniz sizi seviyor, bunu hissediyoruz! ❤️", "Küçük adımlar, büyük başarılar. Aferin! 👏",
    "İşleriniz gül gibi gidiyor! 🌹", "Yorulmadan çalıştınız, bunun karşılığını alacaksınız! 🌟", "Bir sonraki dönem daha da güzel olacak! 🚀",
    "Her gün biraz daha büyüyoruz! 🌱", "Mükemmel bir takım çalışması! 🤝", "Başarılarınızın devamını dileriz! 🍀", "Siz harikasınız, devam! 🌻",
    "Bu emek boşa gitmeyecek! 🙌", "Çiçek gibi bir performans! 🌼", "Gururlandık, teşekkürler! 🏆", "Güzel işler, güzel insanlar! ✨",
    "Hedefe adım adım yaklaşıyoruz! 🎯", "Sabırla, azimle, gülümsemeyle! 😊", "Kazanan bir ekipsiniz! 🥇", "Her yeni gün yeni bir fırsat! ☀️",
    "Müşteri memnuniyeti sizinle başlar! 💚", "İyi ki varsınız, iyi ki çalışıyorsunuz! 🫶", "Tatlı bir başarı hikâyesi yazıyorsunuz! 🍫",
    "Bereketli olsun, bol kazançlar! 💰", "Dinlenin, yarın yeniden çiçek açacağız! 🌙", "Yolunuz açık, kazancınız bol olsun! 🛤️",
    "Harika gidiyorsunuz, sakın durmayın! 🔥", "Çalışkan eller, güzel çiçekler! 🌺", "Teşekkürler, bu dönem de çok iyiydi! 🌈")


def report_data(kind="day", today=None):
    """What happened in the period, as data: orders placed, COMPLETED (shipped in the period), ON-GOING (right now, any age) and CANCELLED / REJECTED (in the period)."""
    s, e, title = period_bounds(kind, today)
    rows = orders()
    inside = lambda ts: bool(ts) and s <= datetime.fromisoformat(ts) < e
    done, lost = jget("doneat") or {}, jget("lostat") or {}
    when = lambda r, d: d[str(r["order_id"])] if str(r["order_id"]) in d and d[str(r["order_id"])][:4] > "2000" else r["timestamp"]  # no record: the day it was placed
    ongoing = [r for r in rows if r["status"] in ONGOING]
    return {"kind": kind, "title": title, "start": s, "end": e,
            "placed": [r for r in rows if inside(r["timestamp"])],
            "completed": [r for r in rows if r["status"] == "completed" and inside(when(r, done))],
            "ongoing": ongoing,
            "cancelled": [r for r in rows if r["status"] in LOST_STATUSES and inside(when(r, lost))],
            "stage": {k: sum(1 for r in ongoing if r["status"] in v) for k, v in (("waiting for you", ("pending", "quote", "payment_review")), ("waiting for the customer", ("quoted", "awaiting_payment")), ("paid, to deliver", ("confirmed",)))}}


def report_value(r):
    return D(r["payable"] or r["total"] or 0)


def report(kind="day", today=None):
    """The report as text for the chat: the numbers, the orders, and a Turkish encouraging line at the end."""
    import random
    d = report_data(kind, today)
    val, total = report_value, lambda rs: fmt_tl(sum((report_value(r) for r in rs), D(0)))
    line = lambda r: f"• #{r['order_id']} {_who(r)} · {_items_text(r)[:60]}" + (f" · {fmt_tl(val(r))}" if val(r) else "")
    block = lambda head, rs: f"{head} ({len(rs)})" + (f" · {total(rs)}\n" + "\n".join(line(r) for r in rs[:8]) + ("\n…" if len(rs) > 8 else "") if rs else " – none")
    out = [f"{'🌙' if kind == 'day' else '📊'} *{d['title']}*",
           f"🛒 Orders placed: *{len(d['placed'])}*" + (f" · {total(d['placed'])}" if d["placed"] else ""),
           block("✅ *Completed*", d["completed"]),
           block("⏳ *On-going* (now)", d["ongoing"]) + ("\n   " + " · ".join(f"{n} {k}" for k, n in d["stage"].items() if n) if d["ongoing"] else ""),
           block("❌ *Cancelled / rejected*", d["cancelled"]),
           random.choice(CHEERS_TR)]
    return "\n\n".join(out)


def combos(pools, budget, limit=6):
    """One product from each pool (a list of {name, price, ...} per category), total price <= budget (a TOTAL for the whole basket, not per category).
    All sums are exact Decimal. Returns up to `limit` different combos, the fullest use of the budget first, no product more than twice: [(products, total)]."""
    import itertools
    budget = D(budget)
    if not pools or any(not p for p in pools) or budget <= 0:
        return []
    def sample(pool, n=12):  # cheapest first, thinned evenly so a big category cannot blow up the search
        pool = sorted((p for p in pool if p.get("price") and D(p["price"]) > 0), key=lambda p: D(p["price"]))
        return pool if len(pool) <= n else [pool[round(i * (len(pool) - 1) / (n - 1))] for i in range(n)]
    pools = [sample(p) for p in pools]
    ok = [(c, sum(D(p["price"]) for p in c)) for c in itertools.product(*pools)]
    ok = sorted(((c, t) for c, t in ok if t <= budget), key=lambda x: (-x[1], [p["name"] for p in x[0]]))
    out, used = [], {}
    for c, t in ok:
        if all(used.get(p["id"], 0) < 2 for p in c):
            out.append((c, t))
            for p in c:
                used[p["id"]] = used.get(p["id"], 0) + 1
        if len(out) >= limit:
            break
    return out
