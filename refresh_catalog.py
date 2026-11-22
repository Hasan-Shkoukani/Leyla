"""Full catalog refresh: wipe the scrape cache, re-scrape both sites -> merge -> graph.json -> photos -> matcher index. Stops at the first failing step.
While it runs customers get the 'maintenance' reply (kv `maintenance`) and are re-texted at the end via `back`.
Run by the admin tool `update_products` and the nightly job (02:00-05:00). By hand: python refresh_catalog.py (no maintenance mode)."""
import glob, json, shutil, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import core

ROOT = Path(__file__).parent
_lock = threading.Lock()
STEPS = [("scrape kibris", ["scraper/scrape_kibris.py"]), ("build graph", ["build_graph.py"]),  # one scraper, see its rules (names of all 3 languages straight from the pages, prices, badges, same-day)
         ("download photos", ["-m", "dl.download_images"]), ("rebuild photo index", ["-m", "dl.build_index"])]
BACKED = ("data/graph.json", "data/raw/site.json")


def names():
    """{id: name} of the scraped products in graph.json (the owner's own items live elsewhere and are never touched)."""
    try:
        return {n["id"]: n["name"] for n in json.loads((ROOT / "data/graph.json").read_text(encoding="utf-8"))["nodes"] if n.get("type") == "product"}
    except Exception:
        return {}


def count():
    return len(names())


def report(old, new):
    """'3 new, 1 removed, 2 renamed' plus the names (first 8 of each)."""
    add, gone = [new[i] for i in new if i not in old], [old[i] for i in old if i not in new]
    ren = [f"{old[i]} -> {new[i]}" for i in new if i in old and old[i] != new[i]]
    if not (add or gone or ren):
        return "No product changed."
    part = lambda label, xs: f"{len(xs)} {label}" + (": " + "; ".join(xs[:8]) + ("..." if len(xs) > 8 else "") if xs else "")
    return " | ".join(part(l, x) for l, x in (("new", add), ("removed", gone), ("renamed", ren)) if x)


def drop_stale_photos():
    """Photo folders of products that no longer exist (the owner's own c-N items stay)."""
    keep = set(names()) | set(core.customs())
    for d in (ROOT / "dl/data/catalog").glob("*"):
        if d.is_dir() and d.name not in keep:
            shutil.rmtree(d)


def refresh_prices(ids=None):
    """Read the price and stock of every product (or only `ids`) from its page and STORE them (prices.json): this is the only place prices are read. Returns (read, asked).
    An item whose page cannot be read keeps its previous stored price."""
    import prices
    nodes = core.graph()[0]
    items = [(n["id"], n.get("urls") or [n["url"]]) for n in nodes.values() if n["type"] == "product" and n.get("url") and not n.get("custom") and (not ids or n["id"] in ids)]
    got = prices.fetch_many(items, workers=4)
    core.save_prices({k: v for k, v in got.items() if v})
    return sum(1 for v in got.values() if v), len(items)


def busy():
    return _lock.locked()


PROGRESS = {"step": "", "since": 0, "stopping": False, "pages": 0}
_stop = threading.Event()
_proc = None
OFFLINE_STEPS = ("scrape kibris", "merge sites", "merge duplicates", "kibris only", "drop removed products", "build graph")  # the part that finishes a stopped update (no waiting on the shop)


def request_stop():
    """The owner stops a running update. The step that is running is ended at once; what was updated so far is merged with the previous list (see run). False if nothing runs."""
    if not busy():
        return False
    _stop.set()
    PROGRESS["stopping"] = True
    p = _proc
    if p and p.poll() is None:
        p.terminate()
    return True


def _step(args, stoppable=True, offline=False):
    """Run one pipeline step. Returns (return code, last output); the return code is None when the step was ended because the owner stopped the update."""
    global _proc
    import os
    env = dict(os.environ)
    if offline:
        env["SCRAPE_OFFLINE"] = "1"  # the scraper reads its cache only: pages that were not fetched stay as they were
    _proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    while True:
        try:
            out, err = _proc.communicate(timeout=1)
            if stoppable and _stop.is_set() and _proc.returncode != 0:  # ended by the stop request itself, not a failure
                return None, ""
            return _proc.returncode, (err or out)
        except subprocess.TimeoutExpired:
            if stoppable and _stop.is_set():
                _proc.terminate()
                try:
                    _proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    _proc.kill()
                    _proc.communicate()
                return None, ""


def _valid_json(f):
    try:
        json.loads(Path(f).read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def keep_previous_products():
    """Products of the previous list (site.json.bak) that are missing from the new site.json are added back as they were. Returns how many."""
    f, b = ROOT / "data/raw/site.json", ROOT / "data/raw/site.json.bak"
    if not (f.exists() and b.exists()):
        return 0
    new, old = json.loads(f.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
    have = {p.get("product_id") for p in new["products"]}
    missing = [p for p in old["products"] if p.get("product_id") not in have]
    if missing:
        new["products"] += missing
        new["product_count"] = len(new["products"])
        core.atomic_write(f, json.dumps(new, ensure_ascii=False, indent=1))
    return len(missing)


def finish_stopped(at_step):
    """The update was stopped while `at_step` ran. A step that was cut off may have left a half-written file (the .bak copy comes back), and when the stop came before the catalog was rebuilt,
    the pages already fetched (fresh) are merged with the pages of the previous run (kept as cache.prev) and the catalog is rebuilt from that mix, without waiting on the shop.
    The photo download, the photo index and the price reading are left for the next update. Returns a sentence."""
    for p in BACKED:
        if (ROOT / p).exists() and not _valid_json(ROOT / p) and (ROOT / (p + ".bak")).exists():
            shutil.copy(ROOT / (p + ".bak"), ROOT / p)
    names_ = [n for n, _ in STEPS]
    if names_.index(at_step) > names_.index("build graph"):
        return "Stopped after the new product list was built; only the photos and prices were left for the next update."
    fresh = 0
    for d in ("cache", "cache2"):
        cur, prev = ROOT / "scraper" / d, ROOT / "scraper" / (d + ".prev")
        cur.mkdir(parents=True, exist_ok=True)
        fresh += len(list(cur.glob("*.html")))
        if prev.is_dir():
            for f in prev.glob("*.html"):
                if not (cur / f.name).exists():
                    shutil.copy(f, cur / f.name)  # the page was not fetched again yet: the previous copy stands in
    PROGRESS["pages"] = fresh
    kept = 0
    for name, args in STEPS:
        if name in OFFLINE_STEPS:
            rc, out = _step(args, stoppable=False, offline=True)
            if rc:
                for p in BACKED:
                    if (ROOT / (p + ".bak")).exists():
                        shutil.copy(ROOT / (p + ".bak"), ROOT / p)
                return f"Stopped; merging the partial update failed at '{name}' ({out.strip()[-200:]}), so the previous list was kept."
            if name in ("scrape kibris", "kibris only"):
                kept = keep_previous_products()  # products whose page was not reached keep their previous details (check_links, next, still drops the ones the shop removed)
    return f"Stopped. {fresh} page(s) had been re-read and merged with the previous list; {kept} product(s) kept their previous details because their page was not reached yet."


def run(back=None):
    """Full re-scrape (about 1-2 h: 2-5 s per page, on purpose). `back` = None means no maintenance mode. Returns a one-line report.
    It can be stopped (request_stop): what was already updated is merged with the previous list, nothing is lost."""
    if not _lock.acquire(blocking=False):
        return "A refresh is already running."
    _stop.clear()
    PROGRESS.update(step="starting", since=time.time(), stopping=False, pages=0)
    if back:
        core.kset("maintenance", "1")
    try:
        old = names()
        before = len(old)
        for d in ("cache", "cache2"):  # a full re-read: the old pages move aside (they stand in for the pages a stopped update did not reach), every page is fetched again
            cur, prev = ROOT / "scraper" / d, ROOT / "scraper" / (d + ".prev")
            if prev.is_dir():
                shutil.rmtree(prev)
            if cur.is_dir():
                cur.rename(prev)
            cur.mkdir(parents=True, exist_ok=True)
        for p in BACKED:
            if (ROOT / p).exists():
                shutil.copy(ROOT / p, ROOT / (p + ".bak"))
        for name, args in STEPS:
            PROGRESS.update(step=name, since=time.time())
            rc, out = _step(args)
            if rc is None:
                return finish_stopped(name)
            if rc:
                return f"Update failed at '{name}': {out.strip()[-300:]}. Products are unchanged unless the graph step already ran."
            if name == "build graph" and before and count() < 0.7 * before:  # site down / blocked: never publish a half-empty catalog
                n = count()
                for p in BACKED:
                    shutil.copy(ROOT / (p + ".bak"), ROOT / p)
                return f"Update cancelled: the scrape found only {n} products (was {before}), so the old catalog was kept."
            if name == "build graph":
                drop_stale_photos()
        PROGRESS.update(step="prices", since=time.time())
        ok, asked = refresh_prices()  # every price too, so the catalog and its prices are from the same night
        return f"Products updated: {before} -> {count()}. " + report(old, names()) + f" Prices read: {ok}/{asked}" + (" (the rest will show as 'the team confirms the price' until their pages read again)." if ok < asked else ".")
    finally:
        PROGRESS.update(step="", stopping=False)
        _lock.release()
        if back:
            core.kset("maintenance", "0")
            core.kset("refresh_at", str(time.time()))
            back()


def _price_line(rec):
    pid = str(rec["product_id"])
    ok, _ = refresh_prices({pid})
    p = core.product(pid)
    e = core.live_prices([p]).get(pid) if p else None
    return f"Price updated: {core.fmt_tl(e['price'])}." + ("" if e["in_stock"] else " It is OUT OF STOCK on the website.") if ok and e else "The price could not be read from the page: the stored price was kept."


def refresh_one(pid, force=False):
    """Re-read ONE scraped product from its own page (name, description, photos) into site.json, rebuild the graph (seconds), and refresh its photos in the background."""
    import httpx, re, html as H
    f = ROOT / "data/raw/site.json"
    site = json.loads(f.read_text(encoding="utf-8"))
    rec = next((p for p in site["products"] if str(p.get("product_id")) == str(pid)), None)
    if not rec or not rec.get("url"):
        return f"{pid} has no page to read (not a scraped item)."
    r = httpx.get(rec["url"], headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "tr,en;q=0.8"}, timeout=20, follow_redirects=True)
    if r.status_code == 404:
        return "That page no longer exists on the website. Hide the item with update_item (available = no), or run update_products."
    r.raise_for_status()
    ld = {}
    for b in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', r.text, re.S):
        try:
            d = json.loads(b)
        except ValueError:
            continue
        for x in d if isinstance(d, list) else d.get("@graph", [d]) if isinstance(d, dict) else []:
            if isinstance(x, dict) and x.get("@type") == "Product":
                ld = x
    if not ld.get("name"):
        raise ValueError("the page has no readable product data")
    imgs = ld.get("image") or rec.get("images") or []
    imgs = [imgs] if isinstance(imgs, str) else list(imgs)
    old = (rec.get("name"), rec.get("description"), rec.get("images"), rec.get("names"))
    rec.update(name=H.unescape(ld["name"]).strip(), description=H.unescape(ld.get("description") or rec.get("description") or ""), images=imgs)
    names, descs = dict(rec.get("names") or {}), dict(rec.get("descriptions") or {})  # the shop's own name in every language, from each language's page (never translated)
    names["tr"], descs["tr"] = rec["name"], rec["description"]
    for u in rec.get("urls") or []:
        if u.get("lang") in ("en", "ru") and u.get("url"):
            try:
                x = httpx.get(u["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=20, follow_redirects=True)
                x.raise_for_status()
                for b_ in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', x.text, re.S):
                    try:
                        d_ = json.loads(b_)
                    except ValueError:
                        continue
                    for y in d_ if isinstance(d_, list) else d_.get("@graph", [d_]) if isinstance(d_, dict) else []:
                        if isinstance(y, dict) and y.get("@type") == "Product" and y.get("name"):
                            names[u["lang"]], descs[u["lang"]] = H.unescape(y["name"]).strip(), H.unescape(y.get("description") or "")
            except Exception:
                pass  # that language keeps what it had
    rec.update(names=names, descriptions=descs)
    px = _price_line(rec)
    if old == (rec["name"], rec["description"], rec["images"], rec["names"]) and not force:
        return f"{rec['name']}: details already up to date. {px}"
    f.write_text(json.dumps(site, ensure_ascii=False, indent=1), encoding="utf-8")
    subprocess.run([sys.executable, "build_graph.py"], cwd=ROOT, check=True, capture_output=True)
    threading.Thread(target=lambda: [subprocess.run([sys.executable, "-m", m], cwd=ROOT, capture_output=True) for m in ("dl.download_images", "dl.build_index")], daemon=True).start()
    return f"Refreshed {rec['name']} from its page (name, description, photos). {px} The photo matcher updates in the background."


def rebuild():
    subprocess.run([sys.executable, "build_graph.py"], cwd=ROOT, check=True, capture_output=True)


def reset_one(pid):
    """Discard the owner's hand edits of ONE item and read its page again: the item is back to exactly what the website says."""
    core.clear_override(pid)
    return refresh_one(pid, force=True)


def restore_one(pid):
    """Bring a removed item back NOW: allowed again and the catalog rebuilt at once (with the details saved at the last update), then its page is read again for fresh details."""
    if not core.restore_product(pid):
        return "That item was not removed."
    rebuild()
    try:
        return refresh_one(pid, force=True).replace("Refreshed", "Brought back and refreshed", 1)
    except Exception as e:
        return f"Brought back with the details saved at the last update (the website page could not be read now: {e!r})."


SITEMAP = "https://kibrisciceksepetim.com/sitemap.xml"


def sitemap_products(xml):
    """{slug: lastmod} of the product pages listed in the shop's sitemap (the Turkish URL of each product)."""
    import re
    out = {}
    for blk in re.findall(r"<url>(.*?)</url>", xml, re.S):
        m = re.search(r"<loc>https?://[^/<]+/tr/products/([^/<?#\s]+)</loc>", blk)
        if m:
            lm = re.search(r"<lastmod>([^<]*)</lastmod>", blk)
            out[m.group(1)] = lm.group(1) if lm else ""
    return out


def fetch_sitemap():
    import httpx
    r = httpx.get(SITEMAP, headers={"User-Agent": "Mozilla/5.0"}, timeout=40, follow_redirects=True)
    r.raise_for_status()
    return r.text


def watch_once(notify, fetch=None, refresh=None):
    """The shop site has no way to push events, so the sitemap (one cheap request, no page scraping) is checked every few minutes:
    a page whose lastmod changed is re-read on its own (no downtime), new / removed products are reported to the owner (adding them is the full update).
    The first check only stores the starting picture. Returns a one-line result."""
    cur = sitemap_products((fetch or fetch_sitemap)())
    if len(cur) < 20:  # a broken or partial answer must never look like 'everything was removed'
        return f"sitemap looks incomplete ({len(cur)} products): ignored"
    snap = core.jget("sitemap_snap")
    core.kset("sitemap_snap", json.dumps(cur))
    core.kset("watch_ts", core.now().isoformat(timespec="seconds"))
    if not snap:
        return f"watching {len(cur)} products"
    new = [s for s in cur if s not in snap]
    gone = [s for s in snap if s not in cur]
    changed = [s for s in cur if s in snap and cur[s] != snap[s]]
    known, done = names(), []
    for s in changed[:10]:
        pid = "k-" + s
        if pid in known:
            try:
                (refresh or refresh_one)(pid)
                done.append(s)
            except Exception as e:
                core.log(f"watch: could not refresh {s}: {e!r}")
    if new or gone:
        lines = []
        if new:
            lines.append(f"🆕 {len(new)} new product(s) on the website: " + ", ".join(new[:6]) + ("…" if len(new) > 6 else ""))
        if gone:
            lines.append(f"🗑 {len(gone)} product(s) no longer on the website: " + ", ".join(gone[:6]) + ("…" if len(gone) > 6 else ""))
        notify("\n".join(lines) + "\nTo add / drop them, press 'Update product list' in the dashboard (Products) or tell me 'update products' (it can take the agent off for 1-3 hours).")
    return f"{len(new)} new, {len(gone)} removed, {len(done)} page(s) refreshed"


def start_watcher(notify):
    """Background thread: the sitemap check every WATCH_MINUTES (30); skipped while a full update runs."""
    import os

    def loop():
        while True:
            time.sleep(60 if not core.kget("watch_ts") else int(os.environ.get("WATCH_MINUTES", "30")) * 60)
            if busy():
                continue
            try:
                watch_once(notify)
            except Exception as e:
                core.log(f"site watch failed: {e!r}")
    threading.Thread(target=loop, daemon=True, name="site-watch").start()


def due(n, last_day):
    """Run tonight? Inside the 02:00-05:00 window (North Cyprus clock) and not already run today."""
    return 2 <= n.hour < 5 and last_day != f"{n:%Y-%m-%d}"


def start_daily(notify, back):
    """Background thread: one full refresh EVERY NIGHT, started only between 02:00 and 05:00 in North Cyprus (Asia/Nicosia follows the summer/winter clock change by itself).
    The day it last ran is stored, so a restart neither skips a night nor runs twice in one."""
    def loop():
        while True:
            n = datetime.now(ZoneInfo("Asia/Nicosia"))
            if due(n, core.kget("refresh_day")):
                core.kset("refresh_day", f"{n:%Y-%m-%d}")  # marked on the attempt: a failing night is not retried every ten minutes
                msg = run(back)
                if not msg.startswith("A refresh is already"):
                    notify("Daily product update: " + msg)
            time.sleep(600)
    threading.Thread(target=loop, daemon=True, name="daily-refresh").start()


if __name__ == "__main__":
    print(run())
