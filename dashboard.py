"""Admin web dashboard: orders, payments, CRM. Login with username + password; only the leader admin creates accounts.
Every action that concerns a customer goes through the same functions the WhatsApp admin chat uses (agent.decide, confirm_payment, ...), so the customer gets the same WhatsApp message.
Open /dashboard. The first start creates the leader account admin@kibris (password 1234): change it from the Users tab."""
import asyncio, atexit, hashlib, hmac, json, os, re, secrets, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

import httpx
from datetime import date
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

import agent, core

router = APIRouter()
LEADER = "admin@kibris"
USERS_LOCK = threading.Lock()  # dash_users.json is read, changed and written back: one change at a time
SESS, FAILS = {}, {}  # token -> {user, exp, v}; (ip, user) -> [timestamps]. Sessions are also kept in dash_sessions.json, so a server restart does not log anyone out
TTL = 24 * 3600  # a login lasts 24 hours, counted from the login
_sess_loaded = False


def _ui_version():
    """Changes when the dashboard screen itself (static/dashboard.html) is updated: sessions made on an older screen must log in again, so nobody keeps running an outdated page."""
    try:
        s = (Path(__file__).parent / "static/dashboard.html").stat()
        return f"{s.st_mtime_ns}-{s.st_size}"
    except OSError:
        return ""


def _sess_load():
    global _sess_loaded
    if _sess_loaded:
        return
    _sess_loaded = True
    try:
        SESS.update(json.loads(core.P("dash_sessions.json").read_text(encoding="utf-8")))
    except Exception:
        pass


def _sess_save():
    now = time.time()
    try:
        p = core.P("dash_sessions.json")
        core.atomic_write(p, json.dumps({t: s for t, s in SESS.items() if s["exp"] > now}))
        os.chmod(p, 0o600)  # it holds login tokens
    except Exception:
        pass


def _sess(request):
    """The valid session of this request, or None: unknown, older than 24 h, or made on an outdated dashboard screen."""
    _sess_load()
    s = SESS.get(request.cookies.get("dash", ""))
    return s if s and s["exp"] > time.time() and s.get("v") == _ui_version() else None
DASH = "dashboard"  # the "admin number" the shared admin functions answer to: their texts are captured for the page instead of going to WhatsApp


# ------------------------------------------------------------------ accounts
def _hash(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def users():
    p = core.P("dash_users.json")
    if not p.exists():
        salt = secrets.token_hex(16)
        core.atomic_write(p, json.dumps({LEADER: {"salt": salt, "hash": _hash("1234", salt), "role": "leader", "default": True}}))
    return json.loads(p.read_text(encoding="utf-8"))


def save_users(d):
    core.atomic_write(core.P("dash_users.json"), json.dumps(d, indent=1))


def set_password(d, name, pw):
    salt = secrets.token_hex(16)
    d[name] = {**d.get(name, {"role": "admin"}), "salt": salt, "hash": _hash(pw, salt)}
    d[name].pop("default", None)


LANG_IDX = {"en": 0, "tr": 1, "ru": 2}
TX = {  # the messages this page shows to the owner, in the language of the dashboard (header X-Lang): en, tr, ru
    "shipped": ("this order is shipped: a change now would be a new order", "bu sipariş gönderildi: şimdi değişiklik yeni bir sipariş olur", "заказ уже отправлен: изменение теперь было бы новым заказом"),
    "closed": ("this order is {st}: a change now would be a new order", "bu sipariş {st}: şimdi değişiklik yeni bir sipariş olur", "заказ {st}: изменение теперь было бы новым заказом"),
    "unreadable": ("the items of this order cannot be read", "bu siparişin ürünleri okunamıyor", "товары этого заказа не удаётся прочитать"),
    "unknown_product": ("unknown product {id}", "bilinmeyen ürün {id}", "неизвестный товар {id}"),
    "no_price": ("there is no stored price for {name}", "{name} için kayıtlı fiyat yok", "для {name} нет сохранённой цены"),
    "need_item": ("an order needs at least one item: reject or cancel it instead", "bir siparişte en az bir ürün olmalı: bunun yerine reddedin veya iptal edin", "в заказе должен быть хотя бы один товар: лучше отклоните или отмените его"),
    "no_change": ("No change.", "Değişiklik yok.", "Без изменений."),
    "items_changed": ("Items changed. {summary}{told}", "Ürünler değiştirildi. {summary}{told}", "Товары изменены. {summary}{told}"),
    "told": (" The customer was told.", " Müşteriye bildirildi.", " Клиенту сообщили."),
    "not_changed": ("the order could not be changed", "sipariş değiştirilemedi", "заказ не удалось изменить"),
    "bad_area": ("unknown delivery area", "bilinmeyen teslimat bölgesi", "неизвестный район доставки"),
    "custom_only_open": ("custom items can only be changed before the order is confirmed", "özel ürünler yalnızca sipariş onaylanmadan önce değiştirilebilir", "свои товары можно менять только до подтверждения заказа"),
    "custom_list_bad": ("custom items must be sent as a list", "özel ürünler liste olarak gönderilmelidir", "свои товары нужно отправить списком"),
    "custom_object_bad": ("each custom item must have valid details", "her özel ürünün bilgileri geçerli olmalı", "у каждого своего товара должны быть корректные данные"),
    "custom_item_bad": ("each custom item needs a name, quantity, and positive price", "her özel ürün için ad, miktar ve pozitif fiyat girin", "для каждого своего товара укажите название, количество и положительную цену"),
    "area_wins": (" NOTE: the address names another town, so the delivery area stays '{got}'. Change the address text to move it to '{want}'.",
                  " NOT: adres başka bir kasabayı belirtiyor, bu yüzden teslimat bölgesi '{got}' olarak kalıyor. '{want}' bölgesine geçirmek için adres metnini değiştirin.",
                  " ВНИМАНИЕ: в адресе указан другой город, поэтому район доставки остаётся '{got}'. Измените текст адреса, чтобы перенести заказ в '{want}'."),
    "phone_bad": ("the {what} '{raw}' does not look right ({why}): write it with the country code, e.g. +90 5xx xxx xx xx", "{what} '{raw}' doğru görünmüyor ({why}): ülke koduyla yazın, örn. +90 5xx xxx xx xx",
                  "{what} «{raw}» выглядит неверно ({why}): напишите с кодом страны, например +90 5xx xxx xx xx"),
    "w_account": ("account phone number", "hesabın telefon numarası", "номер телефона аккаунта"), "phone_taken": ("that number already belongs to the account {who}", "bu numara zaten {who} hesabına ait", "этот номер уже принадлежит аккаунту {who}"),
    "w_customer": ("customer number", "müşteri numarası", "номер клиента"), "w_sender": ("sender phone", "gönderen telefonu", "телефон отправителя"),
    "w_recipient": ("recipient phone", "alıcı telefonu", "телефон получателя"), "w_phone": ("phone", "telefon", "телефон"),
    "old_nan": ("the old price must be a number", "eski fiyat bir sayı olmalı", "старая цена должна быть числом"),
    "old_lt": ("the old price (before) must be higher than the price (after)", "eski fiyat (önce) yeni fiyattan (sonra) yüksek olmalı", "старая цена (до) должна быть выше цены (после)"),
    "person_name": ("the person's name is needed", "kişinin adı gerekli", "нужно имя человека"), "person_gone": ("that person is not saved", "bu kişi kayıtlı değil", "этого человека нет в сохранённых"),
    "st_rejected": ("rejected", "reddedildi", "отклонён"), "st_cancelled": ("cancelled", "iptal edildi", "отменён"),
}


def lx(request):
    lang = LANG_IDX.get((request.headers.get("x-lang") or "en")[:2], 0)
    if core.kget("dashlang", "0") != str(lang):
        core.kset("dashlang", lang)
    return lang


def tx(request, key, **kw):
    return TX[key][lx(request)].format(**kw)


def phone_problem(request, what_key, raw, why):
    return tx(request, "phone_bad", what=tx(request, what_key), raw=raw, why=core.phone_why(why, lx(request)))


def me(request: Request):
    s = _sess(request)
    if not s:
        raise HTTPException(401, "login required")
    if request.method not in ("GET", "HEAD") and request.headers.get("X-Dash") != "1":  # with SameSite=Strict this keeps other sites from acting for a logged-in admin
        raise HTTPException(403, "bad request")
    u = users().get(s["user"])
    if not u:
        raise HTTPException(401, "account removed")
    return {"name": s["user"], **u}


def leader(request: Request):
    u = me(request)
    if u["role"] != "leader":
        raise HTTPException(403, "only the leader admin can do this")
    return u


@router.get("/dashboard")
def page():
    return FileResponse(Path(__file__).parent / "static/dashboard.html", headers={"Cache-Control": "no-store"})  # a browser must never keep an old copy of the page


@router.post("/api/dash/login")
async def login(request: Request, response: Response):
    b = await request.json()
    name, pw = str(b.get("username", "")).strip(), str(b.get("password", ""))
    key = (request.client.host if request.client else "?", name)
    FAILS[key] = [t for t in FAILS.get(key, []) if t > time.time() - 600]
    if len(FAILS[key]) >= 5:
        raise HTTPException(429, "too many attempts, wait 10 minutes")
    u = users().get(name)
    if not u or not hmac.compare_digest(_hash(pw, u["salt"]), u["hash"]):
        FAILS[key].append(time.time())
        raise HTTPException(401, "wrong username or password")
    tok = secrets.token_urlsafe(32)
    _sess_load()
    SESS[tok] = {"user": name, "exp": time.time() + TTL, "v": _ui_version()}
    _sess_save()
    response.set_cookie("dash", tok, max_age=TTL, httponly=True, samesite="strict", secure=request.url.scheme == "https")
    if "fullscreen" in b:  # the full screen preference chosen on the login page is the user's saved preference
        d = users()
        d[name]["fullscreen"] = bool(b["fullscreen"])
        save_users(d)
    core.log(f"dashboard: {name} logged in")
    return {"ok": True}


@router.post("/api/dash/logout")
def logout(request: Request, response: Response):
    _sess_load()
    SESS.pop(request.cookies.get("dash", ""), None)
    _sess_save()
    response.delete_cookie("dash")
    return {"ok": True}


@router.get("/api/dash/me")
def whoami(request: Request):
    u = me(request)
    ph = _member_photo(u["name"])
    nick = users().get(u["name"], {}).get("nickname") or users().get(u["name"], {}).get("display") or ""
    return {"name": u["name"], "nick": nick, "role": u["role"], "default_password": bool(u.get("default")), "photo": ph.exists(), "v": int(ph.stat().st_mtime) if ph.exists() else 0,
            "fullscreen": bool(users().get(u["name"], {}).get("fullscreen"))}


@router.get("/api/dash/users")
def list_users(request: Request):
    leader(request)
    return [{"name": n, "role": u["role"]} for n, u in users().items()]


QUOTES_TR = ("Tek başımıza bir damlayız, birlikte bir okyanus.", "Başarı, aynı yöne kürek çeken ellerin eseridir.", "Bir çiçek bahar getirmez, ama birlikte açan çiçekler bir bahçe olur.",
             "Yalnız yürüyen hızlı gider, birlikte yürüyenler çok uzağa.", "Ekip, bireylerin gücünü ortak bir hayale çevirir.", "Birlik olduğumuz yerde her zorluk çiçek açar.",
             "Güzel işler, güzel insanların omuz omuza vermesiyle doğar.", "Her el bir yaprak, hepimiz bir çiçeğiz.", "Birbirimize güvendiğimiz sürece hiçbir fırtına bizi devirmez.",
             "Takım ruhu, sıradan günleri bile unutulmaz kılar.", "Paylaşılan emek hafifler, paylaşılan başarı büyür.", "Aynı bahçenin çiçekleriyiz; birimizin mutluluğu hepimizi güzelleştirir.")


def good_name(n):
    """A username: any case, any letters or symbols, but NO spaces (3-60 characters). The free text for cards is the nickname."""
    return 3 <= len(n) <= 60 and not re.search(r"[\s\x00-\x1f\x7f]", n)


def _member_photo(user):
    from urllib.parse import quote
    return core.P("team") / f"{quote(user, safe='')}.jpg"  # percent-encoded: a name can never point outside the team folder


ASSIST_LOCK = __import__("threading").Lock()


def reset_private_assistant_day(user):
    day = core.now().date().isoformat()
    key = "dashboard_assistant_day:" + user["name"]
    with ASSIST_LOCK:
        if core.kget(key) != day:
            core.alog_clear("assistant_" + user["name"])
            core.kset("aex:" + DASH + ":" + user["name"], "[]")
            core.kset(key, day)


@router.post("/api/dash/assistant")
async def assistant(request: Request):
    """The global admin chat of the page: the same assistant the owner talks to on WhatsApp / Telegram, answering here. Returns what it said (texts and files)."""
    u = me(request)
    b = await request.json()
    text = str(b.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "write something")
    import asyncio
    live = bool(b.get("live"))  # live mode = the shared real admin chat; otherwise the private conversation of this user
    if not live:
        reset_private_assistant_day(u)
    did = DASH if live else f"{DASH}:{u['name']}"
    core.alog_add("live" if live else "assistant_" + u["name"], "user", text, by=u["name"])
    def work():
        with ASSIST_LOCK:  # one question at a time
            n0 = len(core.FEED)
            agent.admin_entry(did, text, "")
            return [{"kind": e["kind"], "text": e["body"], "file": e.get("file")} for e in core.FEED[n0:] if e.get("to") == did and e["kind"] in ("text", "document", "image")]
    try:
        out = await asyncio.to_thread(work)
    except Exception as e:
        core.log(f"dashboard assistant failed: {e!r}")
        out = [{"kind": "text", "text": "⚠️ The assistant could not answer just now. Try again in a moment.", "file": None}]
    core.log(f"dashboard: {u['name']} asked the assistant")
    return {"replies": out}


@router.get("/api/dash/admin-chat")
def admin_chat_feed(request: Request, since: int = 0, tail: int = 0, before: int = 0, limit: int = 20):
    """The LIVE admin chat, all of it stored for ever. `since` = the `next` of the last call (what is new); `tail` = only the last N events (the first call); `before` = an event id: the `limit` events older than it
    (the page asks for these while the user scrolls up, without end). `more` = there are older events than the ones returned."""
    me(request)
    n = core.alog_slice("live", 0, 0)[0]  # the total only: nothing is parsed
    lim = max(1, min(limit, 200))
    start, stop = (max(0, before - 1 - lim), max(0, before - 1)) if before else (since, n) if since else (max(0, n - tail), n) if tail else (0, n)
    rows = core.alog_slice("live", start, stop)[1]
    acct = {core.canon_phone(x["phone"]): nm for nm, x in users().items() if x.get("phone")}
    who = lambda by: by if by in users() else acct.get(core.canon_phone(by), by) if by else ""  # a WhatsApp / Telegram number is shown as the account that owns it
    ev = [{"id": i + 1, "ts": e["ts"][5:16].replace("T", " "), "to": {"user": "me", "assistant": "assistant"}.get(e["role"], "alert"), "by": who(e.get("by")) if e["role"] == "user" else "agent", "kind": e["kind"], "body": e["body"], "file": e.get("file")}
          for i, e in rows]
    return {"events": ev, "next": n, "more": bool(ev) and ev[0]["id"] > 1}


@router.get("/api/dash/assistant/history")
def assistant_history(request: Request, tail: int = 20, before: int = 0, limit: int = 20):
    """The logged-in user's private assistant chat for today, paged newest-first on entry and older on demand."""
    u = me(request)
    reset_private_assistant_day(u)
    name = "assistant_" + u["name"]
    n = core.alog_slice(name, 0, 0)[0]
    lim = max(1, min(limit, 100))
    start, stop = (max(0, before - 1 - lim), max(0, before - 1)) if before else (max(0, n - max(1, min(tail, 100))), n)
    page = [{"id": i + 1, "role": "me" if e["role"] == "user" else "bot", "text": e["body"], "file": e.get("file"), "ts": e["ts"]} for i, e in core.alog_slice(name, start, stop)[1]]
    return {"messages": page, "day": core.now().date().isoformat(), "more": bool(page) and page[0]["id"] > 1}


@router.get("/api/dash/file/{name}")
def outbox_file(name: str, request: Request, inline: int = 0):
    """A file the assistant produced for the page (a PDF report, a photo). inline=1 shows it in the viewer, otherwise it is a download."""
    me(request)
    p = core.P("outbox") / Path(name).name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name.split("_", 1)[-1], content_disposition_type="inline" if inline else "attachment")


@router.get("/api/dash/team")
def team(request: Request):
    """The team list every employee can see: photo, name, role and a Turkish quote about teamwork."""
    me(request)
    return {"members": [{"user": n, "role": u["role"], "name": u.get("nickname") or u.get("display") or n, "title": u.get("title") or "", "quote": u.get("quote") or "", "phone": u.get("phone") or "", "photo": _member_photo(n).exists(), "pv": int(_member_photo(n).stat().st_mtime) if _member_photo(n).exists() else 0}
                        for n, u in users().items()], "suggestions": list(QUOTES_TR), "agent": agent_card()}


def agent_card():
    """Leyla, the agent, as a card of the team: her name / role / quote / photo (editable like any card) and her health."""
    c = core.jget("agentcard") or {}
    return {"name": c.get("nickname") or "Leyla", "title": c.get("title") or "Sales agent", "quote": c.get("quote") or "", "photo": _member_photo("leyla agent").exists(), "pv": int(_member_photo("leyla agent").stat().st_mtime) if _member_photo("leyla agent").exists() else 0,
            "state": agent_state(), "health": core.health_checks()}


@router.patch("/api/dash/agent/card")
async def agent_card_edit(request: Request):
    """Only a top admin (the leader) can change Leyla's card: name, role, quote, photo."""
    if me(request)["role"] != "leader":
        raise HTTPException(403, "only a top admin can edit this card")
    b = await request.json()
    c = core.jget("agentcard") or {}
    for key, n in (("nickname", 60), ("title", 60), ("quote", 300)):
        if key in b:
            c[key] = " ".join(str(b[key]).split())[:n]
    core.kset("agentcard", json.dumps(c, ensure_ascii=False))
    ph = _member_photo("leyla agent")
    if b.get("photo"):
        import base64
        ph.parent.mkdir(exist_ok=True)
        ph.write_bytes(agent.jpeg(base64.b64decode(str(b["photo"]).split(",")[-1]), 600))
    if b.get("remove_photo"):
        ph.unlink(missing_ok=True)
    return {"ok": True}


@router.get("/api/dash/reports/{kind}")
def progressive_report(kind: str, request: Request, inline: int = 0):
    """Every member of the dashboard can download the running week's / month's report (PDF, English and Turkish): rebuilt on every download, so it always includes everything up to now."""
    me(request)
    if kind not in ("week", "month"):
        raise HTTPException(404)
    import docs_pdf
    p = docs_pdf.progressive(kind)
    return FileResponse(p, media_type="application/pdf", filename=f"{kind}ly_report_{p.stem.split('_')[1]}.pdf", headers={"Cache-Control": "no-store"}, content_disposition_type="inline" if inline else "attachment")


@router.get("/api/dash/agent/photo")
def agent_photo(request: Request, w: int = 0, v: str = ""):
    me(request)
    ph = _member_photo("leyla agent")
    if not ph.exists():
        raise HTTPException(404)
    return _img(ph, w, v)


@router.patch("/api/dash/team/{user:path}")
async def team_edit(user: str, request: Request):
    """Everybody edits their own card; only a top admin (the leader) edits other people's: photo, name, role, quote."""
    u = me(request)
    if u["name"] != user and u["role"] != "leader":
        raise HTTPException(403, "only a top admin can edit other people's cards")
    d = users()
    if user not in d:
        raise HTTPException(404, "no such account")
    b = await request.json()
    for key, field, n in (("nickname", "nickname", 60), ("title", "title", 60), ("quote", "quote", 300)):
        if key in b:
            d[user][field] = " ".join(str(b[key]).split())[:n]  # the nickname is free text (spaces, any case): it is the name on the card, the username stays the login
            d[user].pop("display", None)
    if "phone" in b:  # the number this person writes to the admin chat from (WhatsApp / Telegram): their messages there are shown under this account
        raw = str(b["phone"] or "").strip()
        if not raw:
            d[user].pop("phone", None)
        else:
            if raw.lower().startswith("tg:"):
                ph, why = "tg:" + re.sub(r"\D", "", raw), "" if re.sub(r"\D", "", raw) else "empty"
            else:
                ph, why = core.check_phone(raw)
            if why:
                raise HTTPException(400, phone_problem(request, "w_account", raw, why))
            other = next((n for n, x in d.items() if n != user and x.get("phone") and core.canon_phone(x["phone"]) == core.canon_phone(ph)), None)
            if other:
                raise HTTPException(409, tx(request, "phone_taken", who=other))
            d[user]["phone"] = ph
    if b.get("photo"):
        import base64
        _member_photo(user).parent.mkdir(exist_ok=True)
        _member_photo(user).write_bytes(agent.jpeg(base64.b64decode(str(b["photo"]).split(",")[-1]), 600))
    if b.get("remove_photo"):
        _member_photo(user).unlink(missing_ok=True)
    save_users(d)
    return {"ok": True}


@router.get("/api/dash/team/{user:path}/photo")
def team_photo(user: str, request: Request, w: int = 0, v: str = ""):
    me(request)
    if user not in users() or not _member_photo(user).exists():
        raise HTTPException(404)
    return _img(_member_photo(user), w, v)


@router.post("/api/dash/users")
async def create_user(request: Request):
    leader(request)
    b = await request.json()
    name, pw = str(b.get("username", "")).strip(), str(b.get("password", ""))
    if not good_name(name) or len(pw) < 6:
        raise HTTPException(400, "username 3-60 characters without spaces (any case) and a password of at least 6 characters")
    d = users()
    if name in d:
        raise HTTPException(409, "that username exists")
    with USERS_LOCK:
        d = users()
        if name in d:
            raise HTTPException(409, "that username exists")
        set_password(d, name, pw)
        d[name]["role"] = "admin"
        save_users(d)
    core.log(f"dashboard: account {name} created")
    return {"ok": True}


@router.post("/api/dash/users/{name:path}/password")
async def change_password(name: str, request: Request):
    u = me(request)
    if u["name"] != name and u["role"] != "leader":
        raise HTTPException(403, "not allowed")
    b = await request.json()
    pw = str(b.get("password", ""))
    d = users()
    if name not in d:
        raise HTTPException(404)
    if u["name"] == name and not hmac.compare_digest(_hash(str(b.get("current", "")), d[name]["salt"]), d[name]["hash"]):  # changing your own password needs the current one (the leader resetting someone else's does not)
        raise HTTPException(400, "the current password is not right")
    if len(pw) < 6:
        raise HTTPException(400, "password too short")
    with USERS_LOCK:
        d = users()
        set_password(d, name, pw)
        save_users(d)
    for t in [t for t, s in SESS.items() if s["user"] == name and t != request.cookies.get("dash")]:
        SESS.pop(t)
    _sess_save()
    return {"ok": True}


@router.post("/api/dash/users/{name:path}/rename")
async def rename_user(name: str, request: Request):
    """Change a username: the leader can rename anyone, everyone can rename themselves. The password, role, card and photo move with it and nobody is logged out."""
    u = me(request)
    if u["name"] != name and u["role"] != "leader":
        raise HTTPException(403, "not allowed")
    new = str((await request.json()).get("new_name", "")).strip()
    if not good_name(new):
        raise HTTPException(400, "username 3-60 characters without spaces (any case)")
    d = users()
    if name not in d:
        raise HTTPException(404)
    if new == name:
        return {"ok": True, "name": new}
    if new in d:
        raise HTTPException(409, "that username exists")
    with USERS_LOCK:
        d = users()
        if name not in d or new in d:
            raise HTTPException(409, "that username exists or the account is gone")
        d[new] = d.pop(name)
        if "display" in d[new]:  # an older card name becomes the nickname
            d[new].setdefault("nickname", d[new].pop("display"))
        save_users(d)
    old_photo = _member_photo(name)
    if old_photo.exists():
        old_photo.rename(_member_photo(new))
    for s in SESS.values():
        if s["user"] == name:
            s["user"] = new
    _sess_save()
    core.log(f"dashboard: account {name} renamed to {new}")
    return {"ok": True, "name": new}


@router.delete("/api/dash/users/{name:path}")
def delete_user(name: str, request: Request):
    leader(request)
    d = users()
    if name not in d or d[name].get("role") == "leader":  # decided by role, not by the original name (the leader can be renamed)
        raise HTTPException(400, "cannot remove that account")
    d.pop(name)
    save_users(d)
    for t in [t for t, s in SESS.items() if s["user"] == name]:
        SESS.pop(t)
    _sess_save()
    return {"ok": True}


# ------------------------------------------------------------------ helpers
def shared(fn, *a, **k):
    """Run a function written for the WhatsApp admin chat; return what it would have told the admin."""
    n = len(core.FEED)
    fn(DASH, *a, **k)
    return [e["body"] for e in core.FEED[n:] if e.get("to") == DASH]


def tell(phone, key, **kw):
    """A short fixed message to the customer in their language (Turkish or English), through the normal WhatsApp path."""
    lg_ = agent.lang_of(phone)
    T = {"edit": ("Your order #{oid} was updated by the team 🌸\n{changes}", "#{oid} numaralı siparişiniz ekibimiz tarafından güncellendi 🌸\n{changes}", "Ваш заказ #{oid} обновлён командой 🌸\n{changes}"),
         "removed": ("Sorry, we had to cancel your order #{oid} 🌸 {excuse}", "Üzgünüz, #{oid} numaralı siparişinizi iptal etmek zorunda kaldık 🌸 {excuse}", "К сожалению, нам пришлось отменить ваш заказ #{oid} 🌸 {excuse}"),
         "created": ("Hello! The team created order #{oid} for you 🌸\n{summary}\nTell me if anything should change.", "Merhaba! Ekibimiz sizin için #{oid} numaralı siparişi oluşturdu 🌸\n{summary}\nDeğişmesi gereken bir şey varsa yazın.", "Здравствуйте! Команда создала для вас заказ #{oid} 🌸\n{summary}\nНапишите, если что-то нужно изменить."),
         }[key]
    agent.reply(phone, agent.pick(T, lg_).format(**kw))


def _line_names():
    """{the name an order line was stored with: {tr, en, ru}} so the dashboard shows every order line in its own language."""
    return {core._nm(n["name"]): {"tr": n.get("name_tr") or n["name"], "en": n.get("name_en") or "", "ru": n.get("name_ru") or ""} for n in core.graph()[0].values() if n["type"] == "product"}


def row_out(r, rows=None):
    L = core.row_lines(r) or []
    rows = rows if rows is not None else core.orders()
    nm = _line_names()
    return {**r, "area_key": (core.delivery_calc(r) or {}).get("key") or "", "lines": [{"name": n, "names": nm.get(n, {}), "qty": q, "unit": str(u)} for n, q, u in L], "summary": core.order_block(r), "lang": ("en", "tr", "ru")[agent.lang_of(r["customer_number"])],
            "lost_recent": core.lost_recent(r), "late": core.is_late(r), "external": core.is_external(r), "pay_state": core.pay_state(r) if (r["payable"] or r["paid"]) else "", "addon_of": core.addon_of(r),
            "addons": [{"id": int(x["order_id"]), "state": core.pay_state(x), "payable": x["payable"], "status": x["status"]} for x in rows if core.addon_of(x) == int(r["order_id"])],
            "prio": int(core.kget(f"prio:{r['order_id']}") or 0), "pos": int(core.kget(f"pos:{r['order_id']}")) if core.kget(f"pos:{r['order_id']}") else None}


def one(oid):
    r = core.order_row(oid)
    if not r:
        raise HTTPException(404, "no such order")
    return r


# ------------------------------------------------------------------ orders
@router.get("/api/dash/orders")
def orders(request: Request):
    me(request)
    rows = core.orders()
    board = [r for r in rows if r["status"] != "completed" and (r["status"] not in core.LOST_STATUSES or core.lost_recent(r))]
    return [row_out(r, rows) for r in reversed(board)]


@router.get("/api/dash/analytics")
def analytics(request: Request):
    me(request)
    from datetime import timedelta
    rows = core.orders()
    won = [r for r in rows if r["status"] in ("confirmed", "completed")]
    won_ids = {r["order_id"] for r in won}
    lost = [r for r in rows if r["status"] in core.LOST_STATUSES]
    months = {}
    for r in rows:
        month = r["timestamp"][:7]
        item = months.setdefault(month, {"orders": 0, "revenue": 0})
        item["orders"] += 1
        if r["order_id"] in won_ids:
            item["revenue"] += float(r["payable"] or r["total"] or 0)
    today = core.now().date()
    week = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    paid_by_day = {}
    for r in won:
        day = r["timestamp"][:10]
        if day:
            paid_by_day[day] = paid_by_day.get(day, 0) + float(r["payable"] or r["total"] or 0)
    orders_by_day = {}
    for r in rows:
        day = r["timestamp"][:10]
        if day:
            orders_by_day[day] = orders_by_day.get(day, 0) + 1
    series = lambda start, end: [{"date": (start + timedelta(days=i)).isoformat(), "revenue": paid_by_day.get((start + timedelta(days=i)).isoformat(), 0), "orders": orders_by_day.get((start + timedelta(days=i)).isoformat(), 0)} for i in range((end - start).days)]
    return {"count": len(rows), "won": len(won), "lost": len(lost), "open": len(rows) - len(won) - len(lost),
            "revenue": sum(float(r["payable"] or r["total"] or 0) for r in won), "months": months,
            "periods": {"week": series(week, week + timedelta(days=7)), "month": series(month_start, next_month)}}


@router.get("/api/dash/history")
def order_history(request: Request, offset: int = 0, limit: int = 25):
    me(request)
    rows = sorted(core.orders(), key=lambda r: r["timestamp"], reverse=True)
    offset, limit = max(0, offset), max(1, min(100, limit))
    page = rows[offset:offset + limit]
    all_rows = core.orders()
    return {"count": len(rows), "orders": [row_out(r, all_rows) for r in page], "next": offset + len(page), "more": offset + len(page) < len(rows)}


TASKS_LOCK = threading.Lock()


def create_agent_task(title, assigned_date, deadline="", bullets=None, assigned=None):
    title = str(title or "").strip()[:160]
    if not title:
        raise ValueError("task title is required")
    dates = [str(x or "") for x in (assigned_date, deadline)]
    for value in dates:
        if not value:
            raise ValueError("assignment date and deadline are required")
        if value:
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("dates must use YYYY-MM-DD")
            except ValueError as e:
                raise ValueError("dates must use YYYY-MM-DD") from e
    if dates[1] < dates[0]:
        raise ValueError("deadline must be on or after the assignment date")
    users_ = set(users())
    assigned_ = [str(x) for x in (assigned or ["everyone"]) if str(x) == "everyone" or str(x) in users_][:50] or ["everyone"]
    if "everyone" in assigned_:
        assigned_ = ["everyone"]
    p = core.P("tasks.json")
    with TASKS_LOCK:
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            rows = []
        task = {"id": secrets.token_hex(8), "title": title, "priority": 0, "assigned_date": dates[0], "deadline": dates[1],
                "bullets": [{"text": str(x).strip()[:240], "done": False} for x in (bullets or []) if str(x).strip()][:30],
                "status": "todo", "assigned": assigned_, "created_by": "agent", "created_at": int(time.time()), "source": "agent"}
        rows.append(task)
        core.atomic_write(p, json.dumps(rows, ensure_ascii=False, indent=2))
    return task


@router.get("/api/dash/tasks")
def tasks(request: Request):
    me(request)
    try:
        return json.loads(core.P("tasks.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []


@router.post("/api/dash/tasks")
async def add_task(request: Request):
    user = me(request)
    b = await request.json()
    title = str(b.get("title", "")).strip()[:160]
    if not title:
        raise HTTPException(400, "task title is required")
    for field in ("assigned_date", "deadline"):
        value = str(b.get(field, ""))
        if not value:
            raise HTTPException(400, f"{field} is required")
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError:
            raise HTTPException(400, f"invalid {field}; use YYYY-MM-DD")
    if b["deadline"] < b["assigned_date"]:
        raise HTTPException(400, "deadline must be on or after the assignment date")
    status = str(b.get("status", "todo"))
    if status not in ("todo", "in_progress", "done"):
        raise HTTPException(400, "invalid task status")
    priority = b.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int) or priority not in (0, 1, 2, 3):
        raise HTTPException(400, "priority must be between 0 and 3")
    assigned = b.get("assigned", "everyone")
    if not isinstance(assigned, list):
        assigned = [assigned]
    known = set(users())
    assigned = [str(x)[:80] for x in assigned if str(x).strip() and (str(x) == "everyone" or str(x) in known)][:50] or ["everyone"]
    if "everyone" in assigned:
        assigned = ["everyone"]
    bullets = [{"text": str(x.get("text", "")).strip()[:240], "done": bool(x.get("done"))} if isinstance(x, dict) else {"text": str(x).strip()[:240], "done": False} for x in b.get("bullets", []) if (str(x.get("text", "")).strip() if isinstance(x, dict) else str(x).strip())][:30] if isinstance(b.get("bullets", []), list) else []
    with TASKS_LOCK:
        p = core.P("tasks.json")
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            rows = []
        task = {"id": secrets.token_hex(8), "title": title, "priority": priority, "assigned_date": str(b.get("assigned_date", ""))[:10],
                "deadline": str(b.get("deadline", ""))[:10], "bullets": bullets, "status": status,
                "assigned": assigned, "created_by": user["name"], "created_at": int(time.time()), "source": "admin"}
        rows.append(task)
        core.atomic_write(p, json.dumps(rows, ensure_ascii=False, indent=2))
    return task


@router.patch("/api/dash/tasks/{task_id}")
async def edit_task(task_id: str, request: Request):
    me(request)
    b = await request.json()
    with TASKS_LOCK:
        p = core.P("tasks.json")
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(404, "task not found")
        task = next((x for x in rows if x.get("id") == task_id), None)
        if task is None:
            raise HTTPException(404, "task not found")
        if "priority" in b:
            priority = b["priority"]
            if isinstance(priority, bool) or not isinstance(priority, int) or priority not in (0, 1, 2, 3):
                raise HTTPException(400, "priority must be between 0 and 3")
            task["priority"] = priority
        if "status" in b and b["status"] in ("todo", "in_progress", "done"):
            task["status"] = b["status"]
        for field in ("title", "assigned_date", "deadline"):
            if field in b:
                value = str(b[field]).strip()
                if field == "title":
                    if not value:
                        raise HTTPException(400, "task title is required")
                    task[field] = value[:160]
                elif value:
                    try:
                        time.strptime(value, "%Y-%m-%d")
                    except ValueError:
                        raise HTTPException(400, f"invalid {field}")
                    task[field] = value
                else:
                    task[field] = ""
        if "assigned_date" in b or "deadline" in b:
            for field in ("assigned_date", "deadline"):
                try:
                    if date.fromisoformat(task[field]).isoformat() != task[field]:
                        raise ValueError
                except ValueError:
                    raise HTTPException(400, f"{field} is required in YYYY-MM-DD format")
            if task["deadline"] < task["assigned_date"]:
                raise HTTPException(400, "deadline must be on or after the assignment date")
        if "assigned" in b:
            assigned = b["assigned"] if isinstance(b["assigned"], list) else [b["assigned"]]
            known = set(users())
            task["assigned"] = [str(x) for x in assigned if str(x) == "everyone" or str(x) in known][:50] or ["everyone"]
            if "everyone" in task["assigned"]:
                task["assigned"] = ["everyone"]
        if "bullets" in b and isinstance(b["bullets"], list):
            task["bullets"] = [{"text": str(x.get("text", "")).strip()[:240], "done": bool(x.get("done"))} for x in b["bullets"] if isinstance(x, dict) and str(x.get("text", "")).strip()][:30]
        core.atomic_write(p, json.dumps(rows, ensure_ascii=False, indent=2))
    return task


@router.delete("/api/dash/tasks/{task_id}")
def delete_task(task_id: str, request: Request):
    me(request)
    with TASKS_LOCK:
        p = core.P("tasks.json")
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(404, "task not found")
        new_rows = [x for x in rows if x.get("id") != task_id]
        if len(rows) == len(new_rows):
            raise HTTPException(404, "task not found")
        core.atomic_write(p, json.dumps(new_rows, ensure_ascii=False, indent=2))
    return {"ok": True}


def shown(p, lang):
    """The item's name for the admin in the dashboard's language: the shop's own name in that language (tr / en / ru, copied from the website), the next available one if it has none."""
    order = {"en": ("name_en", "name_tr", "name"), "ru": ("name_ru", "name_en", "name_tr", "name")}.get(lang, ("name_tr", "name", "name_en"))
    return next((p[k] for k in order if p.get(k)), "")


def cached(p):
    """The stored price of an item: (price, age in seconds, in stock) or None. Nothing is read from the network."""
    e = core.live_prices([p]).get(p["id"])
    return (float(e["price"]), int(time.time() - e["ts"]) if e["ts"] else None, e["in_stock"]) if e else None


@router.get("/api/dash/products")
def products(request: Request, q: str = "", lang: str = "tr"):
    """Catalog search for the order form: names and the CACHED price (if there is one). No page is read here."""
    me(request)
    out = []
    for b in core.find_flowers(q, limit=1000, lang="both", with_prices=False):  # every match, not a first page
        c = cached(core.product(b["id"]) or b)
        n_ = core.product(b["id"]) or b
        out.append({"id": b["id"], "name": shown(core.product(b["id"]) or b, lang) or b.get("name_tr"), "category": b.get("category"), "badges": n_.get("badges") or [], "old_price": n_.get("old_price"), "same_day": n_.get("same_day") is not False,
                    "price": c[0] if c else None, "age": c[1] if c else None, "in_stock": c[2] if c else None})
    return out


@router.get("/api/dash/catalog")
def catalog(request: Request, q: str = "", category: str = "", lang: str = "tr"):
    """EVERY item of the catalog (scraped and the owner's own), filterable by words and category. Prices are the cached ones: nothing is read from the shop here."""
    me(request)
    toks = [t for t in re.split(r"\W+", core.fold(q)) if t]
    out, cats, ovr = [], set(), core.overrides()
    for n in core.graph()[0].values():
        if n["type"] != "product":
            continue
        cats.add(n.get("category") or "")
        hay = core.fold(" ".join([n["id"], n["name"], n.get("name_en") or "", n.get("name_tr") or "", n.get("name_ru") or "", n.get("category") or "", *(n.get("aliases") or [])]))
        if (category and n.get("category") != category) or not all(t in hay for t in toks):
            continue
        c = cached(n)
        ph_ = _first_photo(str(n["id"]))
        out.append({"id": n["id"], "name": shown(n, lang), "names": {"tr": n.get("name_tr") or n["name"], "en": n.get("name_en") or "", "ru": n.get("name_ru") or ""},
                    "same_day": n.get("same_day") is not False, "old_price": n.get("old_price"), "badges": n.get("badges") or [], "category": n.get("category") or "", "url": n.get("url") or "",
                    "custom": bool(n.get("custom")), "hidden": n.get("available") is False,
                    "price": float(n["fixed_price"]) if n.get("custom") else float(n["pinned_price"]) if n.get("pinned_price") is not None else (c[0] if c else None), "pinned": n.get("pinned_price") is not None, "age": None if n.get("custom") else (c[1] if c else None),
                    "in_stock": True if n.get("custom") else (c[2] if c else None), "edited": sorted(ovr.get(str(n["id"]), {})), "hay": hay, "photo": bool(ph_), "pv": int(ph_.stat().st_mtime) if ph_ else 0})
    if not _WARM["on"]:
        _WARM["on"] = True
        threading.Thread(target=_warm_thumbs, daemon=True, name="thumbs").start()
    return {"categories": sorted(c for c in cats if c), "items": sorted(out, key=lambda x: x["name"].lower())}


def _first_photo(pid):
    return next(iter(sorted(f for f in (core.ROOT / "dl/data/catalog" / Path(pid).name).glob("*.*") if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))), None)


def _img(path, w, v):
    """An image for the page: `w` = a small cached thumbnail; a URL that carries its version (`v`, changes when the photo does) is cached by the browser for a year, so a list of 250 photos is loaded once."""
    hdr = {"Cache-Control": "private, max-age=31536000, immutable"} if v else {"Cache-Control": "no-cache"}
    return FileResponse(core.thumb(path, min(max(int(w), 32), 600)) if w else path, headers=hdr)


@router.get("/api/dash/photo/{pid}")
def photo(pid: str, request: Request, w: int = 0, v: str = ""):
    me(request)
    f = _first_photo(pid)
    if not f:
        raise HTTPException(404)
    return _img(f, w, v)


_WARM = {"on": False}


def _warm_thumbs():
    """Build the thumbnails of every catalog photo in the background the first time the product list is asked for, so the list does not wait for them."""
    for n in list(core.graph()[0].values()):
        if n["type"] == "product":
            try:
                f = _first_photo(str(n["id"]))
                if f:
                    core.thumb(f, 120)
            except Exception:
                pass


@router.post("/api/dash/catalog/{pid}/price")
def read_price_now(pid: str, request: Request):
    """The admin asks for this item's price to be read from its page and stored now (this item only)."""
    u = me(request)
    p = core.product(pid)
    if not p:
        raise HTTPException(404, "unknown item")
    import refresh_catalog
    ok, _ = refresh_catalog.refresh_prices({pid})
    e = core.live_prices([core.product(pid)]).get(pid)
    if not ok or not e:
        raise HTTPException(409, "the price could not be read from the shop page: the stored price was kept")
    core.log(f"dashboard: {u['name']} updated the price of {pid}")
    return {"price": float(e["price"]), "in_stock": e["in_stock"]}


@router.post("/api/dash/catalog-refresh/stop")
def catalog_stop(request: Request):
    """Stop the running update: what was already updated is merged with the previous list."""
    u = me(request)
    import refresh_catalog
    core.log(f"dashboard: {u['name']} stopped the product update")
    return {"stopping": refresh_catalog.request_stop()}


@router.post("/api/dash/orders/{oid}/ship")
def ship(oid: int, request: Request):
    """Out for delivery: the order moves to the History and the customer is texted (the same function the owner's chat uses)."""
    me(request)
    return act(oid, agent.mark_shipped, oid)


def _own(pid):
    c = core.customs()
    if pid not in c:
        raise HTTPException(404, "that is not one of your own items")
    return c


def _own_out(pid, it):
    return {"id": pid, **it, "colors": ", ".join(it.get("colors") or []), "hidden": (core.product(pid) or {}).get("available") is False,
            "photo": bool(_first_photo(pid))}


def _own_fields(it, b, request):
    from build_graph import categorize
    if "name" in b:
        if not str(b["name"]).strip():
            raise HTTPException(400, "the name is needed")
        it["name"] = str(b["name"]).strip()
    if "price" in b:
        try:
            it["price"] = float(core.parse_amount(b["price"]))
        except ValueError:
            raise HTTPException(400, "the price must be a number")
    if "same_day" in b:
        it["same_day"] = bool(b["same_day"])
    if "old_price" in b:  # the BEFORE price of a promotion; empty = not a promotion. `price` is always the price to pay
        if str(b["old_price"]).strip() == "":
            it.pop("old_price", None)
        else:
            try:
                old = float(core.parse_amount(b["old_price"]))
            except ValueError:
                raise HTTPException(400, tx(request, "old_nan"))
            if old <= float(it.get("price") or 0):
                raise HTTPException(400, tx(request, "old_lt"))
            it["old_price"] = old
    if "category" in b and str(b["category"]).strip():
        cat = categorize(str(b["category"]), it["name"])
        it["category"] = cat if cat != "other" else (core.fold(str(b["category"])).strip() or "other")
    for k_, k2 in (("description", "description"), ("link", "url")):
        if k_ in b:
            it[k2] = str(b[k_]).strip()
    if "colors" in b:
        it["colors"] = [core.fold(x) for x in re.split(r"[,/]", str(b["colors"])) if x.strip()]


def _own_photo(pid, data):
    import base64
    raw = base64.b64decode(str(data).split(",")[-1])
    core.set_custom_photo(pid, agent.jpeg(raw))
    import threading
    threading.Thread(target=agent.reindex, daemon=True).start()


@router.get("/api/dash/custom/{pid}")
def own_get(pid: str, request: Request):
    me(request)
    return _own_out(pid, _own(pid)[pid])


@router.post("/api/dash/custom")
async def own_add(request: Request):
    """Add one of the owner's own items (not on the website): name, price, category and description are needed."""
    u = me(request)
    b = await request.json()
    miss = [k for k in ("name", "price", "category", "description") if not str(b.get(k) or "").strip()]
    if miss:
        raise HTTPException(400, "needed: " + ", ".join(miss))
    try:
        price = core.parse_amount(b["price"])
    except ValueError:
        raise HTTPException(400, "the price must be a number")
    cid = core.add_custom(str(b["name"]), price, str(b["category"]), str(b["description"]), str(b.get("colors") or ""), str(b.get("link") or ""))
    extra = {k: b[k] for k in ("same_day", "old_price") if k in b}
    if extra:
        c = core.customs()
        try:
            _own_fields(c[cid], extra, request)
        except HTTPException:
            del c[cid]
            core.save_customs(c)
            raise
        core.save_customs(c)
    if b.get("photo"):
        _own_photo(cid, b["photo"])
    core.log(f"dashboard: {u['name']} added own item {cid}")
    return {"id": cid}


@router.patch("/api/dash/custom/{pid}")
async def own_edit(pid: str, request: Request):
    u = me(request)
    b = await request.json()
    c = _own(pid)
    _own_fields(c[pid], b, request)
    core.save_customs(c)
    if "hidden" in b:
        core.set_override(pid, "available", "no" if b["hidden"] else "yes")
    if b.get("photo"):
        _own_photo(pid, b["photo"])
    core.log(f"dashboard: {u['name']} edited own item {pid}")
    return _own_out(pid, core.customs()[pid])


@router.delete("/api/dash/custom/{pid}")
def own_delete(pid: str, request: Request):
    u = me(request)
    c = _own(pid)
    del c[pid]
    core.save_customs(c)
    import shutil
    shutil.rmtree(core.ROOT / "dl/data/catalog" / pid, ignore_errors=True)
    core.log(f"dashboard: {u['name']} deleted own item {pid}")
    return {"result": "deleted"}


def _scraped(pid):
    n = core.product(pid)
    if not n or n.get("custom"):
        raise HTTPException(404, "not an item of the shop's website")
    return n


@router.get("/api/dash/catalog/{pid}")
def item_get(pid: str, request: Request):
    me(request)
    n = _scraped(pid)
    return {"id": pid, "name": n["name"], "price": n.get("pinned_price"), "category": n.get("category") or "", "description": n.get("description") or "",
            "colors": ", ".join(n.get("colors") or []), "hidden": n.get("available") is False, "same_day": n.get("same_day") is not False, "old_price": n.get("old_price"),
            "badges": n.get("badges") or [], "names": {"tr": n.get("name_tr") or n["name"], "en": n.get("name_en") or "", "ru": n.get("name_ru") or ""}}


@router.patch("/api/dash/catalog/{pid}")
async def item_edit(pid: str, request: Request):
    """Hand edits of a website item. Only what really changed is saved (and so pinned: a re-scrape never overwrites it); an empty price gives the page price back."""
    u = me(request)
    b = await request.json()
    n = _scraped(pid)
    cur = {"name": n["name"], "description": n.get("description") or "", "category": n.get("category") or "", "colors": ", ".join(n.get("colors") or [])}
    for k, old in cur.items():
        if k in b and str(b[k]).strip() != old and str(b[k]).strip():
            if not core.set_override(pid, k, str(b[k])):
                raise HTTPException(400, f"could not set {k}")
    if "price" in b:
        if str(b["price"]).strip() == "":
            core.unpin(pid, "price")
        elif str(b["price"]).strip() != str(n.get("pinned_price") or "") and not core.set_override(pid, "price", str(b["price"])):
            raise HTTPException(400, "the price must be a positive number")
    if "same_day" in b and bool(b["same_day"]) != (n.get("same_day") is not False):
        core.set_override(pid, "same_day", "yes" if b["same_day"] else "no")
    if "hidden" in b and bool(b["hidden"]) != (n.get("available") is False):
        core.set_override(pid, "available", "no" if b["hidden"] else "yes")
    core.log(f"dashboard: {u['name']} edited item {pid}")
    return item_get(pid, request)


@router.delete("/api/dash/catalog/{pid}")
def item_remove(pid: str, request: Request):
    """Remove a website item for good: no re-scrape ever adds it back."""
    u = me(request)
    _scraped(pid)
    core.remove_product(pid)
    core.log(f"dashboard: {u['name']} removed item {pid} for good")
    return {"result": "removed"}


@router.get("/api/dash/catalog-removed")
def removed_list(request: Request):
    me(request)
    return [{"id": k, **v} for k, v in sorted(core.removed().items(), key=lambda kv: -kv[1].get("ts", 0))]


@router.post("/api/dash/catalog/{pid}/restore")
async def item_restore(pid: str, request: Request):
    """Bring a removed item back right now, with its details read from the website again."""
    u = me(request)
    import refresh_catalog as _rc
    if _rc.busy():
        raise HTTPException(409, "a full product update is running: try again when it has finished")
    import refresh_catalog
    out = await asyncio.to_thread(refresh_catalog.restore_one, pid)
    core.log(f"dashboard: {u['name']} brought item {pid} back")
    return {"result": out}


@router.post("/api/dash/catalog/{pid}/reset")
async def item_reset(pid: str, request: Request):
    """Throw the hand edits away and read the item from the website again: back to its original state."""
    u = me(request)
    import refresh_catalog as _rc
    if _rc.busy():
        raise HTTPException(409, "a full product update is running: try again when it has finished")
    _scraped(pid)
    import refresh_catalog
    out = await asyncio.to_thread(refresh_catalog.reset_one, pid)
    core.log(f"dashboard: {u['name']} reset item {pid} to the website")
    return {"result": out}


# ------------------------------------------------------------------ simulation sandbox
# A second copy of the app in TEST mode (no WhatsApp / Telegram tokens, its own empty state folder, listening only on 127.0.0.1). The dashboard shows its simulation page
# through the /sandbox proxy below, so a simulated customer can never reach a real customer, a real order or the real chat history.
SB = {"proc": None}
SB_LOCK = asyncio.Lock()  # two clicks on "start" must not start two processes
SB_PORT = int(os.environ.get("SANDBOX_PORT", "8001"))
SB_DIR = Path(tempfile.gettempdir()) / "flowers-sandbox-state"


def _sb_running():
    p = SB["proc"]
    return bool(p and p.poll() is None)


def _sb_up():
    try:
        return httpx.get(f"http://127.0.0.1:{SB_PORT}/config", timeout=1.5).status_code == 200
    except Exception:
        return False


def _sb_stop(reset=False):
    p = SB["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
    SB["proc"] = None
    if reset:
        shutil.rmtree(SB_DIR, ignore_errors=True)


atexit.register(_sb_stop)


@router.get("/api/dash/sandbox/status")
async def sandbox_status(request: Request):
    me(request)
    up = _sb_running() and await asyncio.to_thread(_sb_up)
    return {"running": up, "starting": _sb_running() and not up}  # "starting": the process is there, the app inside is still loading


@router.post("/api/dash/sandbox/start")
async def sandbox_start(request: Request):
    u = me(request)
    async with SB_LOCK:
        return await _sandbox_start(u)


def _sb_spawn():
    """Start the sandbox process if it is not already running (does not wait for it)."""
    if _sb_running() or _sb_up():
        return
    SB_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("overrides.json", "custom.json", "removed.json", "prices.json"):  # the shop's own edits and its stored prices are copied, so the simulation sees the same catalog and can place orders
        if core.P(f).exists():
            shutil.copy(core.P(f), SB_DIR / f)
    env = {**os.environ, "STATE_DIR": str(SB_DIR), "WAHA_API_KEY": "", "WAHA_WEBHOOK_SECRET": "", "WHATSAPP_TOKEN": "", "PHONE_NUMBER_ID": "", "TELEGRAM_TOKEN": "", "TELEGRAM_SECRET": "", "PUBLIC_URL": "", "ADMIN_TELEGRAM": "", "DAILY_REFRESH": "0",
           "SANDBOX_PRELOAD": "0", "DL_PRELOAD": "0"}  # (the sandbox never starts a sandbox of its own, and does not load the photo model)
    SB["proc"] = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(SB_PORT)], cwd=str(core.ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sandbox_preload():
    """Called when the server starts: the simulation is already warming up in the background when someone opens its tab."""
    threading.Thread(target=_sb_spawn, daemon=True, name="sandbox-preload").start()


async def _sandbox_start(u):
    if not _sb_running():
        if await asyncio.to_thread(_sb_up):
            raise HTTPException(409, f"something else is already using port {SB_PORT}")
        await asyncio.to_thread(_sb_spawn)
        core.log(f"dashboard: {u['name']} started the simulation sandbox")
    for _ in range(80):  # up to 40 seconds: the app loads its catalog and libraries first
        if await asyncio.to_thread(_sb_up):
            return {"running": True}
        await asyncio.sleep(0.5)
    _sb_stop()
    raise HTTPException(504, "the simulation did not start")


@router.post("/api/dash/sandbox/stop")
async def sandbox_stop(request: Request):
    me(request)
    b = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    await asyncio.to_thread(_sb_stop, bool(b.get("reset")))
    return {"running": False}


async def _forward(method, url, query, ctype, body):
    async with httpx.AsyncClient(timeout=60) as c:
        return await c.request(method, url + (("?" + query) if query else ""), headers={"Content-Type": ctype} if ctype else {}, content=body)


@router.api_route("/sandbox/{path:path}", methods=["GET", "POST"])
async def sandbox_proxy(path: str, request: Request):
    """The sandbox's pages and test endpoints, for a logged-in dashboard user only (the sandbox itself listens on 127.0.0.1)."""
    if not _sess(request):
        raise HTTPException(401, "login required")
    if not re.fullmatch(r"embed/(sim|chat)|config|sim/products|sim/admin/clear|feed|message|scenarios\.json|(?:download|view)/[^/]+", path):  # only what the simulation pages use: never the sandbox's own dashboard or API
        raise HTTPException(404)
    if not _sb_running():
        raise HTTPException(503, "start the simulation first")
    r = await _forward(request.method, f"http://127.0.0.1:{SB_PORT}/{path}", request.url.query, request.headers.get("content-type"), await request.body())
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"), headers={"Cache-Control": "no-store"})


@router.get("/api/dash/catalog-status")
def catalog_status(request: Request):
    me(request)
    import refresh_catalog
    pr = refresh_catalog.PROGRESS
    return {"running": refresh_catalog.busy(), "step": pr["step"], "since": pr["since"], "stopping": pr["stopping"], "maintenance": core.kget("maintenance") == "1",
            "checked": core.kget("watch_ts"), "products": refresh_catalog.count(), "prices_at": max((e.get("ts", 0) for e in core.price_store().values()), default=0)}


@router.post("/api/dash/catalog-refresh")
async def catalog_refresh(request: Request):
    """The big button: re-scrape the shop (details, new and removed products, every price, photos). The agent shows customers the 'we are updating' notice meanwhile (1-3 hours).
    Needs the word `execute` typed by the owner (second confirmation); it can be stopped."""
    u = me(request)
    b = await request.json() if request.headers.get("content-length", "0") not in ("0", "") else {}
    if str(b.get("confirm") or "").strip().lower() != "execute":
        raise HTTPException(400, "type the word execute to confirm")
    import refresh_catalog, threading
    if refresh_catalog.busy():
        return {"started": False, "result": "A refresh is already running."}
    agent.set_down_notice(bool(b.get("notify", True)))
    core.log(f"dashboard: {u['name']} started a full product refresh")
    threading.Thread(target=lambda: core.tell_admins("Product update finished: " + refresh_catalog.run(agent.end_maintenance)), daemon=True, name="manual-refresh").start()
    return {"started": True}


@router.post("/api/dash/catalog/{pid}/update")
def update_item(pid: str, request: Request):
    """Re-read ONE item from its page (name, description, photos, price) without a full refresh."""
    u = me(request)
    import refresh_catalog as _rc
    if _rc.busy():
        raise HTTPException(409, "a full product update is running: try again when it has finished")
    import refresh_catalog
    core.log(f"dashboard: {u['name']} updated item {pid}")
    try:
        return {"result": refresh_catalog.refresh_one(pid)}
    except Exception as e:
        raise HTTPException(409, f"could not update: {e}")


@router.post("/api/dash/products/price")
async def product_price(request: Request):
    """The admin clicked a product: its stored price (from the last update) and how old it is."""
    me(request)
    p = core.product((await request.json()).get("product_id"))
    if not p:
        raise HTTPException(404, "unknown product")
    e = core.live_prices([p]).get(p["id"])
    if not e:
        raise HTTPException(409, "there is no stored price for this item: update it, or add it as a custom item")
    return {"id": p["id"], "price": float(e["price"]), "in_stock": e["in_stock"], "source": "stored", "age": int(time.time() - e["ts"]) if e["ts"] else None}


@router.post("/api/dash/orders")
async def add_order(request: Request):
    u = me(request)
    b = await request.json()
    external = bool(b.get("external"))  # entered by the owner for someone who ordered in person
    raw = str(b.get("phone", "")).strip()
    phone = core.norm(raw) if raw.startswith(("tg:", "ext:")) else core.check_phone(raw)[0] if raw else ""
    if raw and not phone and not raw.startswith(("tg:", "ext:")):
        raise HTTPException(400, phone_problem(request, "w_customer", raw, core.check_phone(raw)[1]))
    for k_ in ("sender_phone", "recipient_phone"):  # every typed number: +90 is added when there is no country code, a number that sounds off is refused
        if str(b.get(k_) or "").strip():
            ok_, why_ = core.check_phone(b[k_])
            if not ok_:
                raise HTTPException(400, phone_problem(request, "w_sender" if k_ == "sender_phone" else "w_recipient", b[k_], why_))
            b[k_] = "+" + ok_
    if external and not phone:
        if not str(b.get("name", "")).strip():
            raise HTTPException(400, "the name of the person is needed")
        phone = core.norm("ext:" + str(b["name"]))
    if not (phone.startswith("ext:") or (phone.isdigit() and len(phone) >= 8)):
        raise HTTPException(400, "customer number needed")
    prods = []
    for it in b.get("items") or []:
        p = core.product(it.get("product_id"))
        if not p:
            raise HTTPException(400, tx(request, "unknown_product", id=it.get("product_id")))
        prods.append((p, max(1, int(it.get("quantity") or 1))))
    custom = str(b.get("custom") or "").strip()
    custom_price = ""
    if b.get("custom_price") not in (None, ""):
        try:
            amount = core.parse_amount(b["custom_price"])
            if amount <= 0:
                raise ValueError("price must be positive")
            custom_price = core.mstr(amount)
        except (ValueError, TypeError):
            raise HTTPException(400, "custom item price must be a valid positive amount")
    if custom_price and not custom:
        raise HTTPException(400, "custom item name needed")
    if not custom_price and custom:
        raise HTTPException(400, "custom item price needed")
    dtype = str(b.get("delivery_type") or "")
    if dtype not in ("normal", "jet", "pickup"):
        raise HTTPException(400, "choose normal, jet, or pick-up delivery")
    if dtype == "jet" and not str(b.get("delivery_time") or "").strip():
        raise HTTPException(400, "jet delivery time needed")
    if dtype == "jet":
        schedule = {k: "" for k in core.OCOLS}
        schedule.update({k: str(b.get(k) or "") for k in ("customer_number", "address", "delivery_date", "delivery_area")})
        schedule.update(delivery_type="normal", order_id=max((int(r["order_id"]) for r in core.orders()), default=0) + 1)
        standard = core.delivery_calc(schedule)["departure"]
        requested = re.sub(r"^(\d):", r"0\1:", str(b["delivery_time"]).strip())[:5]
        if standard and requested == standard:
            dtype = b["delivery_type"] = "normal"
            b["delivery_time"] = ""
    if not prods and not custom:
        raise HTTPException(400, "add at least one item")
    lines = []
    if prods:  # the admin picked these: use the cached price, read the page only for what is not cached (the same numbers the picker showed)
        es = core.live_prices([p for p, _ in prods])
        bad = [p["name"] for p, _ in prods if not es.get(p["id"])]
        if bad:
            raise HTTPException(409, "price could not be read: " + ", ".join(bad))
        lines = [(core.priced(p, es[p["id"]], phone), n) for p, n in prods]
    core.register(phone)
    if b.get("name"):
        core.kset("name:" + phone, str(b["name"]))
    oid = core.order_create(phone, str(b.get("name") or ""), lines, "", "", (str(b.get("notes") or "") + (" | external: in person" if external else "")).strip(" |"), custom=custom, custom_price=custom_price)
    f = {k: str(b[k]) for k in ("sender_name", "sender_phone", "recipient_name", "recipient_phone", "address", "delivery_date", "delivery_type", "delivery_time", "delivery_area") if b.get(k)}
    if b.get("delivery_type") == "normal":
        f["delivery_time"] = ""
    if external and b.get("name") and not f.get("sender_name"):
        f["sender_name"] = str(b["name"])
    if external and not phone.startswith("ext:") and not f.get("sender_phone"):
        f["sender_phone"] = phone
    if f:
        core.order_form_update(oid, **f)
    msgs = []
    if external:
        core.set_profile(phone, name=str(b.get("name") or ""), relationship="in person customer")
        msgs = shared(agent.external_followup, oid, str(b.get("paid") or ""), str(b.get("expect_message") or ""))
    core.log(f"dashboard: {u['name']} added order #{oid} for {phone}")
    if b.get("notify") and not external:
        tell(phone, "created", oid=oid, summary=core.order_summary(core.order_row(oid)))
    return {"order_id": oid, "messages": msgs}


EDITABLE = ("customer_name", "sender_name", "sender_phone", "recipient_name", "recipient_phone", "address", "delivery_area", "delivery_date", "delivery_time", "delivery_type", "notes", "delivery_fee", "status")  # every line of the customer's form, plus the card message (see below)
VISIBLE = ("address", "delivery_area", "delivery_date", "delivery_time", "recipient_name", "recipient_phone", "delivery_type", "card_message")
NOTE_RX = re.compile(r"\s*\| note: [^|]*")  # the card message lives inside `notes` as " | note: <text>"


@router.get("/api/dash/delivery-areas")
def delivery_areas(request: Request):
    """The delivery areas (key + English label) for the order editor's city list."""
    me(request)
    return [{"key": k, "label": a["label"]["en"]} for k, a in core._json("delivery.json")["areas"].items()]


@router.post("/api/dash/delivery-quote")
async def delivery_quote(request: Request):
    me(request)
    b = await request.json()
    row = {k: "" for k in core.OCOLS}
    row.update({k: str(b.get(k) or "") for k in ("customer_number", "address", "delivery_date", "delivery_type", "delivery_time", "delivery_area", "notes", "subtotal", "total")})
    row["order_id"] = max((int(r["order_id"]) for r in core.orders()), default=0) + 1
    calc = core.delivery_calc(row)
    return {"fee": float(calc["fee"]) if calc["fee"] is not None else None, "area": calc["key"], "departure": calc["departure"], "problems": calc["problems"]}


@router.patch("/api/dash/orders/{oid}")
async def edit_order(oid: int, request: Request):
    u = me(request)
    b = await request.json()
    cur = one(oid)
    rows, changes = core.orders(), {}
    base = [dict(r) for r in rows]
    card_old = (re.findall(r"\| note: ([^|]*)", cur["notes"]) or [""])[-1].strip()
    if "notes" in b or "card_message" in b:  # `notes` is the free notes without the card message; the card message is its own field
        plain, card = str(b.get("notes", NOTE_RX.sub("", cur["notes"]).strip())).strip(), str(b.get("card_message", card_old)).strip()
        b["notes"] = (plain + (f" | note: {card}" if card else "")).strip()
        if card != card_old:
            changes["card_message"] = (card_old, card)
    for k_ in ("sender_phone", "recipient_phone"):
        if str(b.get(k_) or "").strip():
            if str(b[k_]).strip() == cur[k_] or core.check_phone(b[k_])[0] == core.check_phone(cur[k_])[0] != "":  # untouched (even if it was saved wrong earlier): not a change
                b.pop(k_)
                continue
            ok_, why_ = core.check_phone(b[k_])
            if not ok_:
                raise HTTPException(400, phone_problem(request, "w_sender" if k_ == "sender_phone" else "w_recipient", b[k_], why_))
            b[k_] = "+" + ok_
    if b.get("delivery_area") and b["delivery_area"] not in core._json("delivery.json")["areas"]:
        raise HTTPException(400, tx(request, "bad_area"))
    for r in rows:
        if int(r["order_id"]) == oid:
            for k in EDITABLE:
                if k in b and str(b[k]) != r[k]:
                    if k == "delivery_fee" and str(b[k]).strip():
                        try:
                            b[k] = str(core.parse_amount(b[k]))
                        except ValueError:
                            raise HTTPException(400, "delivery fee must be a number")
                    changes[k] = (r[k], str(b[k]))
                    r[k] = str(b[k])
    if not changes:
        return {"result": tx(request, "no_change")}
    res = core.apply_orders_edit(rows, base)
    if res.startswith("REFUSED"):
        raise HTTPException(400, res)
    if cur["status"] in core.OPEN and "delivery_fee" not in changes and any(k in changes for k in ("address", "delivery_area", "delivery_type", "delivery_time")):
        core.order_form_update(oid)  # a new address / delivery choice changes the area and the fee
    core.log(f"dashboard: {u['name']} edited order #{oid}: {', '.join(changes)}")
    shown = [f"{k.replace('_', ' ')}: {v[1]}" for k, v in changes.items() if k in VISIBLE]
    if b.get("notify") and shown:
        why = core.change_explain(cur, one(oid), ("en", "tr", "ru")[agent.lang_of(cur["customer_number"])])  # a price that moved is explained, whatever the field
        tell(cur["customer_number"], "edit", oid=oid, changes="\n".join(shown) + ("\n\n" + why if why else ""))
    got = one(oid)["delivery_area"]
    if b.get("delivery_area") and got != b["delivery_area"]:  # the town written in the address decides the area: say so instead of silently keeping the old one
        res += tx(request, "area_wins", got=got, want=b["delivery_area"])
    return {"result": res}


ITEM_NOTE = {  # what the customer is asked after the team changed their items, by stage (en, tr, ru)
    "open": ("Please have a look: if it is all right, tell me and the team will confirm it.", "Lütfen bir göz atın: uygunsa bana yazın, ekibimiz onaylasın.", "Посмотрите, пожалуйста: если всё верно, напишите мне, и команда подтвердит."),
    "unpaid": ("The first quotation no longer counts. Please check it and tell me if it is right: the team confirms it again and sends you a new quotation.",
               "İlk teklif artık geçerli değil. Lütfen kontrol edin, doğruysa bana yazın: ekibimiz yeniden onaylayıp size yeni bir teklif göndersin.",
               "Первое предложение больше не действует. Проверьте, пожалуйста, и напишите, если всё верно: команда снова подтвердит и пришлёт новое предложение."),
}


@router.post("/api/dash/orders/{oid}/items")
async def edit_items(oid: int, request: Request):
    """The owner changes the ITEMS of an order at any stage, the same way as a customer's change. Body: {items: [{name | product_id, quantity}], notify}.
    An item already on the order keeps the price it was sold at; a new one gets the stored catalog price. Before the final payment confirmation it is applied at once and the order goes back to the
    confirmation step (the customer is asked to check it); after the payment was confirmed the owner's edit IS the approval: a higher total asks the customer for the difference only, a lower one asks how the balance should come back."""
    u = me(request)
    b = await request.json()
    cur = one(oid)
    st, ph = cur["status"], cur["customer_number"]
    if st in core.LOST_STATUSES or st == "completed":
        raise HTTPException(409, tx(request, "shipped") if st == "completed" else tx(request, "closed", st=tx(request, "st_" + st)))
    old = core.row_lines(cur)
    if old is None:
        raise HTTPException(409, tx(request, "unreadable"))
    custom_changed = "custom" in b
    custom_rows, custom_text, custom_total = [], "", ""
    if custom_changed:
        if st not in core.OPEN:
            raise HTTPException(409, tx(request, "custom_only_open"))
        if not isinstance(b.get("custom"), list):
            raise HTTPException(400, tx(request, "custom_list_bad"))
        total = core.D(0)
        for item in b.get("custom") or []:
            if not isinstance(item, dict):
                raise HTTPException(400, tx(request, "custom_object_bad"))
            name = str(item.get("name") or "").strip()
            try:
                qty = int(item.get("quantity") or 0)
                price = core.D(core.parse_amount(item.get("price")))
            except (TypeError, ValueError):
                raise HTTPException(400, tx(request, "custom_item_bad"))
            if not name or qty < 1 or price <= 0:
                raise HTTPException(400, tx(request, "custom_item_bad"))
            custom_rows.append((name, qty))
            total += price * qty
        custom_text = "; ".join(f"{qty} × {name}" for name, qty in custom_rows)
        custom_total = core.mstr(total) if custom_rows else ""
    have = {n: (q, un) for n, q, un in old}
    merged = {}
    for it in b.get("items") or []:
        q = int(it.get("quantity") or 0)
        if q <= 0:
            continue
        nm = str(it.get("name") or "")
        if nm in have:
            unit = have[nm][1]
        else:
            p = core.product(it.get("product_id"))
            if not p:
                raise HTTPException(400, tx(request, "unknown_product", id=it.get("product_id")))
            e = core.live_prices([p]).get(p["id"])
            if not e:
                raise HTTPException(409, tx(request, "no_price", name=p["name"]))
            nm, unit = core._nm(p["name"]), core.D(e["price"])
            if nm in have:
                unit = have[nm][1]  # already on the order: it keeps the price it was sold at
        merged[nm] = (merged.get(nm, (0, unit))[0] + q, unit)
    lines = [(n, q, un) for n, (q, un) in merged.items()]
    if not lines and not (custom_text if custom_changed else cur["custom_items"]):
        raise HTTPException(400, tx(request, "need_item"))
    if [(n, q, un) for n, q, un in old] == lines and not custom_changed:
        return {"result": tx(request, "no_change")}
    lg = agent.lang_of(ph)
    lang = ("en", "tr", "ru")[lg]
    notify = bool(b.get("notify", True))
    req = {"kind": "change", "reason": f"changed by the team ({u['name']})", "lines": [[n, q, core.mstr(un)] for n, q, un in lines]}
    if custom_changed:
        req["custom_items"], req["custom_price"] = custom_text, custom_total
    if st in core.OPEN:  # not confirmed yet: applied at once
        row, _, _ = core.order_set_lines(oid, [({"name": n, "price": un}, q) for n, q, un in lines], custom_items=custom_text if custom_changed else None, custom_price=custom_total if custom_changed else None)
        why, stage = core.change_explain(cur, row, lang), "open"
    elif st in ("awaiting_payment", "payment_review"):  # quotation sent, payment not confirmed: back to the confirmation step, a new quotation follows
        if not core.request_change(oid, ph, req):
            raise HTTPException(409, tx(request, "not_changed"))
        row, _, why_not = core.change_apply(oid, requote=True)
        if not row:
            core.change_drop(oid)
            raise HTTPException(409, why_not)
        for k_ in (f"bok:{oid}", f"fok:{oid}", f"subm:{oid}", f"shown:{oid}", f"announced:{oid}", f"formsent:{oid}"):
            core.kset(k_, "")
        core.kset(f"edited:{oid}", "had a receipt" if st == "payment_review" else "1")
        why, stage = core.change_explain(cur, row, lang), "unpaid"
    else:  # confirmed (paid): the owner's edit is the approval; the same code that approves a customer's request does the money part and tells the customer
        if not core.request_change(oid, ph, req):
            raise HTTPException(409, tx(request, "not_changed"))
        msgs = shared(agent.approve_change, oid)
        if core.change_state(core.order_row(oid) or {}) == "" and (core.order_row(oid) or {}).get("change_request"):
            core.change_drop(oid)  # refused (e.g. minimum order): nothing was applied, no request is left waiting
        core.log(f"dashboard: {u['name']} changed the items of paid order #{oid}")
        return {"result": " ".join(msgs) or "Done."}
    core.log(f"dashboard: {u['name']} changed the items of order #{oid} ({st})")
    if notify:
        tell(ph, "edit", oid=oid, changes=(why + "\n\n" if why else "") + agent.pick(ITEM_NOTE[stage], lg))
    return {"result": tx(request, "items_changed", summary=core.order_summary(core.order_row(oid)), told=tx(request, "told") if notify else "")}


@router.delete("/api/dash/orders/{oid}")
async def remove_order(oid: int, request: Request):
    u = me(request)
    b = await request.json() if request.headers.get("content-length", "0") not in ("0", "") else {}
    cur = one(oid)
    base = core.orders()
    res = core.apply_orders_edit([r for r in base if int(r["order_id"]) != oid], base)
    core.log(f"dashboard: {u['name']} removed order #{oid}")
    excuse = str(b.get("excuse") or "").strip()
    if b.get("notify", bool(excuse)):
        tell(cur["customer_number"], "removed", oid=oid, excuse=excuse)
    return {"result": res}


@router.post("/api/dash/orders/order")
async def order_cards(request: Request):
    """The order of the cards in a board column after the admin dragged one up or down (top = most urgent)."""
    me(request)
    for i, oid in enumerate((await request.json()).get("ids") or []):
        if core.order_row(oid):
            core.kset(f"pos:{int(oid)}", str(i))
    return {"ok": True}


@router.post("/api/dash/crm/order")
async def crm_cards(request: Request):
    me(request)
    phones = (await request.json()).get("phones") or []
    with core.STATE_LOCK:  # read, change, write back: one at a time
        d = core.customers()
        for i, p in enumerate(phones):
            d.setdefault(core.canon_phone(p), {})["pos"] = i
        core.atomic_write(core.P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))
    return {"ok": True}


@router.get("/api/dash/orders/{oid}/pdf")
def order_pdf(oid: int, request: Request, kind: str = "quotation", inline: int = 0):
    """The quotation or invoice of an order as a PDF for the admin to open (it is built from the order as it is now)."""
    me(request)
    if kind not in ("quotation", "invoice"):
        raise HTTPException(400, "kind must be quotation or invoice")
    import docs_pdf
    r = one(oid)
    return FileResponse(docs_pdf.build(r, kind, "tr" if agent.lang_of(r["customer_number"]) == 1 else "en"), media_type="application/pdf", filename=f"{kind}_{oid}.pdf", content_disposition_type="inline" if inline else "attachment")


@router.post("/api/dash/orders/{oid}/priority")
async def priority(oid: int, request: Request):
    me(request)
    one(oid)
    core.kset(f"prio:{oid}", str(max(0, min(3, int((await request.json()).get("value") or 0)))))
    return {"ok": True}


def act(oid, fn, *a, **k):
    one(oid)
    return {"messages": shared(fn, *a, **k)}


def set_fee(oid, value):
    """The admin changed this order's delivery fee (0 = free) in the confirm / price screen."""
    if value in (None, ""):
        return
    try:
        fee = core.parse_amount(value)
    except ValueError:
        raise HTTPException(400, "delivery fee must be a number")
    cur = one(oid)
    if cur["delivery_fee"] != "" and core.D(cur["delivery_fee"]) == core.D(str(fee)):
        return
    if not core.set_delivery_fee(oid, fee):
        raise HTTPException(409, "the delivery fee can only be changed before the first confirmation")


@router.post("/api/dash/orders/{oid}/confirm")
async def confirm(oid: int, request: Request):
    """First confirmation. `amount_to_pay` (optional): the admin's own final total; higher than items + delivery becomes a Service fee line, lower a Discount line in the customer's message."""
    me(request)
    b = await request.json()
    cur = one(oid)
    set_fee(oid, b.get("delivery_fee"))
    if b.get("custom_total") not in (None, ""):
        cur = one(oid)
        if cur["status"] not in core.OPEN or not cur["custom_items"] or cur["custom_price"]:
            raise HTTPException(409, tx(request, "custom_only_open"))
        try:
            custom_total = core.D(core.parse_amount(b["custom_total"]))
        except (TypeError, ValueError):
            raise HTTPException(400, tx(request, "custom_item_bad"))
        if custom_total <= 0:
            raise HTTPException(400, tx(request, "custom_item_bad"))
        row, why = core.order_price_custom(oid, core.D(cur["subtotal"] or 0) + custom_total, status="pending")
        if not row:
            raise HTTPException(400, tx(request, "custom_item_bad"))
    if b.get("amount_to_pay") not in (None, ""):
        try:
            amt = core.parse_amount(b["amount_to_pay"])
        except ValueError:
            raise HTTPException(400, "amount to pay must be a number")
        cur = one(oid)
        if cur["payable"] and core.D(str(amt)) != core.D(cur["payable"]) and not core.set_payable(oid, amt):
            raise HTTPException(409, "the amount cannot be changed: the order is not waiting for its first confirmation or its delivery fee is not known")
    return act(oid, agent.decide, True, oid, "", bool(b.get("force")))


@router.post("/api/dash/orders/{oid}/reject")
async def reject(oid: int, request: Request):
    me(request)
    b = await request.json()
    return act(oid, agent.decide, False, oid, str(b.get("reason") or ""))


def breakdown(r, custom_price=None):
    """The price of an order piece by piece, calculated here (Decimal, never by a model): catalog lines, the custom part, delivery, totals. No discounts exist."""
    D = core.D
    L = core.row_lines(r) or []
    sub = sum((q * u for _, q, u in L), D(0))
    prof = core.customers().get(core.norm(r["customer_number"]), {})
    custom = D(str(custom_price)).quantize(D("0.01")) if custom_price not in (None, "") else (D(r["custom_price"]) if r["custom_price"] and r["status"] == "quoted" else None)
    goods = sub + custom if custom is not None else (sub if not r["custom_items"] else None)
    calc = core.delivery_calc({**r, "total": core.mstr(goods) if goods is not None else "", "custom_price": core.mstr(custom) if custom is not None else ""})
    fee = calc["fee"]
    return {"lines": [{"name": n, "qty": q, "unit": float(u), "total": float(q * u)} for n, q, u in L], "subtotal": float(sub),
            "custom_items": r["custom_items"], "custom_price": float(custom) if custom is not None else None,
            "goods_total": float(goods) if goods is not None else None,
            "delivery": {"type": r["delivery_type"] or "normal", "time": r["delivery_time"], "address": r["address"], "area": calc["key"], "area_label": core._json("delivery.json")["areas"].get(calc["key"], {}).get("label", {}).get("en", ""), "fee": float(fee) if fee is not None else None,
                         "problems": calc["problems"]},
            "payable": float(goods + fee) if goods is not None and fee is not None else None}


@router.post("/api/dash/orders/{oid}/pricing")
async def pricing(oid: int, request: Request):
    """The breakdown shown while the admin sets a price; call it again with custom_price to see the new totals."""
    me(request)
    b = await request.json() if request.headers.get("content-length", "0") not in ("0", "") else {}
    return breakdown(one(oid), b.get("custom_price"))


@router.post("/api/dash/orders/{oid}/price")
async def price(oid: int, request: Request):
    me(request)
    b = await request.json()
    r = one(oid)
    set_fee(oid, b.get("delivery_fee"))
    r = one(oid)
    if b.get("custom_price") not in (None, ""):  # the price of the custom part: the total is worked out here
        bd = breakdown(r, b["custom_price"])
        if bd["goods_total"] is None:
            raise HTTPException(400, "price of the custom part needed")
        amount = bd["goods_total"]
    else:
        try:
            amount = core.parse_amount(b.get("amount"))
        except ValueError:
            raise HTTPException(400, "amount needed")
    if r["status"] == "quoted" and r["total"] and core.D(r["total"]) == core.D(str(amount)).quantize(core.D("0.01")):
        return {"messages": ["Already sent with this total."]}  # a double click must not text the customer twice
    return act(oid, agent.set_price, oid, amount, str(b.get("note") or ""), bool(b.get("force")))


@router.post("/api/dash/orders/{oid}/payment")
async def payment(oid: int, request: Request):
    me(request)
    b = await request.json()
    if b.get("ok"):
        return act(oid, agent.confirm_payment, oid)
    return act(oid, agent.reject_payment, oid, str(b.get("reason") or ""))


# ------------------------------------------------------------------ the agent: status and on/off
def agent_state():
    return "maintenance" if core.kget("maintenance") == "1" else "stopped" if core.kget("agent_stopped") == "1" else "on"


def waiting_list():
    return [{"phone": p, "name": n, "text": t} for p, n, t in agent.missed()]


@router.get("/api/dash/agent")
def agent_status(request: Request):
    me(request)
    last = core.kget("refresh_at")
    return {"state": agent_state(), "waiting": waiting_list(), "refreshed_hours_ago": round((time.time() - float(last)) / 3600, 1) if last else None}


@router.post("/api/dash/agent")
async def agent_switch(request: Request):
    u = me(request)
    b = await request.json()
    on = bool(b.get("on"))
    if on:
        agent.resume_now()
    else:
        agent.stop_now(bool(b.get("notify", True)))  # notify = send the 'we are unavailable' message to customers who write meanwhile
    core.log(f"dashboard: {u['name']} turned the agent {'on' if on else 'off'}")
    return {"state": agent_state(), "waiting": waiting_list()}


@router.post("/api/dash/agent/backlog")
async def agent_backlog(request: Request):
    """After the agent was off: answer every customer who wrote meanwhile (yes) or drop their kept messages (no)."""
    u = me(request)
    yes = bool((await request.json()).get("answer"))
    n = agent.answer_missed_now(yes)
    core.log(f"dashboard: {u['name']} {'answered' if yes else 'skipped'} {n} missed chat(s)")
    return {"n": n, "state": agent_state()}


# ------------------------------------------------------------------ CRM
def _kind(sts):
    """Where a person is by their orders alone: lead (nothing in progress), open (an order in progress), customer (a confirmed or delivered order)."""
    return "customer" if sts & {"confirmed", "completed"} else "open" if sts & {"pending", "quote", "quoted", "awaiting_payment", "payment_review"} else "lead"


@router.get("/api/dash/crm")
def crm(request: Request):
    me(request)
    names = {k[5:]: v for k, v in core.q("SELECT k,v FROM kv WHERE k LIKE 'name:%'")}
    stat = core.chat_stats()
    prof, out, rows = core.customers(), [], core.orders()
    for phone, first in core.q("SELECT phone, first_ts FROM customers"):
        mine = [r for r in rows if r["customer_number"] == phone]
        sts = {r["status"] for r in mine}
        kind = _kind(sts)
        p = prof.get(phone, {})
        if p.get("hidden"):  # removed from the CRM: their orders and chat history stay in History and Analytics
            continue
        mine_ = sorted(mine, key=lambda r: r["timestamp"])
        last_addr = next((r["address"] for r in reversed(mine_) if r["address"] and not r["address"].startswith("PICKUP")), "")
        last_form = next((r for r in reversed(mine_) if r["sender_name"] or r["address"] or r["delivery_type"]), {})
        out.append({"phone": phone, "company": p.get("company", ""), "location": p.get("location") or last_addr, "location_auto": not p.get("location") and bool(last_addr), "last_form": {k: last_form.get(k, "") for k in ("sender_name", "sender_phone", "recipient_name", "recipient_phone", "address", "delivery_date", "delivery_area", "delivery_type", "delivery_time")},
                    "key_ok": phone.startswith(("tg:", "ext:")) or bool(core.check_phone(phone)[0]), "form_phone": p.get("phone") or next((r["sender_phone"] for r in reversed(mine_) if r["sender_phone"]), ""), "recipients": core.people(phone, 50), "name": p.get("name") or names.get(phone, ""), "first": first, "messages": stat.get(phone, (0, ""))[0], "last": stat.get(phone, (0, ""))[1],
                    "handle": core.handle_of(phone), "orders": len(mine), "wants": time.time() - float(core.kget(f"lead:{phone}", 0) or 0) < 86400 and not mine, "kind": kind, "channel": "telegram" if phone.startswith("tg:") else "person" if phone.startswith("ext:") else "whatsapp", "relationship": p.get("relationship", ""), "notes": p.get("notes", ""), "character": p.get("character", ""), "character_by": p.get("character_by", ""),
                    "stage": p.get("stage") if p.get("stage") and p.get("stage_kind") == kind else kind, "prio": int(p.get("prio") or 0), "pos": p.get("pos")})
    return sorted(out, key=lambda x: x["last"] or "", reverse=True)


@router.get("/api/dash/crm/{phone}/orders")
def crm_orders(phone: str, request: Request):
    me(request)
    customer = core.canon_phone(phone)
    rows = core.orders()
    mine = sorted((r for r in rows if r["customer_number"] == customer), key=lambda r: r["timestamp"], reverse=True)
    return [row_out(r, rows) for r in mine]


def _person_fields(request, b):
    name = str(b.get("name") or "").strip()
    if not name:
        raise HTTPException(400, tx(request, "person_name"))
    ph = str(b.get("phone") or "").strip()
    if ph:
        ok_, why_ = core.check_phone(ph)
        if not ok_:
            raise HTTPException(400, phone_problem(request, "w_recipient", ph, why_))
        ph = "+" + ok_
    return {"name": name, "phone": ph, "address": str(b.get("address") or "").strip(), "relation": str(b.get("relation") or "").strip(), "note": str(b.get("note") or "").strip()}


@router.post("/api/dash/crm/{phone}/people")
async def person_add(phone: str, request: Request):
    """Add a person the customer sends to (or update the one with the same name): the permanent memory the agent uses when the customer says "send it to X again"."""
    u = me(request)
    f = _person_fields(request, await request.json())
    k = core.remember_person(phone, f["name"], f["phone"], f["address"], f["relation"], f["note"])
    core.log(f"dashboard: {u['name']} saved {f['name']} for {phone}")
    return {"key": k}


@router.patch("/api/dash/crm/{phone}/people/{key:path}")
async def person_edit(phone: str, key: str, request: Request):
    u = me(request)
    b = await request.json()
    f = _person_fields(request, b)
    nk = core.edit_person(phone, key, **f)
    if not nk:
        raise HTTPException(404, tx(request, "person_gone"))
    core.log(f"dashboard: {u['name']} changed {f['name']} of {phone}")
    return {"key": nk}


@router.delete("/api/dash/crm/{phone}/people/{key:path}")
def person_forget(phone: str, key: str, request: Request):
    u = me(request)
    core.forget_person(phone, key)
    core.log(f"dashboard: {u['name']} removed {key} of {phone}")
    return {"result": "removed"}


@router.delete("/api/dash/crm/{phone}")
def crm_remove(phone: str, request: Request):
    """Take a person off the CRM board. Nothing else is deleted: their orders, the History, the Analytics and their chats stay. They come back by themselves if they write again."""
    u = me(request)
    ph = core.canon_phone(phone)
    core.set_profile(ph, hidden=True)
    core.log(f"dashboard: {u['name']} removed {ph} from the CRM")
    return {"result": "removed"}


@router.patch("/api/dash/crm/{phone}")
async def crm_edit(phone: str, request: Request):
    u = me(request)
    b = await request.json()
    ph = core.canon_phone(phone)
    moved_to = ""
    with core.STATE_LOCK:  # read, change, write back: one at a time
        d = core.customers()
        c = d.setdefault(ph, {})
        for k in ("name", "relationship", "notes", "company", "location"):
            if k in b:
                c[k] = str(b[k])
        if "phone" in b:  # the customer's real phone number (a Telegram customer has none from Telegram), same rules as in the admin chat
            v = str(b["phone"]).strip()
            shown_ = c.get("phone") or next((r["sender_phone"] for r in reversed(core.orders()) if r["customer_number"] == ph and r["sender_phone"]), "")
            if not v:
                c.pop("phone", None)
            elif v != shown_:
                ok_, why_ = core.check_phone(v)
                if not ok_:
                    raise HTTPException(400, phone_problem(request, "w_phone", v, why_))
                new_, msg_ = core.set_contact_phone(ph, v, c)
        tgt = core.check_phone(c.get("phone") or "")[0] if "phone" in b else ""
        if tgt and tgt != ph and not ph.startswith(("tg:", "ext:")) and (b.get("move_chat") or not core.check_phone(ph)[0]):
            moved_to = tgt  # the chat number moves to it (a chat id that is not a valid number, e.g. saved without its country code, always moves), also when the phone itself was saved earlier
        if "character" in b:  # edited by the owner = locked against the automatic note; emptied = the automatic note may write it again
            ch = str(b["character"]).strip()[:240]
            if ch != c.get("character", ""):
                c["character"] = ch
                c["character_by"] = "owner" if ch else "auto"
                if not ch:
                    c.pop("character_n", None)
        if b.get("stage") in ("lead", "open", "customer", "lost"):  # the board column the owner dragged the customer to
            c["stage"] = b["stage"]
            c["stage_kind"] = _kind({r["status"] for r in core.orders() if r["customer_number"] == ph})  # remembered: the placement holds until the customer moves on by themselves
        if "prio" in b:
            c["prio"] = max(0, min(3, int(b["prio"] or 0)))
        core.atomic_write(core.P("customers.json"), json.dumps(d, ensure_ascii=False, indent=1))
    if c.get("name"):
        core.kset("name:" + ph, c["name"])
    core.log(f"dashboard: {u['name']} edited CRM {ph}")
    msg = core.move_customer(ph, moved_to, lx(request)) if moved_to and moved_to != ph else ""
    if msg:
        core.log(f"dashboard: {u['name']} moved {ph} to {moved_to}: {msg}")
    return {"ok": True, "messages": [msg] if msg else []}


@router.get("/api/dash/crm/{phone}/chat")
def chat(phone: str, request: Request, tail: int = 20, before: int = 0, limit: int = 20):
    """A customer's chat, newest page first (`tail`), older pages on demand (`before` = a message id)."""
    me(request)
    out = [{"id": i + 1, "role": r, "text": c} for i, (ts, r, c) in enumerate(core.chat_log(core.canon_phone(phone)))]
    page = [m for m in out if m["id"] < before][-max(1, min(limit, 100)):] if before else out[-max(1, min(tail, 100)):]
    return {"messages": page, "more": bool(page) and page[0]["id"] > 1}


@router.post("/api/dash/crm/{phone}/message")
async def send_message(phone: str, request: Request):
    u = me(request)
    text = str((await request.json()).get("text") or "").strip()
    if not text:
        raise HTTPException(400, "empty message")
    agent.reply(core.canon_phone(phone), text)
    core.log(f"dashboard: {u['name']} messaged {phone}")
    return {"ok": True}
