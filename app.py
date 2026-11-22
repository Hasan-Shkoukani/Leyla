"""FastAPI: WhatsApp Meta Cloud webhook + mock-mode test UI. Run: uvicorn app:app --port 8000"""
import asyncio
import json, time
import os
import hashlib, hmac
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
import agent, core  # noqa: E402  (after load_dotenv so env is set)
from core import cfg, norm  # noqa: E402

app = FastAPI()
app.mount("/assets", StaticFiles(directory=str(Path(__file__).parent / "static/assets")), name="assets")  # self-hosted fonts: no third-party request from the dashboard
import dashboard  # noqa: E402
app.include_router(dashboard.router)
GRAPH = f"https://graph.facebook.com/{cfg('GRAPH_API_VERSION', 'v27.0')}"


def _meta_test_allowed(to):
    if cfg("META_TEST_MODE") != "1":
        return
    allowed = {x.strip().lstrip("+") for x in cfg("META_TEST_RECIPIENTS").split(",") if x.strip()}
    if str(to).lstrip("+") not in allowed:
        raise RuntimeError("META_TEST_MODE blocks WhatsApp recipients outside META_TEST_RECIPIENTS")
def is_mock():
    """Test mode (the chat test pages are on): no real channel is connected."""
    return not (cfg("WAHA_API_KEY") or (cfg("WHATSAPP_TOKEN") and cfg("PHONE_NUMBER_ID")) or cfg("TELEGRAM_TOKEN"))


class WhatsApp:
    """Thin Meta Cloud API adapter (sync httpx; called from worker threads or briefly from the loop)."""
    def _h(self):
        return {"Authorization": "Bearer " + cfg("WHATSAPP_TOKEN")}

    def _post(self, payload):
        _meta_test_allowed(payload.get("to", ""))
        r = httpx.post(f"{GRAPH}/{cfg('PHONE_NUMBER_ID')}/messages", headers=self._h(), json={"messaging_product": "whatsapp", **payload}, timeout=30)
        r.raise_for_status()
        return r.json().get("messages", [{}])[0].get("id")  # wamid: lets admin replies be tied to alerts

    def send_text(self, to, body):
        return self._post({"to": to, "type": "text", "text": {"body": body[:4096]}})

    def send_doc(self, to, path, caption=""):
        _meta_test_allowed(to)
        with open(path, "rb") as f:
            r = httpx.post(f"{GRAPH}/{cfg('PHONE_NUMBER_ID')}/media", headers=self._h(), data={"messaging_product": "whatsapp"},
                           files={"file": (Path(path).name, f, __import__("mimetypes").guess_type(path)[0] or "application/octet-stream")}, timeout=60)
        r.raise_for_status()
        return self._post({"to": to, "type": "document", "document": {"id": r.json()["id"], "filename": Path(path).name.split("_", 1)[-1], "caption": caption}})

    def send_image(self, to, path, caption=""):
        _meta_test_allowed(to)
        with open(path, "rb") as f:
            r = httpx.post(f"{GRAPH}/{cfg('PHONE_NUMBER_ID')}/media", headers=self._h(), data={"messaging_product": "whatsapp"},
                           files={"file": (Path(path).name, f, "image/jpeg")}, timeout=60)
        r.raise_for_status()
        return self._post({"to": to, "type": "image", "image": {"id": r.json()["id"], "caption": caption[:1024]}})

    def send_template(self, to, name, params):
        return self._post({"to": to, "type": "template", "template": {"name": name, "language": {"code": cfg("TEMPLATE_LANG", "en")},
                    "components": [{"type": "body", "parameters": [{"type": "text", "text": p} for p in params]}]}})

    def typing(self, msg_id):  # marks the customer's message read and shows "typing..." (fire and forget: never blocks or breaks a reply)
        try:
            self._post({"status": "read", "message_id": msg_id, "typing_indicator": {"type": "text"}})
        except Exception:
            pass

    def download(self, media_id):
        meta = httpx.get(f"{GRAPH}/{media_id}", headers=self._h(), timeout=30).json()
        r = httpx.get(meta["url"], headers=self._h(), timeout=60)
        r.raise_for_status()
        return r.content, meta.get("mime_type", "")


@app.on_event("startup")
async def _pool():  # each chat turn waits on OpenAI most of the time, so many can run at once; the default pool is only cpu+4 threads
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(int(cfg("WORKER_THREADS", "64"))))
    agent.LOOP = asyncio.get_running_loop()


@app.on_event("startup")
def _startup():
    if cfg("WAHA_API_KEY"):
        from waha import Waha
        core.TRANSPORT = Waha()
    elif cfg("WHATSAPP_TOKEN") and cfg("PHONE_NUMBER_ID"):
        core.TRANSPORT = WhatsApp()
    else:
        core.TRANSPORT = None
    if os.environ.get("SANDBOX_PRELOAD", "1") != "0" and not is_mock():  # live server only: the simulation copy warms up now, so its dashboard tab opens at once
        dashboard.sandbox_preload()
    if os.environ.get("WA_CHECK", "1") != "0" and cfg("WHATSAPP_TOKEN") and not cfg("WAHA_API_KEY"):  # an expired WhatsApp token is noticed within minutes (and every 6 hours), not days later
        import threading, time as _t

        def _wa_loop():
            while True:
                try:
                    core.whatsapp_check()
                except Exception as e:
                    core.log(f"whatsapp check failed: {e!r}")
                _t.sleep(6 * 3600)
        threading.Thread(target=_wa_loop, daemon=True, name="wa-check").start()
    if os.environ.get("RESEND_CHECK", "1") != "0":  # a sales order that never reached an owner (a restart, a channel that was down) is sent again, once
        import threading, time as _t3
        def _resend_loop():
            while True:
                try:
                    done = agent.resend_unnotified()
                    if done:
                        core.log(f"sales orders sent after a delay: {done}")
                except Exception as e:
                    core.log(f"resend check failed: {e!r}")
                _t3.sleep(60)
        threading.Thread(target=_resend_loop, daemon=True, name="resend-check").start()
    if os.environ.get("LATE_CHECK", "1") != "0":  # a paid order not marked shipped one hour after its delivery time: the owners are told once
        import threading, time as _t2
        def _late_loop():
            while True:
                try:
                    core.late_check()
                except Exception as e:
                    core.log(f"late check failed: {e!r}")
                _t2.sleep(300)
        threading.Thread(target=_late_loop, daemon=True, name="late-check").start()
    if os.environ.get("DL_PRELOAD", "1") != "0":  # load the photo model now, in the background: the first photo after a restart used to wait 6-8 s for it
        import threading
        threading.Thread(target=lambda: __import__("dl.predict", fromlist=["_load"])._load(), daemon=True, name="dl-preload").start()
    msg = core.restore_orders_if_broken()
    if msg:
        core.log(msg)
        core.tell_admins("⚠️ " + msg)
    if cfg("TELEGRAM_TOKEN"):
        import tgbot
        core.TG = tgbot.Telegram(cfg("TELEGRAM_TOKEN"))
        core.TG.on_sent = core.remember_msg
        if cfg("PUBLIC_URL") and cfg("TELEGRAM_SECRET"):  # tell Telegram where to deliver messages (done again at every start: the ngrok address may have changed)
            try:
                core.TG.set_webhook(cfg("PUBLIC_URL").rstrip("/") + "/telegram/webhook", cfg("TELEGRAM_SECRET"))
            except Exception as e:
                core.log(f"telegram webhook not set: {e!r}")
    for u in core.tgq_all():  # Telegram messages that were received but not yet handed to the agent when the server stopped
        if agent.LOOP:
            agent.LOOP.call_soon_threadsafe(_tg_spawn, u)
    if core.kget("maintenance") == "1":  # the app died in the middle of a refresh: reopen and re-text the customers who wrote meanwhile
        core.kset("maintenance", "0")
        agent.end_maintenance()
        core.tell_admins("The product update was interrupted by a restart. Ask me to update the products again.")  # (never lets a failing message stop the server from starting)
    elif core.kget("agent_stopped") != "1":  # messages that arrived before a restart and were not answered yet (and the ones kept during an update): answered now
        agent.flush_inbox()
    else:  # stopped by the owner: what was in flight waits like the rest for his decision
        core.atomic_write(core.P("inbox.json"), json.dumps([{**x, "live": False} for x in core.inbox()]))
    if not is_mock() and os.environ.get("DAILY_REFRESH", "1") != "0":  # re-scrape the shop (details, new products and every price) every night 02:00-05:00; the owner gets the result
        import refresh_catalog
        refresh_catalog.start_daily(core.tell_admins, agent.end_maintenance)
        if os.environ.get("WATCH_SITE", "1") != "0":  # 'event listener': new / changed / removed products are noticed without a full update
            refresh_catalog.start_watcher(core.tell_admins)
    def nightly():  # one backup a day, AFTER the nightly product update (02:00-05:00) has finished, so it holds the new product list and prices; at 05:00 it goes ahead whatever happened
        import time as _t
        import refresh_catalog  # imported HERE: the one at startup only runs when the nightly product update is on (DAILY_REFRESH), and a missing name made every round fail
        from datetime import datetime
        from zoneinfo import ZoneInfo
        while True:
            try:
                n = datetime.now(ZoneInfo("Asia/Nicosia"))
                today = n.strftime("%Y-%m-%d")
                if core.backup_due(n.hour, today, core.kget("refresh_day"), refresh_catalog.busy(), (core.P("backups") / today).exists()):
                    try:
                        info = core.backup_info(core.backup())
                        core.ok("backup")
                    except Exception as e:  # a failed backup is told to the owners at once, never silent
                        core.problem("backup", e)
                        core.kset("backup_last", json.dumps({"ok": False, "error": repr(e)[:200], "ts": _t.time()}))
                        core.tell_admins(core.backup_report({"ok": False, "error": repr(e)[:200]}))
                    else:
                        core.tell_admins(core.backup_report(info))
                core.chat_prune()
            except Exception as e:
                core.problem("backup", e)
            _t.sleep(600)
    def summaries():  # opening and closing message to the owner, once each per day (OPEN_TIME / CLOSE_TIME, North Cyprus clock)
        import time as _t
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        while True:
            try:
                n = datetime.now(ZoneInfo("Asia/Nicosia"))
                try:
                    __import__("docs_pdf").cleanup_reports()  # a finished week / month report is deleted, the next one starts
                except Exception as e:
                    core.log(f"report cleanup failed: {e!r}")
                for kind, hhmm in (("opening", cfg("OPEN_TIME", "08:00")), ("closing", cfg("CLOSE_TIME", "23:00"))):
                    at = n.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:5]), second=0, microsecond=0)
                    key = f"summary:{n:%Y-%m-%d}:{kind}"
                    if timedelta(0) <= n - at < timedelta(hours=2) and not core.kget(key):  # a late restart never sends the morning one at night
                        core.kset(key, "1")
                        if kind == "opening":
                            agent.send_report("opening")  # the morning reminder as plain text
                        else:
                            agent.send_report("day")  # the night report: a PDF like the invoices
                            for rep, due_ in (("week", n.weekday() == 6), ("month", (n + timedelta(days=1)).month != n.month)):  # Sunday night: the week; the last night of the month: the month
                                if due_ and not core.kget(f"summary:{n:%Y-%m-%d}:{rep}"):
                                    core.kset(f"summary:{n:%Y-%m-%d}:{rep}", "1")
                                    agent.send_report(rep)
            except Exception as e:
                core.problem("summary", e)
            _t.sleep(60)
    if os.environ.get("SUMMARIES", "1") != "0" and not is_mock():
        __import__("threading").Thread(target=summaries, daemon=True, name="summaries").start()
    if os.environ.get("BACKUPS", "1") != "0":
        __import__("threading").Thread(target=nightly, daemon=True, name="backup").start()


@app.get("/health")
def health():
    """Liveness for an uptime monitor: 200 when the database and the catalog load, 503 otherwise."""
    try:
        core.q("SELECT 1")
        n = sum(1 for x in core.graph()[0].values() if x["type"] == "product")
        last = core.kget("refresh_at")
        return {"ok": True, "products": n, "maintenance": core.kget("maintenance") == "1", "agent_stopped": core.kget("agent_stopped") == "1",
                "days_since_refresh": round((__import__("time").time() - float(last)) / 86400, 1) if last else None}
    except Exception as e:
        raise HTTPException(503, repr(e))


# ------------------------------------------------------------ webhook
@app.get("/webhook")
def verify(request: Request):
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == cfg("VERIFY_TOKEN") and cfg("VERIFY_TOKEN"):
        return Response(p.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(403)


def sig_ok(body: bytes, header: str):
    secret = cfg("APP_SECRET")
    want = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest() if secret else None
    return bool(want) and hmac.compare_digest(want, header or "")


SEEN = set()  # ponytail: in-memory dedupe of Meta retries; lost on restart


@app.post("/webhook")
async def receive(request: Request):
    body = await request.body()
    if not sig_ok(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(403, "bad signature")
    wa = core.TRANSPORT or WhatsApp()
    for entry in (await request.json()).get("entry", []):
        for ch in entry.get("changes", []):
            v = ch.get("value", {})
            names = {c["wa_id"]: c.get("profile", {}).get("name", "") for c in v.get("contacts", [])}
            tags = {c.get("wa_id"): (c.get("profile", {}).get("username") or c.get("username") or "") for c in v.get("contacts", [])}  # WhatsApp user names, when Meta sends them (best effort: the field may be absent)
            for msg in v.get("messages", []):
                if cfg("META_TEST_MODE") == "1" and msg.get("from", "").lstrip("+") not in {x.strip().lstrip("+") for x in cfg("META_TEST_RECIPIENTS").split(",") if x.strip()}:
                    continue
                if msg["id"] in SEEN:
                    continue
                SEEN.add(msg["id"])
                if hasattr(wa, "typing"):
                    asyncio.get_running_loop().run_in_executor(None, wa.typing, msg["id"])
                core.remember_msg(msg["id"], (msg.get("text") or {}).get("body") or (msg.get(msg.get("type")) or {}).get("caption") or f"[{msg.get('type')}]")  # so a later reply that quotes it can be understood
                base = {"sender": msg["from"], "name": names.get(msg["from"], ""), "username": tags.get(msg["from"], ""), "reply_to": msg.get("context", {}).get("id", "")}  # `from` is verified by the signature
                t = msg.get("type")
                try:
                    if t == "text":
                        await agent.inbound({**base, "kind": "text", "text": msg["text"]["body"]})
                    elif t in ("audio", "image"):
                        if t == "image" and norm(msg["from"]) in core.admins():
                            core.kset("admin_photo_pending", str(time.time()))
                        data, mime = await asyncio.to_thread(wa.download, msg[t]["id"])
                        if msg[t].get("caption"):
                            await agent.inbound({**base, "kind": "text", "text": msg[t]["caption"]})
                        await agent.inbound({**base, "kind": t, "data": data, "mime": msg[t].get("mime_type") or mime,
                                             "filename": "voice.ogg" if t == "audio" else "photo.jpg"})
                    elif t == "location":  # a shared WhatsApp location becomes the address: a map link plus whatever name the sender attached
                        loc = msg["location"]
                        await agent.inbound({**base, "kind": "text", "text": f"[shared location] https://www.google.com/maps?q={loc.get('latitude')},{loc.get('longitude')} {loc.get('name', '')} {loc.get('address', '')}".strip()})
                    elif t == "document":  # a PDF receipt etc.
                        data, mime = await asyncio.to_thread(wa.download, msg[t]["id"])
                        await agent.inbound({**base, "kind": "document", "data": data, "mime": msg[t].get("mime_type") or mime, "filename": msg[t].get("filename") or "file"})
                    # other types (stickers, locations...) ignored
                except Exception as e:
                    core.log(f"webhook error: {e!r}")
    return {"ok": True}


# Local WAHA test connection.
WAHA_SEEN = set()  # shortcut: dedupe resets on restart, persist ids before using this in production.


@app.post("/waha/webhook")
async def waha_webhook(request: Request):
    secret = cfg("WAHA_WEBHOOK_SECRET")
    if not cfg("WAHA_API_KEY") or not secret or not hmac.compare_digest(request.headers.get("X-Waha-Secret", ""), secret):
        raise HTTPException(403, "bad secret")
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(400, "invalid event")
    if data.get("event") != "message" or data.get("session") != "default":
        return {"ok": True}
    m = data.get("payload")
    if not isinstance(m, dict):
        raise HTTPException(400, "invalid message")
    if m.get("fromMe") is not False:
        return {"ok": True}
    chat = m.get("from", "")
    if isinstance(chat, str) and chat.endswith("@lid"):
        raw = m.get("_data")
        info = raw.get("Info") if isinstance(raw, dict) else None
        chat = info.get("SenderAlt", "") if isinstance(info, dict) else ""
    if not isinstance(chat, str) or "@" not in chat:
        return {"ok": True}
    sender, domain = chat.rsplit("@", 1)
    sender = sender.split(":", 1)[0]  # WAHA GOWS uses device-qualified JIDs for LID SenderAlt values.
    if not sender.isdigit() or domain not in ("c.us", "s.whatsapp.net"):
        return {"ok": True}  # groups and status updates are not customer conversations
    if cfg("META_TEST_MODE") == "1" and sender not in {x.strip().lstrip("+") for x in cfg("META_TEST_RECIPIENTS").split(",") if x.strip()}:
        return {"ok": True}
    mid = m.get("id")
    if not isinstance(mid, str) or not mid:
        raise HTTPException(400, "missing id")
    if mid in WAHA_SEEN:
        return {"ok": True}
    text = m.get("body") or ""
    if not isinstance(text, str):
        raise HTTPException(400, "invalid text")
    reply = m.get("replyTo") or {}
    reply_id = reply.get("id", "") if isinstance(reply, dict) else ""
    quoted = (reply.get("body") or reply.get("caption") or "") if isinstance(reply, dict) else ""
    if reply_id and quoted:
        core.remember_msg(reply_id, quoted)
    elif reply_id:
        try:
            old = await asyncio.to_thread(Waha().get_message, chat, reply_id)
            core.remember_msg(reply_id, old.get("body") or (old.get("media") or {}).get("caption") or "[attachment]")
        except Exception as e:  # a quote is still delivered even when WAHA has expired its local message store
            core.log(f"WAHA quoted message lookup failed: {e!r}")
    base = {"sender": sender, "name": m.get("_data", {}).get("Info", {}).get("PushName", ""), "reply_to": reply_id}
    core.remember_msg(mid, text or "[attachment]")
    if m.get("hasMedia"):
        media = m.get("media") or {}
        url = media.get("url") if isinstance(media, dict) else None
        if not url:
            raise HTTPException(503, "media not downloaded by WAHA")
        from waha import Waha
        body, mime = await asyncio.to_thread(Waha().download, url)
        mime = media.get("mimetype") or mime
        kind = "image" if mime.startswith("image/") else "audio" if mime.startswith("audio/") else "document"
        if text:
            await agent.inbound({**base, "kind": "text", "text": text})
        await agent.inbound({**base, "kind": kind, "data": body, "mime": mime, "filename": media.get("filename") or "attachment"})
    elif text:
        await agent.inbound({**base, "kind": "text", "text": text})
    if len(WAHA_SEEN) >= 10000:
        WAHA_SEEN.clear()
    WAHA_SEEN.add(mid)
    return {"ok": True}


# ------------------------------------------------------------ Telegram
TG_SEEN = set()  # update ids already handled (Telegram retries a delivery that was not answered quickly)


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    secret = cfg("TELEGRAM_SECRET")
    if not secret or not hmac.compare_digest(secret, request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")):
        raise HTTPException(403, "bad secret")
    u = await request.json()
    if u.get("update_id") in TG_SEEN:
        return {"ok": True}
    TG_SEEN.add(u.get("update_id"))
    m = u.get("message")
    if not m or m.get("chat", {}).get("type") != "private":  # only one-to-one chats; groups are ignored
        return {"ok": True}
    await asyncio.to_thread(core.tgq_add, u)  # written down BEFORE Telegram is told "received": it will not send this message again
    _tg_spawn(u)  # answer Telegram at once and work in the background: a slow reply made Telegram time out and retry
    return {"ok": True}


def _tg_spawn(u):
    t = asyncio.create_task(_tg_run(u))
    TG_TASKS.add(t)
    t.add_done_callback(TG_TASKS.discard)


async def _tg_run(u):
    try:
        await _tg_handle(u["message"])
    finally:
        await asyncio.to_thread(core.tgq_done, u.get("update_id"))


TG_TASKS = set()  # keeps the background tasks alive until they finish


async def _tg_handle(m):
    chat = m["chat"]["id"]
    who = f"tg:{chat}"
    frm = m.get("from", {})
    name = " ".join(x for x in (frm.get("first_name"), frm.get("last_name")) if x) or frm.get("username", "")
    reply = m.get("reply_to_message", {}).get("message_id")
    base = {"sender": who, "name": name, "username": frm.get("username", ""), "reply_to": f"tg:{chat}:{reply}" if reply else ""}
    text = m.get("text") or m.get("caption") or ""
    core.remember_msg(f"tg:{chat}:{m.get('message_id')}", text or ("[photo]" if m.get("photo") else "[voice note]" if m.get("voice") or m.get("audio") else "[file]" if m.get("document") else ""))
    tg = core.TG
    try:
        if text.strip().lower().startswith("/id"):  # lets the owner find his id for ADMIN_TELEGRAM
            core.send_text(who, f"Your Telegram id: {chat}")
            return {"ok": True}
        if tg:
            asyncio.ensure_future(asyncio.to_thread(tg.typing, chat))  # not awaited: the agent must not wait for Telegram's round trip
        if text.strip().lower().startswith("/start"):
            text = "hi"
        if m.get("photo"):
            if who in core.admins():  # a photo from an owner is still downloading: a "send it" typed right after must wait for it
                core.kset("admin_photo_pending", str(time.time()))
            if text:
                await agent.inbound({**base, "kind": "text", "text": text})
            await agent.inbound({**base, "kind": "image", "data": await asyncio.to_thread(tg.download, m["photo"][-1]["file_id"]), "mime": "image/jpeg", "filename": "photo.jpg"})
        elif m.get("voice") or m.get("audio"):
            v = m.get("voice") or m.get("audio")
            await agent.inbound({**base, "kind": "audio", "data": await asyncio.to_thread(tg.download, v["file_id"]), "mime": v.get("mime_type", "audio/ogg"), "filename": "voice.ogg", "duration": v.get("duration", 0)})
        elif m.get("document"):
            d = m["document"]
            if text:
                await agent.inbound({**base, "kind": "text", "text": text})
            await agent.inbound({**base, "kind": "document", "data": await asyncio.to_thread(tg.download, d["file_id"]), "mime": d.get("mime_type", ""), "filename": d.get("file_name", "file")})
        elif m.get("location"):
            loc = m["location"]
            await agent.inbound({**base, "kind": "text", "text": f"[shared location] https://www.google.com/maps?q={loc['latitude']},{loc['longitude']}"})
        elif m.get("contact"):  # the customer shared their phone number with the contact button
            await agent.inbound({**base, "kind": "text", "text": f"My phone number is +{core.norm(m['contact'].get('phone_number'))}"})
        elif text:
            await agent.inbound({**base, "kind": "text", "text": text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        core.log(f"telegram webhook error: {e!r}")
    return {"ok": True}


# ------------------------------------------------------------ mock UI (same inbound path as the webhook)
@app.get("/")
def index():
    return RedirectResponse("/dashboard?tab=chat" if is_mock() else "/dashboard")  # the test chat is a tab of the dashboard now


@app.get("/simulate")
def simulate():
    need_mock()
    return RedirectResponse("/dashboard?tab=sim")


@app.get("/embed/chat")
def embed_chat():
    need_mock()
    return FileResponse(Path(__file__).parent / "static/index.html", headers={"Cache-Control": "no-store"})


@app.get("/embed/sim")
def embed_sim():
    need_mock()
    return FileResponse(Path(__file__).parent / "static/simulate.html", headers={"Cache-Control": "no-store"})


@app.get("/sim/products")
def sim_products():
    need_mock()
    return [{"name": n.get("name_en") or n["name"], "id": n["id"]} for n in core.graph()[0].values() if n["type"] == "product"]


@app.get("/scenarios.json")
def scenarios():
    return FileResponse(Path(__file__).parent / "static/scenarios.json")


@app.get("/config")
def config():
    a = sorted(core.admins())
    return {"mock": is_mock(), "admin": a[0] if a else "", "debounce": cfg("DEBOUNCE_SECONDS", "3")}


def need_mock():
    if not is_mock():
        raise HTTPException(404)


@app.post("/message")
async def message(sender: str = Form(..., alias="from"), name: str = Form(""), text: str = Form(""),
                  duration: float = Form(0), reply_to: str = Form(""), file: UploadFile | None = File(None)):
    need_mock()
    base = {"sender": sender, "name": name, "reply_to": reply_to}
    if text:
        await agent.inbound({**base, "kind": "text", "text": text})
    if file is not None:
        mime = file.content_type or ""
        kind = "audio" if mime.startswith("audio") or mime == "video/webm" else ("image" if mime.startswith("image") else "document")
        await agent.inbound({**base, "kind": kind, "data": await file.read(), "mime": mime,
                             "filename": file.filename or "", "duration": duration})
    return {"queued": True}


@app.get("/feed")
def get_feed(since: int = 0):
    need_mock()
    return core.FEED[since:]


@app.post("/sim/admin/clear")
def clear_sim_admin():
    need_mock()
    if agent.DEB.active or agent.DEB.buf:
        raise HTTPException(409, "Wait for the current replies to finish, then clear the chat.")
    for key in ("aex", "afocus", "aflow", "lastalert"):
        core.kset(key, "")
    with core.CHAT_LOCK:
        for phone in core.admins():
            core._chat_file(phone).unlink(missing_ok=True)
            core.CHAT.pop(phone, None)
    return {"since": len(core.FEED)}


@app.get("/view/{name}")
def view(name: str):
    """The same file as /download, but shown in the browser (PDF viewer, image, audio) instead of saved."""
    need_mock()
    p = core.P("outbox") / Path(name).name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name.split("_", 1)[-1], content_disposition_type="inline")


@app.get("/download/{name}")
def download(name: str):
    need_mock()
    p = core.P("outbox") / Path(name).name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name.split("_", 1)[-1])
