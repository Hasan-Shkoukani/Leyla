"""FastAPI: WhatsApp Meta Cloud webhook + mock-mode test UI. Run: uvicorn app:app --port 8000"""
import asyncio
import json, time
import os, signal, sys
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
    return not (cfg("WHATSAPP_TOKEN") and cfg("PHONE_NUMBER_ID"))


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


def request_refresh_restart():
    """Ask Uvicorn to drain its requests and run the shutdown hook."""
    os.kill(os.getpid(), signal.SIGTERM)


@app.on_event("shutdown")
def restart_after_refresh():
    dashboard._sb_stop()  # Close the owned simulation before exec replaces its parent.
    if core.kget("refresh_restart") == "ready":
        # Uvicorn closes listening sockets and drains requests before lifespan shutdown.
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])


def resume_pending_messages():
    completed = core.kget("refresh_restart") == "ready"
    if core.kget("maintenance") == "1":
        core.kset("maintenance", "0")
        core.kset("refresh_restart", "")
        agent.end_maintenance()
        core.tell_admins("The nightly product and photo-index update is complete. The server restarted; queued customer messages are resuming." if completed else "The product update was interrupted by a restart. Queued customer messages are resuming; ask me to update the products again.")
    elif core.kget("agent_stopped") != "1":
        agent.flush_inbox()
    else:
        core.atomic_write(core.P("inbox.json"), json.dumps([{**x, "live": False} for x in core.inbox()]))


@app.on_event("startup")
async def _pool():  # each chat turn waits on OpenAI most of the time, so many can run at once; the default pool is only cpu+4 threads
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(int(cfg("WORKER_THREADS", "64"))))
    agent.LOOP = asyncio.get_running_loop()


@app.on_event("startup")
def _startup():
    if cfg("WHATSAPP_TOKEN") and cfg("PHONE_NUMBER_ID"):
        core.TRANSPORT = WhatsApp()
    else:
        core.TRANSPORT = None
    if os.environ.get("SANDBOX_PRELOAD", "1") != "0" and not is_mock():  # live server only: the simulation copy warms up now, so its dashboard tab opens at once
        dashboard.sandbox_preload()
    if os.environ.get("WA_CHECK", "1") != "0" and cfg("WHATSAPP_TOKEN"):  # an expired WhatsApp token is noticed within minutes
        import threading, time as _t

        def _wa_loop():
            while True:
                try:
                    core.whatsapp_check()
                except Exception as e:
                    core.log(f"whatsapp check failed: {e!r}")
                _t.sleep(6 * 3600)
        threading.Thread(target=_wa_loop, daemon=True, name="wa-check").start()
    if os.environ.get("RESEND_CHECK", "1") != "0":  # an order request that never reached an owner (a restart, a channel that was down) is sent again, once
        import threading, time as _t3
        def _resend_loop():
            while True:
                try:
                    done = agent.resend_unnotified()
                    if done:
                        core.log(f"order requests sent after a delay: {done}")
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
    resume_pending_messages()
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
                            agent.send_report("day")  # the night report: a PDF like the receipts
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
    return {"mock": is_mock(), "admin": a[0] if a and is_mock() else "", "debounce": cfg("DEBOUNCE_SECONDS", "0.25")}


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
