"""Local WAHA transport; clear WAHA_API_KEY to use the Meta adapter again."""
import base64
import mimetypes
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
from core import cfg


class Waha:
    def start(self):
        """Start the existing local container and restore its saved default session."""
        base = cfg("WAHA_URL", "http://127.0.0.1:3000").rstrip("/")
        url = urlsplit(base)
        if url.hostname not in ("localhost", "127.0.0.1", "::1") or url.port != 3000:
            return
        subprocess.run(["docker", "start", "waha"], check=True, stdout=subprocess.DEVNULL, timeout=30)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                response = httpx.get(base + "/api/sessions/default", headers=self._headers(), timeout=2)
            except httpx.TransportError:
                time.sleep(1)
                continue
            response.raise_for_status()
            status = response.json()["status"]
            if status == "STOPPED":
                response = httpx.post(base + "/api/sessions/default/start", headers=self._headers(), timeout=10)
                response.raise_for_status()
            elif status != "STARTING":
                if status == "FAILED":
                    raise RuntimeError("WAHA default session failed; check the WAHA dashboard.")
                return status
            time.sleep(1)
        raise RuntimeError("WAHA did not become ready within 30 seconds; check docker logs waha.")

    def _headers(self):
        return {"X-Api-Key": cfg("WAHA_API_KEY")}

    def _post(self, endpoint, to, **payload):
        r = httpx.post(cfg("WAHA_URL", "http://127.0.0.1:3000").rstrip("/") + "/api/" + endpoint,
                       headers=self._headers(), json={"session": "default", "chatId": to + "@c.us", **payload}, timeout=60)
        r.raise_for_status()
        mid = r.json().get("id")
        return mid.get("_serialized") if isinstance(mid, dict) else mid

    def send_text(self, to, body):
        return self._post("sendText", to, text=body[:4096])

    def _media(self, endpoint, to, path, caption):
        p = Path(path)
        return self._post(endpoint, to, caption=caption, file={"filename": p.name,
                          "mimetype": mimetypes.guess_type(str(p))[0] or "application/octet-stream",
                          "data": base64.b64encode(p.read_bytes()).decode()})

    def send_doc(self, to, path, caption=""):
        return self._media("sendFile", to, path, caption)

    def send_image(self, to, path, caption=""):
        return self._media("sendImage", to, path, caption)

    def download(self, url):
        # Media must come from our WAHA file store; never send the API key to a webhook-supplied host.
        parsed = urlsplit(url)
        if not parsed.path.startswith("/api/files/") or ".." in parsed.path:
            raise ValueError("invalid WAHA media path")
        target = cfg("WAHA_URL", "http://127.0.0.1:3000").rstrip("/") + parsed.path
        r = httpx.get(target, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "")

    def get_message(self, chat_id, message_id):
        """Read a quoted message from WAHA when it predates our local reply cache."""
        chat = quote(chat_id, safe="")
        mid = quote(message_id, safe="")
        r = httpx.get(cfg("WAHA_URL", "http://127.0.0.1:3000").rstrip("/") +
                      f"/api/default/chats/{chat}/messages/{mid}",
                      headers=self._headers(), params={"downloadMedia": "true"}, timeout=20)
        r.raise_for_status()
        return r.json()
