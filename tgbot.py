"""Telegram Bot API adapter with the same methods as the WhatsApp one (send_text / send_image / send_doc), plus webhook setup and file download.
A Telegram customer is the id "tg:<chat id>"; the bot can only message people who started it (/start), and there are no templates and no 24-hour window."""
import html as H
import re
import threading
import time
from pathlib import Path

import httpx


def to_html(text):
    """WhatsApp-style text (*bold*, `code`) -> Telegram HTML. Links and +numbers are made tappable by Telegram itself."""
    t = H.escape(text, quote=False)
    t = re.sub(r"\*([^*\n]+)\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"<i>\1</i>", t)  # _italic_ (snake_case words are left alone)
    return re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)


def chunks(text, n=4000):
    """Telegram refuses messages over 4096 characters: split on blank lines, then hard."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        while len(para) > n:
            out.append(para[:n]); para = para[n:]
        if len(cur) + len(para) + 2 > n and cur:
            out.append(cur); cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    return out + ([cur] if cur else [])


class Telegram:
    def __init__(self, token):
        self.on_sent = None
        self._tl = threading.local()
        self.api, self.files = f"https://api.telegram.org/bot{token}", f"https://api.telegram.org/file/bot{token}"

    def _call(self, method, data=None, files=None):
        for k in range(3):  # a dropped connection ("connection reset by peer") is retried: files are re-read, texts are small
            try:
                if files:
                    [f[1].seek(0) for f in files.values() if hasattr(f[1], "seek")]
                r = httpx.post(f"{self.api}/{method}", data=data, files=files, timeout=60)
                break
            except httpx.TransportError:
                if k == 2:
                    raise
                time.sleep(1 + k)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"telegram {method}: {j.get('description')}")
        return j["result"]

    @staticmethod
    def chat(to):
        return str(to)[3:]

    def refs(self):
        """Ids of the parts of the last message this thread sent (a long text is split in several Telegram messages)."""
        return list(getattr(self._tl, "refs", []))

    def send_text(self, to, body):
        ref = None
        self._tl.refs = []
        for part in chunks(body):
            m = self._call("sendMessage", {"chat_id": self.chat(to), "text": to_html(part), "parse_mode": "HTML", "link_preview_options": '{"is_disabled": false}'})
            mid = f"tg:{self.chat(to)}:{m['message_id']}"
            self._tl.refs.append(mid)
            ref = ref or mid
            if self.on_sent:
                self.on_sent(mid, part)  # every part of a long answer can be replied to, not only the first
        return ref

    def send_image(self, to, path, caption=""):
        with open(path, "rb") as f:
            m = self._call("sendPhoto", {"chat_id": self.chat(to), "caption": to_html(caption)[:1024], "parse_mode": "HTML"}, {"photo": (Path(path).name, f, "image/jpeg")})
        return f"tg:{self.chat(to)}:{m['message_id']}"

    def send_doc(self, to, path, caption=""):
        name = re.sub(r"^\d{6}(\d{6})?_", "", Path(path).name)  # outbox files carry a time prefix
        with open(path, "rb") as f:
            m = self._call("sendDocument", {"chat_id": self.chat(to), "caption": to_html(caption)[:1024], "parse_mode": "HTML"}, {"document": (name, f)})
        return f"tg:{self.chat(to)}:{m['message_id']}"

    def typing(self, chat_id):
        try:
            self._call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except Exception:
            pass

    def download(self, file_id):
        info = self._call("getFile", {"file_id": file_id})
        r = httpx.get(f"{self.files}/{info['file_path']}", timeout=60)
        r.raise_for_status()
        return r.content

    def set_webhook(self, url, secret):
        return self._call("setWebhook", {"url": url, "secret_token": secret, "allowed_updates": '["message"]', "drop_pending_updates": "false"})
