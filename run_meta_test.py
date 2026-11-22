"""Run the real Meta Cloud API test number against a separate order/chat store."""
import os
from pathlib import Path

from dotenv import dotenv_values


root = Path(__file__).resolve().parent
env = {**dotenv_values(root / ".env"), **os.environ}
required = ("META_TEST_ACCESS_TOKEN", "META_TEST_PHONE_NUMBER_ID", "META_TEST_RECIPIENTS",
           "META_TEST_ADMIN_NUMBER", "META_TEST_VERIFY_TOKEN", "META_TEST_APP_SECRET")
missing = [key for key in required if not env.get(key)]
if missing:
    raise SystemExit("Add these Meta test-number settings to .env: " + ", ".join(missing))

digits = lambda value: "".join(c for c in str(value) if c.isdigit())
recipients = [digits(x) for x in env["META_TEST_RECIPIENTS"].split(",") if digits(x)]
admin = digits(env["META_TEST_ADMIN_NUMBER"])
if admin not in recipients:
    raise SystemExit("META_TEST_ADMIN_NUMBER must be included in META_TEST_RECIPIENTS")

env.update({
    "WHATSAPP_TOKEN": env["META_TEST_ACCESS_TOKEN"],
    "PHONE_NUMBER_ID": env["META_TEST_PHONE_NUMBER_ID"],
    "ADMIN_NUMBERS": admin,
    "VERIFY_TOKEN": env["META_TEST_VERIFY_TOKEN"],
    "APP_SECRET": env["META_TEST_APP_SECRET"],
    "META_TEST_MODE": "1",
    "META_TEST_RECIPIENTS": ",".join(recipients),
    "GRAPH_API_VERSION": env.get("GRAPH_API_VERSION") or "v27.0",
    "STATE_DIR": env.get("META_TEST_STATE_DIR") or str(root / "test-state" / "meta-cloud"),
    "WAHA_API_KEY": "",
    "TELEGRAM_TOKEN": "",
    "TELEGRAM_SECRET": "",
    "WA_CHECK": "0",
    "DAILY_REFRESH": "0",
    "SUMMARIES": "0",
    "RESEND_CHECK": "0",
    "LATE_CHECK": "0",
    "SANDBOX_PRELOAD": "0",
})
os.chdir(root)
print("Meta Cloud API test mode; test-only state:", env["STATE_DIR"])
print("Allowed recipient numbers:", ", ".join(recipients))
print("Set Meta's HTTPS webhook callback to <public URL>/webhook and use META_TEST_VERIFY_TOKEN.")
os.execvpe(str(root / ".venv" / "bin" / "uvicorn"),
           [str(root / ".venv" / "bin" / "uvicorn"), "app:app", "--host", "0.0.0.0", "--port", "8000"], env)
