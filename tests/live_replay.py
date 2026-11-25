"""Replays customer / owner chats through the CURRENT code with the LIVE model (spends OpenAI tokens: ask first).
Each chat runs in its own temp STATE_DIR with blank channel tokens: nothing reaches WhatsApp / Telegram and no real order is touched.
Usage: .venv/bin/python tests/live_replay.py flowers_and_bears arabic french   (several names run in parallel; no name = list them)
Chats live in tests/live_scenarios.json: ["C", text] customer, ["A", text] owner, ["SETSTATUS", id, status] simulates a receipt arriving.
Transcripts (bot replies, tool calls, what the interpreter read, final orders) go to <out_dir>/<name>.json and are printed."""
import asyncio, json, os, shutil, subprocess, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = json.load(open(os.path.join(ROOT, "tests/live_scenarios.json"), encoding="utf-8"))


def show(path):
    for e in json.load(open(path, encoding="utf-8")):
        tag = {"customer": "👤 C", "admin": "🛠 A", "customer_bot": "🤖→C", "admin_bot": "🤖→A", "note": "   ·", "log": "   LOG"}[e["side"]]
        print(tag, e["text"][:600].replace("\n", " / "))


def main(names, out_dir):
    if len(names) > 1:  # one process per chat, all at once
        procs = [subprocess.Popen([sys.executable, __file__, n, out_dir], stdout=subprocess.DEVNULL, stderr=open(os.path.join(out_dir, n + ".log"), "w")) for n in names]
        for n, p in zip(names, procs):
            p.wait()
            print(f"=================== {n} (exit {p.returncode})")
            if os.path.exists(os.path.join(out_dir, n + ".json")):
                show(os.path.join(out_dir, n + ".json"))
        return
    name = names[0]
    sys.path.insert(0, ROOT); os.chdir(ROOT)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    tmp = tempfile.mkdtemp(prefix="replay_")
    for f in ("prices.json", "removed.json", "overrides.json", "custom.json"):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), tmp)
    admin, cust = "905550009999", "90555" + str(abs(hash(name)) % 10**7).zfill(7)  # made-up numbers, never a real customer
    for k, v in dict(STATE_DIR=tmp, ADMIN_NUMBERS=admin, ADMIN_TELEGRAM="", GRAPH_PATH=os.path.join(ROOT, "data/graph.json"), DEBOUNCE_SECONDS="0.3", NOTIFY_ASYNC="0", RESEND_CHECK="0",
                     CHARACTER_AUTO="0", DL_PRELOAD="0", SANDBOX_PRELOAD="0", WA_CHECK="0", RATE_MESSAGES="1000", META_TEST_MODE="0", WAHA_API_KEY="", WAHA_WEBHOOK_SECRET="",
                     WHATSAPP_TOKEN="", PHONE_NUMBER_ID="", TELEGRAM_TOKEN="", TELEGRAM_SECRET="", PUBLIC_URL="").items():
        os.environ[k] = v
    import core, agent
    core.TG = None
    interpret = agent.interpret_message

    def interpret_logged(phone, text, st):
        d = interpret(phone, text, st)
        core.log("interpreted: " + json.dumps(d, ensure_ascii=False)[:1200])
        return d
    agent.interpret_message = interpret_logged
    seen, events = 0, []

    def drain():
        nonlocal seen
        for e in core.FEED[seen:]:
            body = e.get("body") or ""
            if e.get("kind") == "log" and ("tools:" in body or "interpreted:" in body or "failed" in body):
                events.append({"side": "log", "text": body[:1500]})
            if e.get("kind") in ("text", "document") and e.get("to") in (cust, admin):
                events.append({"side": "customer_bot" if e["to"] == cust else "admin_bot", "text": body.strip(), "file": e.get("file")})
        seen = len(core.FEED)

    async def feed(m):
        agent.DEB = agent.Debouncer(agent.process_batch)
        await agent.inbound(m)
        await asyncio.gather(*list(agent.DEB.t.values()), return_exceptions=True)

    for s in SCEN[name]:
        if s[0] == "SETSTATUS":
            r = core.order_row(int(s[1]))
            if not r or r["status"] not in ("awaiting_payment", "payment_review"):  # a receipt only arrives for an order waiting for its payment
                events.append({"side": "note", "text": f"[setup] SKIPPED: order #{s[1]} is {r['status'] if r else 'missing'}, not waiting for a payment"})
                continue
            core.order_update(int(s[1]), ("awaiting_payment", "payment_review"), status=s[2])
            events.append({"side": "note", "text": f"[setup] order #{s[1]} set to {s[2]} (simulates the receipt arriving)"})
            continue
        events.append({"side": "customer" if s[0] == "C" else "admin", "text": s[1]})
        t0 = time.time()
        try:
            asyncio.run(feed({"sender": cust if s[0] == "C" else admin, "name": "Customer" if s[0] == "C" else "Owner", "kind": "text", "text": s[1]}))
        except Exception as e:
            events.append({"side": "note", "text": f"EXCEPTION {e!r}"})
        drain()
        events.append({"side": "note", "text": f"({time.time() - t0:.0f}s)"})
    events.append({"side": "note", "text": "FINAL ORDERS: " + " | ".join(f"#{r['order_id']} {r['status']} {r['items']} notes={r['notes'][-80:]} paid={r['paid']}" for r in core.orders())})
    json.dump(events, open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args.pop() if args and os.path.isdir(args[-1]) else tempfile.mkdtemp(prefix="replays_")
    if not args or any(a not in SCEN for a in args):
        sys.exit("chats: " + ", ".join(SCEN))
    main(args, out)
    if len(args) == 1:
        show(os.path.join(out, args[0] + ".json"))
    print("transcripts in", out)
