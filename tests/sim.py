"""Simulated user + admin against the running mock server. Usage: .venv/bin/python tests/sim.py [scenario numbers...]
Prints the transcript. Needs the server on :8000 (mock mode)."""
import json, random, re, sys, time
import httpx

B = "http://127.0.0.1:8000"
S = json.load(open("static/scenarios.json", encoding="utf-8"))
c = httpx.Client(base_url=B, timeout=60)
admin = re.sub(r"\D", "", c.get("/config").json()["admin"])
since = len(c.get("/feed").json()) and c.get("/feed").json()[-1]["id"] or 0


def drain(num, wait_first=45, quiet=5):
    """Collect feed events until quiet seconds pass after the first one."""
    global since
    out, t0, last = [], time.time(), None
    while True:
        for e in c.get("/feed", params={"since": since}).json():
            since = e["id"]
            if e.get("kind") != "log":
                out.append(e); last = time.time()
        now = time.time()
        if (last and now - last > quiet) or (not last and now - t0 > wait_first):
            return out
        time.sleep(0.7)


def show(evs, num):
    for e in evs:
        to = re.sub(r"\D", "", e.get("to") or "")
        who = "ADMIN <-" if to == admin else "  bot  ->"
        tag = f"[{e['tpl']}] " if e.get("kind") == "template" else ""
        print(f"   {who} {tag}{(e.get('body') or '').strip()}" + ("  📎" + e["file"] if e.get("file") else ""))


pick = {int(a) for a in sys.argv[1:]}
used_phones, used_names = set(), set()
first_names = ["Ayse", "Mehmet", "Elif", "Can", "Zeynep", "Hasan", "Maria", "John", "Sara", "Omer", "Leyla", "Murat", "Deniz", "Natasha", "Ahmed", "Selin"]
last_names = ["Demir", "Kaya", "Yilmaz", "Smith", "Brown", "Ivanov", "Petrova", "Arslan", "Aydin", "Miller"]
for sc in S:
    n = int(sc["name"].split(".")[0])
    if pick and n not in pick:
        continue
    num = "9053" + str(random.randint(10**7, 10**8 - 1))
    while num in used_phones:
        num = "9053" + str(random.randint(10**7, 10**8 - 1))
    used_phones.add(num)
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    while name in used_names:
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
    used_names.add(name)
    print(f"\n=== {sc['name']}  (customer {num})")
    oid = None
    for st in sc["steps"]:
        if "c" in st:
            text = st["c"].replace("{long}", "I love flowers and want to tell you everything about my event. " * 40)
            print(f" 👤 {text[:160]}")
            c.post("/message", data={"from": num, "name": name, "text": text})
            if st.get("fast"):
                time.sleep(0.6); continue
            evs = drain(num)
        else:
            text = st["a"].replace("{oid}", oid or "1")
            print(f" 🛠  admin: {text}")
            c.post("/message", data={"from": admin, "name": "Admin", "text": text})
            evs = drain(admin)
        for e in evs:
            m = re.search(r"Order #(\d+)", e.get("body") or "")
            if m: oid = m.group(1)
        show(evs, num)
