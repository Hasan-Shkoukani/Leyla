"""Run model-written Python on the catalog and nothing else.
Isolation (Linux): bubblewrap with NO network, NO host filesystem except a read-only /usr (for the interpreter) and a read-only
/sandbox folder holding data.json (the public catalog) + the code; empty environment; CPU/memory limits; hard timeout; output cap.
So the code cannot see .env, orders.csv, chats, or other customers, and cannot reach the internet.
ponytail: Linux + bwrap only; if bwrap is missing run_code() refuses and the agent falls back to its typed catalog tools."""
import json, resource, shutil, subprocess, sys, tempfile
from pathlib import Path

RUNNER = '''
import json, sys, io, contextlib, math, statistics, collections, itertools, re, datetime
d = json.load(open("/sandbox/data.json"))
g = {"products": d["products"], "business": d["business"], "rates": d.get("rates", {}), "json": json, "math": math, "statistics": statistics,
     "collections": collections, "itertools": itertools, "re": re, "datetime": datetime}
if d.get("edit"):
    g["orders"] = d["orders"]
buf = io.StringIO()
ok = True
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(open("/sandbox/code.py").read(), "query", "exec"), g)
except BaseException as e:
    ok = False
    buf.write(f"ERROR: {type(e).__name__}: {e}")
sys.stdout.write(buf.getvalue()[:6000])
if d.get("edit") and ok:
    sys.stdout.write("\\n\\x00ORDERS\\x00" + json.dumps(g["orders"], ensure_ascii=False, default=str))
'''


def available():
    return bool(shutil.which("bwrap"))


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def run_code(code, data, timeout=8):
    return _run(code, data, timeout)[0]


def run_edit(code, data, timeout=8):
    """Like run_code but `orders` (a copy of orders.csv) is available and returned: (printed text, new orders list or None). The host validates before writing."""
    return _run(code, {**data, "edit": True}, timeout)


def _run(code, data, timeout):
    """code: Python source using `products` (list of dicts), `business` (dict); print() the answer. Returns the printed text."""
    if not available():
        return "ERROR: code sandbox unavailable on this machine", None
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        (d / "code.py").write_text(code, encoding="utf-8")
        (d / "runner.py").write_text(RUNNER, encoding="utf-8")
        cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--clearenv",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", str(d), "/sandbox", "--chdir", "/sandbox",
               "/usr/bin/python3", "-I", "/sandbox/runner.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, preexec_fn=_limits, env={})
        except subprocess.TimeoutExpired:
            return "ERROR: the query took too long (limit 8 s); make it simpler", None
        if r.returncode and not r.stdout.strip():
            return "ERROR: the query was stopped (time or memory limit); make it simpler", None
        out, _, tail = r.stdout.partition("\n\x00ORDERS\x00")
        new = json.loads(tail) if tail else None
        return out.strip()[:6000] or "(no output)", new
