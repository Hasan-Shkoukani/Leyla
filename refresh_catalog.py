"""Local catalog compatibility helpers. Website scraping and price fetching are disabled."""
from pathlib import Path
import subprocess
import sys
import core

ROOT = Path(__file__).parent
PROGRESS = {'step': '', 'since': 0, 'stopping': False, 'pages': 0}


def names():
    return {p['id']: p['name'] for p in core.graph()[0].values() if p.get('type') == 'product'}


def count():
    return len(names())


def busy():
    return False


def request_stop():
    return False


def refresh_prices(ids=None):
    products = [p for p in core.graph()[0].values() if p.get('type') == 'product' and (not ids or p['id'] in ids)]
    stored = core.live_prices(products)
    return sum(stored.get(p['id']) is not None for p in products), len(products)


def run(back=None, restart=None):
    try:
        core._load.cache_clear()
        return f'Local catalog reloaded: {count()} products. Website scraping is not included.'
    finally:
        if back:
            back()


def refresh_one(pid, force=False):
    if not core.product(pid):
        return 'Unknown local product.'
    return 'NOT UPDATED: website scraping is not included. Edit the local catalog or update the product in the dashboard.'


def rebuild():
    subprocess.run([sys.executable, 'build_graph.py', 'data/site.sample.json'], cwd=ROOT, check=True, capture_output=True)


def reset_one(pid):
    core.clear_override(pid)
    core._load.cache_clear()
    return 'Product overrides removed; the local catalog values are used.'


def restore_one(pid):
    if not core.restore_product(pid):
        return 'That item was not removed.'
    core._load.cache_clear()
    return 'Product restored from local data.'


def start_daily(*args, **kwargs):
    return None


def start_watcher(*args, **kwargs):
    return None
