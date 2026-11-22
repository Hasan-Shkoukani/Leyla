"""Reading prices from the shop's own product pages (schema.org JSON-LD Product.offers). Used ONLY by the update jobs:
the nightly full update, the admin's "update everything" / "update this item" buttons. Customers never wait on a page: the prices they see are the ones
stored by the last update (core.price_store, prices.json). A page that cannot be read gives None and the old stored price stays.
ponytail: HTML is parsed with a regex over JSON-LD blocks; if a site drops JSON-LD the price cannot be read (the old one stays) until the parser is extended."""
import json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
_lock = threading.Lock()
_host_slots, _host_last = {}, {}


def parse_page(html):
    """(price Decimal, in_stock bool) from the page's JSON-LD Product.offers; ValueError if there is none or it is not Turkish Lira."""
    for block in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for x in data if isinstance(data, list) else [data]:
            if not isinstance(x, dict) or x.get("@type") != "Product":
                continue
            offers = x.get("offers")
            offers = offers[0] if isinstance(offers, list) and offers else offers
            if not isinstance(offers, dict):
                continue
            raw = offers.get("price", offers.get("lowPrice"))
            if raw is None or str(offers.get("priceCurrency", "TRY")).upper() not in ("TRY", "TL"):
                continue
            try:
                price = Decimal(str(raw).replace(",", ""))
            except InvalidOperation:
                continue
            if price <= 0:
                continue
            return price, "OutOfStock" not in str(offers.get("availability", ""))
    raise ValueError("no usable JSON-LD price on the page")




def parse_flags(html):
    """What the product page shows next to its price, read from the block between the product title (<h1>) and the stock line, the same in every language of the site:
    old_price = the crossed-out price of a discounted item (None if it has none; the JSON-LD price is always the price to pay), same_day = the green "delivered today" chip is there.
    A page without that block gives {"old_price": None, "same_day": None} (unknown: nothing stored is changed)."""
    i = html.find("<h1")
    j = html.find('role="status"', i) if i >= 0 else -1
    if i < 0 or j < 0 or j - i > 6000:
        return {"old_price": None, "same_day": None}
    block = html[html.find("</h1>", i):j]
    old = None
    m = re.search(r'<span[^>]*line-through[^>]*>\s*[^\d<]*(\d[\d.,\s\u00a0\u202f]*)', block)  # "₺3.850" (tr), "₺3,850" (en), "3 850 ₺" (ru)
    if m:
        try:
            from build_graph import num
            old = Decimal(str(num(re.sub(r"[\s\u00a0\u202f]", "", m.group(1)))))  # "₺3.850" / "₺3.850,50" / "₺3,850"
        except (InvalidOperation, ValueError, TypeError):
            old = None
    return {"old_price": old, "same_day": bool(re.search(r'<span[^>]*inline-flex[^>]*rounded-full[^>]*>', block))}


def _polite(host):
    """At most 4 requests in flight per host, and at least 0.15 s between two starts."""
    import httpx  # noqa: F401  (imported by fetch_one)
    slot = _host_slots.setdefault(host, threading.Semaphore(4))
    slot.acquire()
    with _lock:
        wait = _host_last.get(host, 0) + 0.15 - time.time()
        _host_last[host] = max(time.time(), _host_last.get(host, 0) + 0.15)
    if wait > 0:
        time.sleep(wait)
    return slot




def fetch_one(url):
    """Network boundary: GET the product page and parse it -> {"price": Decimal, "in_stock": bool}. Up to 3 tries on timeouts, 429 and 5xx. Raises on failure."""
    import httpx
    slot = _polite(urlparse(url).netloc)
    last = "no answer"
    try:
        for attempt in range(3):
            try:
                r = httpx.get(url, headers={"User-Agent": UA, "Accept-Language": "tr,en;q=0.8"}, timeout=10, follow_redirects=True)
            except httpx.TransportError as e:
                last = repr(e)
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
                try:
                    time.sleep(min(float(r.headers.get("Retry-After", 0.6 * (attempt + 1))), 5))
                except ValueError:
                    time.sleep(1)
                continue
            r.raise_for_status()
            price, in_stock = parse_page(r.text)
            fl = parse_flags(r.text)
            return {"price": price, "in_stock": in_stock, "old_price": fl["old_price"] if fl["old_price"] and fl["old_price"] > price else None, "same_day": fl["same_day"]}
        raise RuntimeError(last)
    finally:
        slot.release()


def fetch_many(items, workers=4):
    """items = [(key, [urls])] -> {key: {"price": Decimal, "in_stock": bool, "ts": time} | None}. Each product's links are tried in order; pages are read in parallel but politely (see _polite)."""
    def one(it):
        for u in it[1]:
            try:
                got = fetch_one(u)
                return {"price": Decimal(str(got["price"])), "in_stock": bool(got["in_stock"]), "ts": time.time(), "old_price": got.get("old_price"), "same_day": got.get("same_day")}
            except Exception:
                continue
        return None
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return dict(zip((k for k, _ in items), ex.map(one, items)))
