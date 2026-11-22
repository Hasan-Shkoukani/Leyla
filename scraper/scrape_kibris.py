#!/usr/bin/env python3
"""THE catalog scraper: kibrisciceksepetim.com -> data/raw/site.json (what build_graph.py reads). Polite and re-runnable (cache2/, wiped by a full refresh).

RULES (every one is checked by tests/test_core.py, keep them true):
 1. NAMES are copied from the product's own page in each language (tr, en, ru): the JSON-LD Product.name, HTML-unescaped and stripped, nothing else. Never translated, edited, merged
    with another site's name or guessed. A page that cannot be read this time keeps the name the previous run copied from it; a language that never had one stays None, and the whole run
    fails when more than 3 % of the products miss a name.
 2. PRICE is the price to pay (JSON-LD offers.price). A discounted item also has `old_price` (the crossed-out price shown before it); a product WITHOUT a crossed-out price has old_price None.
 3. SAME DAY: `same_day` is True only when the page shows the "delivered today" chip next to the price; the products without it (a few) are False.
 4. BADGES (promotional, featured, heartfelt, bestSeller, new) come from the shop's own /badges/<slug> listing pages (all their ?page=N pages), never guessed from a product.
 4b. A product that is NOT on the shop's sitemap any more was hidden or disabled by the shop's admin: it is left out here too (the catalog drops it; the agent never offers it). A product that is on
    the sitemap but whose page failed to load right now keeps its previous record: a network hiccup never hides a live product.
 5. A product keeps the id it already has in data/graph.json (matched by its page slug), a new one is "k-<slug>"; ids are never renumbered (photos, stored prices and orders point at them).
 6. The BLOG (the index and every post) is stored word for word in all three languages: the only shop text the agent may quote. The other information pages (faq, about, contact, delivery, privacy, terms) are read in
    Turkish only, as before, for the contact details; their text is never given to the agent (the owner's own delivery rules and policies are in data/).
Only the shop's own pages are read: no other site, no translation service."""
import hashlib, html as H, json, os, random, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
LANGS = ("tr", "en", "ru")
BASE = "https://kibrisciceksepetim.com"
BADGES = ("promotional", "featured", "heartfelt", "bestSeller", "new")
BAD = re.compile(r"/api/|/cart|/login|/register|/profile|/admin|/orders/track|/order-received")  # robots.txt; ?page=N is allowed, other queries are not
CACHE = ROOT / "cache2"
OUT = ROOT.parent / "data/raw"


def clean(s):
    return re.sub(r"\s+", " ", H.unescape(str(s or ""))).strip()


def parse_product(html, lang_url):
    """One product page -> {name, description, price, availability, images, ...}. Pure function of the page (see the rules above)."""
    from bs4 import BeautifulSoup
    import prices
    s = BeautifulSoup(html, "html.parser")
    ld = {}
    for sc in s.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(sc.string)
        except Exception:
            continue
        for x in d if isinstance(d, list) else d.get("@graph", [d]):
            if isinstance(x, dict) and x.get("@type") == "Product":
                ld = x
    off = ld.get("offers") or {}
    off = off[0] if isinstance(off, list) and off else off
    imgs = ld.get("image") or []
    imgs = [imgs] if isinstance(imgs, str) else list(imgs)
    for m in re.findall(r'https?://[^"\'\s\\<>]+?\.(?:jpe?g|png|webp)', html):
        if "/catalog/" in m and m not in imgs:
            imgs.append(m)
    h1 = s.find("h1")
    name = clean(ld.get("name") or (h1.get_text(" ", strip=True) if h1 else ""))
    price = off.get("price")
    fl = prices.parse_flags(html)
    return {"name": name or None, "description": clean(ld.get("description")), "price": float(price) if price not in (None, "") else None,
            "availability": (off.get("availability") or "").split("/")[-1], "images": imgs, "sku": ld.get("sku"),
            "old_price": float(fl["old_price"]) if fl["old_price"] else None, "same_day": fl["same_day"], "url": lang_url}


def main():
    import requests
    from bs4 import BeautifulSoup
    CACHE.mkdir(exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    stats = {"net": 0, "blocked": 0}

    def get(url, lang="tr"):
        assert not BAD.search(url), url
        f = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".html")
        if f.exists():
            return f.read_text(encoding="utf-8")
        if os.environ.get("SCRAPE_OFFLINE"):  # finishing a stopped update: never wait on the shop
            return None
        sess.headers["Accept-Language"] = lang
        for i in range(5):
            time.sleep(random.uniform(2, 5))
            stats["net"] += 1
            try:
                r = sess.get(url, timeout=40)
            except requests.RequestException as e:
                print("err", e, file=sys.stderr)
                time.sleep(2 ** i * 3)
                continue
            if r.status_code == 200:
                f.write_text(r.text, encoding="utf-8")
                return r.text
            if r.status_code == 404:
                return None
            stats["blocked"] += 1
            time.sleep(int(r.headers.get("Retry-After", 0)) if r.headers.get("Retry-After", "").isdigit() else 2 ** i * 5)
            if stats["blocked"] >= 8:
                sys.exit("repeatedly blocked, stopping")
        print("giving up", url, file=sys.stderr)
        return None

    locs = [H.unescape(u) for u in re.findall(r"<loc>(.*?)</loc>", get(BASE + "/sitemap.xml") or "")]
    slugs = [u.rsplit("/", 1)[1] for u in locs if u.startswith(BASE + "/tr/products/")]
    if not slugs:
        sys.exit("the sitemap lists no products: nothing changed")

    ids = {}  # rule 5: the ids the catalog already uses
    try:
        for n in json.loads((OUT.parent / "graph.json").read_text(encoding="utf-8"))["nodes"]:
            if n.get("type") == "product":
                for u in [n.get("url")] + list(n.get("urls") or []):
                    m = re.search(r"/products/([^/?#]+)", str(u or ""))
                    if m:
                        ids[m.group(1)] = n["id"]
    except Exception:
        pass

    badges, warnings = {}, []  # rule 4
    for b in BADGES:
        seen, q, count, want = set(), [f"{BASE}/tr/badges/{b}"], 0, None
        while q:
            u = q.pop(0)
            if u in seen:
                continue
            seen.add(u)
            h = get(u)
            if not h:
                continue
            if want is None and (m := re.search(r'"@type":"ItemList"[^{}]*?"numberOfItems":(\d+)', h)):
                want = int(m.group(1))  # the shop says how many products the collection has: checked below
            for m in re.findall(r'href="/tr/products/([^"/?#]+)(?:\?[^"]*)?"', h):  # the cards of page 2+ carry "?page=N" on their links
                if b not in badges.setdefault(m, []):
                    badges[m].append(b)
                    count += 1
            q += [BASE + m for m in re.findall(rf'href="(/tr/badges/{b}\?page=\d+)"', h)]
        if want is not None and want != count:
            warnings.append(f"badge {b}: the shop says {want} products, {count} were found on its pages")
    prev = {}  # the previous run's records (by slug): what a page that fails to load right now falls back to
    try:
        for q in json.loads((OUT / "site.json").read_text(encoding="utf-8")).get("products", []):
            m = re.search(r"/products/([^/?#]+)", str(q.get("url") or ""))
            if m:
                prev[m.group(1)] = q
    except Exception:
        pass
    products, missing = [], []
    for slug in slugs:
        per = {lang: (lambda h, u: parse_product(h, u) if h else None)(get(f"{BASE}/{lang}/products/{slug}", lang), f"{BASE}/{lang}/products/{slug}") for lang in LANGS}
        tr = per["tr"]
        if not tr or not tr["name"]:  # the page did not load: the live product keeps its previous record (it is on the sitemap), or counts as missing if it never had one
            if slug in prev:
                products.append(prev[slug])
            missing.append(slug) if slug not in prev else None
            continue
        pn = prev.get(slug, {}).get("names") or {}
        for l in LANGS:
            if not per[l] or not per[l]["name"]:
                if pn.get(l):
                    per[l] = {**(per[l] or {}), "name": pn[l], "description": (prev[slug].get("descriptions") or {}).get(l), "url": f"{BASE}/{l}/products/{slug}"}
                else:
                    missing.append(slug)
                    break
        products.append({"product_id": ids.get(slug) or "k-" + slug, "name": tr["name"],
                         "names": {l: (per[l] or {}).get("name") for l in LANGS}, "descriptions": {l: (per[l] or {}).get("description") for l in LANGS},
                         "description": tr["description"], "url": tr["url"], "urls": [{"site": "kibris", "lang": l, "url": per[l]["url"]} for l in LANGS if per[l]],
                         "price": tr["price"], "old_price": tr["old_price"] if tr["old_price"] and tr["price"] and tr["old_price"] > tr["price"] else None,
                         "same_day": tr["same_day"], "badges": badges.get(slug, []), "currency": "TL", "images": tr["images"], "availability": tr["availability"],
                         "sku": tr["sku"], "source": ["kibris"]})
    if len(missing) > 0.03 * len(slugs):  # rule 1
        sys.exit(f"{len(missing)} of {len(slugs)} products have no readable name in some language ({missing[:5]}...): nothing changed")

    pages, blog = {}, {}  # rule 6
    for u in locs:
        m = re.match(rf"{re.escape(BASE)}/(tr|en|ru)/(blog(?:/[^/]+)?|faq|about|contact|privacy|terms|delivery(?:/[^/]+)?)$", u)
        if not m or (m.group(2) not in ("blog",) and not m.group(2).startswith("blog/") and m.group(1) != "tr"):
            continue
        if h := get(u, m.group(1)):
            s = BeautifulSoup(h, "html.parser")
            main = s.find("main") or s
            h1 = s.find("h1")
            rec = {"lang": m.group(1), "title": clean(h1.get_text(" ", strip=True) if h1 else ""), "text": clean(main.get_text(" ", strip=True))}
            (blog if m.group(2).startswith("blog") else pages)[u] = rec
    allh = "".join((get(u) or "") for u in [BASE + "/tr"] + [u for u in pages if "/tr/" in u])
    links = lambda pat: sorted({H.unescape(a["href"]) for a in BeautifulSoup(allh, "html.parser").find_all("a", href=re.compile(pat))})
    home = BeautifulSoup(get(BASE + "/tr") or "", "html.parser")
    biz = {"whatsapp_links": links(r"wa\.me|whatsapp"), "tel_links": links(r"^tel:"), "mailto": links(r"^mailto:"), "social": links(r"facebook\.com/|instagram\.com/|twitter\.com/\w|youtube\.com/\w|tiktok"),
           "map_links": sorted(set(re.findall(r'https?://(?:www\.)?google\.com/maps[^"\'<> \\]*|https://maps\.app\.goo\.gl/[^"\'<> \\]*', allh))),
           "emails_in_text": sorted(set(re.findall(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.sub(r"<[^>]+>", " ", allh)))),
           "footer_text": clean(home.find("footer").get_text(" ", strip=True)) if home.find("footer") else "", "pages": pages, "blog": blog, "source": BASE + "/tr"}
    out = {"scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"), "site": BASE, "sites": [BASE], "business": biz, "product_count": len(products), "warnings": warnings, "products": products}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "site.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, OUT / "site.json")
    print(f"done {len(products)} products, {sum(1 for p in products if p['badges'])} with a badge, {sum(1 for p in products if p['old_price'])} discounted, "
          f"{sum(1 for p in products if p['same_day'] is False)} not same-day, {len(blog)} blog pages, net {stats['net']}, names missing in {len(missing)}" + "".join(f"\nWARNING {w}" for w in warnings))


if __name__ == "__main__":
    main()
