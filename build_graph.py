"""Build data/graph.json from the scraped site JSON.
usage: python build_graph.py [site.json]   (or env SITE_JSON; falls back to data/site.sample.json)
Raw shape expected: {"products":[{id?,name,category,tags?,price,currency,sale_price,colors,url,images,description}],
                     "business":{name,phone,whatsapp,email,address,maps_url,hours,delivery,about,website}}"""
import html, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).parent
TR = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iisSgGuUoOcC")


def fold(s):  # lowercase + strip Turkish diacritics
    return str(s or "").translate(TR).lower()


# first match wins; name is checked before site tags (which are messy)
CATS = [
    ("teddy bears", ("ayi", "teddy", "bear", "pelus")),
    ("chocolate", ("cikolata", "chocolate", "choco")),
    ("apology boxes", ("ozur", "apolog", "sorry")),
    ("fruit baskets", ("meyve", "fruit")),
    ("wreaths", ("celenk", "wreath")),
    ("orchids", ("orkide", "orchid")),
    ("lilies", ("lilyum", "lily", "lilies", "zambak")),
    ("daisies", ("papatya", "daisy", "daisies")),
    ("roses", ("gul", "rose")),
    ("plants", ("bitki", "plant", "sukulent", "succulent", "kaktus", "cactus", "yucca", "drasena", "dracena", "dracaena", "sansevieria", "pasa kilici", "melek kanadi", "ficus", "monstera", "bonsai", "aloe", "palmiye")),
]
COLORS = {
    "red": ("kirmizi", "red"), "burgundy": ("bordo", "burgundy"), "pink": ("pembe", "pink"),
    "white": ("beyaz", "white"), "yellow": ("sari", "yellow"), "purple": ("mor", "purple", "lila", "lilac"),
    "orange": ("turuncu", "orange"), "blue": ("mavi", "blue"), "peach": ("seftali", "peach"),
    "mixed": ("karisik", "mixed", "renkli", "rainbow"),
}


def categorize(*texts):
    for t in texts:  # name first, then category, then tags
        t = fold(t)
        for cat, kws in CATS:
            if any(re.search(r"\b" + k, t) for k in kws):
                return cat
    return "other"


def colors_of(text):
    t = fold(text)
    return [c for c, kws in COLORS.items() if any(re.search(r"\b" + k, t) for k in kws)]


def num(x):  # ponytail: handles "1.250,00 TL" / "1,250.00" / 1250; not every locale
    if x in (None, ""):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^\d.,]", "", str(x))
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if len(s.split(",")[-1]) <= 2 else s.replace(",", "")
    elif s.count(".") > 1 or ("." in s and len(s.split(".")[-1]) == 3):
        s = s.replace(".", "")
    return float(s) if s else None




def norm_business(b):
    """Scraper output (wa links, pages dict) -> flat fields the lookup tool reads. Idempotent on already-flat input."""
    if "pages" not in b:
        return b
    pg = {u: v.get("text", "") for u, v in b["pages"].items() if isinstance(v, dict)}
    pick = lambda key: next((t for u, t in pg.items() if key in u), "")
    wa = next((m.group(1) for l in b.get("whatsapp_links") or [] if (m := re.search(r"wa\.me/\+?(\d+)", l))), None)
    maps = (b.get("map_links") or [None])[0]  # prefer a found maps link, else None
    website = os.environ.get("SHOP_BASE_URL", "https://example.com").rstrip("/")
    out = {"name": b.get("name") or os.environ.get("SHOP_NAME", "Leyla Demo Shop"), "company": b.get("company") or os.environ.get("SHOP_NAME", "Leyla Demo Shop"),
            "phone": f"+{wa}" if wa else None, "whatsapp": f"https://wa.me/{wa}" if wa else None,
            "email": (b.get("emails_in_text") or [None])[0], "address": b.get("address", ""),
            "maps_url": maps,  # ponytail: site has no Google Maps link; owner must supply one, never invent it
            "hours": b.get("hours", ""), "social": b.get("social"),
            "delivery": pick("delivery")[:1500], "about": pick("about")[:1500], "website": website}
    info = {}  # the shop's BLOG word for word, per language: {"blog": {"tr": {title, text}, ...}, "blog/<post>": ...}. No other page of the shop is given to the agent.
    for u, v in (b.get("blog") or {}).items():
        m = re.match(r".*/(tr|en|ru)/(.+)$", u)
        if m and isinstance(v, dict):
            info.setdefault(m.group(2), {})[m.group(1)] = {"title": v.get("title", ""), "text": (v.get("text") or "")[:6000]}
    out["info"] = info
    o = ROOT / "data/business_override.json"  # owner-supplied facts win over scraped ones; "_drop" removes scraped keys
    if o.is_file():
        ov = json.loads(o.read_text(encoding="utf-8"))
        for k in ov.pop("_drop", []):
            out.pop(k, None)
        out.update(ov)
    return out


def removed():
    """Items the owner removed for good: {id: {name, url, ts}}. Lives in the state folder, so no re-scrape or rebuild can bring them back."""
    root = Path(__file__).parent
    base = Path(os.environ["STATE_DIR"]) if os.environ.get("STATE_DIR") else (root / "state" if (root / "state").is_dir() else root)  # the same folder core.P() writes to
    p = base / "removed.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build(raw):
    nodes, edges, seen = [], [], set()
    gone = removed()
    gone_urls = {r["url"] for r in gone.values() if r.get("url")}

    def node(id_, **kw):
        if id_ not in seen:
            seen.add(id_)
            nodes.append({"id": id_, **kw})

    for i, p in enumerate(raw.get("products", [])):
        pid = str(p.get("product_id") or p.get("id") or f"p{i}")  # DL model must use the same ids
        if pid in gone or p.get("url") in gone_urls:  # removed by the owner: never put back, whatever the shop site still lists
            continue
        tags = p.get("tags") or []
        cat = categorize(p.get("name"), p.get("category"), " ".join(tags) if isinstance(tags, list) else tags)
        cols = []
        for c in (p.get("colors") or []) :
            cols += colors_of(c) or [fold(c)]
        cols = list(dict.fromkeys(cols or colors_of(p.get("name"))))
        links = []  # Local catalog entries never link to an external shop.
        nm = p.get("names") or {}  # the shop's own name in each language, copied by the scraper: never translated, no other source
        de = p.get("descriptions") or {}
        node(pid, type="product", name=html.unescape(nm.get("tr") or p.get("name", "")).strip(), category=cat, colors=cols,
             aliases=p.get("aliases") or [], name_en=html.unescape(nm.get("en") or "").strip() or None, name_tr=html.unescape(nm.get("tr") or "").strip() or None,
             name_ru=html.unescape(nm.get("ru") or "").strip() or None, urls=links, url=links[0] if links else None,
             image=f"/api/dash/photo/{pid}", description=html.unescape(p.get("description") or "")[:300],
             description_en=html.unescape(de.get("en") or "")[:300] or None, description_ru=html.unescape(de.get("ru") or "")[:300] or None,
             badges=list(p.get("badges") or []), same_day=p.get("same_day"), before=p.get("old_price"),
             **({"pinned_price": float(p["fixed_price"])} if p.get("fixed_price") is not None else {}))
        node("cat:" + cat, type="category", name=cat)
        edges.append({"from": pid, "to": "cat:" + cat, "rel": "in_category"})
        for c in cols:
            node("color:" + c, type="color", name=c)
            edges.append({"from": pid, "to": "color:" + c, "rel": "has_color"})
    biz = norm_business(raw.get("business") or raw.get("business_info") or {})
    node("business", type="business", **biz)
    return {"nodes": nodes, "edges": edges}


def main(src=None, out=None):
    src = Path(src or (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("SITE_JSON") or "")
    if not src.is_file():
        src = ROOT / "data/site.sample.json"
    g = build(json.loads(Path(src).read_text(encoding="utf-8")))
    out = Path(out or ROOT / "data/graph.json")
    out.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{src.name} -> {out}: {sum(n['type']=='product' for n in g['nodes'])} products")


if __name__ == "__main__":
    main()
