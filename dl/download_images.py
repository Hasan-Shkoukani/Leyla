"""Download catalog photos listed in ../data/raw/site.json. Polite: 1 req/sec, skips existing.
Usage: python -m dl.download_images [--delay 1.0]
Expects site.json items with `product_id` and `images` (list of urls or {url|src}).
"""
import argparse, html, json, shutil, time
from pathlib import Path
from dl import config


def _products(d):
    # ponytail: tolerate {"products": [...]} or a bare list; adjust if site.json differs
    return d.get("products", d) if isinstance(d, dict) else d


def _url(i):
    return html.unescape(i if isinstance(i, str) else (i.get("url") or i.get("src") or ""))


def main():
    import requests
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(config.SITE_JSON))
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    items = _products(json.loads(Path(a.site).read_text()))
    s = requests.Session()
    s.headers["User-Agent"] = "flowers-agent-research/0.1 (catalog matching)"
    for p in items:
        pid = str(p["product_id"])
        urls = [u for u in map(_url, p.get("images") or []) if u]
        mf = config.CATALOG_DIR / pid / "urls.json"  # the photo list as downloaded last time; a different list = the shop changed its photos
        if mf.exists() and json.loads(mf.read_text()) != urls:
            shutil.rmtree(mf.parent)
        for n, img in enumerate(p.get("images") or []):
            url = _url(img)
            if not url:
                continue
            ext = Path(url.split("?")[0]).suffix.lower()
            out = config.CATALOG_DIR / pid / f"{n}{ext if ext in config.IMG_EXTS else '.jpg'}"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                r = s.get(url, timeout=30); r.raise_for_status()
                out.write_bytes(r.content)
            except Exception as e:
                print("FAIL", url, e)
            time.sleep(a.delay)
        if urls and mf.parent.exists():
            mf.write_text(json.dumps(urls))
    print("done")


if __name__ == "__main__":
    main()
