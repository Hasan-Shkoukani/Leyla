"""Google Maps link -> verified delivery area. Small and isolated: resolves a (short) map link, reads the coordinates, asks the Google Geocoding API for the place names
(key in the env var GOOGLE_MAPS_API_KEY) and matches them against data/delivery.json. Anything unverified -> {"status": "manual"}: the agent then asks the customer for the city.
Nothing here guesses a city and no fee is ever set from an unverified location."""
import json, re
from urllib.parse import parse_qs, unquote, urlparse

import core
from core import cfg

LINK = re.compile(r"https?://(?:maps\.app\.goo\.gl|goo\.gl/maps|(?:www\.|maps\.)?google\.[a-z.]+/maps|www\.google\.[a-z.]+/maps)[^\s]*", re.I)
HOSTS = ("google.com", "goo.gl", "maps.app.goo.gl", "google.com.tr", "google.com.cy")  # only these are ever fetched (no other host: no server-side request forgery)
NUM = r"(-?\d{1,3}\.\d{3,})"


def find_link(text):
    m = LINK.search(text or "")
    return m.group(0).rstrip(").,]") if m else ""


def _allowed(url):
    h = (urlparse(url).hostname or "").lower()
    return any(h == d or h.endswith("." + d) for d in HOSTS) or bool(re.fullmatch(r"(www\.|maps\.)?google\.[a-z.]+", h))


def coords(url):
    """(lat, lng) written in the link itself, or None."""
    u = unquote(url)
    for rx in (rf"@{NUM},{NUM}", rf"!3d{NUM}!4d{NUM}", rf"[?&](?:q|ll|query|center|destination)={NUM},{NUM}"):
        m = re.search(rx, u)
        if m and abs(float(m.group(1))) <= 90 and abs(float(m.group(2))) <= 180:
            return float(m.group(1)), float(m.group(2))
    return None


def resolve(url, get=None):
    """Follow the redirects of a short link (max 5, only Google hosts, 5 s). Returns the final URL or '' when it cannot be followed."""
    import httpx
    get = get or httpx.get
    for _ in range(5):
        if not _allowed(url):
            return ""
        if coords(url):
            return url
        try:
            r = get(url, follow_redirects=False, timeout=5)
        except Exception:
            return ""
        nxt = r.headers.get("location")
        if not nxt:
            return url
        url = nxt if nxt.startswith("http") else f"{urlparse(url).scheme}://{urlparse(url).netloc}{nxt}"
    return ""


def names_at(lat, lng, get=None):
    """Place names (locality, district, neighbourhood...) from the Geocoding API; [] when the key is missing or the call fails / is over quota. Cached by rounded coordinates."""
    key = cfg("GOOGLE_MAPS_API_KEY")
    if not key:
        return []
    ck = f"geo:{lat:.3f},{lng:.3f}"
    if core.kget(ck):
        return json.loads(core.kget(ck))
    import httpx
    try:
        j = (get or httpx.get)("https://maps.googleapis.com/maps/api/geocode/json", params={"latlng": f"{lat},{lng}", "key": key, "language": "en"}, timeout=5).json()
    except Exception:
        return []
    if j.get("status") != "OK":  # ZERO_RESULTS, OVER_QUERY_LIMIT, REQUEST_DENIED...: no answer, never a guess
        return []
    out = []
    for res in j.get("results", [])[:3]:
        for c in res.get("address_components", []):
            if set(c.get("types", [])) & {"locality", "administrative_area_level_2", "administrative_area_level_1", "sublocality", "neighborhood", "postal_town"}:
                out.append(c["long_name"])
    out = list(dict.fromkeys(out))
    if out:
        core.kset(ck, json.dumps(out, ensure_ascii=False))
    return out


def locate(text, get=None, geocode_get=None):
    """{"status": "ok", "area": key, "label": ..., "place": "Girne", "url": link} | {"status": "unsupported", "place": ..} | {"status": "manual", "why": ..} | None (no map link in the text)."""
    link = find_link(text)
    if not link:
        return None
    final = resolve(link, get)
    c = coords(final) if final else None
    if not c:
        return {"status": "manual", "why": "the map link could not be resolved to a location", "url": link}
    names = names_at(*c, get=geocode_get)
    if not names:
        return {"status": "manual", "why": "the location could not be verified (no map service answer)", "url": link}
    area = core.detect_area(" ".join(names))
    if not area:
        return {"status": "unsupported", "place": names[0], "url": link}
    return {"status": "ok", "area": area, "label": core._json("delivery.json")["areas"][area]["label"]["en"], "place": names[0], "url": link}
