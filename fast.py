"""Fast path: the messages that need no thinking are answered by code in well under a second (no model call). Everything else returns None and goes to the model.
English and Turkish only (the shop's two languages); any other language never matches and goes to the model. Facts come from the same data the model uses."""
import random, re

import core
from build_graph import fold

TR_CH = re.compile(r"[çğıöşüÇĞİÖŞÜ]")  # no re.I here: it would make i match ı/İ
TR_RE = re.compile(r"\b(merhaba|selam\w*|teşekkür\w*|tesekkur\w*|günaydın|nerede\w*|adres\w*|konum\w*|harita|telefon\w*|iletişim|çalışma|saat\w*|açık|kapalı|fiyat\w*|fotoğraf\w*|sağ|var|mı|mi|mu|mü|nedir|kaç\w*|göster\w*|resim|lütfen|iyi|slm|mrb|tşk|eyvallah)\b", re.I)
PUNCT = r"[ !.,?🌸💐🙏❤️😊]*"
THANKS = re.compile(r"^((ok(ay)?|great|perfect|cool|tamam|süper|harika|tamamdır)[ ,.!]+)?(thanks?( you)?( so much| a lot)?|thank u|thx|ty|teşekkür(ler| ederim)?|tesekkur(ler)?( ederim)?|çok teşekkürler|sağ ol|sag ol|eyvallah|tşk)" + PUNCT + "$", re.I)
GREET = re.compile(r"^(hi+|hello+|hey+|hiya|good (morning|afternoon|evening)|merhaba|selam(lar)?|günaydın|gunaydin|iyi (akşamlar|günler|aksamlar|gunler)|mrb|slm)" + PUNCT + "$", re.I)
HOURS = re.compile(r"\b(opening|closing|working|business) hours?\b|\bwhat time\b.*\b(open|close)|\bare you open\b|\bwhen (do|are) you (open|close)|çalışma saat|açılış|kapanış|saat kaçta (?:açı|kapan)|kaçta (açıl|kapan)|açık mı|kaça kadar|kacta (acil|kapan)|calisma saat", re.I)
MINE = re.compile(r"\b(same|this|that|these|my|mine|his|her|their|our|new|another|other|different|deliver\w*|send it|send her|send him|here|there|aynı|bu|şu|benim|onun|teslim\w*|buraya|oraya|yeni)\b|\[shared location\]", re.I)  # "same location", "my address", "deliver to this location": the CUSTOMER's place, not a question about the shop
WHERE = re.compile(r"\bwhere (are|is)\b.*\b(you|shop|store)\b|\b(your )?(address|location|directions)\b|\bgoogle maps?\b|\b(send|share).*\bmap\b|\bwhere.*located\b|nerede|neredesiniz|adres|konum|harita|yol tarif", re.I)
PHONE = re.compile(r"\b(your|shop'?s?) (phone|number|contact|email|e-mail)\b|\b(the )?(shop|store|company|flower ?shop)('?s)? (phone|number)\b|\bcontact (info|details)\b|\bhow (can|do) i (call|contact|reach)\b|telefon numara|iletişim|sizi nasıl ara|numaranız", re.I)
OTHER = re.compile(r"\b(my|mine|me|want|need|tomorrow|today|tonight|use|yarın|yarin|bugün|bugun|istiyorum|benim)\b|order|sipari|bouquet|buket|flower|çiçek|cicek|rose|gül|gul\b|price|fiyat|deliver|teslim|photo|fotoğraf|pay|ödeme|odeme|cancel|iptal|chocolate|çikolata|teddy|ayı|gift|hediye|birthday|doğum|how much|kaç (tl|lira|para)|cheap|ucuz|expensive|pahal", re.I)
SHOW = re.compile(r"\b(photo|picture|pic|image|show|see|fotoğraf\w*|resim\w*|göster\w*)\b", re.I)
ASK = re.compile(r"\b(price|cost|how much|do you have|have you got|fiyat\w*|kaç|var mı|nedir|ne kadar)\b", re.I)

GREETS = [("Hi, lovely to meet you! I'm Leyla 🌸 Fresh flowers and same-day delivery across North Cyprus. What's the occasion, or who are the flowers for? 💐",
           "Merhaba, tanıştığımıza çok sevindim! Ben Leyla 🌸 Taze çiçekler ve Kuzey Kıbrıs'ın her yerine aynı gün teslimat. Hangi vesileyle, çiçekler kimin için? 💐"),
          ("Hello and welcome! 🌸 Leyla here: fresh flowers and same-day delivery all over North Cyprus. Who are we making smile today? 💐",
           "Merhaba, hoş geldiniz! 🌸 Ben Leyla: taze çiçekler ve tüm Kuzey Kıbrıs'a aynı gün teslimat. Bugün kimi mutlu ediyoruz? 💐")]
AGAIN = [("Hi again 🌸 How can I help with your flowers?", "Tekrar merhaba 🌸 Çiçeklerinizle ilgili nasıl yardımcı olabilirim?")]
WELCOME = [("You're very welcome! 🌸 I'm here whenever you need flowers.", "Rica ederim! 🌸 Çiçek lazım olduğunda buradayım."),
           ("My pleasure 💐 Just message me anytime!", "Ne demek, benim için zevkti 💐 İstediğiniz zaman yazın!")]


CYR = re.compile(r"[\u0400-\u04FF]")


def _tr(text):
    return 1 if TR_CH.search(text) or TR_RE.search(text) else 0


def _hours(b, tr):
    h = b.get("hours") or ""
    m = re.search(r"\d\d:\d\d\s*[-–]\s*\d\d:\d\d", h)
    if not m:
        return None
    rng = re.sub(r"\s*[-–]\s*", " - ", m.group(0))
    every = bool(re.search(r"every day|monday to sunday|mon-sun", h, re.I))
    if tr:
        return f"Her gün {rng} saatleri arasında açığız 🌸" if every else f"Çalışma saatlerimiz: {rng} 🌸"
    return f"We're open every day, {rng} 🌸" if every else f"Our opening hours: {h} 🌸"


def _where(b, tr):
    if not b.get("address"):
        return None
    line = f"{b['address']} 🌸" if not tr else f"Adresimiz: {b['address']} 🌸"
    return line + (f"\n\n{b['maps_url']}" if b.get("maps_url") else "")


def _phone(b, tr):
    if not b.get("phone"):
        return None
    return (f"You can call or WhatsApp us on {b['phone']}" + (f", or email {b['email']}" if b.get("email") else "") + " 🌸") if not tr else \
           (f"Bizi {b['phone']} numarasından arayabilir veya WhatsApp'tan yazabilirsiniz" + (f", e-posta: {b['email']}" if b.get("email") else "") + " 🌸")


def _product(phone, text):
    """'show me a photo of <exact product name>' / 'price of <exact name>': the catalog item whose full name is in the message."""
    t = re.sub(r"[^\w ]", " ", fold(text))
    t = " " + re.sub(r"\s+", " ", t) + " "
    best = None
    for n in core.graph()[0].values():
        if n["type"] != "product" or n.get("available") is False:
            continue
        for nm in {n["name"], n.get("name_en"), n.get("name_tr"), *(n.get("aliases") or [])}:
            f = re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", fold(nm or ""))).strip()
            specific = len(f.split()) >= 3 or bool(re.search(r"\d", f))  # "papatya buketi" is a kind of item, not one product: it goes to the agent, which shows the options
            if len(f) >= 8 and specific and f" {f} " in t and (best is None or len(f) > best[0]):
                best = (len(f), n, nm)
    return best[1:] if best else None


LANG_OF = None  # set by agent.py: (phone, text) -> 0 en / 1 tr / 2 ru, the language of the CHAT (a line that shows no language keeps the chat's)


FORM_FIELDS = re.compile(r"(?im)^\s*(?:(?:your|sender|recipient)(?:'s)? (?:name|phone)|city(?: / town)?|delivery address|address|card message(?: \(optional\))?|gönderen (?:adı|telefonu)|alıcı (?:adı|telefonu)|şehir(?: / ilçe)?|adres|kart notu(?: \(isteğe bağlı\))?)\s*:")


PRICE_WORDS = re.compile(r"\b(price|cost|how much|fiyat\w*|ne kadar|kaç|kac)\b", re.I)
DELIVER_Q = re.compile(r"\b(deliver\w*|delivery|teslimat\w*|g[oö]nder\w*)\b", re.I)


SUGGEST = re.compile(r"\b(recommend\w*|suggest\w*|gifts?|birthday|anniversary|mum|mom|mother|wife|girlfriend|romantic|oner\w*|hediye|dogum gunu|yildonumu|anne\w*|tavsiye\w*)\b", re.I)
SUGGEST_NOT = re.compile(r"\b(order|deliver\w*|pay\w*|address|note|card|photo|status|cancel|change|refund)\b|sipari|teslim|ödeme|odeme|adres|iptal|\d{6,}", re.I)
AROUND = re.compile(r"(?:around|about|approx\w*|roughly|~|civar\w*|yaklaşık|yaklasik)\s*(\d[\d.,]*)\s*(?:tl|try|₺|lira)?", re.I)


def _suggestions(text):
    """Up to 5 products for an open 'what do you recommend / a gift for X' message: near the budget if one is named (under N, or around N: 60%-100% of it), else the shop's heartfelt / popular collection. [] = let the model handle it."""
    t = text.strip()
    amt = (BUDGET_MAX.search(t) or AROUND.search(t))
    if amt:
        top = _num(amt.group(1))
        hits = core.find_flowers(query="", max_price=top, min_price=top * 0.6, limit=5, sort="price_desc")
    else:
        coll = collection_for(t) or ("heartfelt" if re.search(r"gift|hediye|romantic|birthday|anniversary|dogum|yildonumu", fold(t)) else "bestSeller")
        pool = sorted(core.badge_products(coll), key=lambda n: n["name"])
        es = core.live_prices(pool[:12], show=True) if pool else {}
        hits = [p for p in pool[:12] if es.get(p["id"]) and es[p["id"]].get("price") is not None and es[p["id"]].get("in_stock") is not False][:5]
    return [x for x in (core.product(h["id"]) for h in hits) if x and x.get("available") is not False][:5]


def _counted_match(text):
    """'5 red roses': the ONE catalog hit whose name carries that same number as a whole number ('5-Piece', not '15'); None when there is no number or not exactly one such hit."""
    m = re.search(r"(?<![\d.,])(\d{1,3})(?![\d.,])", text)
    plan = plan_search(text)
    if not m or not plan or not plan["query"] or plan.get("max_price") or plan.get("min_price") or plan.get("sort"):
        return None
    n = m.group(1)
    hits = [core.product(h["id"]) for h in core.find_flowers(**plan)]
    hits = [h for h in hits if h and h.get("available") is not False]
    want = set(plan["query"].lower().split()) & {c for h in hits for c in h.get("colors") or []}  # "red": the hit must have that colour
    ok = [h for h in hits if want <= set(h.get("colors") or []) and len(re.findall(r"\d+", h.get("name") or "")) == 1  # a name with one number ("4 ... and 5 ...") is not "5"
          and any(re.search(rf"(?<![\d.,]){n}(?![\d.,])", x or "") for x in (h.get("name"), h.get("name_en")))]
    return ok[0] if len(ok) == 1 else None


def answer(phone, text):
    """-> ("text", str) | ("photo", product, caption, short line) | None (give it to the model)."""
    text = text.strip()
    if FORM_FIELDS.search(text):
        return None
    lg = LANG_OF(phone, text) if LANG_OF else (2 if CYR.search(text) else _tr(text))
    if text and len(text) <= 100 and "\n" not in text and TODAY_DELIVER.search(text) and not text.startswith("["):  # "can you deliver today / right now?": a direct answer from the shop's clock, no model
        mine = [r for r in core.orders() if r["customer_number"] == phone and r["status"] not in ("rejected", "cancelled")]
        key = core.detect_area(text) or ((core.detect_area(mine[-1]["address"]) or mine[-1]["delivery_area"]) if mine else None)
        return ("text", core.today_answer(lg, key or None, mine[-1] if mine else None))
    if not text or len(text) > 100 or "\n" in text or text.startswith("[") or CYR.search(text) or lg == 2:  # Russian is answered by the model, in Russian (the canned replies are English / Turkish)
        return None
    tr = 1 if re.match(r"\s*(merhaba|selam|g[uü]nayd[ıi]n|iyi (ak[sş]amlar|g[uü]nler))", text, re.I) else lg  # a Turkish greeting is answered in Turkish even without Turkish letters
    busy = any(r["customer_number"] == phone and r["status"] in core.OPEN + core.LOCKED for r in core.orders())
    if THANKS.match(text) and not busy:
        return ("text", random.choice(WELCOME)[tr])
    if GREET.match(text):
        if busy:  # an order is open: a plain hello is still answered at once by code, with a nudge to the order
            return ("text", ("Hi again 🌸 I'm here, how can I help with your order?", "Tekrar merhaba 🌸 Buradayım, siparişinizle ilgili nasıl yardımcı olabilirim?")[tr])
        return ("text", (random.choice(AGAIN) if core.history(phone) else random.choice(GREETS))[tr])
    if not OTHER.search(text):  # shop facts: one or several in the same message; anything else in it goes to the model
        b = core.business_info()
        parts = [f(b, tr) for rx, f in ((HOURS, _hours), (WHERE, _where), (PHONE, _phone)) if rx.search(text) and not (rx is WHERE and MINE.search(text))]
        if parts and all(parts):
            return ("text", "\n\n".join(parts))
    if not busy and (SHOW.search(text) or ASK.search(text)) and not any(rx.search(text) for _, rx in CURRENCY_WORDS):  # "...in euros?" needs the conversion: the model answers that, not this lira-only shortcut
        found = _product(phone, text)
        if found:
            p, nm = found
            tr = min(LANG_OF(phone, re.sub(re.escape(nm), " ", text, flags=re.I)), 1) if LANG_OF else _tr(re.sub(re.escape(nm), " ", text, flags=re.I))  # the product's own name must not decide the language; a text that shows none keeps the chat's
            e = core.live_prices([p], show=True).get(p["id"])
            if e and e.get("in_stock") is not False:
                name = core.display(p, "tr" if tr else "en")
                cap = f"{name} - {core.fmt_tl(e['price'])}\n{p.get('url') or ''}".strip()
                if SHOW.search(text):
                    return ("photo", p, cap, "Buyurun 🌸 Beğendiniz mi, sipariş verelim mi?" if tr else "Here it is 🌸 Shall I put it together for you?")
                return ("text", cap + ("\n\nSipariş verelim mi? 🌸" if tr else "\n\nShall I put it together for you? 🌸"))
    if not busy and PRICE_WORDS.search(fold(text)) and not any(rx.search(text) for _, rx in CURRENCY_WORDS):  # "how much is 5 red roses": ONE product whose name has that exact number is answered by code; anything less clear goes to the model
        p = _counted_match(text)
        e = core.live_prices([p], show=True).get(p["id"]) if p else None
        if e and e.get("price") is not None and e.get("in_stock") is not False:
            return ("text", f"{core.display(p, 'tr' if tr else 'en')} - {core.fmt_tl(e['price'])}\n{p.get('url') or ''}".strip() + ("\n\nSipariş verelim mi? 🌸" if tr else "\n\nShall I put it together for you? 🌸"))
    if not busy and DELIVER_Q.search(fold(text)) and not NOT_SIMPLE.search(text):  # "do you deliver to Kyrenia?": the area's rules, straight from delivery.json
        key = core.detect_area(text) or core.detect_area(" ".join(re.sub(r"y[ae]$", "", w) for w in fold(text).split()))  # Turkish "to": magusaya, girneye
        a = core._json("delivery.json")["areas"].get(key) if key else None
        if a:
            label = a["label"].get("tr" if tr else "en") or a["label"]["en"]
            jet = core.fmt_tl(a["jet_fee"]) if a["jet_fee"] is not None else None
            if tr:
                return ("text", f"Evet, {label} bölgesine teslimat yapıyoruz 🌸 Normal teslimat ücretsiz ve {a['departure']}'de çıkıyor" + (f"; belirli saatte (jet) teslimat +{jet}" if jet else "; jet ücretini ekibimiz teyit eder") + f". Minimum sipariş {core.fmt_tl(a['minimum'])} (sadece ürünler).")
            return ("text", f"Yes, we deliver to {label} 🌸 Normal delivery is free and leaves at {a['departure']}" + (f"; jet delivery to a specific time is +{jet}" if jet else "; the team confirms the jet fee") + f". Minimum order is {core.fmt_tl(a['minimum'])} in products.")
    if not busy and len(text) <= 160 and SUGGEST.search(fold(text)) and not SUGGEST_NOT.search(text) and not any(rx.search(text) for _, rx in CURRENCY_WORDS):  # "a gift for my mum around 1500 TL", "what do you recommend?": the options come from code, no model
        picks = _suggestions(text)
        if picks:
            return ("options", picks)
    if not busy and not SHOW.search(text) and search_hint(phone, text) and not any(rx.search(text) for _, rx in CURRENCY_WORDS):
        plan = plan_search(text)
        hits = core.find_flowers(**plan) if plan else []
        options = [core.product(p["id"]) for p in hits if core.product(p["id"])]
        if options:
            return ("options", options)
    return None


SEARCH = re.compile(r"\b(rose|roses|orchid|orchids|daisy|daisies|tulip|lily|lilies|sunflower|bouquet|bouquets|flowers?|plants?|chocolates?|teddy|bear|cake|gifts?|gül|gul|orkide|papatya|lale|buket|çiçek|cicek|çikolata|cikolata|ayıcık|ayicik|pasta|hediye|bitki)\b", re.I)
NOT_SIMPLE = re.compile(r"\d|under|below|over|above|between|cheap|expensive|budget|altında|altinda|üstünde|arasında|ucuz|pahal|bütçe|butce|order|sipari|take|alıyorum|alayım|istiyorum|i want|i'd like", re.I)


COLLECTIONS = (  # the shop's own collections, asked for in plain words (en / tr / ru); the first that matches wins
    ("promotional", re.compile(r"\b(discounts?|discounted|promotions?|promotional|promo|offers|deals?|on sale|sales?|cheaper|indirim\w*|kampanya\w*|f[ıi]rsat\w*|скидк\w*|акци[яий]\w*|распродаж\w*)\b", re.I)),
    ("heartfelt", re.compile(r"\b(heartfelt|lovely gift|thoughtful gift|sweet gift|gift from the heart|romantic|kalpten\w*|duygusal\w*|i[çc]ten|romantik|от всего сердца|романтичн\w*|трогательн\w*|милый подарок)\b", re.I)),
    ("bestSeller", re.compile(r"\b(best[- ]?sellers?|most popular|popular|top sellers?|most ordered|[çc]ok satan\w*|pop[üu]ler\w*|хит\w*|популярн\w*|бестселлер\w*)\b", re.I)),
    ("new", re.compile(r"\b(new arrivals?|what'?s new|newest|latest|just added|new products?|yeni (?:gelen|[üu]r[üu]n)\w*|yenilik\w*|новинк\w*|новые|новое)\b", re.I)),
    ("featured", re.compile(r"\b(featured|your favou?rites?|recommend\w*|[öo]ne [çc][ıi]kan\w*|[öo]ner\w*|рекоменд\w*|посоветуйте|посоветуете)\b", re.I)),
)


def collection_for(text):
    """'do you have any discounts?', 'I need a lovely gift', 'what is popular?': the shop collection this message asks for (promotional, heartfelt, bestSeller, new, featured), else None. A short message only."""
    t = text.strip()
    if len(t) > 160 or "\n" in t or t.startswith(("[", "(replying")):
        return None
    return next((c for c, rx in COLLECTIONS if rx.search(t)), None)


def search_hint(phone, text):
    """A plain 'do you have orchids?' style question: the catalog hits, so the model can answer in ONE call (it may still search again). None = no hint."""
    text = text.strip()
    if len(text) > 100 or "\n" in text or not SEARCH.search(text) or NOT_SIMPLE.search(text):
        return None
    if any(r["customer_number"] == phone and r["status"] in core.OPEN + core.LOCKED for r in core.orders()):
        return None
    return text


BUDGET_MAX = re.compile(r"(?:under|below|less than|max(?:imum)?|up to|within|at most|budget(?:\s+(?:of|is))?|altında|altinda|en fazla)\s*(?:of\s*)?(\d[\d.,]*)\s*(?:tl|try|₺|lira)?", re.I)
BUDGET_MIN = re.compile(r"(?:over|above|more than|at least|minimum|üstünde|ustunde|en az)\s*(\d[\d.,]*)\s*(?:tl|try|₺|lira)?", re.I)
CHEAP = re.compile(r"\b(cheapest|least expensive|lowest price|en ucuz)\b", re.I)
PRICEY = re.compile(r"\b(most expensive|priciest|highest price|en pahalı|en pahali)\b", re.I)
FILLER = re.compile(r"\b(i|we|want|will|would|like|take|get|buy|order|show|see|look|photos?|pictures?|pics?|images?|me|us|something|anything|do|you|have|any|what|is|are|the|a|an|of|for|my|your|please|can|could|how|much|does|cost|thing|things|one|var|mı|mi|mu|mu\?|hangisi|bana|göster|goster|tl|try|need|sell|too|with|some|there|got|give|find|looking|look|in|and|to|dollars?|usd|euros?|eur|pounds?|gbp)\b", re.I)
CURRENCY_WORDS = (("USD", re.compile(r"\b(dollars?|usd|dolar)\b|\$", re.I)), ("EUR", re.compile(r"\b(euros?|eur|avro)\b|€", re.I)), ("GBP", re.compile(r"\b(pounds?|gbp|sterlin\w*)\b|£", re.I)))
ORDER_INTENT = re.compile(r"\b(take|order|buy|want|add|i'd like|i would like|istiyorum|alayım|alayim|alıyorum|aliyorum|sipariş|siparis)\b|\b(yes|evet)\b", re.I)


def _num(x):
    return float(core.parse_amount(x.strip(".,")))  # "1.500" and "1,500" are fifteen hundred; "2,5" is two and a half


DETAILS_CUT = re.compile(r"\s(?:for\s+[A-ZÇĞİÖŞÜ][\w'’]+|\+?\d[\d ]{7,}|(?:tomorrow|today|tonight|yarın|bugün|deliver\w*|my name|name is|address)\b).*$", re.S)


def plan_search(text):
    """A message about products: the arguments for flowers_lookup, read from the words (budget, cheapest / priciest, the product words), so the search runs in code BEFORE the model and the model
    can answer, or place the order, in ONE call. None = not a product message (the model searches itself if it needs to)."""
    t = text.strip()
    if len(t) > 160 or "\n" in t or t.startswith(("[", "(replying")):  # a quoted reply is about the quoted message: the model reads the chat
        return None
    mx, mn = BUDGET_MAX.search(t), BUDGET_MIN.search(t)
    sort = "price_asc" if CHEAP.search(t) else "price_desc" if PRICEY.search(t) else ""
    if not (SEARCH.search(t) or mx or mn or sort):
        return None
    q = BUDGET_MIN.sub(" ", BUDGET_MAX.sub(" ", t))
    q = CHEAP.sub(" ", PRICEY.sub(" ", q))
    q = DETAILS_CUT.sub("", q)  # the recipient, phone, address and date that follow the product words are not part of the search
    q = " ".join(FILLER.sub(" ", re.sub(r"[?!.,;:]", " ", q)).split())
    q = re.sub(r"^1\s+(?=[a-zA-ZçğıöşüÇĞİÖŞÜ])", "", q)  # "1 pink rose bouquet": the leading 1 is "one", not a product number
    out = {"query": q, "limit": 5}
    cur = next((c for c, rx in CURRENCY_WORDS if rx.search(t)), "")
    if cur:  # "...in euros?": the search itself returns the amounts in that currency, so the model does not have to remember to convert
        out["currency"] = cur
    if mx:
        out["max_price"] = _num(mx.group(1))
    if mn:
        out["min_price"] = _num(mn.group(1))
    if sort:
        out["sort"] = sort
    return out


KINDS = (("flowers", re.compile(r"\b(flowers?|bouquets?|roses?|daisy|daisies|lil(?:y|ies)|çiçek\w*|buket\w*|gül\w*)\b", re.I)),
         ("chocolate", re.compile(r"\b(chocolates?|çikolata\w*)\b", re.I)),
         ("teddy bears", re.compile(r"\b(teddy( bears?)?|plush|ayıcık\w*|oyuncak\w*)\b", re.I)))


def multi_kinds(text, plan):
    """["flowers", "chocolate"] when ONE message asks for two or more kinds of product for one budget (the code then builds the combinations); [] otherwise."""
    if not plan or not plan.get("max_price") or plan.get("currency") not in (None, "", "TRY", "TL") or len(text) > 160:
        return []
    ks = [k for k, rx in KINDS if rx.search(text)]
    return ks if len(ks) >= 2 else []


YES = re.compile(r"\b(yes|yeah|yep|yup|ok(?:ay)?|sure|perfect|great|good|fine|correct|right|confirm(?:ed)?|go ahead|do it|looks good|sounds (?:good|perfect|great|right)|that'?s (?:it|all|right|correct|perfect|everything)|evet|tamam(?:d[iı]r)?|olur|onay(?:l[ıi]yorum|la)?|do[gğ]ru|harika|m[uü]kemmel|g[uü]zel|да|хорошо|отлично|подтверждаю|верно)\b", re.I)
CHANGE = re.compile(r"\b(but|wait|actually|change|add|also|remove|instead|rather|except|only one|cancel|what|why|how|when|where|which|not|don'?t|değiştir\w*|ekle\w*|çıkar\w*|ama|hay[ıi]r|вместо|добавь|но|нет)\b|\?", re.I)


def is_confirmation(text):
    """A short plain yes ("yes", "perfect", "no this sounds perfect :)", "evet tamam"): affirmative words, nothing that asks, changes or refuses. 'no, this/that ...' in front of an affirmative is a yes."""
    t = re.sub(r"^\s*no[,.! ]+(?=(this|that|it|everything|that's|thats)\b)", "", (text or "").strip(), flags=re.I)
    return 0 < len(t) <= 60 and bool(YES.search(t)) and not CHANGE.search(t)


INJECTION = re.compile(r"ignore (?:all |any |the |your )?(?:previous |prior |above |earlier )?(?:instructions|rules|prompt)|(?:system|developer|hidden) (?:prompt|message|instructions)|(?:reveal|show|print|repeat|tell me|leak) (?:me )?(?:your|the) (?:prompt|instructions|rules|tools|configuration|system)|you are now\b|from now on you|act as (?:a |an |the )?(?!customer)|pretend (?:to be|you are|that)|developer mode|jailbreak|\bDAN\b|forget (?:everything|all|your)|new instructions|override (?:your|the|all)|i am (?:the|your) (?:owner|admin|developer|boss)|as the (?:owner|admin)|talimatlar[ıi]n[ıi]? (?:unut|yok say)|önceki talimat|системн\w+ (?:промпт|инструкц)|игнорируй", re.I)
TASK = re.compile(r"\b(?:write|draft|compose|make up|generate|create)\b[^.?!]{0,30}\b(?:poem|essay|story|song|code|script|program|email|e-mail|letter|speech|article|report|cv|resume|rap)\b|\b(?:translate|solve|homework|python|javascript|html|sql|algorithm|equation|derivative|integral|capital of|who (?:is|was) the president|stock|bitcoin|crypto|weather|football score)\b|\b(?:şiir yaz|kod yaz|ödev|çevir)\b|\b(?:напиши (?:стих|код|эссе)|реши задачу)\b", re.I)
REMIND = re.compile(r"\b(?:remind|reminder|did (?:you|they) (?:tell|inform|send|ask)|have (?:you|they) (?:told|confirmed|seen)|any (?:news|update)|is it confirmed|still waiting|confirm (?:it|my order|the order)|check (?:my|the) order|ask (?:the )?(?:team|admin|owner)|tell (?:the )?(?:team|admin|owner)|hatırlat\w*|onayland[ıi] m[ıi]|ekibe (?:söyle|sor)|напомни\w*|подтвердил\w*)\b", re.I)


SINGULAR_REF = re.compile(r"\b(?:the|that|this|your|suggested)\s+(plant|bear|teddy|chocolate|bouquet|cake|basket|box|orchid|rose)\b(?!s)", re.I)


TODAY_DELIVER = re.compile(r"(?=.*\?|^\s*(?:can|could|is it possible|do you|are you|will you|mümkün|olur mu|можно|возможно))(?:(?=.*\b(?:today|tonight|right now|same[- ]day|asap|now)\b)(?=.*\b(?:deliver\w*|send|get (?:it|them))\b)|(?=.*\bbug[uü]n\b)(?=.*(?:teslim|g[oö]nder|ula[sş]))|(?=.*\bbu ak[sş]am\b)(?=.*teslim)|(?=.*сегодня)(?=.*(?:достав|привез|довез)))", re.I | re.S)


GREET_LEAD = re.compile(r"^\s*(hi+|hello+|hey+|hiya|good (?:morning|afternoon|evening)|merhaba|selam(?:lar)?|günaydın|gunaydin|iyi (?:akşamlar|günler|aksamlar|gunler))\b[\s,.!…:;-]*", re.I)
SHORT_GREETS = ("Hi, lovely to meet you! I'm Leyla 🌸", "Merhaba, tanıştığımıza çok sevindim! Ben Leyla 🌸")


def split_head(phone, text):
    """A long message ("Hi... I would like to order roses. What are your hours?"): the greeting and the simple shop questions are answered AT ONCE by code (no model);
    what is left goes to the agent. Returns (replies, the part answered, the rest)."""
    t = text.strip()
    if FORM_FIELDS.search(t) or len(t) > 600 or t.startswith("[") or CYR.search(t) or (LANG_OF and LANG_OF(phone, t) == 2):  # Russian chats are answered by the model, in Russian
        return [], "", t
    busy = any(r["customer_number"] == phone and r["status"] in core.OPEN + core.LOCKED for r in core.orders())
    replies, answered = [], []
    m = GREET_LEAD.match(t)
    if m and not busy and len(t) > m.end() and not core.history(phone):  # a first greeting only; later ones are left to the agent
        replies.append(SHORT_GREETS[1 if re.match(r"\s*(merhaba|selam|g[uü]nayd[ıi]n|iyi )", m.group(1), re.I) else min(LANG_OF(phone, t), 1) if LANG_OF else _tr(t)])  # a Turkish greeting is answered in Turkish even without Turkish letters ("iyi aksamlar")
        answered.append(t[:m.end()].strip())
        t = t[m.end():]
    b, kept = core.business_info(), []
    for ch in re.split(r"(?<=[.!?…])\s+|\n+", t):
        c = ch.strip()
        if c and len(c) <= 100 and not OTHER.search(c):
            parts = [f(b, min(LANG_OF(phone, c), 1) if LANG_OF else _tr(c)) for rx, f in ((HOURS, _hours), (WHERE, _where), (PHONE, _phone)) if rx.search(c) and not (rx is WHERE and MINE.search(c))]
            if parts and all(parts):
                replies.append("\n\n".join(parts))
                answered.append(c)
                continue
        kept.append(ch)
    return replies, " ".join(answered), " ".join(kept).strip()


def requested_delivery_time(text):
    """Recognise a standalone request to change delivery time; other wording stays with the agent."""
    # shortcut: this fast path handles explicit English time changes; other languages use the normal tools.
    m = re.fullmatch(r"\s*(?:actually[, ]*)?(?:(?:can|could|may) (?:i|we|you) )?(?:(?:i|we) (?:would like|want|wanna) (?:it )?)?(?:please )?(?:(?:(?:change|switch|move)(?: (?:it|delivery|the delivery|the time))? to|(?:deliver|delivery)(?: it)? at|make it|at) )?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?: please)?[?.! ]*", text, re.I)
    if not m or not (m[2] or m[3] or re.search(r"\b(?:at|deliver(?:y)?(?: it)?)\s+\d{1,2}\b", text, re.I)):
        return None
    hour, minute = int(m[1]), int(m[2] or 0)
    if minute > 59 or (m[3] and not 1 <= hour <= 12) or (not m[3] and hour > 23):
        return None
    if m[3]:
        hour = hour % 12 + (12 if m[3].lower() == "pm" else 0)
    elif hour <= 12 and re.search(r"\b(?:at|deliver(?:y)?(?: it)?)\s+\d{1,2}\b", text, re.I):
        hour %= 12
        if hour <= core.now().hour:
            hour += 12
    return f"{hour:02d}:{minute:02d}"


MONTH_NAMES = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
               "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
               "ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6, "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
               "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}


def far_date(text, today=None, days=None):
    """The date the customer names with a day and a month ("25 December", "December 25th", "25.12") when it lies further ahead than the shop takes orders (MAX_DAYS_AHEAD, 2 by default); None otherwise."""
    from datetime import date, timedelta
    today = today or core.now().date()
    mx = int(core.cfg("MAX_DAYS_AHEAD", "2")) if days is None else days
    t = core.fold(text or "")
    cands = []
    for d, mon in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([^\W\d_]{3,9})\b", t):
        cands.append((int(d), MONTH_NAMES.get(mon)))
    for mon, d in re.findall(r"\b([^\W\d_]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b", t):
        cands.append((int(d), MONTH_NAMES.get(mon)))
    for d, m in re.findall(r"\b(\d{1,2})[./](\d{1,2})\b(?![./]\d)", t):
        if 1 <= int(m) <= 12:
            cands.append((int(d), int(m)))
    for d, m in cands:
        if not m or not 1 <= d <= 31:
            continue
        for y in (today.year, today.year + 1):
            try:
                dt = date(y, m, d)
            except ValueError:
                continue
            if dt >= today:
                if dt > today + timedelta(days=mx):
                    return dt
                break
    return None
