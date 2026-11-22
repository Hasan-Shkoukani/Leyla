"""Quotation (sent at the owner's first confirmation, with the bank details) and invoice (sent after the owner confirms the payment) as PDFs, always in English AND Turkish (lang "bi").
doc_data() is the content, build() draws it; both are pure code, no model. Font: DejaVu Sans (data/fonts), which has the Turkish letters."""
from pathlib import Path

from fpdf import FPDF

import core

FONTS = core.ROOT / "data/fonts"
L = {  # label -> (en, tr)
    "quotation": ("QUOTATION", "TEKLİF"), "invoice": ("INVOICE", "FATURA"), "no": ("No.", "No"), "date": ("Date", "Tarih"),
    "customer": ("Customer", "Müşteri"), "delivery_to": ("Delivery to", "Teslimat"), "when": ("Delivery date", "Teslimat tarihi"),
    "item": ("Item", "Ürün"), "qty": ("Qty", "Adet"), "unit": ("Unit price", "Birim fiyat"), "total": ("Total", "Tutar"),
    "goods": ("Items total", "Ürün toplamı"), "delivery": ("Delivery", "Teslimat"), "free": ("free", "ücretsiz"),
    "fee": ("Service fee", "Hizmet bedeli"), "discount": ("Discount", "İndirim"), "topay": ("TOTAL TO PAY", "ÖDENECEK TOPLAM"),
    "bank": ("Bank transfer details", "Banka havale bilgileri"), "holder": ("Account holder", "Hesap sahibi"), "bankname": ("Bank", "Banka"),
    "iban": ("IBAN", "IBAN"), "paid": ("PAID", "ÖDENDİ"), "paidamt": ("Amount paid", "Ödenen tutar"), "thanks": ("Thank you for your order.", "Siparişiniz için teşekkürler."),
    "order": ("Order", "Sipariş"), "notes": ("Note", "Not"),
    "salesorder": ("SALES ORDER", "SATIŞ SİPARİŞİ"), "sender": ("Sender", "Gönderen"), "recipient": ("Recipient", "Alıcı"), "address": ("Address", "Adres"),
    "dtype": ("Delivery type", "Teslimat türü"), "status": ("Status", "Durum"), "giftnote": ("Card note", "Kart notu"), "waiting": ("Waiting for your first confirmation", "İlk onayınız bekleniyor"),
    "normal": ("normal", "normal"), "jet": ("jet", "jet"), "pickup": ("pick-up at the shop", "mağazadan teslim"),
}


def t(k, lang):
    """lang = en | tr | bi (both: "English / Türkçe", the form the PDFs always use)."""
    en, tr = L[k]
    return (en if en == tr else f"{en} / {tr}") if lang == "bi" else tr if lang == "tr" else en


def both(en, tr):
    return en if en == tr or not tr else f"{en} / {tr}" if en else tr


def info(row, kind, lang):
    """The detail rows above the item table: (label key, value). The sales order (for the owner) carries every detail of the order form."""
    who = lambda n, ph: " · ".join(x for x in (n, core.plus(ph) if ph else "") if x)
    cust = f"{row['customer_name'] or ''} {core.plus(row['customer_number'])}".strip()
    area = (core._json("delivery.json")["areas"].get(row["delivery_area"]) or {})
    delivery_time = row["delivery_time"] or (area.get("departure", "") if (row["delivery_type"] or "normal") == "normal" else "")
    when = f"{row['delivery_date']} {delivery_time}".strip()
    if kind != "salesorder":
        return [("customer", cust), ("delivery_to", ", ".join(x for x in (row["recipient_name"], core.plus(row["recipient_phone"]) if row["recipient_phone"] else "", row["address"] if not row["address"].startswith("PICKUP") else "") if x)),
                ("when", when), ("dtype", t(row["delivery_type"] or "normal", lang)), ("order", f"#{row['order_id']}")]
    notes = " · ".join(dict.fromkeys(x.strip().removeprefix("note:").strip() for x in (row["notes"] or "").split("|") if x.strip().startswith("note:")))
    area = (core._json("delivery.json")["areas"].get(row["delivery_area"]) or {}).get("label", {})
    return [("order", f"#{row['order_id']}  ·  {row['timestamp'][:16].replace('T', ' ')}"), ("customer", cust), ("sender", who(row["sender_name"], row["sender_phone"])),
            ("recipient", who(row["recipient_name"], row["recipient_phone"])), ("address", row["address"]), ("when", when),
            ("dtype", ("JET REQUEST: APPROVE ONLY IF POSSIBLE · " if (row["delivery_type"] or "") == "jet" and lang != "tr" else "JET TALEBİ: YALNIZCA MÜMKÜNSE ONAYLAYIN · " if (row["delivery_type"] or "") == "jet" else "") + t(row["delivery_type"] or "normal", lang) + (f" · {both(area.get('en', ''), area.get('tr', '')) if lang == 'bi' else area.get('tr' if lang == 'tr' else 'en', '')}" if area else "")),
            ("giftnote", notes), ("status", t("waiting", lang))]


def doc_data(row, kind, lang):
    """kind = quotation | invoice -> everything that goes on the page, as plain data (so it can be tested without opening the PDF)."""
    biz, pay, D = core.business_info(), core._json("payment.json"), core.D
    by_name = {core._nm(p["name"]): p for p in core.graph()[0].values() if p.get("type") == "product"}
    name_ = lambda nm: (both(core.display(by_name[nm], "en"), core.display(by_name[nm], "tr")) if lang == "bi" else core.display(by_name[nm], lang)) if nm in by_name else nm
    items = [(name_(nm), q, u, q * u) for nm, q, u in (core.row_lines(row) or [])]
    for nm_, pr_ in core.custom_rows(row):
        items.append((nm_, 1, D(pr_ or 0), D(pr_ or 0)))
    adj = core.adjustment(row)
    areas = core._json("delivery.json")["areas"]
    a = areas.get(row["delivery_area"])
    detail_rows = info(row, kind, lang)
    return {
        "kind": kind, "title": t(kind, lang), "number": f"{ {'quotation': 'Q', 'invoice': 'INV', 'salesorder': 'SO'}[kind] }-{int(row['order_id']):05d}", "date": core.now().strftime("%d.%m.%Y"),
        "shop": [biz.get("company") or biz.get("name", ""), biz.get("address", ""), " · ".join(x for x in (biz.get("phone"), biz.get("email")) if x), biz.get("website", "")],
        "customer": f"{row['customer_name'] or ''} {core.plus(row['customer_number'])}".strip(),
        "delivery_to": ", ".join(x for x in (row["recipient_name"], core.plus(row["recipient_phone"]) if row["recipient_phone"] else "", row["address"] if not row["address"].startswith("PICKUP") else "") if x),
        "when": dict(detail_rows)["when"],
        "items": items, "goods": row["total"], "delivery_label": (both(a["label"]["en"], a["label"]["tr"]) if lang == "bi" else a["label"]["tr" if lang == "tr" else "en"]) if a else "", "delivery_fee": row["delivery_fee"],
        "adjustment": adj, "payable": row["payable"], "paid": row["paid"],
        "bank": {"holder": pay["holder"], "bank": pay["bank"], "iban": pay["iban"], "extra": pay.get("extra", ""), "receipt": both(pay["receipt_line"]["en"], pay["receipt_line"]["tr"]) if lang == "bi" else pay["receipt_line"]["tr" if lang == "tr" else "en"]} if kind == "quotation" else None,
        "order": f"#{row['order_id']}",
        "info": detail_rows,
    }


def build(row, kind, lang, out=None):
    d = doc_data(row, kind, lang)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, 18)
    pdf.add_font("DV", "", str(FONTS / "DejaVuSans.ttf")); pdf.add_font("DV", "B", str(FONTS / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    plum = (113, 75, 103)
    pdf.set_font("DV", "B", 17); pdf.set_text_color(*plum); pdf.cell(0, 9, d["shop"][0], new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DV", "", 9); pdf.set_text_color(90, 90, 90)
    for line in d["shop"][1:]:
        if line:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5); pdf.set_draw_color(*plum); pdf.set_line_width(0.6); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)
    bi = lang == "bi"
    pdf.set_text_color(30, 30, 30); pdf.set_font("DV", "B", 17 if bi else 20); pdf.cell(110, 10, d["title"])
    pdf.set_font("DV", "", 10); pdf.cell(0, 5, f"{t('no', lang)} {d['number']}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(110, 5, ""); pdf.cell(0, 5, f"{t('date', lang)}: {d['date']}", align="R", new_x="LMARGIN", new_y="NEXT"); pdf.ln(4)
    for k, v in d["info"]:
        if v:
            pdf.set_font("DV", "B", 9 if bi else 10); pdf.cell(60 if bi else 38, 6, t(k, lang)); pdf.set_font("DV", "", 10); pdf.multi_cell(0, 6, v, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_fill_color(241, 232, 238)
    W = (85, 25, 40, 40) if bi else (100, 20, 35, 35)

    def head():
        pdf.set_font("DV", "B", 8.5 if bi else 10); pdf.set_text_color(30, 30, 30)
        for w, k, al in zip(W, ("item", "qty", "unit", "total"), "LCRR"):
            pdf.cell(w, 8, t(k, lang), fill=True, align=al)
        pdf.ln(8); pdf.set_font("DV", "", 10)
    head()
    for name, q, u, tot in d["items"]:  # a long name wraps: the row grows with it, and the table continues on the next page (with its header) when the page is full
        lines = len(pdf.multi_cell(W[0], 7, str(name), dry_run=True, output="LINES")) or 1
        h = 7 * lines
        if pdf.get_y() + h > 297 - 22:
            pdf.add_page(); head()
        y = pdf.get_y(); pdf.multi_cell(W[0], 7, str(name), new_x="RIGHT", new_y="TOP", max_line_height=7)
        pdf.set_xy(10 + W[0], y); pdf.cell(W[1], 7, str(q), align="C"); pdf.cell(W[2], 7, core.fmt_tl(u), align="R"); pdf.cell(W[3], 7, core.fmt_tl(tot), align="R")
        pdf.set_xy(10, y + h); pdf.set_draw_color(225, 225, 225); pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    def row_(label, value, bold=False):
        pdf.set_font("DV", "B" if bold else "", 11 if bold else 10); pdf.cell(155, 7, label, align="R"); pdf.cell(35, 7, value, align="R", new_x="LMARGIN", new_y="NEXT")
    row_(t("goods", lang), core.fmt_tl(d["goods"] or 0))
    if d["delivery_fee"] != "":
        row_(f"{t('delivery', lang)} {('(' + d['delivery_label'] + ')') if d['delivery_label'] else ''}".strip(), core.fmt_tl(d["delivery_fee"]) if core.D(d["delivery_fee"]) else t("free", lang))
    if d["adjustment"]:
        row_(t("fee" if d["adjustment"] > 0 else "discount", lang), ("+" if d["adjustment"] > 0 else "-") + core.fmt_tl(abs(d["adjustment"])))
    pdf.ln(1); pdf.set_text_color(*plum); row_(t("topay", lang), core.fmt_tl(d["payable"] or 0), True); pdf.set_text_color(30, 30, 30)
    if kind == "invoice":
        pdf.ln(3); pdf.set_font("DV", "B", 22 if bi else 26); pdf.set_text_color(40, 140, 80); pdf.cell(0, 14, t("paid", lang), align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("DV", "", 10); pdf.set_text_color(30, 30, 30); pdf.cell(0, 6, f"{t('paidamt', lang)}: {core.fmt_tl(d['paid'] or d['payable'] or 0)}", align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6); pdf.cell(0, 6, t("thanks", lang), new_x="LMARGIN", new_y="NEXT")
    elif kind == "quotation":
        b = d["bank"]; pdf.ln(6); pdf.set_fill_color(248, 244, 247); pdf.set_font("DV", "B", 11); pdf.cell(0, 8, t("bank", lang), fill=True, new_x="LMARGIN", new_y="NEXT")
        for k, v in (("holder", b["holder"]), ("bankname", b["bank"]), ("iban", b["iban"] + (f"  {b['extra']}" if b["extra"] else ""))):
            pdf.set_font("DV", "", 10); pdf.cell(60 if bi else 38, 7, t(k, lang)); pdf.set_font("DV", "B", 10); pdf.cell(0, 7, v, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2); pdf.set_font("DV", "", 9.5); pdf.set_text_color(90, 90, 90); pdf.multi_cell(0, 6, b["receipt"], new_x="LMARGIN", new_y="NEXT")
    out = out or core.P("documents") / f"{kind}_{int(row['order_id'])}.pdf"
    Path(out).parent.mkdir(exist_ok=True)
    pdf.output(str(out))
    return out


def build_report(kind="day", today=None, out=None):
    """The day / week / month report as a PDF in the same style as the quotation and the invoice (English and Turkish), ending with a Turkish encouraging line."""
    import random
    D_ = core.report_data(kind, today)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, 18)
    pdf.add_font("DV", "", str(FONTS / "DejaVuSans.ttf")); pdf.add_font("DV", "B", str(FONTS / "DejaVuSans-Bold.ttf"))
    pdf.add_page()
    plum = (113, 75, 103)
    biz = core.business_info()
    pdf.set_font("DV", "B", 17); pdf.set_text_color(*plum); pdf.cell(0, 9, biz.get("company") or biz.get("name", ""), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DV", "", 9); pdf.set_text_color(90, 90, 90)
    for line in (biz.get("address", ""), " · ".join(x for x in (biz.get("phone"), biz.get("email")) if x), biz.get("website", "")):
        if line:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5); pdf.set_draw_color(*plum); pdf.set_line_width(0.6); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)
    names = {"day": ("Z-REPORT", "Z RAPORU"), "week": ("WEEKLY REPORT", "HAFTALIK RAPOR"), "month": ("MONTHLY REPORT", "AYLIK RAPOR")}[kind]
    pdf.set_text_color(30, 30, 30); pdf.set_font("DV", "B", 17); pdf.cell(125, 10, f"{names[0]} / {names[1]}")
    pdf.set_font("DV", "", 10); pdf.cell(0, 5, f"No. RP-{D_['start']:%Y%m%d}-{kind[0].upper()}", align="R", new_x="LMARGIN", new_y="NEXT")
    last = D_["end"] - core.timedelta(days=1)
    sofar = kind != "day" and core.now().replace(tzinfo=None) < D_["end"]  # a week / month that is still running: the report grows until it is over
    pdf.cell(125, 5, ""); pdf.cell(0, 5, f"{D_['start']:%d.%m.%Y}" + (f" – {last:%d.%m.%Y}" if kind != "day" else ""), align="R", new_x="LMARGIN", new_y="NEXT")
    if sofar:
        pdf.set_font("DV", "", 9); pdf.set_text_color(*plum); pdf.cell(125, 5, ""); pdf.cell(0, 5, f"so far / şimdiye kadar: {core.now():%d.%m.%Y %H:%M}", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    val = core.report_value
    tot = lambda rs: sum((val(r) for r in rs), core.D(0))
    pdf.set_fill_color(241, 232, 238); pdf.set_font("DV", "B", 9)
    for w, h_ in ((47.5, "Orders placed / Verilen"), (47.5, "Completed / Tamamlanan"), (47.5, "On-going / Devam eden"), (47.5, "Cancelled / İptal")):
        pdf.cell(w, 8, h_, fill=True, align="C")
    pdf.ln(8); pdf.set_font("DV", "B", 12)
    for w, k in ((47.5, "placed"), (47.5, "completed"), (47.5, "ongoing"), (47.5, "cancelled")):
        pdf.cell(w, 9, str(len(D_[k])), align="C")
    pdf.ln(10)

    COLS = ((26, "Order / Sipariş", "L"), (46, "Customer / Müşteri", "L"), (83, "Items / Ürünler", "L"), (35, "Amount / Tutar", "R"))

    def thead():
        pdf.set_text_color(30, 30, 30); pdf.set_fill_color(241, 232, 238); pdf.set_font("DV", "B", 8.5)
        for w, h_, al in COLS:
            pdf.cell(w, 7, h_, fill=True, align=al)
        pdf.ln(7); pdf.set_font("DV", "", 9)

    def lines_of(w, text):
        return len(pdf.multi_cell(w, 6, str(text), dry_run=True, output="LINES")) or 1

    def table(head, rs):  # every row wraps (name and items), grows with its text, and the table continues on the next page with its header: no cap, no overlap
        if pdf.get_y() > 297 - 50:
            pdf.add_page()
        pdf.set_font("DV", "B", 11); pdf.set_text_color(*plum); pdf.multi_cell(0, 8, f"{head}  ({len(rs)})  ·  {core.fmt_tl(tot(rs))}", new_x="LMARGIN", new_y="NEXT")
        thead()
        for r in rs:
            who, items = (r["customer_name"] or core.plus(r["customer_number"])), core._items_text(r)
            h = 6 * max(lines_of(COLS[1][0], who), lines_of(COLS[2][0], items)) + 1
            if pdf.get_y() + h > 297 - 22:
                pdf.add_page(); thead()
            y = pdf.get_y()
            pdf.set_font("DV", "", 9); pdf.set_text_color(30, 30, 30)
            pdf.set_xy(10, y); pdf.cell(COLS[0][0], 6, f"#{r['order_id']}")
            pdf.set_xy(10 + COLS[0][0], y); pdf.multi_cell(COLS[1][0], 6, who, new_x="RIGHT", new_y="TOP", max_line_height=6)
            pdf.set_xy(10 + COLS[0][0] + COLS[1][0], y); pdf.multi_cell(COLS[2][0], 6, items, new_x="RIGHT", new_y="TOP", max_line_height=6)
            pdf.set_xy(200 - COLS[3][0], y); pdf.cell(COLS[3][0], 6, core.fmt_tl(val(r)) if val(r) else "–", align="R")
            pdf.set_xy(10, y + h); pdf.set_draw_color(225, 225, 225); pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        if not rs:
            pdf.set_font("DV", "", 9); pdf.cell(0, 7, "–  none / yok", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
    table("Completed / Tamamlanan", D_["completed"])
    table("On-going / Devam eden", D_["ongoing"])
    table("Cancelled, rejected / İptal, reddedilen", D_["cancelled"])
    pdf.set_font("DV", "B", 11); pdf.set_text_color(*plum)
    pdf.multi_cell(0, 8, "".join(ch for ch in random.choice(core.CHEERS_TR) if ord(ch) < 0x2500).strip(), align="C", new_x="LMARGIN", new_y="NEXT")  # the PDF font has no emoji
    out = out or core.P("documents") / f"report_{kind}_{D_['start']:%Y%m%d}.pdf"
    Path(out).parent.mkdir(exist_ok=True)
    pdf.output(str(out))
    return out


def progressive(kind):
    """The report of the week / month that is running NOW, rebuilt from the orders each time it is asked for (so it keeps growing as the period passes). Only the current one is kept: the file of a finished week / month is deleted."""
    if kind not in ("week", "month"):
        raise ValueError(kind)
    start = core.period_bounds(kind)[0]
    d = core.P("reports"); d.mkdir(exist_ok=True)
    for f in d.glob(f"{kind}_*.pdf"):
        if f.name != f"{kind}_{start:%Y%m%d}.pdf":
            f.unlink(missing_ok=True)
    return build_report(kind, out=d / f"{kind}_{start:%Y%m%d}.pdf")


def cleanup_reports():
    """Delete the stored week / month reports of finished periods (called from the server loop, so a new week or month always starts clean)."""
    d = core.P("reports")
    for kind in ("week", "month"):
        cur = f"{kind}_{core.period_bounds(kind)[0]:%Y%m%d}.pdf"
        for f in d.glob(f"{kind}_*.pdf") if d.exists() else []:
            if f.name != cur:
                f.unlink(missing_ok=True)
