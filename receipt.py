"""Payment receipt check. Pure code, no model: text is pulled out of a photo (OCR hook), PDF, DOCX or pasted text, then the amount, the receiving account and the date are compared
with the order. The result only tells the owner what it SEES: the owner always does the real confirmation in the bank account."""
import io, re, zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal

import core

MONTHS = {"ocak": 1, "subat": 2, "mart": 3, "nisan": 4, "mayis": 5, "haziran": 6, "temmuz": 7, "agustos": 8, "eylul": 9, "ekim": 10, "kasim": 11, "aralik": 12,
          "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
          "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
KEYWORDS = re.compile(r"\b(transaction|transfer|reference|receipt|payment|paid|dekont|havale|eft|fast|islem|referans|odeme|iban|hesap|account|bank|banka|gonderen|alici|sender|beneficiary|tutar|amount|toplam)\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\s?[A-Z0-9]{1,4}\b")


def extract_text(kind, data=None, filename="", text="", ocr=None):
    """(text, why_not): the readable text of what the customer sent, or '' with the reason it could not be read."""
    if kind == "text":
        return text, ""
    name = (filename or "").lower()
    try:
        if kind == "image":
            out = ocr(data) if ocr else ""
            return (out or ""), ("" if out else "the photo could not be read automatically")
        if name.endswith(".pdf") or (data or b"")[:4] == b"%PDF":
            from pypdf import PdfReader
            out = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages[:5])
            return out, ("" if out.strip() else "the PDF has no readable text (a scan or a picture)")
        if name.endswith(".docx") or (data or b"")[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                xml = z.read("word/document.xml").decode("utf-8", "ignore")
            out = re.sub(r"<[^>]+>", " ", re.sub(r"</w:p>", "\n", xml))
            return out, ("" if out.strip() else "the document is empty")
        if name.endswith((".txt", ".csv")):
            return data.decode("utf-8", "ignore"), ""
    except Exception as e:
        return "", f"the file could not be opened ({type(e).__name__})"
    return "", "this file type cannot be read automatically"


def _amounts(text):
    out = []
    for m in re.findall(r"\d[\d.,\s]*\d|\d", text):
        try:
            out.append(Decimal(str(core.parse_amount(m.replace(" ", "")))).quantize(Decimal("0.01")))
        except (ValueError, ArithmeticError):
            pass
    return out


def _dates(text, today):
    found = []
    for d, mo, y in re.findall(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", text):
        found.append((int(y), int(mo), int(d)))
    for y, mo, d in re.findall(r"\b(\d{4})[./-](\d{1,2})[./-](\d{1,2})\b", text):
        found.append((int(y), int(mo), int(d)))
    for d, mon, y in re.findall(r"\b(\d{1,2})\s+([a-z]{3,9})\s+(\d{4})\b", text):
        if mon in MONTHS:
            found.append((int(y), MONTHS[mon], int(d)))
    out = []
    for y, mo, d in found:
        try:
            out.append(date(y, mo, d))
        except ValueError:
            pass
    return out


def check(text, row, now=None):
    """The receipt against the order: {relevant, amount, account, date, notes, verdict}. verdict: valid | problems | irrelevant. Each check is True / False / None (not visible)."""
    now = now or core.now()
    t = core.fold(text)
    t = re.sub(r"[ \t]+", " ", t)
    pay = core._json("payment.json")
    iban = re.sub(r"\s", "", pay["iban"]).upper()
    flat = re.sub(r"\s", "", (text or "").upper())
    ibans = [re.sub(r"\s", "", i) for i in IBAN.findall((text or "").upper())]
    holder = [w for w in core.fold(pay["holder"]).split() if len(w) > 2]
    notes, res = [], {"relevant": False, "amount": None, "account": None, "date": None}
    want = Decimal(row["payable"] or row["total"] or 0).quantize(Decimal("0.01"))
    amts = _amounts(text or "")
    # amount
    if want and want in amts:
        res["amount"] = True
    elif amts and re.search(r"\b(tl|try|usd|eur|gbp)\b|[₺$€£]|tutar|amount|toplam|total", t):
        res["amount"] = False
        notes.append(f"amount: the order needs {core.fmt_tl(want)}, the receipt shows " + ", ".join(sorted({core.fmt_tl(a) for a in amts if a >= 10 and a != want})[:3] or ["no matching amount"]))
    # receiving account
    if iban in flat:
        res["account"] = True
    elif holder and all(w in t for w in holder):
        res["account"] = True
        notes.append("account: the holder name matches, the IBAN is not shown")
    elif ibans:
        res["account"] = False
        notes.append(f"account: the receipt shows another IBAN (ending {ibans[0][-4:]}), ours ends {iban[-4:]}")
    # date: not before the order was placed, not in the future
    placed = datetime.fromisoformat(row["timestamp"]).date() if row.get("timestamp") else now.date() - timedelta(days=30)
    ds = _dates(text or "", now.date())
    if ds:
        okd = [d for d in ds if placed <= d <= now.date() + timedelta(days=1)]
        res["date"] = bool(okd)
        if not okd:
            notes.append(f"date: {ds[0]:%d.%m.%Y} is " + ("before the order was placed" if ds[0] < placed else "in the future"))
    res["own"] = bool(re.search(r"quotation\s*/\s*teklif|sales order\s*/\s*satis siparisi|invoice\s*/\s*fatura|\b(q|so|inv)-\d{5}\b", t))  # the customer sent back the document WE sent
    res["relevant"] = (bool(KEYWORDS.search(t)) or res["amount"] is not None or res["account"] is not None or bool(ibans)) and not res["own"]
    res["notes"] = notes
    if not res["relevant"]:
        res["verdict"] = "irrelevant"
    elif res["amount"] is True and res["account"] is not False and res["date"] is not False and res["account"] is True:
        res["verdict"] = "valid"
    else:
        res["verdict"] = "problems"
    return res


def lines(res, lang=0):
    """The checklist the owner reads: one line per check."""
    words = (("Amount matches the order", "Amount does NOT match", "Amount not found in it", "Goes to our account", "Goes to ANOTHER account", "Receiving account not found in it", "Date is plausible", "Date is wrong", "No date found in it"),
             ("Tutar siparişle eşleşiyor", "Tutar eşleşmiyor", "Tutar bulunamadı", "Hesabımıza gidiyor", "BAŞKA bir hesaba gidiyor", "Alıcı hesap bulunamadı", "Tarih makul", "Tarih yanlış", "Tarih bulunamadı"),
             ("Сумма совпадает с заказом", "Сумма НЕ совпадает", "Сумма не найдена", "Перевод на наш счёт", "Перевод на ДРУГОЙ счёт", "Реквизиты получателя не найдены", "Дата выглядит правдоподобно", "Дата неверна", "Дата не найдена"))[lang]
    mark = lambda v, ok, no, unk: ("✅ " + ok) if v is True else ("❌ " + no) if v is False else ("➖ " + unk)
    notes = []
    for note in res["notes"]:
        amount = re.match(r"amount: the order needs (.*), the receipt shows (.*)", note)
        other_iban = re.match(r"account: the receipt shows another IBAN \(ending (\d+)\), ours ends (\d+)", note)
        if amount:
            notes.append((f"⚠️ Amount: order needs {amount[1]}; receipt shows {amount[2]}.",
                          f"⚠️ Tutar: sipariş tutarı {amount[1]}; dekontta {amount[2]} görünüyor.",
                          f"⚠️ Сумма: по заказу нужно {amount[1]}, в квитанции указано {amount[2]}.")[lang])
        elif note == "account: the holder name matches, the IBAN is not shown":
            notes.append(("⚠️ Account: the holder name matches, but the IBAN is not shown.",
                          "⚠️ Hesap: hesap sahibi adı eşleşiyor ancak IBAN görünmüyor.",
                          "⚠️ Счёт: имя владельца совпадает, но IBAN не указан.")[lang])
        elif other_iban:
            notes.append((f"⚠️ Account: another IBAN ends {other_iban[1]}; ours ends {other_iban[2]}.",
                          f"⚠️ Hesap: başka IBAN'ın sonu {other_iban[1]}; bizimkinin sonu {other_iban[2]}.",
                          f"⚠️ Счёт: другой IBAN оканчивается на {other_iban[1]}, наш — на {other_iban[2]}.")[lang])
        else:
            date = re.match(r"date: (\d{2}\.\d{2}\.\d{4}) is (before the order was placed|in the future)", note)
            if date:
                notes.append((f"⚠️ Date: {date[1]} is {date[2]}.",
                              f"⚠️ Tarih: {date[1]} {('sipariş verilmeden önce' if date[2].startswith('before') else 'gelecekte')}.",
                              f"⚠️ Дата: {date[1]} — {('раньше оформления заказа' if date[2].startswith('before') else 'в будущем')}.")[lang])
            else:
                notes.append(f"⚠️ {note}")
    return "\n".join([mark(res["amount"], *words[0:3]), mark(res["account"], *words[3:6]), mark(res["date"], *words[6:9])]
                     + notes)
