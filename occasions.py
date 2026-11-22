"""Occasion dates, resolved by code: "one day before Mother's Day", "the day before Valentine's", "next Christmas". Local calendar first (country rules, no web call for a fixed
date); an occasion that is not in the calendar is reported as unknown so the agent asks for the exact date. Nothing here invents a date."""
import re
from datetime import date, timedelta

import core

COUNTRY = "cy"  # the shop's own market: North Cyprus follows the Turkish calendar
WORDS = {
    "mothers_day": r"mother'?s? ?day|mothering sunday|anneler g[uü]n[uü]|день матери",
    "valentines": r"valentine'?s?( day)?|sevgililer g[uü]n[uü]|день святого валентина|день влюблённых",
    "womens_day": r"(international )?women'?s? day|kad[iı]nlar g[uü]n[uü]|8 mart|8 march|международный женский день|8 марта",
    "new_year_eve": r"new year'?s? eve|y[iı]lba[sş][iı] ak[sş]am[iı]|новогодняя ночь",
    "new_year": r"new year'?s? day|new year|y[iı]lba[sş][iı]|новый год",
    "christmas": r"christmas|noel|рождество",
    "fathers_day": r"father'?s? ?day|babalar g[uü]n[uü]|день отца",
    "teachers_day": r"teachers'? day|[oö][gğ]retmenler g[uü]n[uü]|день учителя",
}
LABEL = {"mothers_day": "Mother's Day", "valentines": "Valentine's Day", "womens_day": "International Women's Day", "new_year_eve": "New Year's Eve", "new_year": "New Year's Day",
         "christmas": "Christmas", "fathers_day": "Father's Day", "teachers_day": "Teachers' Day"}
COUNTRY_WORDS = {"uk": r"\b(uk|england|britain|british|london)\b", "us": r"\b(usa|america|american)\b", "ru": r"\b(russia|russian|россия|росси)\w*"}
NUM = {"a": 1, "one": 1, "bir": 1, "two": 2, "iki": 2, "three": 3, "üç": 3, "uc": 3, "four": 4, "dört": 4, "five": 5, "beş": 5}
OFFSET = re.compile(r"(?:(\d+|a|one|two|three|four|five|bir|iki|üç|uc|dört|beş)\s+days?\s+(before|after|prior to)|the day (before|after)|(\d+|bir|iki|üç)\s+g[uü]n (?:[oö]nce|sonra)|накануне|за (\d+) д)", re.I)


def nth_weekday(y, month, weekday, n):
    d = date(y, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
    return d


def easter(y):  # Western Easter (anonymous Gregorian algorithm)
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo, da = divmod(h + l - 7 * m + 114, 31)
    return date(y, mo, da + 1)


def occurrence(key, year, country=COUNTRY):
    """The date of an occasion in one year, by country rule; None when the calendar does not know it."""
    if key == "mothers_day":
        if country == "uk":
            return easter(year) - timedelta(days=21)  # Mothering Sunday: 4th Sunday of Lent
        if country == "ru":
            return nth_weekday(year, 11, 6, 5) if nth_weekday(year, 11, 6, 5).month == 11 else nth_weekday(year, 11, 6, 4)  # last Sunday of November
        return nth_weekday(year, 5, 6, 2)  # Turkey / North Cyprus / USA: 2nd Sunday of May
    fixed = {"valentines": (2, 14), "womens_day": (3, 8), "new_year_eve": (12, 31), "new_year": (1, 1), "christmas": (12, 25), "teachers_day": (11, 24)}
    if key in fixed:
        return date(year, *fixed[key])
    if key == "fathers_day":
        return nth_weekday(year, 6, 6, 3)
    return None


def resolve(text, today, country=None):
    """[{phrase, occasion, date, note}]: every occasion named in the text, with the date it means (the next one that is not in the past, shifted by 'N days before/after').
    country: the one named in the text (UK, USA, Russia) or the shop's own; when it was not named and the answer differs by country, the note says which one was assumed."""
    t = (text or "").lower()
    country = country or next((c for c, rx in COUNTRY_WORDS.items() if re.search(rx, t)), "")
    named = bool(country)
    country = country or COUNTRY
    out = []
    for key, rx in WORDS.items():
        m = re.search(rx, t, re.I)
        if not m or (key == "new_year" and re.search(WORDS["new_year_eve"], t, re.I)):  # "new year's eve" is not also "new year"
            continue
        seg = t[max(0, m.start() - 40):m.start()]
        off = OFFSET.search(seg) or OFFSET.search(t[m.end():m.end() + 25])
        delta = 0
        if off:
            word = (off.group(1) or off.group(4) or off.group(5) or "1").lower()
            n = int(word) if word.isdigit() else NUM.get(word, 1)
            low = off.group(0).lower()
            delta = -n if re.search(r"before|prior|önce|once|накануне|за ", low) else n
        for y in (today.year, today.year + 1, today.year + 2):
            occ = occurrence(key, y, country)
            if occ and occ + timedelta(days=delta) >= today:
                break
        d = occ + timedelta(days=delta)
        note = f"{LABEL[key]} {y} is {occ:%A %d %B %Y}" + (f" in {country.upper()}" if named else " (North Cyprus / Turkey calendar)")
        if delta:
            note += f"; {abs(delta)} day(s) {'before' if delta < 0 else 'after'} = {d:%A %d %B %Y}"
        if key == "mothers_day" and not named:
            note += ". Mother's Day differs by country (UK, Russia and others use other dates): if the customer or the recipient is from elsewhere, ask which country"
        out.append({"phrase": m.group(0), "occasion": key, "date": d.isoformat(), "note": note})
    return out
