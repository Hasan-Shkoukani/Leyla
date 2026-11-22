# WhatsApp templates (admin alerts outside the 24h window)

Category: **UTILITY**, language `en` (set `TEMPLATE_LANG`). Create these in Meta Business Manager and wait for approval.
Variables are plain text, no newlines (the code collapses whitespace and truncates to 200 chars). Free-form text is
used instead whenever the admin messaged the number in the last 24h (any command opens the window; replying to a template
alert, e.g. `yes 12`, also does).

## 1. `order_alert`
```
New flower order #{{1}} from {{2}}. Items: {{3}}. Delivery: {{4}}. Reply "yes {{1}}" to confirm or "no {{1}} reason" to decline.
```
{{1}} order number, {{2}} customer number + name, {{3}} items and total (TL), {{4}} address + delivery date.

## 2. `issue_alert`
```
Customer {{1}} needs the team. Type: {{2}}. Details: {{3}}. Tell the assistant what to answer.
```
{{1}} customer number, {{2}} custom / unknown / issue, {{3}} short summary (also used for errors and photo-cap warnings).

## 3. `limit_warning`
```
Message from customer {{1}} was held for you: {{2}}, length {{3}}. Excerpt: {{4}}. Chat is paused; tell the assistant to un-mute them when done.
```
{{1}} customer number, {{2}} "Long text" / "Long voice note", {{3}} chars or seconds, {{4}} excerpt (100 chars).

## 4. `admin_notice`
```
Flower shop update: {{1}}
```
{{1}} any short notice (spare; currently unused by the code, reserved for daily summaries).
