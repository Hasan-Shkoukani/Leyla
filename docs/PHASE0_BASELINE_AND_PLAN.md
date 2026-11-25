# Phase 0: source verification, baseline and implementation plan

No project source file was edited in this phase (only this report was added). Evidence: `AGENT_TECHNICAL_AUDIT.md`, the source, a full test run, a live-model baseline (18 turns, 3 scripted customers, temp state, no real orders) and two probes.

## 1. Test baseline (measured)
`pytest tests/`: **281 passed, 0 failed, 17 warnings, 53 s.**

## 2. Performance baseline (measured today, gpt-5.6-luna, reasoning none, debounce 3 s, customer-visible time)
| Metric | Value |
|---|---|
| Turns | 18 (6 messages x 3 customers) |
| End-to-end median / p95 | **5.9 s / 10.3 s** |
| Turns answered by code (greeting, hours) | 6 of 18, each **3.03 s = the debounce alone** |
| Turns that used the model | 12 of 18 |
| Model calls per model turn (avg) | 2.58 (range 1-6) |
| Input tokens per model turn (avg) | 31,187 (single-call turns: about 11.3-11.6k, i.e. the audit's about 11.9k fixed overhead is confirmed) |
| Cached input tokens per model turn (avg) | 23,566 (provider prompt cache works) |
| Output tokens per model turn (avg) | 220 |
| Time inside model calls per model turn (avg) | 4.69 s |
| Worst turns | multi-category budget request: 6 calls, 72.6k input tokens, 13.1 s; order + confirm turns: 3-5 calls, 35-63k tokens, 7.6-10.3 s |
Raw per-turn data: `scratchpad/audit/baseline_llm.json` (tool names per turn included).

## 3. Audit claims verified against the code
* 32 customer tools, 63 owner tools: confirmed by introspection.
* Fast-model path exists (`agent.py:70`, `FAST_MODEL=` empty in `.env`, so disabled; chosen only for "simple" turns where code already did the lookup, `agent.py:1150`).
* **`gpt-5.6-luna` + `reasoning_effort` != none + function tools = HTTP 400** (reproduced). Reasoning cannot be raised on this model without the Responses API or another model.
* Telegram prompt contradiction confirmed: `agent.py:1116` always tells the model "you do NOT know their phone number, ask for it", while `sender_known()` (`agent.py:381`) already fills it for returning customers.
* Debounce is a flat `asyncio.sleep(DEBOUNCE_SECONDS)` restarted on every message (`agent.py:1446`): it costs 3 s even for messages answered by code.

## 4. New findings from this phase
1. **Lost-update race (reproduced).** The dashboard edit path reads `core.orders()` outside the lock, then `apply_orders_edit(rows)` writes that stale copy: an agent write in between (e.g. a payment confirmation: status `confirmed`, paid 5) is silently reverted to `pending` / empty paid. Concurrent creation is fine (40 parallel `order_create` calls: 40 unique ids, no errors).
2. **No payment ledger.** Money is one `paid` column on the order row; there is no payment history, no record of additional payments, refunds or adjustments. The master prompt's "payment history preserved / additional payment recorded separately / double refund prevention" needs a new append-only ledger (design below).
3. **Current change workflow differs from the requested one.** `core.change_apply()` writes the change into the confirmed order at approval time and, if it costs more, sets the status back to `awaiting_payment` immediately; the requested rule is "update the order only after the difference is verified". Lower price: current code writes a note `refund due` and silently sets `paid` to the new amount.
4. **No geocoding exists.** A Google Maps link is only stored as the address text; the delivery area is found by matching place names in that text (`core.detect_area`, `data/delivery.json`, 9 areas with minimum order and jet fee), so a map link gives no area, no fee and no minimum check. Nothing resolves short links or calls a Google API.
5. **No "gift" concept.** Admin instructions reach customers only through `message_customer` / `ask_customer`; there is no field on the order and nothing prevents the model from pricing or inventing a gift.
6. **Statuses.** `cancelled` appears 32 times and `rejected` 21 times in the code and dashboard, and both are already treated as Lost (`LOST_STATUSES`), so writing `rejected` for an approved cancellation is technically safe and backward compatible (existing `cancelled` rows stay valid).
7. **Possible duplicate-work bug (to investigate in Phase 1/4):** in the live baseline one customer's plain catalog order ("1 pink rose bouquet of 5 ...") made the agent call `request_quote` and `place_order` in the same turn.
8. **Identity.** Linking by user tag or phone exists (`core.link`, `linked_ids`) but the WhatsApp payload for hidden-number users is unverified.

## 5. Channel status (actual inspection)
* **Telegram: operational path.** Token set; webhook registered with Telegram (`pending_update_count` 0; its `last_error` is an old 404 from a moment the server was down). Handler: `app.py` `/telegram/webhook` -> `agent.inbound`. Regression-tested in `tests/test_core.py` (fake transport).
* **WhatsApp: implemented in code, not connected.** Complete adapter (`app.py:30`: text, document, image, template, typing, media download, HMAC `X-Hub-Signature-256` check, wa_id/username capture best effort). `.env` has `WHATSAPP_TOKEN`, `PHONE_NUMBER_ID`, `VERIFY_TOKEN`, `APP_SECRET`, but a read-only Graph check returns **401 (token expired)**. Still needed to connect: a fresh permanent token, the webhook URL + verify token registered in Meta, approved message templates (`docs/templates.md`), and confirmation of the hidden-number payload fields. I will not touch credentials or webhooks without your authorization.

## 6. Implementation plan (smallest changes, each phase tested separately)
* **Phase 1, performance:** (a) debounce: skip or shorten it for messages that code answers (greeting/hours) and add an "adaptive" wait (short for a complete sentence, longer only when a message ends mid-thought or arrives in a burst); (b) cut the prompt by merging duplicate rules and removing the contradicting Telegram line; send only the tools a turn can use (e.g. no order tools for catalog questions); (c) stop duplicate `flowers_lookup` calls in one turn (cache per turn); (d) log per-request metrics (model calls, tokens, durations, tool times) through one small wrapper; (e) evaluate the fast path with a routing test and enable it only for greeting/info turns if answers match.
* **Phase 2, multi-category shopping:** extend `budget_basket` into a deterministic combination builder (`combo_search(categories, total_budget)`): independent per-category lookups run concurrently, Python builds one-product-per-category combinations under the total budget and returns formatted lines; the model only presents them.
* **Phase 3, delivery and gifts:** small isolated `geo.py` (resolve short links with a timeout, extract coordinates/place, Google Geocoding with an env key, cache, map to `delivery.json` areas; any failure -> ask the customer for the city); a `gift` field on the order (free text, never priced) set only by an admin tool and shown on invoice/summary.
* **Phase 4, financial workflows:** payment ledger (append-only `payments.jsonl`, one JSON line per event: order id, kind payment/extra/refund/adjustment, amount, status, who, when); change request state machine (requested -> approved/rejected -> awaiting difference -> verified -> applied) keyed by order id; cancellation requires admin approval and ends as `rejected`.
* **Phase 5, reliability:** make every read-modify-write take `_lock` around the read as well (fixes the lost update), route dashboard edits through the same locked function, duplicate guards on change and payment requests, restart-safe request state, owner-tool authorization asserted in Python.
* **Phase 6:** re-run this baseline script and the full suite, compare.

## 7. Decisions I need from you (everything else proceeds without asking)
1. **Cancellation before payment.** You earlier told me a cancel before the final payment confirmation "just cancels". The master prompt says every cancellation goes to the admin and ends as REJECTED. Which rule should win? (My recommendation: admin approval always, status `rejected`; a never-paid order could still be auto-approved if you prefer.)
2. **Lower-price change after payment.** Refund, store credit, or something else? Nothing in the project defines it today (currently the system only writes "refund due" in the notes).
3. **Higher-price change after payment.** Confirm the new rule: the paid order stays as it is until the extra amount is verified, then it is updated and a new receipt is sent (this replaces today's behavior of updating at approval).
4. **Google Geocoding.** Do you have a Google Maps API key with Geocoding enabled? If not, short links can still be resolved and coordinates extracted, but the city cannot be derived reliably, so the agent would ask for the city every time.
5. **Fast model.** May I enable `gpt-4o-mini` for greeting/info turns after the routing test passes (it is a cheaper, weaker model, never used for orders, payments or changes)?
