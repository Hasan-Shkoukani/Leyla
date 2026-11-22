# Flower shop WhatsApp agent (LangGraph)

## Run (mock mode, no WhatsApp/OpenAI creds needed for the UI shell)
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # light deps only
cp .env.example .env                                                  # set OPENAI_API_KEY to talk to a real model
.venv/bin/python build_graph.py            # data/raw/site.json (or site.sample.json) -> data/graph.json
.venv/bin/uvicorn app:app --port 8000      # open http://localhost:8000
.venv/bin/pytest -q                        # tests (fakes only, no network)
```
Empty `WHATSAPP_TOKEN`/`PHONE_NUMBER_ID` = mock mode: UI with Customer tab, Admin tab (the only place files download) and live feed.
Set both, plus `APP_SECRET`/`VERIFY_TOKEN`, and point Meta's webhook at `/webhook` for live mode (`/message` then returns 404).

## Layout
`build_graph.py` graph builder (category cleaning, colors from names) · `core.py` sqlite/graph/orders.csv/stats/alerts ·
`agent.py` LangGraph agent, batching, photo/voice, limits, admin commands · `app.py` FastAPI + Meta adapter · `static/index.html` UI ·
`docs/templates.md` admin templates · `tests/test_core.py`.
State (orders, chats, customers, prices, outbox, backups...) lives in `state/` (STATE_DIR overrides it). Older layouts keep it in the project folder until `./migrate_state.sh` (server stopped) moves it into `state/`.

## Prices
Prices are **stored**, not read while a customer waits. `prices.json` holds each product's last-read price and stock (read from the product's own page: JSON-LD `offers.price`, see `prices.py`) and is merged into the in-memory graph when it loads, so every lookup, budget filter, "cheapest / priciest" (precomputed lists, `core.extremes()`) and order total is a dictionary lookup with no network.
They change only when an update stores new ones: the nightly full update (02:00-05:00, `DAILY_REFRESH=0` turns it off), the dashboard's "Update product list" (everything) or "Update item" / "update price" (one item). An item whose page cannot be read keeps its previous price; an item with no stored price gets no price: the agent says the team will confirm and the order goes through `request_quote`. The Products tab shows when prices were last updated. `orders.csv` records the price used at order time. Hand-set (pinned) prices and the owner's own items are never overwritten by an update.
**Note:** the prices in the scrape file `data/raw/site.json` can differ from the live page (about a quarter did on 2026-11-13), so the stored prices always come from the pages, never from that file.

## Behaviour notes
- Every customer message path: limits -> per-user debounce (`DEBOUNCE_SECONDS`) -> one agent turn. Admin numbers (`ADMIN_NUMBERS`) are routed in code to commands before any LLM; customers never reach orders/stats.
- Admin: plain words only, no typed commands. The assistant has tools to confirm/reject, price, discount, message or ask a customer, mute, send the orders/stats files, change products, set a rate, and an `edit_orders_with_python` tool (sandboxed, validated, with automatic backup + undo) for clearing, unconfirming and editing orders.
- Voice length is checked before transcribing: client `duration` field (the Cloud API gives none; mock UI sends it) and `ffprobe` if installed, larger wins; unknown and >25 MB counts as too long.
- Photos: `dl.predict.predict` first (stub if missing), then vision (`VISION_MODEL`) when unsure; >3 photos in a batch = no model, ask for up to three; `VISION_DAILY_CAP` per customer.
- Product ids in `graph.json` are the scraper's `id` (else `p<index>`); the DL model must return the same ids.

## Catalog upkeep, CRM, health
- `update_products` (owner: "we updated our products") and a nightly job (every night 02:00-05:00 Asia/Nicosia, `DAILY_REFRESH=0` turns it off) run `refresh_catalog.py`: wipe the scrape cache, re-scrape both sites, merge, rebuild the graph, refresh photos, update the matcher index (only new/changed photos). Customers get "maintenance" meanwhile and are re-texted after. A scrape under 70% of the old size is discarded; the owner gets a change report (new / removed / renamed).
- Owner's own items (`custom.json`, ids `c-N`, stored price) are merged into the graph at load, so no re-scrape touches them. `update_item` / `refresh_item` change or re-read ONE item by name or link (several matches: the owner picks).
- `customers.json` (CRM): name, relationship, notes, board stage. There are no standing discounts or free-delivery flags: delivery is free or paid by what the customer chose, and the owner changes one order's fee (0 = free) or the amount to pay when he confirms it (dashboard confirm / price screens, or `set_delivery_fee` in the admin chat).
- `/health`, alerts after 3 failures in a row (openai / whatsapp), nightly backup to `backups/` (14 days). Upcoming occasions (Valentine's, Mother's Day...) are added to the agent's prompt from the North Cyprus date (`core.upcoming_events`).

## Stubbed / not verified live
No real OpenAI or WhatsApp call was ever made (fakes only). The Meta adapter (send/document upload/template/media download) is untested against Meta. Language detection for canned replies is EN/TR only (agent replies follow the customer's language via the model). Templates need Meta approval. Real site data: rerun `build_graph.py` once `data/raw/site.json` exists.

## Simulation
In mock mode open `/simulate`: up to 10 customer windows (own number each) plus the admin window; "random orders from all" sends a random chat from every customer at once.
Catalog = kibrisciceksepetim.com only, read by `scraper/scrape_kibris.py` (its rules are in its docstring).

## Dashboard
`/dashboard`: orders, payments to check, CRM, accounts (English / Türkçe / Русский). Leader account admin@kibris (default password 1234: change it on first login); only the leader creates accounts. Every action that concerns a customer goes through the same code as the WhatsApp admin chat, so the customer gets the same WhatsApp message. Put it behind HTTPS before exposing it.

## Chats are not stored
The conversation with each customer lives in memory only (last 14 days, at most 200 messages; `HISTORY_DAYS`, `HISTORY_MAX`) and is forgotten on a restart; the server keeps orders, not chats (WhatsApp offers no way to read an old chat back, so the agent also gets the customer's past orders as context). The one exception: while the agent cannot answer (catalog refresh, maintenance, stopped from the dashboard or the admin chat) a customer's text is kept in `inbox.json` until it is answered or dropped. Those customers are told once that we are unavailable and may order on the website; when the agent is back the kept messages are answered as normal messages. After a stop, the owner is asked whether to answer them all or skip them. If the server is down, Meta redelivers the messages after the restart.

## Telegram
Works next to WhatsApp (`TELEGRAM_TOKEN`, `TELEGRAM_SECRET`, `PUBLIC_URL`, `ADMIN_TELEGRAM` in `.env`). A Telegram customer is the id `tg:<chat id>`: Telegram gives no phone number, so the agent asks for it. No templates and no 24-hour window: the bot can message anyone who started it. Messages arrive at `/telegram/webhook` (checked with the secret header); `/id` shows a chat id, `/start` greets. Replies are converted from WhatsApp-style `*bold*` to Telegram HTML; quotation and invoice PDFs, photos, voice notes, receipts and shared locations work the same way.

## Persistence
Orders and customers are kept on disk and survive any restart or power cut: `orders.csv` (written atomically, the previous version kept as `orders.csv.prev`, restored at the next start if the file is ever damaged), `chat.db` (customers, contact counts, settings; transactional), `customers.json` (CRM), `custom.json`, `overrides.json`, `dash_users.json`, `inbox.json`, `documents/` (quotation and invoice PDFs), `backups/` (daily, 14 days). Every chat message is stored forever in `chats/<customer>.jsonl` (one JSON document per message, appended and flushed to disk; a customer is a phone number or `tg:<id>`), backed up daily; the agent keeps the latest `HISTORY_MAX` (200) in mind, the owner assistant searches all of it (`chat_history`, `search_chats`). `./clean_state.sh` deletes all of it and now refuses without `--wipe-everything`.


Local WAHA testing: set `WAHA_URL`, `WAHA_API_KEY` and a random `WAHA_WEBHOOK_SECRET`. Run WAHA separately with `docker start waha`; the app only connects to its API. Start ngrok independently with `ngrok http 8000`. Pairing still needs a QR scan if WhatsApp has logged out. Configure the `default` WAHA session to deliver `message` events to `/waha/webhook` with a custom `X-Waha-Secret` header matching that secret. Direct text, photos, audio and document messages use the shared agent. Clear `WAHA_API_KEY` and restart to restore Meta. Dashboard sandbox processes clear WAHA credentials.

## Test through Meta's Cloud API before using the shop number

Create a Meta developer app and use its WhatsApp **test phone number**, test access token, and allowlisted recipient numbers. In `.env`, fill `META_TEST_ACCESS_TOKEN`, `META_TEST_PHONE_NUMBER_ID`, `META_TEST_RECIPIENTS` (comma-separated country-code digits), `META_TEST_ADMIN_NUMBER` (one of those recipients), `META_TEST_VERIFY_TOKEN`, and `META_TEST_APP_SECRET`. The test profile uses a separate `test-state/meta-cloud` store and blocks all outbound / inbound numbers outside the allowlist.

Start the app in its own terminal with `bash run-meta-test.sh`. Run `ngrok http 8000` separately, then configure the Meta webhook callback as `<ngrok HTTPS URL>/webhook`, use the test verify token, and subscribe to `messages`. Message the Meta test number from an allowlisted customer test phone to exercise real Cloud API webhooks, replies, and media. Do not put the shop's production token or phone-number ID in the `META_TEST_*` fields.
