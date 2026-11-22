#!/bin/sh
# Moves the shop's STATE (orders, chats, customers, prices, outbox, backups...) out of the code folder into ./state. Run it ONCE, with the server STOPPED. Safe to run again.
cd "$(dirname "$0")" || exit 1
pgrep -f "uvicorn app:app" >/dev/null && { echo "REFUSED: the server is running. Stop it first, run this, then start it again."; exit 1; }
[ -d state ] && { echo "state/ already exists: nothing to do"; exit 0; }
mkdir state || exit 1
for f in orders.csv orders.csv.prev orders_today.csv active_orders.csv customers.json custom.json overrides.json removed.json prices.json prices_cache.json chat.db daily.csv weekly.csv monthly.csv \
         dash_users.json dash_sessions.json inbox.json tg_updates.json payments.jsonl metrics.jsonl photo_decisions.log admin_photo.jpg \
         chats documents inbox_media outbox reports team incoming order_backups backups; do
  [ -e "$f" ] && mv "$f" state/
done
echo "moved to ./state:"; ls state
