#!/bin/sh
[ "$1" = "--wipe-everything" ] || { echo "REFUSED: this deletes the REAL orders, customers and chats. If you really want that: ./clean_state.sh --wipe-everything"; exit 1; }
# wipe ALL test state: chats and the agent's memory (chat.db), orders and their backups, stats, price overrides, outbox, logs. Keeps code, .env, data/ and the matcher index/thresholds.
cd "$(dirname "$0")" && rm -rf dl/data/catalog/c-* state backups order_backups incoming inbox.json customers.json custom.json admin_photo.jpg chat.db orders.csv orders.csv.prev orders_today.csv active_orders.csv orders_archive_*.csv daily.csv weekly.csv monthly.csv overrides.json outbox server.log photo_decisions.log dl/artifacts/predictions.log && echo "test state cleaned"
