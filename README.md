# Leyla 🌸

An open-source flower shop assistant with a web dashboard, customer chat simulator, optional WhatsApp Meta Cloud API, and photo-based product matching.

- Helps customers choose products and collects delivery details.
- Tracks baskets, custom quotes, orders, payment receipts, and customer contacts.
- Lets staff approve, reject, reopen, and manage orders through chat or the dashboard.
- Supports multilingual conversations and generates order PDFs.
- Uses a local CLIP image index, with optional OpenAI vision fallback.

## Run locally

Requires Python 3.11+ and an OpenAI API key for conversations.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY and a model available to your account in .env.
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard`. The demo login is **admin / admin123**. This shared default is for local testing; change it before exposing the app to a network. With WhatsApp credentials empty, messages stay in the local simulator. Catalog data and photos are local; conversational AI still uses the OpenAI API and may incur charges. Customer replies contain no external links, and online card checkout is disabled.

## Configure your shop

The public release includes fictional demo products and placeholder payment details. Replace `data/graph.json`, `data/business_override.json`, `data/payment.json`, `data/delivery.json`, and `data/persona.md` with your own configuration before taking real orders. Delivery rules currently use a North Cyprus example.

Use the local catalog format in [data/CATALOG.md](data/CATALOG.md) and [data/catalog.schema.json](data/catalog.schema.json). Edit `data/site.sample.json`, then run `python build_graph.py data/site.sample.json`. The scraper file is intentionally empty; there are no scheduled website refresh jobs.

Photo matching is optional: install `dl/requirements.txt`, put your own permitted images in `dl/data/catalog/<product_id>/`, then run `python -m dl.build_index`. Alternatively, import a local folder with `python -m dl.download_images --source /path/to/photos`. Product photos are served from disk only; missing files never trigger a remote download. Model weights and shop photos are not included. WhatsApp uses Meta's official API; WAHA and Telegram are not included.

## License

MIT. Bundled third-party assets retain their own licenses in `LICENSES/` and `static/assets/flower/three/LICENSE`. No customer records, credentials, shop catalog, or private account files are included. Review your configuration, hosting security, and customer-data handling before real use.
