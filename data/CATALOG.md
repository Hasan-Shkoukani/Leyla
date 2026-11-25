# Local catalog format

Use `site.sample.json` as your starting point. `catalog.schema.json` defines the JSON format; `graph.json` is generated for the agent.

```json
{
  "products": [
    {
      "id": "red-roses",
      "name": "Red Rose Bouquet",
      "names": {"en": "Red Rose Bouquet", "tr": "Kırmızı Gül Buketi"},
      "category": "roses",
      "colors": ["red"],
      "fixed_price": 800,
      "currency": "TRY",
      "same_day": true,
      "description": "A red rose bouquet."
    }
  ],
  "business": {"name": "My Demo Shop"}
}
```

- Give every product a unique, stable `id`. If supplied, `product_id` must equal `id`.
- `fixed_price` is the stored price in TL. It must be positive; prices are never fetched from a website.
- Optional `names` are your own product names in English, Turkish and Russian.
- Put photos in `dl/data/catalog/<id>/`, for example `dl/data/catalog/red-roses/0.jpg`. JPEG, PNG, WebP and AVIF are supported. Image URLs are not used.
- Omit `images` or leave it empty. Do not add product URLs; catalog entries and customer replies are local and link-free.
- Run `python build_graph.py data/site.sample.json` after editing the catalog. For optional CLIP matching, install `dl/requirements.txt` and run `python -m dl.build_index` after adding photos.
- Configure delivery rules, payment details and the assistant persona in their separate `data/` files. Placeholder payment details are for testing only.

The scraper file is intentionally empty. The public demo contains no shop photos or customer data; supply images you are allowed to use.
