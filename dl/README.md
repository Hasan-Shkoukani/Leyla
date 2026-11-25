# Flower catalog image matcher (retrieval)

CLIP-style encoder -> embeddings of catalog photos -> cosine nearest neighbours. `predict()` returns top-5 products and `confident` (top score >= threshold AND top1-top2 >= margin).
Run everything from the repo root (the folder containing `dl/`) on your own machine/Colab (GPU helps, CPU works).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r dl/requirements.txt

# 1. Put photos in dl/data/catalog/<product_id>/, or import a local folder:
python -m dl.download_images --source /path/to/photos
# This command imports local files only; it never downloads images.

# 2. (optional) negatives: put 50+ irrelevant images (other florists, screenshots, random photos) in dl/data/negatives/
# 3. (optional, later) labeled customer photos: dl/data/customer/<product_id>/*.jpg
#    Split them: keep ~half out of training for evaluation (move the training half to e.g. dl/data/customer_train
#    and point config.CUSTOMER_DIR users accordingly, or just train before adding the eval photos).

# 4. (optional) light fine-tune of a projection head (CPU ok, small batches; --resume continues)
python -m dl.train --epochs 10 --batch 16
# 5. build the index (base model, or the fine-tuned head if dl/artifacts/head.pt exists)
python -m dl.build_index
# 6. evaluate + choose thresholds (needs dl/data/customer/ held-out photos; negatives recommended)
python -m dl.evaluate
# 7. try it
python -m dl.predict some_photo.jpg
```

Without step 4 the base CLIP is used (a fine one to start with). Without step 6 conservative default thresholds apply (0.85 / 0.05).

## Send back
Copy these from `dl/artifacts/`: `index.npz`, `meta.json`, `thresholds.json`, `head.pt` (if trained), and later `predictions.log` (every predict call is logged: time, top-3, confident), plus the printed output of `dl.evaluate`.
