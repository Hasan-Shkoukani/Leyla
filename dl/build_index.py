"""Embed catalog photos -> artifacts/index.npz (product_ids, embeddings) + meta.json.
Usage: python -m dl.build_index [--head dl/artifacts/head.pt] [--with-customer]
--with-customer also adds labeled customer photos to the index (do NOT use when evaluating on them).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from dl import config, data
from dl.encoder import load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None, help="path to fine-tuned head (default: artifacts/head.pt if exists)")
    ap.add_argument("--with-customer", action="store_true")
    ap.add_argument("--full", action="store_true", help="re-embed everything (default: only new/changed photos, drop deleted ones)")
    a = ap.parse_args()
    items = data.labeled(config.CATALOG_DIR) + (data.labeled(config.CUSTOMER_DIR) if a.with_customer else [])
    assert items, f"no images under {config.CATALOG_DIR}; run download_images first"
    keys = [f"{p.parts[-3]}/{p.parts[-2]}/{p.name}:{p.stat().st_size}" for p, _ in items]  # a replaced photo changes its size, so it is embedded again
    head = a.head or str(config.HEAD_PATH)
    sig = [config.MODEL_NAME, config.PRETRAINED, head, Path(head).stat().st_mtime_ns if Path(head).exists() else 0]
    old = {}
    if not a.full and config.INDEX_PATH.exists() and config.META_PATH.exists():
        z, m = np.load(config.INDEX_PATH), json.loads(config.META_PATH.read_text())
        if "keys" in z.files and m.get("sig") == sig:  # same model/head: old rows stay valid
            old = {str(k): e for k, e in zip(z["keys"], z["embeddings"])}
    todo = [i for i, k in enumerate(keys) if k not in old]
    dev = "cached"
    new = {}
    if todo:
        embed, dev = load_encoder(Path(a.head) if a.head else None)
        for i in range(0, len(todo), 64):  # ponytail: a bad image aborts; add try/except if catalog has corrupt files
            part = todo[i:i + 64]
            for j, e in zip(part, embed([data.load_rgb(items[j][0]) for j in part])):
                new[keys[j]] = e
    embs = np.stack([new[k] if k in new else old[k] for k in keys]).astype("float32")
    config.ARTIFACTS.mkdir(exist_ok=True)
    np.savez(config.INDEX_PATH, product_ids=np.array([pid for _, pid in items]), embeddings=embs, keys=np.array(keys))
    config.META_PATH.write_text(json.dumps({
        "model": config.MODEL_NAME, "pretrained": config.PRETRAINED, "head": head, "sig": sig,
        "n_images": len(items), "n_products": len({p for _, p in items}), "device": dev,
        "built": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print("indexed", len(items), "images,", len(todo), "embedded now")


if __name__ == "__main__":
    main()
