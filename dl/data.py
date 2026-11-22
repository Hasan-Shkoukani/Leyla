"""Dataset layout helpers.
dl/data/catalog/<product_id>/*.jpg   studio photos (from download_images.py)
dl/data/customer/<product_id>/*.jpg  real labeled customer photos (added later)
dl/data/negatives/*.jpg              irrelevant images (for false-confident test)
"""
from pathlib import Path
from dl import config


def list_images(folder: Path):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in config.IMG_EXTS)


def labeled(folder: Path):
    """[(path, product_id)] where product_id = parent folder name."""
    return [(p, p.parent.name) for p in list_images(folder)]


def load_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")
