"""Single place for knobs. No heavy imports here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CATALOG_DIR = ROOT / "data" / "catalog"      # <product_id>/*.jpg (studio photos)
CUSTOMER_DIR = ROOT / "data" / "customer"    # <product_id>/*.jpg (real customer photos, labeled)
NEGATIVES_DIR = ROOT / "data" / "negatives"  # flat folder of irrelevant/other-florist images
SITE_JSON = ROOT.parent / "data" / "raw" / "site.json"

MODEL_NAME = "ViT-B-32"          # any open_clip model name
PRETRAINED = "laion2b_s34b_b79k"  # open_clip pretrained tag
HEAD_PATH = ARTIFACTS / "head.pt"      # optional fine-tuned projection head
INDEX_PATH = ARTIFACTS / "index.npz"
META_PATH = ARTIFACTS / "meta.json"
THRESHOLDS_PATH = ARTIFACTS / "thresholds.json"
LOG_PATH = ARTIFACTS / "predictions.log"

# conservative defaults used until evaluate.py writes thresholds.json
CONFIDENT_SCORE = 0.85
MARGIN = 0.05
TOP_K = 5
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
