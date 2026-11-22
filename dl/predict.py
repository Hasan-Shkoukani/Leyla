"""predict(image_bytes) -> {"matches": [{"product_id", "score"}...top5], "confident": bool}
Degrades to {"matches": [], "confident": False} if torch/open_clip/model/index is missing.
CLI: python -m dl.predict image.jpg
"""
import io, json, sys, time
from dl import config

_state = {}  # cached: {"embed", "pids", "embs"}  or {"failed": True}


def _thresholds():
    t = {"confident_score": config.CONFIDENT_SCORE, "margin": config.MARGIN}
    try:
        t.update(json.loads(config.THRESHOLDS_PATH.read_text()))
    except Exception:
        pass
    return t


def _load():
    if _state:
        return _state
    try:
        import numpy as np
        from dl.encoder import load_encoder
        z = np.load(config.INDEX_PATH)
        embed, _ = load_encoder()
        _state.update(embed=embed, zero_shot=embed.zero_shot, pids=z["product_ids"], embs=z["embeddings"])
    except Exception as e:  # missing torch / weights / index
        print(f"dl.predict: model unavailable ({type(e).__name__}: {e})", file=sys.stderr)
        _state["failed"] = True
    return _state


def _log(res):
    try:
        config.ARTIFACTS.mkdir(exist_ok=True)
        top = ";".join(f"{m['product_id']}:{m['score']:.3f}" for m in res["matches"][:3])
        with open(config.LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{top}\t{res['confident']}\n")
    except Exception:
        pass


def rank(sims, pids, k=config.TOP_K):
    """Max similarity per product (a product may have several photos), top-k, sorted desc."""
    best = {}
    for pid, s in zip(pids, sims):
        pid = str(pid)
        if s > best.get(pid, -2):
            best[pid] = float(s)
    return [{"product_id": p, "score": s} for p, s in sorted(best.items(), key=lambda x: -x[1])[:k]]


def predict(image_bytes: bytes) -> dict:
    res = {"matches": [], "confident": False}
    try:
        st = _load()
        if not st.get("failed"):
            from PIL import Image
            im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            q = st["embed"]([im])[0]
            m = rank(st["embs"] @ q, st["pids"])
            t = _thresholds()
            conf = bool(m) and m[0]["score"] >= t["confident_score"] and \
                (len(m) < 2 or m[0]["score"] - m[1]["score"] >= t["margin"])
            res = {"matches": m, "confident": conf}
    except Exception as e:
        print(f"dl.predict error: {e}", file=sys.stderr)
    _log(res)
    return res


# What a flower shop customer may send: everything the shop sells (flowers, roses, lilies, daisies, plants, wreaths, fruit baskets, chocolate, teddy bears, cakes, vases) and anything that can matter to
# an order (a person holding a bouquet, a map, a chat or transfer screenshot, an address note). Only a photo that is clearly none of these is dropped.
RELEVANT = ["a photo of flowers", "a bouquet of roses", "a flower arrangement in a vase", "a potted plant", "a wreath of flowers", "a basket of fruit", "a box of chocolates",
            "a gift box", "a teddy bear gift", "a bunch of balloons", "a cake", "a person holding a bouquet of flowers", "a woman holding flowers", "a flower in someone's hand",
            "a garden with flowers", "a glass vase or aquarium with flowers", "a wedding or party decoration with flowers", "a florist shop",
            "a screenshot of a map", "a screenshot of a phone chat", "a receipt or bank transfer screenshot", "a handwritten note with an address"]
IRRELEVANT = ["a photo of an exam paper with questions", "a photo of a page of printed text", "a page of a textbook or lecture slides", "a photo of a person's face, a selfie without flowers",
              "a photo of a car", "a photo of food on a plate", "a meme", "a photo of a landscape without flowers", "a photo of a pet", "a screenshot of computer code",
              "a photo of a classroom", "a photo of a laptop or a computer screen", "a photo of a street", "a photo of a document or an ID card"]


def relevance(image_bytes: bytes, min_prob=0.9, max_relevant=0.08) -> dict:
    """Is this photo about the shop at all? Local zero-shot CLIP, no network, no model call. {"irrelevant": True} only when it is clearly something else (a page, an exam, a car...):
    the best description must be an unrelated one with at least `min_prob`, AND all the shop-related descriptions together must be below `max_relevant`, so a person holding a flower,
    a plant in the background or anything unclear stays relevant. Degrades to {"irrelevant": False} if the model is missing."""
    try:
        st = _load()
        if st.get("failed"):
            return {"irrelevant": False}
        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        labels = RELEVANT + IRRELEVANT
        pr = st["zero_shot"](im, labels)
        i = max(range(len(pr)), key=pr.__getitem__)
        mass = sum(pr[:len(RELEVANT)])
        return {"irrelevant": i >= len(RELEVANT) and pr[i] >= min_prob and mass <= max_relevant, "label": labels[i], "prob": round(pr[i], 3), "shop_mass": round(mass, 3)}
    except Exception as e:
        print(f"dl.relevance error: {e}", file=sys.stderr)
        return {"irrelevant": False}


if __name__ == "__main__":
    print(json.dumps(predict(open(sys.argv[1], "rb").read()), indent=2))
