"""Evaluate retrieval + choose thresholds.
Queries: dl/data/customer/<pid>/*.jpg (held out: NOT in the index; build_index without --with-customer).
Negatives: dl/data/negatives/*.jpg. Writes artifacts/thresholds.json.
Usage: python -m dl.evaluate [--max-false-confident 0.02] [--max-wrong-confident 0.05]
"""
import argparse, json
from pathlib import Path
import numpy as np
from dl import config, data
from dl.encoder import load_encoder
from dl.predict import rank


def top(embed, imgs_paths, embs, pids):
    out = []
    for i in range(0, len(imgs_paths), 64):
        q = embed([data.load_rgb(p) for p in imgs_paths[i:i + 64]])
        out += [rank(embs @ v, pids) for v in q]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-false-confident", type=float, default=0.02, help="on negatives")
    ap.add_argument("--max-wrong-confident", type=float, default=0.05, help="on labeled queries")
    a = ap.parse_args()
    z = np.load(config.INDEX_PATH); embs, pids = z["embeddings"], z["product_ids"]
    embed, _ = load_encoder()
    qs = data.labeled(config.CUSTOMER_DIR)
    neg = data.list_images(config.NEGATIVES_DIR) if config.NEGATIVES_DIR.exists() else []
    assert qs, f"put labeled held-out photos in {config.CUSTOMER_DIR}/<product_id>/"
    R = top(embed, [p for p, _ in qs], embs, pids); y = [l for _, l in qs]
    t1 = np.mean([r[0]["product_id"] == l for r, l in zip(R, y)])
    t3 = np.mean([l in [m["product_id"] for m in r[:3]] for r, l in zip(R, y)])
    print(f"queries={len(qs)} top1={t1:.3f} top3={t3:.3f}  negatives={len(neg)}")
    N = top(embed, neg, embs, pids) if neg else []

    def stats(r):  # (top score, margin)
        return r[0]["score"], r[0]["score"] - (r[1]["score"] if len(r) > 1 else 0)
    P = [(*stats(r), r[0]["product_id"] == l) for r, l in zip(R, y)]
    Ns = [stats(r) for r in N]
    best = None
    print("score  margin  coverage  wrong_conf  false_conf_neg")
    for s in np.arange(0.5, 1.0, 0.025):
        for m in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
            conf = [(c, ok) for sc, mg, ok in P if (c := sc >= s and mg >= m)]
            cov = np.mean([sc >= s and mg >= m and ok for sc, mg, ok in P])
            wrong = np.mean([sc >= s and mg >= m and not ok for sc, mg, ok in P])
            fc = np.mean([sc >= s and mg >= m for sc, mg in Ns]) if Ns else 0.0
            if wrong <= a.max_wrong_confident and fc <= a.max_false_confident and (best is None or cov > best[0]):
                best = (cov, float(s), m, wrong, fc)
            if abs(s * 40 % 4) < 1e-6 and m in (0.0, 0.03):  # print a coarse grid only
                print(f"{s:.3f}  {m:.2f}    {cov:.3f}     {wrong:.3f}       {fc:.3f}")
    if best is None:
        print("No threshold meets the limits; keeping conservative defaults (0.9 / 0.08).")
        best = (0, 0.9, 0.08, 0, 0)
    cov, s, m, wrong, fc = best
    print(f"RECOMMENDED confident_score={s:.3f} margin={m} (correct-confident coverage={cov:.3f}, wrong={wrong:.3f}, false-confident on negatives={fc:.3f})")
    if not neg:
        print("WARNING: no negatives; thresholds not checked against irrelevant images.")
    config.ARTIFACTS.mkdir(exist_ok=True)
    config.THRESHOLDS_PATH.write_text(json.dumps({"confident_score": s, "margin": m, "top1": float(t1), "top3": float(t3),
                                                  "n_queries": len(qs), "n_negatives": len(neg)}, indent=2))


if __name__ == "__main__":
    main()
