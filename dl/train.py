"""Optional light fine-tune: frozen CLIP backbone + trainable linear projection head,
supervised-contrastive loss over heavily augmented views (simulates messy customer photos).
Usage: python -m dl.train --epochs 10 --batch 16 [--resume]
Saves artifacts/head.pt every epoch (resume-able). Then rerun build_index.
"""
import argparse, random
from pathlib import Path
from dl import config, data

CKPT = config.ARTIFACTS / "train_ckpt.pt"


def screenshot_border(im):
    """Paste on a UI-like canvas: solid/noisy background + bars, like a phone screenshot."""
    from PIL import Image, ImageDraw
    if random.random() > 0.3:
        return im
    w, h = im.size
    bw, bh = int(w * random.uniform(0, .3)), int(h * random.uniform(.05, .35))
    col = tuple(random.randint(0, 255) for _ in range(3))
    canvas = Image.new("RGB", (w + 2 * bw, h + 2 * bh), col)
    canvas.paste(im, (bw, random.randint(0, 2 * bh)))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, canvas.width, bh // 3], fill=tuple(random.randint(0, 255) for _ in range(3)))
    return canvas


def make_aug():
    import torchvision.transforms as T
    return T.Compose([
        T.Lambda(screenshot_border),
        T.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.6, 1.6)),
        T.RandomRotation(25, fill=(127, 127, 127)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        T.RandomGrayscale(0.05),
    ])


def supcon(z, y, t=0.1):
    import torch
    sim = z @ z.T / t
    eye = torch.eye(len(y), device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    n = pos.sum(1)
    ok = n > 0
    return -((logp * pos).sum(1)[ok] / n[ok]).mean()


def main():
    import torch, open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=10, help="passes over the image list per epoch (few images/product)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = data.labeled(config.CATALOG_DIR) + data.labeled(config.CUSTOMER_DIR)
    # ponytail: if you evaluate on data/customer, move some of it out of training first (see README)
    assert items, "no training images"
    pids = sorted({p for _, p in items}); cls = {p: i for i, p in enumerate(pids)}
    model, _, pre = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad = False
    dim = model.encode_image(torch.zeros(1, 3, 224, 224, device=dev)).shape[-1]
    head = torch.nn.Linear(dim, dim).to(dev)
    with torch.no_grad():
        head.weight.copy_(torch.eye(dim)); head.bias.zero_()  # start = base model
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    ep0 = 0
    if a.resume and CKPT.exists():
        ck = torch.load(CKPT, map_location=dev)
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"]); ep0 = ck["epoch"]
        print("resumed at epoch", ep0)
    aug = make_aug()
    config.ARTIFACTS.mkdir(exist_ok=True)
    cache = {}
    for ep in range(ep0, a.epochs):
        order = items * a.repeat; random.shuffle(order)
        tot = n = 0
        for i in range(0, len(order) - a.batch + 1, a.batch):
            batch = order[i:i + a.batch]
            imgs, ys = [], []
            for p, pid in batch:
                im = cache.setdefault(p, data.load_rgb(p)) if len(cache) < 2000 else data.load_rgb(p)
                for _ in range(2):
                    imgs.append(pre(aug(im))); ys.append(cls[pid])
            x = torch.stack(imgs).to(dev); y = torch.tensor(ys, device=dev)
            with torch.no_grad():
                f = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
            z = torch.nn.functional.normalize(head(f), dim=-1)
            loss = supcon(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / max(n, 1):.4f}")
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, CKPT)
        torch.save({"head": head.state_dict(), "dim": dim}, config.HEAD_PATH)
    print("saved", config.HEAD_PATH)


if __name__ == "__main__":
    main()
