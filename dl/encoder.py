"""Shared encoder loading/embedding. Heavy imports are lazy (inside functions)."""
from dl import config


def load_encoder(head_path=None, device=None):
    """Returns (embed_fn, device). embed_fn(list[PIL.Image]) -> np.ndarray (n, d), L2-normalised."""
    import numpy as np, torch, open_clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED)
    model.eval().to(device)
    head = None
    hp = head_path if head_path is not None else config.HEAD_PATH
    if hp and hp.exists():
        ck = torch.load(hp, map_location=device)
        head = torch.nn.Linear(ck["dim"], ck["dim"]).to(device)
        head.load_state_dict(ck["head"]); head.eval()

    @torch.no_grad()
    def embed(images, bs=32):
        out = []
        for i in range(0, len(images), bs):
            x = torch.stack([preprocess(im) for im in images[i:i + bs]]).to(device)
            f = model.encode_image(x).float()
            f = torch.nn.functional.normalize(f, dim=-1)
            if head is not None:
                f = torch.nn.functional.normalize(head(f), dim=-1)
            out.append(f.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), "float32")

    text_cache = {}  # the labels never change between photos: their text features (and the tokenizer) are computed once, not on every photo (that was 0.9 s of every relevance check)

    @torch.no_grad()
    def zero_shot(image, labels):
        """Plain CLIP (no fine-tuned head): which text label describes the image best. Returns the probability of each label."""
        x = preprocess(image).unsqueeze(0).to(device)
        f = torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1)
        key = tuple(labels)
        if key not in text_cache:
            text_cache[key] = torch.nn.functional.normalize(model.encode_text(open_clip.get_tokenizer(config.MODEL_NAME)(list(labels)).to(device)).float(), dim=-1)
        return (100.0 * f @ text_cache[key].T).softmax(dim=-1)[0].cpu().tolist()

    embed.zero_shot = zero_shot
    return embed, device
