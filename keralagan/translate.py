"""Translate images with a trained generator.

    # a folder or single image, with a full training checkpoint
    python -m keralagan.translate --ckpt checkpoints/mural/latest.pt --input photos/ --out results/mural

    # with an exported generator (from scripts/export_generator.py)
    python -m keralagan.translate --ckpt weights/mural_G_AB.pt --input me.jpg --out results/
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .data import IMG_EXTS
from .models import Generator


def load_generator(ckpt_path, direction="AtoB", device="cpu"):
    """Accepts a full training checkpoint or an exported single-generator file."""
    s = torch.load(ckpt_path, map_location=device)
    if "generator" in s:  # exported file
        state, cfg = s["generator"], s.get("config", {})
    else:
        state, cfg = s["G_AB" if direction == "AtoB" else "G_BA"], s.get("config", {})
    state = {k: v.float() for k, v in state.items()}
    G = Generator(n_blocks=cfg.get("n_blocks", 9)).to(device)
    G.load_state_dict(state)
    G.eval()
    return G, cfg


@torch.no_grad()
def translate_image(G, img: Image.Image, size=256, max_side=1024, device="cpu") -> Image.Image:
    """Run G on a PIL image. Short side is resized to `size` (long side capped at max_side),
    then rounded to a multiple of 4 so the down/up-sampling shapes line up."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    if max(w, h) * scale > max_side:
        scale = max_side / max(w, h)
    nw, nh = max(4, round(w * scale / 4) * 4), max(4, round(h * scale / 4) * 4)
    img = img.resize((nw, nh), Image.BICUBIC)
    x = (TF.to_tensor(img) * 2 - 1).unsqueeze(0).to(device)
    y = G(x)[0].clamp(-1, 1) * 0.5 + 0.5
    arr = (y.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Translate images with KeralaGAN")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    ap.add_argument("--size", type=int, default=None, help="short side for inference (default: training size)")
    ap.add_argument("--side_by_side", action="store_true", help="save input and output next to each other")
    args = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G, cfg = load_generator(args.ckpt, args.direction, dev)
    size = args.size or cfg.get("size", 256)
    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    os.makedirs(args.out, exist_ok=True)
    for p in files:
        inp = Image.open(p).convert("RGB")
        out = translate_image(G, inp, size=size, device=dev)
        if args.side_by_side:
            inp = inp.resize(out.size, Image.BICUBIC)
            canvas = Image.new("RGB", (out.width * 2, out.height))
            canvas.paste(inp, (0, 0)); canvas.paste(out, (out.width, 0))
            out = canvas
        out.save(os.path.join(args.out, p.stem + ".png"))
    print(f"[KeralaGAN] translated {len(files)} image(s) -> {args.out}")


if __name__ == "__main__":
    main()
