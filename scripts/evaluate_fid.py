"""Translate the test set and score it with FID (lower = closer to the real target domain).

Also writes a before/after grid for the README.

    pip install pytorch-fid
    python scripts/evaluate_fid.py --data data/mural --ckpt checkpoints/mural/latest.pt
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keralagan.data import list_images  # noqa: E402
from keralagan.translate import load_generator, translate_image  # noqa: E402


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """FID formula, written here so it works with every SciPy version."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    covmean = np.real(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def center_square(img, size):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grid", default=None, help="path for a before/after grid PNG (default: results/<name>_grid.png)")
    ap.add_argument("--grid_n", type=int, default=6)
    args = ap.parse_args()

    try:
        from pytorch_fid.fid_score import compute_statistics_of_path
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        sys.exit("pytorch-fid is not installed: pip install pytorch-fid")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src, tgt = ("testA", "testB") if args.direction == "AtoB" else ("testB", "testA")
    src_files = list_images(os.path.join(args.data, src))
    tgt_files = list_images(os.path.join(args.data, tgt))
    if not src_files or not tgt_files:
        sys.exit("Test folders are empty.")
    G, _ = load_generator(args.ckpt, args.direction, dev)
    name = os.path.basename(os.path.normpath(args.data))

    with tempfile.TemporaryDirectory() as fake_dir, tempfile.TemporaryDirectory() as real_dir:
        pairs = []
        for i, p in enumerate(src_files):
            inp = center_square(Image.open(p).convert("RGB"), args.size)
            out = translate_image(G, inp, size=args.size, device=dev)
            out.save(os.path.join(fake_dir, f"{i:05d}.png"))
            if i < args.grid_n:
                pairs.append((inp, out))
        for i, p in enumerate(tgt_files):
            center_square(Image.open(p).convert("RGB"), args.size).save(os.path.join(real_dir, f"{i:05d}.png"))
        inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        bs = max(1, min(16, len(src_files), len(tgt_files)))
        m1, s1 = compute_statistics_of_path(fake_dir, inception, bs, 2048, dev, 0)
        m2, s2 = compute_statistics_of_path(real_dir, inception, bs, 2048, dev, 0)
        fid = frechet_distance(m1, s1, m2, s2)

    grid_path = args.grid or os.path.join("results", f"{name}_grid.png")
    os.makedirs(os.path.dirname(grid_path) or ".", exist_ok=True)
    s = args.size
    grid = Image.new("RGB", (s * len(pairs), s * 2), "white")
    for k, (a, b) in enumerate(pairs):
        grid.paste(a, (k * s, 0)); grid.paste(b, (k * s, s))
    grid.save(grid_path)

    result = {"dataset": name, "direction": args.direction, "fid": round(float(fid), 2),
              "n_translated": len(src_files), "n_real": len(tgt_files)}
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{name}_fid.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Grid (top: input, bottom: output) -> {grid_path}")
    if len(src_files) < 50 or len(tgt_files) < 50:
        print("Note: FID is noisy with fewer than ~50 images per side; report it with that caveat.")


if __name__ == "__main__":
    main()
