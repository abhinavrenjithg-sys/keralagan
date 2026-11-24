"""Crop large mural photos into many square training patches.

One big mural photo can give 10-40 useful 256x256 crops, which is how a few dozen
mural photos become a few hundred training images.

    python scripts/make_patches.py --src raw/murals_full --dst raw/mural_patches --size 384 --stride 256
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=384, help="patch side in pixels (before training resize)")
    ap.add_argument("--stride", type=int, default=256, help="step between patches; < size means overlap")
    ap.add_argument("--scales", default="1.0,0.7", help="also cut patches from downscaled copies")
    ap.add_argument("--min_std", type=float, default=18.0, help="skip flat patches (plain wall, sky, borders)")
    args = ap.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    total = kept = 0
    for p in sorted(Path(args.src).rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        base = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        n = 0
        for sc in scales:
            img = base if sc == 1.0 else base.resize((int(base.width * sc), int(base.height * sc)), Image.LANCZOS)
            for y in range(0, img.height - args.size + 1, args.stride):
                for x in range(0, img.width - args.size + 1, args.stride):
                    total += 1
                    crop = img.crop((x, y, x + args.size, y + args.size))
                    if np.asarray(crop.convert("L"), dtype=np.float32).std() < args.min_std:
                        continue
                    crop.save(dst / f"{p.stem}_s{int(sc * 100)}_{y}_{x}.jpg", quality=92)
                    kept += 1
                    n += 1
        print(f"{p.name}: {n} patches")
    print(f"Kept {kept}/{total} patches -> {dst}")


if __name__ == "__main__":
    main()
