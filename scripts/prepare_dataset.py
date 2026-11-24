"""Build a CycleGAN dataset from two raw folders.

Cleans, de-duplicates, resizes and splits images into trainA/testA/trainB/testB.

    python scripts/prepare_dataset.py --a raw/summer --b raw/monsoon --name monsoon
    python scripts/prepare_dataset.py --a raw/people --b raw/mural_patches --name mural --test_frac 0.1
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def process(paths, short_side, min_size, seen):
    """Yield (stem, PIL image) for usable, unique images."""
    skipped = {"small": 0, "dup": 0, "broken": 0}
    for p in paths:
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception:
            skipped["broken"] += 1
            continue
        if min(img.size) < min_size:
            skipped["small"] += 1
            continue
        small = img.resize((16, 16), Image.BILINEAR).convert("L")
        digest = hashlib.md5(small.tobytes()).hexdigest()  # catches exact and near-exact copies
        if digest in seen:
            skipped["dup"] += 1
            continue
        seen.add(digest)
        w, h = img.size
        s = short_side / min(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        yield p.stem, img
    print(f"    skipped: {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="raw folder for domain A (e.g. photos)")
    ap.add_argument("--b", required=True, help="raw folder for domain B (e.g. murals)")
    ap.add_argument("--name", required=True, help="dataset name -> data/<name>/")
    ap.add_argument("--root", default="data")
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--short_side", type=int, default=320, help="downscale so the short side is this (keeps files small)")
    ap.add_argument("--min_size", type=int, default=256, help="drop images smaller than this")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.root) / args.name
    seen = set()
    for dom, src in (("A", args.a), ("B", args.b)):
        paths = collect(src)
        print(f"[{dom}] {len(paths)} files in {src}")
        if not paths:
            sys.exit(f"No images in {src}")
        items = list(process(paths, args.short_side, args.min_size, seen))
        random.shuffle(items)
        n_test = max(1, int(len(items) * args.test_frac))
        for phase, chunk in (("test", items[:n_test]), ("train", items[n_test:])):
            d = out / f"{phase}{dom}"
            d.mkdir(parents=True, exist_ok=True)
            for k, (stem, img) in enumerate(chunk):
                img.save(d / f"{dom}_{k:05d}_{stem[:40]}.jpg", quality=92)
            print(f"    {phase}{dom}: {len(chunk)} images")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
