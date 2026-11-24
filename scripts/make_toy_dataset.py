"""Generate a tiny synthetic 'day -> night' dataset to check the pipeline end to end
before you have collected real images. Domain A: bright skies with a sun and hills.
Domain B: dark skies with a moon. Nothing here is real data.

    python scripts/make_toy_dataset.py --out data/toy --n 120
    python -m keralagan.train --data data/toy --epochs 5 --size 128 --n_blocks 6
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def scene(night: bool, rng: random.Random, size=160):
    img = Image.new("RGB", (size, size))
    d = ImageDraw.Draw(img)
    top = (10, 18, 48) if night else (110, 170, 225)
    bot = (35, 50, 100) if night else (225, 238, 245)
    for y in range(size):
        t = y / size
        d.line([(0, y), (size, y)], fill=tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3)))
    cx, cy, r = rng.randint(20, size - 20), rng.randint(15, 60), rng.randint(9, 16)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(240, 236, 210) if night else (255, 238, 170))
    if night:
        for _ in range(30):
            x, y = rng.randint(0, size), rng.randint(0, size // 2)
            d.point((x, y), fill=(255, 255, 255))
    horizon = rng.randint(int(size * .55), int(size * .7))
    pts = [(0, size)] + [(x, horizon + rng.randint(-14, 8)) for x in range(0, size + 20, 20)] + [(size, size)]
    d.polygon(pts, fill=(18, 30, 50) if night else (80, 140, 80))
    for _ in range(rng.randint(2, 4)):  # palm-ish trees
        x = rng.randint(5, size - 5)
        h = rng.randint(25, 45)
        col = (8, 12, 20) if night else (60, 40, 25)
        d.line([(x, horizon + 10), (x + 3, horizon + 10 - h)], fill=col, width=3)
        leaf = (10, 18, 30) if night else (40, 110, 55)
        for dx in (-12, -6, 6, 12):
            d.line([(x + 3, horizon + 10 - h), (x + 3 + dx, horizon + 18 - h)], fill=leaf, width=3)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    for dom, night in (("A", False), ("B", True)):
        for phase, n in (("train", args.n), ("test", max(4, args.n // 10))):
            d = out / f"{phase}{dom}"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                scene(night, rng).save(d / f"{i:04d}.jpg", quality=92)
    print(f"Toy dataset -> {out}")


if __name__ == "__main__":
    main()
