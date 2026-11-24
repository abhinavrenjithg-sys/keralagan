"""Strip a training checkpoint down to one generator (fp16) for the demo app.

A full checkpoint holds 2 generators, 2 discriminators and optimiser state (~300 MB).
The exported file holds only what inference needs (~22 MB).

    python scripts/export_generator.py --ckpt checkpoints/mural/latest.pt --out weights/mural_G_AB.pt
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    ap.add_argument("--fp32", action="store_true", help="keep full precision")
    args = ap.parse_args()

    s = torch.load(args.ckpt, map_location="cpu")
    state = s["G_AB" if args.direction == "AtoB" else "G_BA"]
    if not args.fp32:
        state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    cfg = dict(s.get("config", {}), direction=args.direction, epoch=s.get("epoch", -1) + 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"generator": state, "config": cfg}, args.out)
    print(f"Exported {args.direction} generator ({os.path.getsize(args.out) / 1e6:.1f} MB) -> {args.out}")


if __name__ == "__main__":
    main()
