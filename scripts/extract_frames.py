"""Pull still frames out of your own videos (e.g. rainy-day or night clips) with ffmpeg.

    python scripts/extract_frames.py --src raw/videos/monsoon --dst raw/monsoon --every 2
"""

import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of videos")
    ap.add_argument("--dst", required=True, help="output folder for frames")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    ap.add_argument("--max_side", type=int, default=720)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it (Colab has it; on Windows: winget install ffmpeg).")
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    vids = [p for p in sorted(Path(args.src).rglob("*")) if p.suffix.lower() in VIDEO_EXTS]
    for v in vids:
        pattern = dst / f"{v.stem}_%04d.jpg"
        vf = f"fps=1/{args.every},scale='if(gt(iw,ih),{args.max_side},-2)':'if(gt(iw,ih),-2,{args.max_side})'"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(v), "-vf", vf, "-q:v", "2", str(pattern)],
                       check=True)
        print(f"{v.name}: {len(list(dst.glob(v.stem + '_*.jpg')))} frames")
    print(f"Done: {len(vids)} video(s) -> {dst}")


if __name__ == "__main__":
    main()
