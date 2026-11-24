"""Assemble a ready-to-upload Hugging Face Space folder.

    python scripts/build_space.py            # -> space/
    cd space && git init && git remote add origin https://huggingface.co/spaces/<you>/KeralaGAN
    git add . && git commit -m "KeralaGAN demo" && git push -u origin main
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space"

README = """---
title: KeralaGAN
emoji: 🎨
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# KeralaGAN

Repaint a photo as a Kerala mural, in monsoon weather, or at night.
A CycleGAN trained on unpaired images. Code: see the GitHub repository.
"""


def main():
    weights = sorted((ROOT / "weights").glob("*_G_AB.pt"))
    if not weights:
        raise SystemExit("No exported weights in weights/. Run scripts/export_generator.py first.")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "weights").mkdir(parents=True)
    shutil.copy(ROOT / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "keralagan", OUT / "keralagan", ignore=shutil.ignore_patterns("__pycache__"))
    for w in weights:
        shutil.copy(w, OUT / "weights" / w.name)
    if (ROOT / "examples").is_dir():
        shutil.copytree(ROOT / "examples", OUT / "examples")
    (OUT / "requirements.txt").write_text("torch\ntorchvision\npillow\nnumpy\n")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Space ready in {OUT} with {len(weights)} model(s): {[w.name for w in weights]}")


if __name__ == "__main__":
    main()
