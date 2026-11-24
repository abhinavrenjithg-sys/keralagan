"""Fast CPU tests: shapes, image pool, one training step, export and translate.

    pip install pytest && pytest -q
"""

import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from keralagan.models import Discriminator, Generator, ImagePool  # noqa: E402
from keralagan.translate import load_generator, translate_image  # noqa: E402


def test_generator_shape():
    G = Generator(n_blocks=2)
    x = torch.randn(2, 3, 64, 64)
    y = G(x)
    assert y.shape == x.shape
    assert y.min() >= -1 and y.max() <= 1


def test_generator_non_square():
    G = Generator(n_blocks=2)
    assert G(torch.randn(1, 3, 64, 96)).shape == (1, 3, 64, 96)


def test_discriminator_patch_grid():
    D = Discriminator()
    out = D(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 1, 30, 30)  # 30x30 grid of 70x70 patch scores


def test_image_pool():
    pool = ImagePool(4)
    for _ in range(10):
        out = pool.query(torch.randn(2, 3, 8, 8))
        assert out.shape == (2, 3, 8, 8)
    assert len(pool.images) == 4


def test_train_export_translate(tmp_path):
    data = tmp_path / "toy"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_toy_dataset.py"), "--out", str(data), "--n", "6"],
                   check=True)
    subprocess.run([sys.executable, "-m", "keralagan.train", "--data", str(data), "--out", str(tmp_path),
                    "--epochs", "2", "--size", "64", "--n_blocks", "2", "--workers", "0", "--max_steps", "3"],
                   check=True, cwd=ROOT)
    ckpt = tmp_path / "checkpoints" / "toy" / "latest.pt"
    assert ckpt.exists()
    assert (tmp_path / "samples" / "toy" / "epoch_002.png").exists()

    # resume continues from the saved epoch
    subprocess.run([sys.executable, "-m", "keralagan.train", "--data", str(data), "--out", str(tmp_path),
                    "--epochs", "3", "--size", "64", "--n_blocks", "2", "--workers", "0", "--max_steps", "2",
                    "--resume"], check=True, cwd=ROOT)
    assert torch.load(ckpt, map_location="cpu")["epoch"] == 2

    exported = tmp_path / "weights" / "toy_G_AB.pt"
    subprocess.run([sys.executable, str(ROOT / "scripts/export_generator.py"), "--ckpt", str(ckpt),
                    "--out", str(exported)], check=True)
    G, cfg = load_generator(exported)
    assert cfg["n_blocks"] == 2
    out = translate_image(G, Image.new("RGB", (130, 90), (120, 160, 210)), size=64)
    assert out.size[1] == 64 and out.size[0] % 4 == 0
