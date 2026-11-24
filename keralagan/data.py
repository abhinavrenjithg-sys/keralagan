"""Unpaired image dataset: <root>/{train,test}{A,B}/*.jpg"""

import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def build_transform(size: int, train: bool):
    ops = [transforms.Resize(int(size * 1.12), transforms.InterpolationMode.BICUBIC)]
    if train:
        ops += [transforms.RandomCrop(size), transforms.RandomHorizontalFlip()]
    else:
        ops += [transforms.CenterCrop(size)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    return transforms.Compose(ops)


def eval_transform(size: int):
    """Resize the short side to `size` and keep the aspect ratio (for inference on full photos)."""
    return transforms.Compose([
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])


class UnpairedDataset(Dataset):
    def __init__(self, root, phase: str = "train", size: int = 256):
        self.A = list_images(os.path.join(root, f"{phase}A"))
        self.B = list_images(os.path.join(root, f"{phase}B"))
        if not self.A or not self.B:
            raise FileNotFoundError(
                f"No images found in {root}/{phase}A or {root}/{phase}B. "
                "Run scripts/prepare_dataset.py first (see data/README.md)."
            )
        self.tf = build_transform(size, train=(phase == "train"))

    def __len__(self):
        return max(len(self.A), len(self.B))

    def __getitem__(self, i):
        a = Image.open(self.A[i % len(self.A)]).convert("RGB")
        b = Image.open(random.choice(self.B)).convert("RGB")  # unpaired on purpose
        return self.tf(a), self.tf(b)
