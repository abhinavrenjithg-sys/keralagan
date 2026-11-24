"""CycleGAN networks: ResNet generator (Johnson et al.) and 70x70 PatchGAN discriminator."""

import random

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """c7s1-64, d128, d256, R256 x n_blocks, u128, u64, c7s1-3 (tanh output in [-1, 1])."""

    def __init__(self, ngf: int = 64, n_blocks: int = 9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):
            layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN: outputs a grid of real/fake scores, one per overlapping patch."""

    def __init__(self, ndf: int = 64):
        super().__init__()

        def blk(i, o, stride, norm=True):
            layers = [nn.Conv2d(i, o, 4, stride, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.net = nn.Sequential(
            *blk(3, ndf, 2, norm=False),
            *blk(ndf, ndf * 2, 2),
            *blk(ndf * 2, ndf * 4, 2),
            *blk(ndf * 4, ndf * 8, 1),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    """N(0, 0.02) init used in the CycleGAN paper."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ImagePool:
    """Buffer of previously generated images; feeding D a history stabilises training."""

    def __init__(self, size: int = 50):
        self.size = size
        self.images = []

    def query(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return imgs.detach()
        out = []
        for img in imgs.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                i = random.randrange(self.size)
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out)
