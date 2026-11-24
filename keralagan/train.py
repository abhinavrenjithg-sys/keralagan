"""Train a CycleGAN on an unpaired dataset.

    python -m keralagan.train --data data/monsoon --epochs 100
    python -m keralagan.train --data data/monsoon --epochs 100 --resume     # after a Colab disconnect
"""

import argparse
import csv
import itertools
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import UnpairedDataset
from .models import Discriminator, Generator, ImagePool, init_weights


def get_args(argv=None):
    ap = argparse.ArgumentParser(description="Train KeralaGAN (CycleGAN)")
    ap.add_argument("--data", required=True, help="dataset root with trainA/trainB")
    ap.add_argument("--name", default=None, help="run name (default: dataset folder name)")
    ap.add_argument("--out", default=".", help="where checkpoints/ samples/ logs/ go (e.g. a Drive folder)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_cyc", type=float, default=10.0)
    ap.add_argument("--lambda_idt", type=float, default=0.5, help="identity weight, as a fraction of lambda_cyc")
    ap.add_argument("--n_blocks", type=int, default=9, help="residual blocks (9 for 256px, 6 for 128px)")
    ap.add_argument("--pool", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=10, help="keep a numbered checkpoint every N epochs")
    ap.add_argument("--max_steps", type=int, default=0, help="stop each epoch after N steps (0 = full epoch; for smoke tests)")
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA (faster on T4)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or os.path.basename(os.path.normpath(args.data))
    ckpt_dir = os.path.join(args.out, "checkpoints", name)
    sample_dir = os.path.join(args.out, "samples", name)
    log_path = os.path.join(args.out, "logs", f"{name}.csv")
    for d in (ckpt_dir, sample_dir, os.path.dirname(log_path)):
        os.makedirs(d, exist_ok=True)

    ds = UnpairedDataset(args.data, "train", args.size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=(dev == "cuda"))
    print(f"[KeralaGAN] {name}: {len(ds.A)} A images, {len(ds.B)} B images, device={dev}")

    G_AB, G_BA = Generator(n_blocks=args.n_blocks).to(dev), Generator(n_blocks=args.n_blocks).to(dev)
    D_A, D_B = Discriminator().to(dev), Discriminator().to(dev)
    for m in (G_AB, G_BA, D_A, D_B):
        m.apply(init_weights)

    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), args.lr, betas=(0.5, 0.999))

    decay_start = args.epochs // 2
    decay_len = max(1, args.epochs - decay_start)
    lam = lambda e: 1.0 - max(0, e - decay_start) / float(decay_len)
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lam)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lam)

    use_amp = args.amp and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    pool_A, pool_B = ImagePool(args.pool), ImagePool(args.pool)
    start = 0

    latest = os.path.join(ckpt_dir, "latest.pt")
    if args.resume and os.path.exists(latest):
        s = torch.load(latest, map_location=dev)
        for k, m in dict(G_AB=G_AB, G_BA=G_BA, D_A=D_A, D_B=D_B, opt_G=opt_G, opt_D=opt_D).items():
            m.load_state_dict(s[k])
        start = s["epoch"] + 1
        sch_G.last_epoch = sch_D.last_epoch = start - 1
        for opt in (opt_G, opt_D):
            for g in opt.param_groups:
                g["lr"] = args.lr * lam(start)
        print(f"[KeralaGAN] resumed at epoch {start}")

    fixed_A, fixed_B = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    fixed_A, fixed_B = fixed_A.to(dev), fixed_B.to(dev)

    new_log = not os.path.exists(log_path) or start == 0
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["epoch", "step", "loss_G", "loss_gan", "loss_cyc", "loss_idt", "loss_D", "lr"])

    for epoch in range(start, args.epochs):
        t0 = time.time()
        for i, (real_A, real_B) in enumerate(loader):
            if args.max_steps and i >= args.max_steps:
                break
            real_A, real_B = real_A.to(dev, non_blocking=True), real_B.to(dev, non_blocking=True)

            # ---- generators
            for p in itertools.chain(D_A.parameters(), D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                fake_B, fake_A = G_AB(real_A), G_BA(real_B)
                pB, pA = D_B(fake_B), D_A(fake_A)
                loss_gan = mse(pB, torch.ones_like(pB)) + mse(pA, torch.ones_like(pA))
                loss_cyc = l1(G_BA(fake_B), real_A) + l1(G_AB(fake_A), real_B)
                if args.lambda_idt > 0:
                    loss_idt = l1(G_AB(real_B), real_B) + l1(G_BA(real_A), real_A)
                else:
                    loss_idt = torch.zeros((), device=dev)
                loss_G = loss_gan + args.lambda_cyc * loss_cyc + args.lambda_cyc * args.lambda_idt * loss_idt
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

            # ---- discriminators
            for p in itertools.chain(D_A.parameters(), D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss_D = 0
                for D, real, fake in ((D_A, real_A, pool_A.query(fake_A)), (D_B, real_B, pool_B.query(fake_B))):
                    pr, pf = D(real), D(fake)
                    loss_D = loss_D + 0.5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

            if i % 50 == 0:
                lr = opt_G.param_groups[0]["lr"]
                row = [epoch, i, loss_G.item(), loss_gan.item(), loss_cyc.item(), loss_idt.item(), float(loss_D), lr]
                log.writerow([f"{v:.5f}" if isinstance(v, float) else v for v in row])
                log_f.flush()
                print(f"[{epoch + 1}/{args.epochs}] step {i}/{len(loader)}  G {row[2]:.3f} "
                      f"(gan {row[3]:.3f} cyc {row[4]:.3f} idt {row[5]:.3f})  D {row[6]:.3f}")

        sch_G.step(); sch_D.step()

        # sample grid: rows = real A | fake B | real B | fake A
        G_AB.eval(); G_BA.eval()
        with torch.no_grad():
            grid = torch.cat([fixed_A, G_AB(fixed_A), fixed_B, G_BA(fixed_B)])
        save_image(grid * 0.5 + 0.5, os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"), nrow=4)
        G_AB.train(); G_BA.train()

        state = dict(G_AB=G_AB.state_dict(), G_BA=G_BA.state_dict(), D_A=D_A.state_dict(), D_B=D_B.state_dict(),
                     opt_G=opt_G.state_dict(), opt_D=opt_D.state_dict(), epoch=epoch,
                     config=dict(size=args.size, n_blocks=args.n_blocks, name=name))
        torch.save(state, latest + ".tmp")
        os.replace(latest + ".tmp", latest)  # atomic: a disconnect mid-save can't corrupt latest.pt
        if (epoch + 1) % args.save_every == 0:
            torch.save(state, os.path.join(ckpt_dir, f"epoch_{epoch + 1:03d}.pt"))
        print(f"[KeralaGAN] epoch {epoch + 1} done in {time.time() - t0:.0f}s, saved {latest}")

    log_f.close()


if __name__ == "__main__":
    main()
