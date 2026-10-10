"""Is a better spine mask worth training for?

The deployed mask stops above the reference lower end vertebra on 53 of 481 AASCE
films, and on those the Cobb MAE is 9.9 deg against 4.0 deg on the rest. Running the
same geometry on a perfect mask (built from the landmarks) gives the ceiling. This
script asks the practical question instead: train a segmentation net on the
landmark-derived masks and see how much of that ceiling it reaches OUT OF FOLD.

Group 5-fold cross-validation over the file-name groups (the same grouping the
article's cluster bootstrap uses), so no film is scored by a net that saw a film
from the same series.

CPU baseline (small net, low resolution):
    python spine_unet.py --epochs 20 --out unet_masks.npz
On the GPU, the configuration that actually answers the question:
    python spine_unet.py --size 768x256 --base 32 --epochs 60 --out unet_masks_gpu.npz
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cobb_oracle as CO      # noqa: E402
import cobb_postproc as CP    # noqa: E402
import levels_analysis as LA  # noqa: E402

IMG_DIR = os.path.join(ROOT, 'aasce_kaggle', 'train')
torch.manual_seed(0)
np.random.seed(0)


def group_of(name):
    m = re.match(r'sunhl-1th-(\d+-\w+-\d+)-(\d+)\s', name)
    return m.group(1) + '/' + m.group(2) if m else name


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(x)))


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.d1, self.d2, self.d3 = Block(1, base), Block(base, base * 2), Block(base * 2, base * 4)
        self.mid = Block(base * 4, base * 8)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        m = self.mid(F.max_pool2d(d3, 2))
        u3 = self.u3(torch.cat([F.interpolate(m, size=d3.shape[-2:], mode='nearest'), d3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, size=d2.shape[-2:], mode='nearest'), d2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, size=d1.shape[-2:], mode='nearest'), d1], 1))
        return self.out(u1)


def load_dataset(names, pts_of, h, w):
    """Images and landmark-derived masks, both at h x w."""
    X = np.zeros((len(names), 1, h, w), np.float32)
    Y = np.zeros((len(names), 1, h, w), np.float32)
    for i, n in enumerate(names):
        im = Image.open(os.path.join(IMG_DIR, n)).convert('L')
        shape = (im.size[1], im.size[0])
        X[i, 0] = np.asarray(im.resize((w, h), Image.BILINEAR), np.float32) / 255.0
        mask = CO.oracle_mask(pts_of[n], shape)
        Y[i, 0] = np.asarray(Image.fromarray(mask * 255).resize((w, h), Image.BILINEAR), np.float32) / 255.0
    X = (X - X.mean(axis=(1, 2, 3), keepdims=True)) / (X.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
    return X, (Y > 0.5).astype(np.float32)


def augment(x, y):
    """Left-right flip, a small similarity transform, and intensity jitter.

    Image and mask get the SAME geometric transform, so the pairing stays honest:
    a mirrored or slightly rotated spine is still a spine with the same mask.
    """
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    n = x.shape[0]
    ang = (torch.rand(n, device=x.device) * 2 - 1) * (5 * np.pi / 180)
    scale = 1 + (torch.rand(n, device=x.device) * 2 - 1) * 0.06
    tx = (torch.rand(n, device=x.device) * 2 - 1) * 0.04
    ty = (torch.rand(n, device=x.device) * 2 - 1) * 0.04
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode='border')
    y = F.grid_sample(y, grid, align_corners=False, padding_mode='zeros')
    x = x * (1 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * 0.1)
    return x, (y > 0.5).float()


def dice_bce(logit, target):
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


def train_fold(Xtr, Ytr, args, device):
    net = UNet(args.base).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    n = len(Xtr)
    for ep in range(args.epochs):
        net.train()
        idx = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, args.batch):
            b = idx[s:s + args.batch]
            x = torch.from_numpy(Xtr[b]).to(device)
            y = torch.from_numpy(Ytr[b]).to(device)
            x, y = augment(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                loss = dice_bce(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(b)
        sched.step()
        yield ep, tot / n, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--size', default='384x128', help='HxW the net trains at')
    ap.add_argument('--base', type=int, default=16)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', default='unet_masks.npz')
    args = ap.parse_args()
    h, w = (int(v) for v in args.size.lower().split('x'))

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count() or 4)
    print('device %s | %dx%d | base %d | %d epochs' %
          (torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
           h, w, args.base, args.epochs), flush=True)

    recs = {}
    for line in open(CP.RUN, encoding='utf-8'):
        r = json.loads(line)
        recs[r['name']] = r
    names = sorted(recs)
    pts_of = LA.reference_points()
    print('loading %d films...' % len(names), flush=True)
    X, Y = load_dataset(names, pts_of, h, w)

    groups = np.array([group_of(n) for n in names])
    uniq = sorted(set(groups))
    rng = np.random.default_rng(3)
    fold_of_group = {g: i % args.folds for i, g in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of_group[g] for g in groups])

    pred = np.zeros((len(names), h, w), np.float32)
    for f in range(args.folds):
        tr, te = folds != f, folds == f
        t0 = time.time()
        last = None
        for ep, loss, net in train_fold(X[tr], Y[tr], args, device):
            last = net
            if (ep + 1) % 5 == 0:
                print('  fold %d epoch %2d loss %.4f (%.0fs)' % (f, ep + 1, loss, time.time() - t0), flush=True)
        last.eval()
        with torch.no_grad():
            out, Xte = [], X[te]
            for s in range(0, len(Xte), args.batch):
                x = torch.from_numpy(Xte[s:s + args.batch]).to(device)
                # test-time augmentation: the film and its mirror
                p = torch.sigmoid(last(x)) + torch.flip(torch.sigmoid(last(torch.flip(x, [3]))), [3])
                out.append((p / 2).float().cpu().numpy()[:, 0])
            pred[te] = np.concatenate(out)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0.0
        print('fold %d done in %.0fs (train %d, test %d, peak %.1f GiB)'
              % (f, time.time() - t0, tr.sum(), te.sum(), peak), flush=True)
        # Without this the allocator carries its fragmented arena into the next fold;
        # on a 12 GiB card that creeps up until the driver starts paging over PCIe and
        # throughput collapses while the GPU still reports 100% busy.
        del last, net, out
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    path = os.path.join(ROOT, 'radtools_run', args.out)
    np.savez_compressed(path, names=np.array(names), pred=pred.astype(np.float16))
    print('saved out-of-fold masks to %s' % path, flush=True)


if __name__ == '__main__':
    main()
