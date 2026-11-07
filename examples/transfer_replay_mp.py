"""Transfer with replay — the v0.6/v0.8 research protocol, on the last
generation of the MP core.

This is the old repo's actual fix for continual learning
(vision/hahino_v08_noswitch.py), ported to MPBrain:

  Phase A: train the digits brain (MNIST, 10 classes).
  Transfer: birth 26 letter classes (MPBrain.birth_classes — digits
            re-headed to 36 with ZERO letter rows: bit-exact old
            behavior), then freeze the digits classes (Law 4).
  Phase B: every step runs TWO passes, exactly like the research code:
            1. letters batch, FULL 36-way CE   (targets 10-35)
            2. digits batch, FULL 36-way CE    (targets 0-9)
           after each backward the freeze hook zeroes the digits'
           gradient segments, so digit weights are immutable while
           still receiving forward traffic. The letters paths' head
           rows for digit classes learn CALIBRATED suppression from
           real digit inputs — that is what the record's
           "digits improved +0.10 through joint sum training" came
           from: forgetting resistance plus replay-side calibration,
           not freezing alone.

  Phase C (3-domain extension): freeze letters, birth 10 fashion
  classes, and replay BOTH previous domains every step (fashion CE
  + letters replay CE + digits replay CE, all 46-way).

  Growth: maintain() every 2 epochs (unfrozen classes only; frozen
  domains are skipped — and the freeze hook now survives rebuilds).

Measured record for this protocol family: digits retained >= 97%
(target met), and in the unfrozen variant digits IMPROVED +0.10.

    python examples/transfer_replay_mp.py
    python examples/transfer_replay_mp.py --epochs 6 --subset 12000

Requires torchvision (datasets auto-download on first run).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from hahino_mp import MPBrain

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIGITS = list(range(10))
LETTERS = list(range(10, 36))


def load_digits():
    here = os.path.dirname(os.path.abspath(__file__))
    tr = datasets.MNIST(os.path.join(here, 'data'), train=True,
                        download=True)
    te = datasets.MNIST(os.path.join(here, 'data'), train=False,
                        download=True)
    x_tr = ((tr.data.float() / 255.0 - 0.1307) / 0.3081).reshape(-1, 784)
    x_te = ((te.data.float() / 255.0 - 0.1307) / 0.3081).reshape(-1, 784)
    return (x_tr, tr.targets.long()), (x_te, te.targets.long())


def load_letters():
    # EMNIST is stored rotated 90 degrees; the research-era fix
    # (established in v0.6 transfer) is transpose + vertical flip.
    here = os.path.dirname(os.path.abspath(__file__))
    tr = datasets.EMNIST(os.path.join(here, 'data'), split='letters',
                         train=True, download=True)
    te = datasets.EMNIST(os.path.join(here, 'data'), split='letters',
                         train=False, download=True)
    y_tr = tr.targets.long() - 1          # labels are 1..26
    y_te = te.targets.long() - 1
    x_tr = tr.data.float().transpose(-1, -2).flip(-2).reshape(-1, 784) \
        / 255.0
    x_te = te.data.float().transpose(-1, -2).flip(-2).reshape(-1, 784) \
        / 255.0
    return (x_tr, y_tr), (x_te, y_te)


def load_fashion():
    here = os.path.dirname(os.path.abspath(__file__))
    tr = datasets.FashionMNIST(os.path.join(here, 'data'), train=True,
                               download=True)
    te = datasets.FashionMNIST(os.path.join(here, 'data'), train=False,
                               download=True)
    x_tr = (tr.data.float() / 255.0).reshape(-1, 784)
    x_te = (te.data.float() / 255.0).reshape(-1, 784)
    return (x_tr, tr.targets.long()), (x_te, te.targets.long())


def maybe_subset(x, y, n):
    if not n:
        return x, y
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(x.shape[0], generator=g)[:n]
    return x[idx], y[idx]


@torch.no_grad()
def accuracy(model, x, y):
    x, y = x.to(DEVICE), y.to(DEVICE)
    correct = total = 0
    for i in range(0, x.shape[0], 2048):
        pred = model(x[i:i + 2048]).argmax(dim=1)
        correct += pred.eq(y[i:i + 2048]).sum().item()
        total += pred.numel()
    return 100. * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8,
                    help='letters-phase epochs (replay included)')
    ap.add_argument('--subset', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Device: {DEVICE}')

    (xd_tr, yd_tr), (xd_te, yd_te) = load_digits()
    (xl_tr, yl_tr), (xl_te, yl_te) = load_letters()
    yl_tr = yl_tr + 10
    yl_te = yl_te + 10
    (xf_tr, yf_tr), (xf_te, yf_te) = load_fashion()
    yf_tr = yf_tr + 36
    yf_te = yf_te + 36
    LETTERS = list(range(10, 36))
    FASHION = list(range(36, 46))

    # ---------------- Phase A: digits ----------------
    print('=== Phase A: digits brain (MNIST) ===')
    model = MPBrain(num_classes=10, in_dim=784).to(DEVICE)
    x, y = maybe_subset(xd_tr, yd_tr, args.subset)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE,
                        shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.pool_adam.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            model.pool_adam.step()
            total_loss += loss.item()
            correct += model(data).argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
        print(f'  [digits ep{epoch}] loss {total_loss/len(loader):.4f} '
              f'acc {100.*correct/total:.2f}% ({time.time()-t0:.1f}s)')
        if epoch > 0 and epoch % 2 == 0 and epoch <= 6:
            model.maintain(loader, epoch, xd_te[:1024].to(DEVICE),
                           verbose=False)
    digits_A = accuracy(model, xd_te, yd_te)
    print(f'  digits: {digits_A:.2f}%\n')

    # ---------------- transfer: birth letters + freeze digits --------
    print('=== Transfer: birth 26 letter classes, freeze digits ===')
    model.freeze_classes(DIGITS)
    model.birth_classes(26)
    print(f'  pool: {model.pool.numel():,} | classes: '
          f'{model.num_classes} (digits immutable)\n')

    # ---------------- Phase B: letters with per-step digits replay ----
    print('=== Phase B: letters + interleaved digits replay '
          '(full 36-way CE) ===')
    xL, yL = maybe_subset(xl_tr, yl_tr, args.subset)
    letters_loader = DataLoader(TensorDataset(xL, yL), batch_size=BATCH_SIZE,
                                shuffle=True)
    digits_loader = DataLoader(TensorDataset(xd_tr, yd_tr),
                               batch_size=BATCH_SIZE, shuffle=True)
    digits_cycle = iter(digits_loader)
    calib = xl_te[:1024].to(DEVICE)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0
        for data, target in letters_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # pass 1: letters batch, full 36-way CE
            model.pool_adam.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            model.pool_adam.step()
            total_loss += loss.item()
            correct += model(data).argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
            # pass 2: digits replay batch, full 36-way CE
            try:
                ddata, dtarget = next(digits_cycle)
            except StopIteration:
                digits_cycle = iter(digits_loader)
                ddata, dtarget = next(digits_cycle)
            ddata, dtarget = ddata.to(DEVICE), dtarget.to(DEVICE)
            model.pool_adam.zero_grad()
            dloss = F.cross_entropy(model(ddata), dtarget)
            dloss.backward()
            model.pool_adam.step()   # freeze hook zeroes digits' grads
        print(f'  [letters+replay ep{epoch}] loss '
              f'{total_loss/len(letters_loader):.4f} '
              f'acc {100.*correct/total:.2f}% ({time.time()-t0:.1f}s)')
        if epoch > 0 and epoch % 2 == 0 and epoch <= 6:
            model.maintain(letters_loader, epoch, calib, verbose=False)

    digits_B = accuracy(model, xd_te, yd_te)
    letters_B = accuracy(model, xl_te, yl_te)
    print(f'\n  digits  after transfer: {digits_B:.2f}%  '
          f'(baseline {digits_A:.2f}%)')
    print(f'  letters after transfer: {letters_B:.2f}%')

    # ---------------- Phase C: fashion with letters+digits replay ----
    print('\n=== Phase C: birth 10 fashion classes, freeze letters, '
          'replay both ===')
    model.freeze_classes(LETTERS)
    model.birth_classes(10)
    print(f'  pool: {model.pool.numel():,} | classes: '
          f'{model.num_classes} (digits+letters immutable)\n')
    xF, yF = maybe_subset(xf_tr, yf_tr, args.subset)
    fashion_loader = DataLoader(TensorDataset(xF, yF), batch_size=BATCH_SIZE,
                                shuffle=True)
    letters_loader_r = DataLoader(TensorDataset(xl_tr, yl_tr),
                                  batch_size=BATCH_SIZE, shuffle=True)
    letters_cycle = iter(letters_loader_r)
    calib = xf_te[:1024].to(DEVICE)
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0
        for data, target in fashion_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # pass 1: fashion batch, full 46-way CE
            model.pool_adam.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            model.pool_adam.step()
            total_loss += loss.item()
            correct += model(data).argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
            # pass 2: letters replay
            try:
                rdata, rtarget = next(letters_cycle)
            except StopIteration:
                letters_cycle = iter(letters_loader_r)
                rdata, rtarget = next(letters_cycle)
            rdata, rtarget = rdata.to(DEVICE), rtarget.to(DEVICE)
            model.pool_adam.zero_grad()
            rloss = F.cross_entropy(model(rdata), rtarget)
            rloss.backward()
            model.pool_adam.step()
            # pass 3: digits replay
            try:
                ddata, dtarget = next(digits_cycle)
            except StopIteration:
                digits_cycle = iter(digits_loader)
                ddata, dtarget = next(digits_cycle)
            ddata, dtarget = ddata.to(DEVICE), dtarget.to(DEVICE)
            model.pool_adam.zero_grad()
            dloss = F.cross_entropy(model(ddata), dtarget)
            dloss.backward()
            model.pool_adam.step()
        print(f'  [fashion+replay ep{epoch}] loss '
              f'{total_loss/len(fashion_loader):.4f} '
              f'acc {100.*correct/total:.2f}% ({time.time()-t0:.1f}s)')
        if epoch > 0 and epoch % 2 == 0 and epoch <= 6:
            model.maintain(fashion_loader, epoch, calib, verbose=False)

    digits_C = accuracy(model, xd_te, yd_te)
    letters_C = accuracy(model, xl_te, yl_te)
    fashion_C = accuracy(model, xf_te, yf_te)
    print(f'\n  digits  final: {digits_C:.2f}%')
    print(f'  letters final: {letters_C:.2f}%')
    print(f'  fashion final: {fashion_C:.2f}%')

    print('\n' + '=' * 60)
    print('TRANSFER + REPLAY REPORT (v08 protocol, last-gen MP core)')
    print('=' * 60)
    print(f'  digits baseline          {digits_A:6.2f}%')
    print(f'  digits after letters     {digits_B:6.2f}%   '
          f'delta {digits_B - digits_A:+.2f}')
    print(f'  digits after fashion     {digits_C:6.2f}%   '
          f'delta {digits_C - digits_A:+.2f}')
    print(f'  letters after its phase  {letters_B:6.2f}%')
    print(f'  letters after fashion    {letters_C:6.2f}%   '
          f'delta {letters_C - letters_B:+.2f}')
    print(f'  fashion                  {fashion_C:6.2f}%')
    print(f'  pool: {model.pool.numel():,} | sub-paths: '
          f'{sum(len(cd) for cd in model.descs)}')
    print('  Protocol: each phase freezes the previous domain(s) and')
    print('  replays them every step (full CE); new paths learn')
    print('  calibrated suppression of older columns from real data.')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'hahino_mp_transfer_replay.pth')
    torch.save({'pool': model.pool.detach().cpu(),
                'descs': model.descs,
                'num_classes': model.num_classes,
                'in_dim': 784,
                'frozen_classes': sorted(model.frozen_classes),
                'digits_baseline': digits_A,
                'digits_after': digits_B,
                'letters_after': letters_B}, out)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
