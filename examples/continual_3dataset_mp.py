"""Continual-learning demo: 3 datasets, class-incremental, no replay.

The non-catastrophic-forgetting protocol from the research project
(Paper 1 transfer, v0.6/v0.8 era), on a single MPBrain:

  Phase A: train digits (MNIST, classes 0-9)
  -> freeze digits                      (freeze sovereignty, Law 4)
  Phase B: birth 26 letter classes (EMNIST Letters), train letters
  -> freeze letters
  Phase C: birth 10 fashion classes (Fashion-MNIST), train fashion

Old classes' parameters are IMMUTABLE after their phase (gradient
mask + growth skip), so retention is measured, not assumed. The
research record for this protocol: digits retained >= 97% after 26
letters were added to a trained digit network; final 3-domain
assembly 93.25% joint.

    python examples/continual_3dataset_mp.py
    python examples/continual_3dataset_mp.py --epochs 4 --subset 12000

Requires torchvision (datasets auto-download on first run).
"""
import argparse
import copy
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

DIGITS = list(range(0, 10))
LETTERS = list(range(10, 36))
FASHION = list(range(36, 46))


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


def train_phase(model, loader, epochs, phase_name):
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        total_loss = correct = total = 0
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.pool_adam.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            model.pool_adam.step()
            total_loss += loss.item()
            correct += logits.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
        print(f'  [{phase_name} ep{epoch}] loss '
              f'{total_loss/len(loader):.4f} '
              f'acc {100.*correct/total:.2f}% '
              f'({time.time()-t0:.1f}s)')


@torch.no_grad()
def per_domain_acc(model, domains):
    """domains: list of (name, (x_te, y_te), class_ids)."""
    model.eval()
    out = {}
    for name, (x, y), ids in domains:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct = total = 0
        for i in range(0, x.shape[0], 2048):
            pred = model(x[i:i + 2048]).argmax(dim=1)
            correct += pred.eq(y[i:i + 2048]).sum().item()
            total += pred.numel()
        out[name] = 100. * correct / total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8,
                    help='training epochs per phase')
    ap.add_argument('--subset', type=int, default=0,
                    help='cap per-phase train size (0 = full)')
    ap.add_argument('--growth-last-epoch', type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Device: {DEVICE}')

    def maybe_subset(x, y):
        if not args.subset:
            return x, y
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(x.shape[0], generator=g)[:args.subset]
        return x[idx], y[idx]

    print('Loading MNIST digits...')
    (xd_tr, yd_tr), (xd_te, yd_te) = load_digits()
    print('Loading EMNIST letters...')
    (xl_tr, yl_tr), (xl_te, yl_te) = load_letters()
    print('Loading Fashion-MNIST...')
    (xf_tr, yf_tr), (xf_te, yf_te) = load_fashion()

    # shift letter/fashion labels into the joint class space
    yl_tr += 10
    yl_te += 10
    yf_tr += 36
    yf_te += 36

    model = MPBrain(num_classes=10, in_dim=784).to(DEVICE)
    print(f'Initial: 10 classes, pool {model.pool.numel():,}\n')

    retention = []

    # ---------------- Phase A: digits ----------------
    print('=== Phase A: digits (MNIST) ===')
    x, y = maybe_subset(xd_tr, yd_tr)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE,
                        shuffle=True)
    train_phase(model, loader, args.epochs, 'digits')
    doms = [('digits', (xd_te, yd_te), DIGITS)]
    acc = per_domain_acc(model, doms)
    retention.append(('after digits (baseline)', acc))
    print(f'  retention: {acc}\n')

    # ---------------- Phase B: letters ----------------
    print('=== Phase B: +26 letters (EMNIST), digits FROZEN ===')
    model.freeze_classes(DIGITS)
    new_ids = model.birth_classes(26)
    assert new_ids == LETTERS
    x, y = maybe_subset(xl_tr, yl_tr)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE,
                        shuffle=True)
    calib = xl_te[:1024].to(DEVICE)
    for epoch in range(args.epochs):
        train_phase(model, loader, 1, 'letters')
        if epoch > 0 and epoch % 2 == 0 \
                and epoch <= args.growth_last_epoch:
            model.maintain(loader, epoch, calib)
    doms = [('digits', (xd_te, yd_te), DIGITS),
            ('letters', (xl_te, yl_te), LETTERS)]
    acc = per_domain_acc(model, doms)
    retention.append(('after letters', acc))
    print(f'  retention: {acc}')
    print(f'  >>> digits retained {acc["digits"]:.2f}% '
          f'(frozen since phase A)\n')

    # ---------------- Phase C: fashion ----------------
    print('=== Phase C: +10 fashion (Fashion-MNIST), digits+letters '
          'FROZEN ===')
    model.freeze_classes(LETTERS)
    new_ids = model.birth_classes(10)
    assert new_ids == FASHION
    x, y = maybe_subset(xf_tr, yf_tr)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE,
                        shuffle=True)
    calib = xf_te[:1024].to(DEVICE)
    for epoch in range(args.epochs):
        train_phase(model, loader, 1, 'fashion')
        if epoch > 0 and epoch % 2 == 0 \
                and epoch <= args.growth_last_epoch:
            model.maintain(loader, epoch, calib)
    doms = [('digits', (xd_te, yd_te), DIGITS),
            ('letters', (xl_te, yl_te), LETTERS),
            ('fashion', (xf_te, yf_te), FASHION)]
    acc = per_domain_acc(model, doms)
    retention.append(('after fashion (final)', acc))
    print(f'  retention: {acc}\n')

    # ---------------- final report ----------------
    print('=' * 60)
    print('CONTINUAL LEARNING REPORT — 3 DATASETS, NO REPLAY')
    print('=' * 60)
    for stage, acc in retention:
        line = '  '.join(f'{k} {v:.2f}%' for k, v in acc.items())
        print(f'  {stage:<26} {line}')
    print(f'\n  Pool: {model.pool.numel():,} params | '
          f'sub-paths: {sum(len(cd) for cd in model.descs)} | '
          f'spawns: {model.spawn_log}')
    print('  Frozen throughout after their phase: digits, letters '
          '(freeze sovereignty, Law 4)')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'hahino_mp_3dataset.pth')
    torch.save({'pool': model.pool.detach().cpu(),
                'descs': model.descs,
                'num_classes': model.num_classes,
                'in_dim': 784,
                'frozen_classes': sorted(model.frozen_classes),
                'retention': retention}, out)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
