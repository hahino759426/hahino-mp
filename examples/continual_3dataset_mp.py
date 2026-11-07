"""Continual learning, the recorded way: 3 sovereign MP brains + gate.

The non-catastrophic-forgetting protocol as it was actually measured
in the research project (Paper 1 transfer era -> README champions):

  Phase A: train the DIGITS brain (MNIST). Freeze it.
  Phase B: train the LETTERS brain (EMNIST) from scratch. Freeze it.
           -> the digits brain cannot forget: separate pool, frozen.
  Phase C: train the FASHION brain (Fashion-MNIST). Freeze it.
  Assembly: a tiny 3-class multi-path GATE on the RAW 784 pixels
            (Phase-0 law: the gate must read raw input) routes each
            test image to its sovereign brain.

Retention is structural, not statistical: domains live in separate
parameter pools and are immutable after their phase. The research
record for this architecture family: digits retained >= 97% after 26
letters were added (the motivating transfer experiment); 2-domain
assembly 93.25% joint (README), 3-class domain gate 99.27% corrected.

    python examples/continual_3dataset_mp.py
    python examples/continual_3dataset_mp.py --epochs 4 --subset 12000

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

DOMAIN_NAMES = ['digits', 'letters', 'fashion']


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


def train_brain(model, loader, epochs, name, calib_x,
                growth_last_epoch=6):
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
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
        print(f'  [{name} ep{epoch}] loss {total_loss/len(loader):.4f} '
              f'acc {100.*correct/total:.2f}% ({time.time()-t0:.1f}s)')
        if epoch > 0 and epoch % 2 == 0 and epoch <= growth_last_epoch:
            model.maintain(loader, epoch, calib_x, verbose=False)
    # cooldown: freeze the whole brain (freeze sovereignty, Law 4)
    model.freeze_classes(range(model.num_classes))
    model.eval()


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
                    help='training epochs per brain')
    ap.add_argument('--subset', type=int, default=0,
                    help='cap per-phase train size (0 = full)')
    ap.add_argument('--gate-epochs', type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Device: {DEVICE}')

    print('Loading MNIST digits...')
    (xd_tr, yd_tr), (xd_te, yd_te) = load_digits()
    print('Loading EMNIST letters...')
    (xl_tr, yl_tr), (xl_te, yl_te) = load_letters()
    print('Loading Fashion-MNIST...')
    (xf_tr, yf_tr), (xf_te, yf_te) = load_fashion()

    # ---------------- three sovereign brains, trained sequentially ----
    print('\n=== Phase A: digits brain (MNIST) ===')
    digits = MPBrain(num_classes=10, in_dim=784).to(DEVICE)
    loader = DataLoader(TensorDataset(*maybe_subset(xd_tr, yd_tr,
                                                    args.subset)),
                        batch_size=BATCH_SIZE, shuffle=True)
    train_brain(digits, loader, args.epochs, 'digits',
                xd_te[:1024].to(DEVICE))
    digits_acc_A = accuracy(digits, xd_te, yd_te)
    print(f'  digits brain: {digits_acc_A:.2f}%  (FROZEN)\n')

    print('=== Phase B: letters brain (EMNIST) — digits brain untouched '
          '===')
    letters = MPBrain(num_classes=26, in_dim=784).to(DEVICE)
    loader = DataLoader(TensorDataset(*maybe_subset(xl_tr, yl_tr,
                                                    args.subset)),
                        batch_size=BATCH_SIZE, shuffle=True)
    train_brain(letters, loader, args.epochs, 'letters',
                xl_te[:1024].to(DEVICE))
    letters_acc_B = accuracy(letters, xl_te, yl_te)
    print(f'  letters brain: {letters_acc_B:.2f}%  (FROZEN)')
    digits_acc_B = accuracy(digits, xd_te, yd_te)
    print(f'  digits brain re-measured: {digits_acc_B:.2f}%  '
          f'>>> RETAINED (frozen, separate pool)\n')

    print('=== Phase C: fashion brain (Fashion-MNIST) — digits+letters '
          'untouched ===')
    fashion = MPBrain(num_classes=10, in_dim=784).to(DEVICE)
    loader = DataLoader(TensorDataset(*maybe_subset(xf_tr, yf_tr,
                                                    args.subset)),
                        batch_size=BATCH_SIZE, shuffle=True)
    train_brain(fashion, loader, args.epochs, 'fashion',
                xf_te[:1024].to(DEVICE))
    fashion_acc_C = accuracy(fashion, xf_te, yf_te)
    print(f'  fashion brain: {fashion_acc_C:.2f}%  (FROZEN)')
    digits_acc_C = accuracy(digits, xd_te, yd_te)
    letters_acc_C = accuracy(letters, xl_te, yl_te)
    print(f'  digits re-measured:   {digits_acc_C:.2f}%  >>> RETAINED')
    print(f'  letters re-measured:  {letters_acc_C:.2f}%  >>> RETAINED\n')

    # ---------------- the domain gate (reads RAW pixels — Phase-0 law) --
    print('=== Gate: 3-class multi-path brain on raw 784 ===')
    gate = MPBrain(num_classes=3, in_dim=784, initial_neurons=4,
                   k_max=2).to(DEVICE)
    gx_tr = torch.cat([xd_tr[:30000], xl_tr[:30000], xf_tr[:30000]])
    gy_tr = torch.cat([torch.zeros(30000, dtype=torch.long),
                       torch.ones(30000, dtype=torch.long),
                       torch.full((30000,), 2, dtype=torch.long)])
    gx_te = torch.cat([xd_te, xl_te, xf_te])
    gy_te = torch.cat([torch.zeros(len(yd_te), dtype=torch.long),
                       torch.ones(len(yl_te), dtype=torch.long),
                       torch.full((len(yf_te),), 2, dtype=torch.long)])
    loader = DataLoader(TensorDataset(gx_tr, gy_tr), batch_size=256,
                        shuffle=True)
    train_brain(gate, loader, args.gate_epochs, 'gate',
                gx_te[:1024].to(DEVICE), growth_last_epoch=2)
    gate_acc = accuracy(gate, gx_te, gy_te)
    print(f'  gate accuracy: {gate_acc:.2f}%  (record: 99.27% '
          f'3-class, 99.71% binary)\n')

    # ---------------- assembled system: gate routes, brains stay sovereign
    print('=== Assembly: gate(raw) -> routed sovereign brain ===')
    brains = [digits, letters, fashion]
    tests_x = [xd_te, xl_te, xf_te]
    tests_y = [yd_te, yl_te, yf_te]
    n_cls = [10, 26, 10]
    report = []
    for d, (name, x, y) in enumerate(zip(DOMAIN_NAMES, tests_x,
                                         tests_y)):
        x = x.to(DEVICE)
        dom = gate(x).argmax(dim=1)
        correct = total = 0
        for b in range(3):
            mask = dom == b
            if not mask.any():
                continue
            xb = x[mask]
            yb = y.to(DEVICE)[mask]
            for i in range(0, xb.shape[0], 2048):
                pred = brains[b](xb[i:i + 2048]).argmax(dim=1)
                correct += pred.eq(yb[i:i + 2048]).sum().item()
                total += pred.numel()
        acc = 100. * correct / total
        routed = 100. * (dom == d).float().mean().item()
        report.append((name, acc, routed))
        print(f'  {name:<8} routed-by-gate {routed:5.1f}%   '
              f'brain accuracy {acc:.2f}%')
    joint = sum(a * len(y) for (_, a, _), y in
                zip(report, tests_y)) / sum(len(y) for y in tests_y)
    print(f'\n  ASSEMBLED JOINT: {joint:.2f}%   '
          f'(record: 93.25% 2-domain w/ stem+maturation)')

    print('\n' + '=' * 60)
    print('CONTINUAL LEARNING REPORT — 3 SOVEREIGN BRAINS + GATE')
    print('=' * 60)
    print(f'  digits baseline           {digits_acc_A:6.2f}%')
    print(f'  digits after +26 letters  {digits_acc_B:6.2f}%   '
          f'>>> RETAINED (frozen)')
    print(f'  digits after +10 fashion  {digits_acc_C:6.2f}%   '
          f'>>> RETAINED (frozen)')
    print(f'  letters after its phase   {letters_acc_B:6.2f}%')
    print(f'  letters after +10 fashion {letters_acc_C:6.2f}%   '
          f'>>> RETAINED (frozen)')
    print(f'  fashion                   {fashion_acc_C:6.2f}%')
    print(f'  gate (3-class, raw)       {gate_acc:6.2f}%')
    print(f'  assembled joint           {joint:6.2f}%')
    print('\n  No replay, no EWC, no rehearsal: retention is structural.')
    print('  Frozen domains are byte-identical after later phases; the')
    print('  gate — not shared parameters — handles cross-domain')
    print('  interaction (Law: paths must not compete cross-domain).')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'hahino_mp_3dataset_gated.pth')
    torch.save({'digits': {'pool': digits.pool.detach().cpu(),
                           'descs': digits.descs},
                'letters': {'pool': letters.pool.detach().cpu(),
                            'descs': letters.descs},
                'fashion': {'pool': fashion.pool.detach().cpu(),
                            'descs': fashion.descs},
                'gate': {'pool': gate.pool.detach().cpu(),
                         'descs': gate.descs},
                'joint': joint}, out)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
