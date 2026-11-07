"""MNIST demo for the Hahino MP brain.

Multi-path digit specialist (10 classes). Same protocol as the
research run hahino_v10_digits_mp.py (Paper 2/3 era): train, test,
maintain every MAINTENANCE_INTERVAL epochs until GROWTH_LAST_EPOCH,
deepcopy the best checkpoint, final sweep.

    python examples/mnist_mp.py            # GPU if available
    python examples/mnist_mp.py --epochs 3 --subset 20000   # quick CPU run

Requires torchvision (MNIST auto-downloads on first run).
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

MAINTENANCE_INTERVAL = 2
GROWTH_LAST_EPOCH = 8
EARLY_STOP_PATIENCE = 3


def load_mnist(subset=0):
    here = os.path.dirname(os.path.abspath(__file__))
    tr = datasets.MNIST(os.path.join(here, 'data'), train=True,
                        download=True)
    te = datasets.MNIST(os.path.join(here, 'data'), train=False,
                        download=True)
    x_tr = ((tr.data.float() / 255.0 - 0.1307) / 0.3081).reshape(-1, 784)
    y_tr = tr.targets.long()
    x_te = ((te.data.float() / 255.0 - 0.1307) / 0.3081).reshape(-1, 784)
    y_te = te.targets.long()
    if subset:
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(x_tr.shape[0], generator=g)[:subset]
        x_tr, y_tr = x_tr[idx], y_tr[idx]
    return x_tr, y_tr, x_te, y_te


def train_epoch(model, loader, epoch):
    model.train()
    total_loss = correct = total = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        model.pool_adam.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        model.pool_adam.step()
        total_loss += loss.item()
        correct += logits.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
        if batch_idx % 400 == 0:
            print(f'Epoch {epoch} [{batch_idx*len(data)}/'
                  f'{len(loader.dataset)}] Loss: {loss.item():.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def test(model, loader):
    model.eval()
    NC = model.num_classes
    conf = torch.zeros(NC, NC, dtype=torch.long)
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        logits = model(data)
        pred = logits.argmax(dim=1)
        for t, p in zip(target.tolist(), pred.tolist()):
            conf[t][p] += 1
    acc = 100. * conf.diag().sum().item() / conf.sum().item()
    per_class = [100. * conf[i][i].item() / conf[i].sum().item()
                 for i in range(NC)]
    print(f'Test accuracy: {acc:.2f}%  per-class: '
          + ' '.join(f'{v:.1f}' for v in per_class))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--subset', type=int, default=0,
                    help='cap the train set size (0 = all 60k)')
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Device: {DEVICE}')
    x_tr, y_tr, x_te, y_te = load_mnist(args.subset)
    print(f'train: {x_tr.shape[0]:,} | test: {x_te.shape[0]:,}')
    train_loader = DataLoader(TensorDataset(x_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=2048,
                             shuffle=False)
    calib_x = x_te[:1024].to(DEVICE)

    model = MPBrain(num_classes=10, in_dim=784).to(DEVICE)
    print(f'Initial: 10 classes x 1 sub-path x 784->4->10, '
          f'pool {model.pool.numel():,}\n')

    start = time.time()
    best_acc, no_improve = 0.0, 0
    best_model = None
    for epoch in range(args.epochs):
        ep_t = time.time()
        train_epoch(model, train_loader, epoch)
        acc = test(model, test_loader)
        print(f'  epoch wall time: {time.time() - ep_t:.1f}s')

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f'\nEarly stopping at epoch {epoch} '
                      f'(best {best_acc:.2f}%)')
                break

        if epoch > 0 and epoch % MAINTENANCE_INTERVAL == 0 \
                and epoch <= GROWTH_LAST_EPOCH:
            model.maintain(test_loader, epoch, calib_x)

    if best_model is not None:
        model = best_model
        print(f'\nFinal sweep on BEST checkpoint ({best_acc:.2f}%)')
    acc = test(model, test_loader)

    print('\nEMERGENT STRUCTURE')
    for c in range(model.num_classes):
        from hahino_mp import describe_path
        K = len(model.descs[c])
        structs = ' | '.join(describe_path(d) for d in model.descs[c])
        print(f'  P{c}: K={K} [{structs}] ({model.class_params(c):,}p)')
    print(f'Total sub-paths: {sum(len(cd) for cd in model.descs)}')
    print(f'Spawns: {model.spawn_log}')
    print(f'Pool: {model.pool.numel():,} | accuracy: {acc:.2f}%')
    print(f'Training time: {time.time() - start:.1f}s')

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'hahino_mp_mnist.pth')
    torch.save({'pool': model.pool.detach().cpu(),
                'descs': model.descs,
                'num_classes': 10,
                'in_dim': 784,
                'accuracy': acc}, out)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
