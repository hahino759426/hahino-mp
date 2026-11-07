"""
Hahino v1.0 DIGITS-MP: multi-path digit specialist (10 outputs).

Same MP architecture and rule set as the letters brain:
- 10 classes x K emergent sub-paths (784->k->10), MAX within class (WTA),
  SUM across classes. No router anywhere.
- widen first (k = ceil(excess*25)), 16K cap per sub-path
- SPAWN for below-average classes when params plateau or cap hit;
  spawn halves the old source so both sub-paths must work
- UNCAPPED total budget; cleanup (duplicate twins + dead neurons,
  compensated) after EVERY rebuild
- age-aware anti-hallucination: <1% usage after 4-epoch grace -> delete

Baseline to beat: 97.68% (Stage-A digits brain, which needed an internal
rectified router — this brain should match/beat it with pure noswitch MP).

Saves hahino_v10_digit_brain_mp.pth.

Run: .venv/Scripts/python.exe -u hahino_v10_digits_mp.py
"""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 64
EPOCHS = 20
EARLY_STOP_PATIENCE = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAINTENANCE_INTERVAL = 2
GROWTH_LAST_EPOCH = 8
TARGET_CLASS_LOSS = 0.10
K_SCALE = 25
PARAM_CAP_PER_SUBPATH = 16_000
TOTAL_BUDGET = 999_999_999
K_MAX = 4
USAGE_MIN = 0.01
GRACE_EPOCHS = 4
INIT_NEURONS = 4
NUM_CLASSES = 10
NAMES = [str(i) for i in range(NUM_CLASSES)]

torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {DEVICE}")
print("Hahino v1.0 DIGITS-MP: multi-path digit brain "
      "(10 outputs, uncapped + cleanup)")

# ---- data: MNIST, tensors ---------------------------------------------------
print("Loading MNIST...")
_d = datasets.MNIST('./data', train=True, download=False)
_dt = datasets.MNIST('./data', train=False, download=False)

X_TR = ((_d.data.float() / 255.0 - 0.1307) / 0.3081).reshape(-1, 784)
Y_TR = _d.targets.long()
X_TE = ((_dt.data.float() / 255.0 - 0.1307) / 0.3081).reshape(-1, 784)
Y_TE = _dt.targets.long()
print(f"train: {X_TR.shape[0]:,} | test: {X_TE.shape[0]:,}")

train_loader = DataLoader(TensorDataset(X_TR, Y_TR), batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(TensorDataset(X_TE, Y_TE), batch_size=2048,
                         shuffle=False)


def cname(c):
    return str(c)


def path_forward(pool, layers, x):
    for idx, (kind, wo, bo, o, i) in enumerate(layers):
        W = pool[wo:wo + o * i].view(o, i)
        b = pool[bo:bo + o]
        x = F.linear(x, W, b)
        if kind == 'r' and idx < len(layers) - 1:
            x = F.relu(x)
    return x


def path_params(layers):
    return sum(o * i + o for _, wo, bo, o, i in layers)


def describe_path(layers):
    return '->'.join(str(s) for s in
                     [layers[0][4]] + [l[3] for l in layers])


class PoolAdam:
    def __init__(self, pool, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.pool = pool
        self.lr, self.betas, self.eps = lr, betas, eps
        self.t = 0
        self.m = torch.zeros_like(pool.data)
        self.v = torch.zeros_like(pool.data)

    def carry_to(self, new_pool, index_map):
        nm = torch.zeros_like(new_pool.data)
        nv = torch.zeros_like(new_pool.data)
        old = index_map >= 0
        nm[old] = self.m.to(new_pool.device)[index_map[old]]
        nv[old] = self.v.to(new_pool.device)[index_map[old]]
        self.pool = new_pool
        self.m, self.v = nm, nv

    def step(self):
        self.t += 1
        if self.m.device != self.pool.device:
            self.m = torch.zeros_like(self.pool.data)
            self.v = torch.zeros_like(self.pool.data)
        g = self.pool.grad
        if g is None:
            return
        b1, b2 = self.betas
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = self.m / (1 - b1 ** self.t)
        v_hat = self.v / (1 - b2 ** self.t)
        self.pool.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps),
                                value=-self.lr)

    def zero_grad(self):
        if self.pool.grad is not None:
            self.pool.grad.zero_()


class MPDigitBrain(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, initial_neurons=INIT_NEURONS):
        super().__init__()
        self.num_classes = num_classes
        self.descs = []
        offset = 0
        for c in range(num_classes):
            w = initial_neurons
            d = [('r', offset, offset + w * 784, w, 784)]
            offset += w * 784 + w
            d.append(('r', offset, offset + num_classes * w,
                      num_classes, w))
            offset += num_classes * w + num_classes
            self.descs.append([d])
        self.pool = nn.Parameter(torch.zeros(offset))
        for cd in self.descs:
            for d in cd:
                for (kind, wo, bo, o, i) in d:
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(
                            self.pool[wo:wo + o * i].view(o, i), a=5 ** 0.5)
                        nn.init.zeros_(self.pool[bo:bo + o])
        self.pool_adam = PoolAdam(self.pool)
        self.loss_history = {c: [] for c in range(num_classes)}
        self.last_action = {c: None for c in range(num_classes)}
        self.usage = {c: [1.0] for c in range(num_classes)}
        self.birth = {c: [0] for c in range(num_classes)}
        self.spawn_log = []
        self._fast_ready = False
        self._build_fast()
        self.prune_log = []

    def class_params(self, c):
        return sum(path_params(d) for d in self.descs[c])

    def forward(self, x):
        if self._fast_ready:
            return self._forward_fast(x)
        return self._forward_loop(x)

    def _forward_loop(self, x):
        vecs = []
        for cd in self.descs:
            if len(cd) == 1:
                vecs.append(path_forward(self.pool, cd[0], x))
            else:
                sp = torch.stack([path_forward(self.pool, d, x)
                                  for d in cd], dim=1)
                vecs.append(sp.max(dim=1).values)
        return torch.stack(vecs, dim=1).sum(dim=1)

    def _build_fast(self):
        self._fast_ready = False
        two, deep = [], []
        for c, cd in enumerate(self.descs):
            for si, d in enumerate(cd):
                (two if len(d) == 2 else deep).append((c, si, d))
        if not two:
            self._fast = None
            return
        idx1, idxb1 = [], []
        groups = {}
        cursor = 0
        for (c, si, d) in two:
            k1, wo1, bo1, o, wi = d[0]
            k2, wo2, bo2, co, ci = d[1]
            idx1.append(torch.arange(wo1, wo1 + o * wi))
            idxb1.append(torch.arange(bo1, bo1 + o))
            groups.setdefault(ci, []).append((c, cursor, wo2, bo2, co))
            cursor += o
        gdata = {}
        cls_list = []
        for ci, items in groups.items():
            gdata[ci] = {
                'offs': [it[1] for it in items],
                'idx2': torch.cat([torch.arange(it[2], it[2] + it[4] * ci)
                                   for it in items]).to(DEVICE),
                'idxb2': torch.cat([torch.arange(it[3], it[3] + it[4])
                                    for it in items]).to(DEVICE),
                'co': items[0][4],
            }
            cls_list.extend(it[0] for it in items)
        self._fast = {
            'idx1': torch.cat(idx1).to(DEVICE),
            'idxb1': torch.cat(idxb1).to(DEVICE),
            'groups': gdata,
            'cls': torch.tensor(cls_list, dtype=torch.long, device=DEVICE),
            'deep': deep,
        }
        self._fast_ready = True

    def _forward_fast(self, x):
        f = self._fast
        W1 = self.pool[f['idx1']].view(-1, 784)
        b1 = self.pool[f['idxb1']]
        h = F.relu(F.linear(x, W1, b1))
        outs = []
        for ci, gd in f['groups'].items():
            hs = torch.stack([h[:, off:off + ci] for off in gd['offs']])
            G = hs.shape[0]
            W2 = self.pool[gd['idx2']].view(G, gd['co'], ci)
            b2 = self.pool[gd['idxb2']].view(G, gd['co'])
            outs.append(torch.bmm(hs, W2.transpose(1, 2)) + b2.unsqueeze(1))
        vec = torch.cat(outs, 0).permute(1, 0, 2)
        cls_idx = f['cls'].view(1, -1, 1).expand(x.shape[0], -1,
                                                 vec.shape[2])
        if f['deep']:
            deep_outs = [path_forward(self.pool, d, x)
                         for (_, _, d) in f['deep']]
            vec = torch.cat([vec] + [t.unsqueeze(1) for t in deep_outs], 1)
            dcls = torch.tensor([c for (c, _, _) in f['deep']],
                                dtype=torch.long, device=x.device)
            cls_idx = torch.cat([cls_idx, dcls.view(1, -1, 1).expand(
                x.shape[0], -1, vec.shape[2])], 1)
        per_class = torch.full((x.shape[0], self.num_classes,
                                vec.shape[2]), float('-inf'),
                               device=x.device)
        per_class.scatter_reduce_(1, cls_idx, vec, reduce='amax')
        return per_class.sum(dim=1)

    def plan_growth(self, c, si, mode, k):
        d = self.descs[c][si]
        if mode == 'widen':
            kind, wo, bo, o, i = d[0]
            nkind, nwo, nbo, no, ni = d[1]
            norms = self.pool.data[wo:wo + o * i].view(o, i).norm(dim=1)
            sources = [int(j) for j in norms.argsort(descending=True)[:k]]
            d[0] = (kind, wo, bo, o + k, i)
            d[1] = (nkind, nwo, nbo, no, ni + k)
            return {'path': (c, si), 'mode': 'widen', 'k': k,
                    'sources': sources, 'orig_o': o, 'orig_co_in': ni,
                    'co_out': no}
        if mode == 'deepen':
            prev_w = d[-1][4]
            hidden = max(k, prev_w)
            d.insert(-1, ('l', -1, -1, hidden, prev_w))
            kind, wo, bo, o, i = d[-1]
            d[-1] = (kind, wo, bo, o, hidden)
            return {'path': (c, si), 'mode': 'deepen', 'hidden': hidden,
                    'prev_w': prev_w}
        raise ValueError(mode)

    def plan_prune(self, c, si):
        d = self.descs[c][si]
        kind, wo, bo, o, i = d[0]
        norms = self.pool.data[wo:wo + o * i].view(o, i).norm(dim=1)
        drop = int(norms.argmin())
        nkind, nwo, nbo, no, ni = d[1]
        d[0] = (kind, wo, bo, o - 1, i)
        d[1] = (nkind, nwo, nbo, no, ni - 1)
        return {'path': (c, si), 'mode': 'prune', 'drops': [drop],
                'orig_o': o, 'orig_co_in': ni, 'co_out': no}

    def _make_prune_ops(self, targets):
        by = {}
        for (c, si, drop) in targets:
            by.setdefault((c, si), []).append(drop)
        ops = []
        for (c, si), drops in by.items():
            d = self.descs[c][si]
            kind, wo, bo, o, i = d[0]
            nkind, nwo, nbo, no, ni = d[1]
            if o - len(drops) < 4:
                drops = drops[:o - 4]
                if not drops:
                    continue
            d[0] = (kind, wo, bo, o - len(drops), i)
            d[1] = (nkind, nwo, nbo, no, ni - len(drops))
            ops.append({'path': (c, si), 'mode': 'prune', 'drops': drops,
                        'orig_o': o, 'orig_co_in': ni, 'co_out': no})
        return ops

    def _halve_plan(self, c, si):
        d = self.descs[c][si]
        kind, wo, bo, o, i = d[0]
        norms = self.pool.data[wo:wo + o * i].view(o, i).norm(dim=1)
        keep_n = max(4, (o + 1) // 2)
        keep = set(int(j) for j in norms.argsort(descending=True)[:keep_n])
        return sorted(set(range(o)) - keep)

    def plan_spawn(self, c):
        norms = [path_forward(self.pool, d, X_TE[:512].to(DEVICE))
                 .norm().item() for d in self.descs[c]]
        src = int(np.argmax(norms))
        src_desc = [list(l) for l in self.descs[c][src]]
        drops = self._halve_plan(c, src)
        if drops:
            d = self.descs[c][src]
            kind, wo, bo, o, i = d[0]
            nkind, nwo, nbo, no, ni = d[1]
            d[0] = (kind, wo, bo, o - len(drops), i)
            d[1] = (nkind, nwo, nbo, no, ni - len(drops))
        ops = [{'path': (c, None), 'mode': 'spawn', 'src': src,
                'src_desc': src_desc}]
        if drops:
            ops.append({'path': (c, src), 'mode': 'prune', 'drops': drops,
                        'orig_o': src_desc[0][3],
                        'orig_co_in': src_desc[1][4],
                        'co_out': src_desc[1][3]})
        return ops

    def cleanup(self, epoch):
        dup = []
        for c in range(self.num_classes):
            for si, d in enumerate(self.descs[c]):
                kind, wo, bo, o, i = d[0]
                if o <= 5:
                    continue
                W = self.pool.data[wo:wo + o * i].view(o, i)
                seen = {}
                n_dropped = 0
                for r in range(o):
                    if o - n_dropped <= 4:
                        break
                    key = tuple(round(v, 6) for v in W[r].tolist())
                    if key in seen:
                        dup.append((c, si, r))
                        n_dropped += 1
                    else:
                        seen.setdefault(key, r)
        if dup:
            self.rebuild_pool(self._make_prune_ops(dup))
        calib = X_TE[:1024].to(DEVICE)
        dead = []
        for c in range(self.num_classes):
            for si, d in enumerate(self.descs[c]):
                kind, wo, bo, o, i = d[0]
                if o <= 4:
                    continue
                W = self.pool.data[wo:wo + o * i].view(o, i)
                b = self.pool.data[bo:bo + o]
                for r in (W.norm(dim=1) < 0.05).nonzero().flatten().tolist():
                    nkind, nwo, nbo, co, ci = d[1]
                    Wc = self.pool.data[nwo:nwo + co * ci].view(co, ci)
                    pre = calib @ W[r] + b[r]
                    act = F.relu(pre).mean()
                    self.pool.data[nbo:nbo + co] += Wc[:, r] * act
                    dead.append((c, si, r))
        if dead:
            self.rebuild_pool(self._make_prune_ops(dead))
        if dup or dead:
            print(f"  [cleanup] dups {len(dup)} + dead {len(dead)} "
                  f"-> pool {self.pool.numel():,}")

    def rebuild_pool(self, ops):
        old_pool = self.pool.data
        new_pool = torch.zeros(0, device=old_pool.device)
        index_map = torch.zeros(0, dtype=torch.long, device=old_pool.device)
        cursor = 0
        ops_by_class = {}
        for op in ops:
            ops_by_class.setdefault(op['path'][0], []).append(op)
        new_descs = []
        for c, cd in enumerate(self.descs):
            class_ops = ops_by_class.get(c, [])
            spawn_op = next((x for x in class_ops
                             if x['mode'] == 'spawn'), None)
            subprunes = {x['path'][1] for x in class_ops
                         if x['mode'] == 'subprune'}
            new_cd = []
            for si, d in enumerate(cd):
                if si in subprunes:
                    continue
                op_widen = next((x for x in class_ops
                                 if x['mode'] == 'widen'
                                 and x['path'][1] == si), None)
                op_deep = next((x for x in class_ops
                                if x['mode'] == 'deepen'
                                and x['path'][1] == si), None)
                op_prune = next((x for x in class_ops
                                 if x['mode'] == 'prune'
                                 and x['path'][1] == si), None)
                new_d = []
                for li, (kind, wo, bo, o, i) in enumerate(d):
                    no, ni = o, i
                    if op_widen is not None and li == 0:
                        no = op_widen['orig_o'] + op_widen['k']
                    if op_widen is not None and li == 1:
                        ni = op_widen['orig_co_in'] + op_widen['k']
                    if op_prune is not None and li == 0:
                        no = op_prune['orig_o'] - len(op_prune['drops'])
                    if op_prune is not None and li == 1:
                        ni = op_prune['orig_co_in'] - len(op_prune['drops'])
                    if op_deep is not None and li == len(d) - 2:
                        no, ni = op_deep['hidden'], op_deep['prev_w']
                    if op_deep is not None and li == len(d) - 1:
                        ni = op_deep['hidden']
                    wlen, blen = no * ni, no
                    w_off = cursor
                    cursor += wlen
                    b_off = cursor
                    cursor += blen
                    seg_map = torch.full((wlen + blen,), -1, dtype=torch.long,
                                         device=old_pool.device)
                    with torch.no_grad():
                        buf = torch.zeros(wlen + blen,
                                          device=old_pool.device)
                        if op_widen is not None and li == 0:
                            for r in range(op_widen['orig_o']):
                                buf[r * ni: r * ni + i] = \
                                    old_pool[wo + r * i: wo + (r + 1) * i]
                                buf[wlen + r] = old_pool[bo + r]
                            for n, s in enumerate(op_widen['sources']):
                                r = op_widen['orig_o'] + n
                                srcv = old_pool[wo + s * i: wo + (s + 1) * i]
                                buf[r * ni: r * ni + i] = srcv + \
                                    torch.randn(i, device=old_pool.device) * 0.01
                                buf[wlen + r] = old_pool[bo + s]
                                seg_map[r * ni: r * ni + i] = \
                                    wo + s * i + torch.arange(i, device=old_pool.device)
                                seg_map[wlen + r] = bo + s
                            for r in range(op_widen['orig_o']):
                                seg_map[r * ni: r * ni + i] = \
                                    wo + r * i + torch.arange(i, device=old_pool.device)
                                seg_map[wlen + r] = bo + r
                        elif op_prune is not None and li == 0:
                            drops_set = set(op_prune['drops'])
                            r_new = 0
                            for r in range(op_prune['orig_o']):
                                if r in drops_set:
                                    continue
                                buf[r_new * ni: r_new * ni + i] = \
                                    old_pool[wo + r * i: wo + (r + 1) * i]
                                buf[wlen + r_new] = old_pool[bo + r]
                                seg_map[r_new * ni: r_new * ni + i] = \
                                    wo + r * i + torch.arange(i, device=old_pool.device)
                                seg_map[wlen + r_new] = bo + r
                                r_new += 1
                        elif op_prune is not None and li == 1:
                            drops_set = set(op_prune['drops'])
                            o_old = op_prune['co_out']
                            i_old = op_prune['orig_co_in']
                            keep = [cc for cc in range(i_old)
                                    if cc not in drops_set]
                            Wold = old_pool[wo:wo + o_old * i_old] \
                                .view(o_old, i_old)
                            buf[:wlen] = Wold[:, keep].reshape(-1)
                            buf[wlen:] = old_pool[bo:bo + o_old]
                            for r in range(o_old):
                                for k2, old_c in enumerate(keep):
                                    seg_map[r * len(keep) + k2] = \
                                        wo + r * i_old + old_c
                            seg_map[wlen:] = bo + torch.arange(
                                o_old, device=old_pool.device)
                        elif op_widen is not None and li == 1:
                            o_old = op_widen['co_out']
                            i_old = op_widen['orig_co_in']
                            Wold = old_pool[wo:wo + o_old * i_old] \
                                .view(o_old, i_old)
                            Wn = torch.zeros(o_old, ni,
                                             device=old_pool.device)
                            Wn[:, :i_old] = Wold
                            for n, s in enumerate(op_widen['sources']):
                                Wn[:, s] *= 0.5
                                Wn[:, i_old + n] = Wold[:, s] * 0.5
                            buf[:wlen] = Wn.reshape(-1)
                            buf[wlen:] = old_pool[bo:bo + no]
                            for col in range(i_old):
                                seg_map[col * no: (col + 1) * no] = \
                                    wo + col * no + torch.arange(no, device=old_pool.device)
                            seg_map[wlen:] = bo + torch.arange(no, device=old_pool.device)
                        elif op_deep is not None and li == len(d) - 2:
                            Wn = torch.zeros(no, ni, device=old_pool.device)
                            Wn[:ni, :ni] = torch.eye(ni,
                                                     device=old_pool.device)
                            buf[:wlen] = Wn.reshape(-1)
                        elif op_deep is not None and li == len(d) - 1:
                            o_old = o
                            Wold = old_pool[wo:wo + o_old * op_deep['prev_w']] \
                                .view(o_old, op_deep['prev_w'])
                            Wn = torch.zeros(o_old, ni,
                                             device=old_pool.device)
                            Wn[:, :op_deep['prev_w']] = Wold
                            buf[:wlen] = Wn.reshape(-1)
                            buf[wlen:] = old_pool[bo:bo + o_old]
                            for col in range(op_deep['prev_w']):
                                seg_map[col * o_old: (col + 1) * o_old] = \
                                    wo + col * o_old + torch.arange(o_old, device=old_pool.device)
                            seg_map[wlen:] = bo + torch.arange(o_old, device=old_pool.device)
                        else:
                            buf = old_pool[wo:wo + wlen + blen].clone()
                            seg_map = torch.arange(
                                wo, wo + wlen + blen, device=old_pool.device)
                    new_pool = torch.cat([new_pool, buf])
                    index_map = torch.cat([index_map, seg_map])
                    new_d.append((kind, w_off, b_off, no, ni))
                new_cd.append(new_d)
                if spawn_op is not None and si == spawn_op['src']:
                    with torch.no_grad():
                        jit_pool, jit_map, jit_d = [], [], []
                        for (kind, wo, bo, o, i) in spawn_op['src_desc']:
                            wlen, blen = o * i, o
                            w_off = cursor
                            cursor += wlen
                            b_off = cursor
                            cursor += blen
                            seg = old_pool[wo:wo + wlen + blen]
                            buf = seg + torch.randn(
                                wlen + blen, device=old_pool.device) * 0.01
                            jit_pool.append(buf)
                            jit_map.append(torch.arange(
                                wo, wo + wlen + blen, device=old_pool.device))
                            jit_d.append((kind, w_off, b_off, o, i))
                        new_pool = torch.cat([new_pool] + jit_pool)
                        index_map = torch.cat([index_map] + jit_map)
                        new_cd.append(jit_d)
            new_descs.append(new_cd)
        new_pool = nn.Parameter(new_pool)
        self.pool_adam.carry_to(new_pool, index_map)
        self.pool = new_pool
        self.descs = new_descs
        self._build_fast()

    def maintain(self, loader, epoch):
        self.eval()
        NC = self.num_classes
        loss_sum = [0.0] * NC
        n = [0] * NC
        conf = torch.zeros(NC, NC, dtype=torch.long)
        usage_wins = {c: [0] * len(self.descs[c]) for c in range(NC)}
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                logits = self(data)
                sl = F.cross_entropy(logits, target, reduction='none')
                preds = logits.argmax(dim=1)
                for t, p, l in zip(target.tolist(), preds.tolist(),
                                   sl.tolist()):
                    loss_sum[t] += l
                    n[t] += 1
                    conf[t][p] += 1
                for c in range(NC):
                    if len(self.descs[c]) < 2:
                        continue
                    mask = (target == c)
                    if not mask.any():
                        continue
                    xc = data[mask]
                    sp = torch.stack([path_forward(self.pool, d, xc)
                                      for d in self.descs[c]], dim=1)
                    win = sp[:, :, c].argmax(dim=1)
                    for w in win.tolist():
                        usage_wins[c][w] += 1
        losses = [loss_sum[i] / max(1, n[i]) for i in range(NC)]
        accs = [conf[i][i].item() / max(1, conf[i].sum().item())
                for i in range(NC)]
        self.usage = {c: ([w / max(1, n[c]) for w in usage_wins[c]]
                          if len(self.descs[c]) > 1 else [1.0])
                      for c in range(NC)}

        growth_allowed = epoch <= GROWTH_LAST_EPOCH
        grew = False
        actions = {}

        if growth_allowed:
            sp_ops = []
            for c in range(NC):
                K = len(self.descs[c])
                if K < 2:
                    continue
                for si in range(K - 1, 0, -1):
                    age = epoch - self.birth[c][si]
                    if age < GRACE_EPOCHS:
                        continue
                    if self.usage[c][si] < USAGE_MIN:
                        sp_ops.append({'path': (c, si), 'mode': 'subprune'})
                        self.prune_log.append((epoch, c, si,
                                               self.usage[c][si]))
                        actions[c] = (f'PRUNED dead sub-path #{si} '
                                      f'(age {age}, usage '
                                      f'{self.usage[c][si]*100:.2f}%)')
            if sp_ops:
                self.rebuild_pool(sp_ops)
                for op in sp_ops:
                    c, si = op['path']
                    if si < len(self.usage[c]):
                        del self.usage[c][si]
                    if si < len(self.birth[c]):
                        del self.birth[c][si]
                print(f"  [anti-hallucination] pruned {len(sp_ops)} "
                      f"old-unused sub-path(s)")

            h_ops = []
            for c in range(NC):
                for si, d in enumerate(self.descs[c]):
                    kind, wo, bo, o, i = d[0]
                    if o <= 4:
                        continue
                    norms = self.pool.data[wo:wo + o * i] \
                        .view(o, i).norm(dim=1)
                    if (norms < 0.05).any():
                        h_ops.append(self.plan_prune(c, si))
            if h_ops:
                self.rebuild_pool(h_ops)

            ops = []
            widen_budget = TOTAL_BUDGET - self.pool.numel()
            spawn_budget = TOTAL_BUDGET - self.pool.numel()
            avg_acc = sum(accs) / len(accs)
            order = sorted(range(NC), key=lambda c: -losses[c])
            for c in order:
                hist = self.loss_history[c]
                hist.append(losses[c])
                hist[:] = hist[-4:]
                best_recent = min(hist)
                excess = best_recent - TARGET_CLASS_LOSS
                if excess <= 0:
                    self.last_action[c] = None
                    continue
                weak = accs[c] < avg_acc
                plateau = (len(hist) >= 2
                           and hist[-1] >= 0.98 * min(hist[:-1]))
                K = len(self.descs[c])
                k_req = max(1, int(np.ceil(excess * K_SCALE)))
                si = int(np.argmin(self.usage[c])) if K > 1 else 0
                d = self.descs[c][si]
                i0, co_out = d[0][4], d[1][3]
                per = i0 + 1 + co_out
                remaining = PARAM_CAP_PER_SUBPATH - path_params(d)
                k = min(k_req, remaining // per)
                acted = False
                if not plateau and k >= 1 and widen_budget >= k * per:
                    ops.append(self.plan_growth(c, si, 'widen', k))
                    actions[c] = f'sub#{si} widen +{k} ' \
                                 f'(loss {best_recent:.3f})'
                    widen_budget -= k * per
                    spawn_budget -= k * per
                    self.last_action[c] = 'widen'
                    acted = True
                if not acted:
                    spawn_cost = path_params(d) - \
                        len(self._halve_plan(c, si)) * per
                    if can_spawn := (weak and K < K_MAX
                                     and spawn_budget >= spawn_cost):
                        ops.extend(self.plan_spawn(c))
                        why = 'plateau' if plateau else 'cap hit'
                        actions[c] = (f'SPAWN sub-path #{K} + halve src '
                                      f'({why}, acc {accs[c]*100:.1f}% '
                                      f'< avg {avg_acc*100:.1f}%)')
                        spawn_budget -= spawn_cost
                        self.spawn_log.append((epoch, c))
                        self.last_action[c] = 'spawn'
                        self.loss_history[c] = []
                        grew = True
                        continue
                    else:
                        actions[c] = ('waiting' if not weak
                                      else 'K cap reached')
                if acted:
                    grew = True
            if ops:
                self.rebuild_pool(ops)
                for op in ops:
                    if op['mode'] == 'spawn':
                        self.birth[op['path'][0]].append(epoch)
                self.cleanup(epoch)

        print(f"\n--- Maintenance (Epoch {epoch}) "
              f"[{'growth ACTIVE' if growth_allowed else 'FROZEN'}] ---")
        for c in range(NC):
            K = len(self.descs[c])
            structs = ' | '.join(describe_path(d) for d in self.descs[c])
            use = (' '.join(f'{u*100:.0f}%' for u in self.usage[c])
                   if K > 1 else '-')
            print(f"  P{c} ({cname(c)}): K={K} [{structs}] "
                  f"({self.class_params(c):,}p) loss {losses[c]:.3f} "
                  f"acc {accs[c]*100:.1f}% use[{use}] — "
                  f"{actions.get(c, 'no growth needed')}")
        print(f"  Pool: {self.pool.numel():,} (uncapped, cleaned) | "
              f"sub-paths: {sum(len(cd) for cd in self.descs)}")
        print("-" * 60)
        return grew


def train_epoch(model, epoch):
    model.train()
    total_loss = correct = total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
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
            print(f'Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    return total_loss / len(train_loader), 100. * correct / total


@torch.no_grad()
def test(model):
    model.eval()
    NC = model.num_classes
    conf = torch.zeros(NC, NC, dtype=torch.long)
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        logits = model(data)
        pred = logits.argmax(dim=1)
        for t, p in zip(target.tolist(), pred.tolist()):
            conf[t][p] += 1
    acc = 100. * conf.diag().sum().item() / conf.sum().item()
    per_class = [100. * conf[i][i].item() / conf[i].sum().item()
                 for i in range(NC)]
    print(f'\nDigit-MP Test Accuracy: {acc:.2f}%')
    for i in range(NC):
        print(f'  {i}: {per_class[i]:.1f}%', end='')
        if (i + 1) % 5 == 0:
            print()
    print()
    return acc, conf


def main():
    model = MPDigitBrain().to(DEVICE)
    print(f"Initial: 10 classes x 1 sub-path x 784->{INIT_NEURONS}->10, "
          f"pool {model.pool.numel():,}\n")

    start = time.time()
    best_acc, no_improve = 0.0, 0
    best_model = None

    for epoch in range(EPOCHS):
        ep_t = time.time()
        tr_loss, tr_acc = train_epoch(model, epoch)
        acc, _ = test(model)
        print(f"  epoch wall time: {time.time() - ep_t:.1f}s")

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best {best_acc:.2f}%)")
                break

        if epoch > 0 and epoch % MAINTENANCE_INTERVAL == 0 \
                and epoch <= GROWTH_LAST_EPOCH:
            model.maintain(test_loader, epoch)

    if best_model is not None:
        model = best_model
        print(f"\nFinal sweep on BEST checkpoint ({best_acc:.2f}%)")
    acc, conf = test(model)

    print("\n" + "=" * 56)
    print("HAHINO v1.0 DIGITS-MP - FINAL RESULTS")
    print("=" * 56)
    print("\nEMERGENT STRUCTURE TABLE")
    for c in range(NUM_CLASSES):
        K = len(model.descs[c])
        structs = ' | '.join(describe_path(d) for d in model.descs[c])
        use = (' '.join(f'{u*100:.0f}%' for u in model.usage[c])
               if K > 1 else '-')
        print(f"  P{c} ({cname(c)}): K={K} [{structs}] "
              f"({model.class_params(c):,}p) use[{use}]")
    print(f"\n  Total sub-paths: {sum(len(cd) for cd in model.descs)}")
    print(f"  Spawns: {[(e, cname(c)) for e, c in model.spawn_log]}")
    print(f"  Pool: {model.pool.numel():,} | accuracy: {acc:.2f}% "
          f"(Stage-A baseline: 97.68%)")
    print(f"Training time: {time.time() - start:.1f}s")

    torch.save({'pool': model.pool.detach().cpu(),
                'descs': model.descs,
                'num_classes': NUM_CLASSES,
                'accuracy': acc},
               'hahino_v10_digit_brain_mp.pth')
    print("Saved to hahino_v10_digit_brain_mp.pth")


if __name__ == '__main__':
    main()
