"""Hahino MP — core multi-path neural network.

Extracted from the Hahino Brain research project (Paper 2: Multi-Path
Architecture / Paper 3: The Growth Architecture). The code in this
module is the measured, proven machinery, kept as close to the
original research implementation as possible. The only changes from
the research driver (`reference/hahino_v10_digits_mp.py`) are
de-globalization:

- module-level DEVICE      -> operations follow `pool.device`
- hardcoded input dim 784  -> `MPBrain(in_dim=...)`
- global X_TE calibration  -> calibration batch passed to
  `maintain()` / `plan_spawn()` / `cleanup()`

Everything else — the flat parameter pool, (offset, shape) path
descriptors, WTA-within-class / SUM-across readout, the growth ladder
(widen/deepen/spawn/prune), function-preserving rebuilds with
optimizer moment carry, duplicate/dead-neuron cleanup, and the
age-aware anti-hallucination prune — is verbatim from the runs that
produced the Paper 2/3 results.

Laws this module embodies (all measured, see README):
  widths are INIT VALUES, not settings — they grow under loss;
  growth is FUNCTION-PRESERVING (post-growth loss delta literally 0.0);
  a path firing on <1% of samples after a grace period is deleted
  (minimum support);
  the optimizer's moments are carried across every pool rebuild.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['path_forward', 'path_params', 'describe_path',
           'PoolAdam', 'MPBrain']


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
    """Adam over the flat pool; moments carried across pool rebuilds."""

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


class MPBrain(nn.Module):
    """Multi-path brain: one sovereign sub-path set per class.

    - K emergent sub-paths per class (in -> k -> num_classes),
      MAX within class (WTA), SUM across classes. No router anywhere.
    - widen first (loss-excess driven), per-sub-path param cap
    - SPAWN for below-average classes on plateau/cap; the source is
      halved so both sub-paths must work
    - function-preserving rebuilds on a flat parameter pool after
      every topology change, with optimizer moment carry-over
    - cleanup (duplicate twins + dead neurons, bias-compensated)
      after every rebuild
    - age-aware anti-hallucination: <1% usage after a grace period ->
      sub-path deleted
    """

    def __init__(self, num_classes=10, in_dim=784, initial_neurons=4,
                 target_class_loss=0.10, k_scale=25,
                 param_cap_per_subpath=16_000, total_budget=999_999_999,
                 k_max=4, usage_min=0.01, grace_epochs=4):
        super().__init__()
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.target_class_loss = target_class_loss
        self.k_scale = k_scale
        self.param_cap_per_subpath = param_cap_per_subpath
        self.total_budget = total_budget
        self.k_max = k_max
        self.usage_min = usage_min
        self.grace_epochs = grace_epochs
        self.descs = []
        offset = 0
        for c in range(num_classes):
            w = initial_neurons
            d = [('r', offset, offset + w * in_dim, w, in_dim)]
            offset += w * in_dim + w
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
        self.frozen_classes = set()
        self._freeze_mask_cache = None
        self._freeze_hook_handle = None
        self._register_freeze_hook()
        self._fast_ready = False
        self._build_fast()
        self.prune_log = []

    # ---- freeze sovereignty (Law 4) ---------------------------------------
    # Frozen classes' parameters become IMMUTABLE: their gradient segments
    # are zeroed by a hook on the flat pool, and every growth/cleanup
    # path skips them. This is the class-level form of the research
    # protocol (digits frozen while letters trained — v0.6/v0.8 era:
    # digits retained >= 97% after 26 letters were added).

    def freeze_classes(self, classes):
        """Freeze (make immutable) the given classes. Irrevocable per run."""
        self.frozen_classes |= set(int(c) for c in classes)
        self._freeze_mask_cache = None

    def _register_freeze_hook(self):
        if getattr(self, '_freeze_hook_handle', None) is not None:
            self._freeze_hook_handle.remove()
        self._freeze_hook_handle = self.pool.register_hook(
            self._apply_freeze)

    def _frozen_mask(self, device):
        key = (self.pool.numel(), frozenset(self.frozen_classes))
        if self._freeze_mask_cache is not None \
                and self._freeze_mask_cache[0] == key \
                and self._freeze_mask_cache[1].device == device:
            return self._freeze_mask_cache[1]
        mask = torch.zeros(self.pool.numel(), dtype=torch.bool,
                           device=device)
        for c in self.frozen_classes:
            if c >= len(self.descs):
                continue
            for d in self.descs[c]:
                for (kind, wo, bo, o, i) in d:
                    mask[wo:wo + o * i] = True
                    mask[bo:bo + o] = True
        self._freeze_mask_cache = (key, mask)
        return mask

    def _apply_freeze(self, grad):
        if not self.frozen_classes:
            return grad
        return grad.masked_fill(
            self._frozen_mask(grad.device), 0.0)

    # ---- continual growth: birth new classes ------------------------------

    def birth_classes(self, n_new, initial_neurons=None):
        """Birth n_new new classes. Function-preserving: every existing
        parameter (frozen or not) is carried into the new pool
        unchanged; existing heads are re-headed to the new class count
        with the new classes' columns ZERO-INIT (the research-sanctioned
        class-growth init — zero-init prevents leak into old columns).
        Only the new classes' blocks carry fresh weights. Returns the
        new class ids."""
        old_total = len(self.descs)
        total = old_total + n_new
        if initial_neurons is None:
            initial_neurons = max(4, min(self.descs[0][0][0][3], 8))
        old_pool = self.pool.data
        dev = old_pool.device
        new_pool = torch.zeros(0, device=dev)
        index_map = torch.zeros(0, dtype=torch.long, device=dev)
        cursor = 0
        new_descs_all = []
        for c, cd in enumerate(self.descs):
            new_cd = []
            for d in cd:
                new_d = []
                for li, (kind, wo, bo, o, i) in enumerate(d):
                    no, ni = o, i
                    if li == len(d) - 1:
                        no = total          # re-head to new class count
                    wlen, blen = no * ni, no
                    w_off = cursor
                    cursor += wlen
                    b_off = cursor
                    cursor += blen
                    buf = torch.zeros(wlen + blen, device=dev)
                    seg = torch.full((wlen + blen,), -1, dtype=torch.long,
                                     device=dev)
                    with torch.no_grad():
                        if li == len(d) - 1:
                            Wold = old_pool[wo:wo + o * i].view(o, i)
                            buf[:o * i] = Wold.reshape(-1)   # old rows
                            seg[:o * i] = wo + torch.arange(
                                o * i, device=dev)
                            buf[wlen:wlen + o] = old_pool[bo:bo + o]
                            seg[wlen:wlen + o] = bo + torch.arange(
                                o, device=dev)
                            # new classes' head rows stay ZERO
                        else:
                            buf = old_pool[wo:wo + wlen + blen].clone()
                            seg = torch.arange(wo, wo + wlen + blen,
                                               device=dev)
                    new_pool = torch.cat([new_pool, buf])
                    index_map = torch.cat([index_map, seg])
                    new_d.append((kind, w_off, b_off, no, ni))
                new_cd.append(new_d)
            new_descs_all.append(new_cd)
        # fresh blocks for the new classes
        fresh_descs = []
        for _ in range(n_new):
            w = initial_neurons
            d = [('r', cursor, cursor + w * self.in_dim, w, self.in_dim)]
            cursor += w * self.in_dim + w
            d.append(('r', cursor, cursor + total * w, total, w))
            cursor += total * w + total
            wlen = d[0][3] * self.in_dim + d[0][3]
            hlen = d[1][3] * d[1][4] + d[1][3]
            fresh = torch.zeros(wlen + hlen, device=dev)
            with torch.no_grad():
                nn.init.kaiming_uniform_(
                    fresh[:d[0][3] * self.in_dim].view(
                        d[0][3], self.in_dim), a=5 ** 0.5)
                nn.init.kaiming_uniform_(
                    fresh[wlen:wlen + d[1][3] * d[1][4]].view(
                        d[1][3], d[1][4]), a=5 ** 0.5)
                # ZERO the new classes' head rows for EXISTING classes
                # (zero-init class growth — prevents leak into old
                # columns; research finding from the v29 fusion-head
                # mechanics note)
                fresh[wlen:wlen + old_total * w] = 0.0
            new_pool = torch.cat([new_pool, fresh])
            index_map = torch.cat([index_map, torch.full(
                (wlen + hlen,), -1, dtype=torch.long, device=dev)])
            fresh_descs.append([[tuple(l) for l in d]])
        self.pool_adam.carry_to(self._swap_pool(new_pool), index_map)
        new_descs_all.extend(fresh_descs)
        self.descs = new_descs_all
        self.num_classes = total
        for c in range(old_total, total):
            self.loss_history[c] = []
            self.last_action[c] = None
            self.usage[c] = [1.0]
            self.birth[c] = [0]
        self._freeze_mask_cache = None
        self._register_freeze_hook()
        self._build_fast()
        return list(range(old_total, total))

    def _swap_pool(self, new_data):
        """Replace the pool Parameter, preserving the freeze hook."""
        new_pool = nn.Parameter(new_data)
        self.pool = new_pool
        return new_pool

    def class_params(self, c):
        return sum(path_params(d) for d in self.descs[c])

    @property
    def device(self):
        return self.pool.device

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
        dev = self.device
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
                                   for it in items]).to(dev),
                'idxb2': torch.cat([torch.arange(it[3], it[3] + it[4])
                                    for it in items]).to(dev),
                'co': items[0][4],
            }
            cls_list.extend(it[0] for it in items)
        self._fast = {
            'idx1': torch.cat(idx1).to(dev),
            'idxb1': torch.cat(idxb1).to(dev),
            'groups': gdata,
            'cls': torch.tensor(cls_list, dtype=torch.long, device=dev),
            'deep': deep,
        }
        self._fast_ready = True

    def _forward_fast(self, x):
        f = self._fast
        if f['idx1'].device != x.device:
            # plain-dict tensors do not follow module .to(device);
            # rebuild the fast-path indices on the input's device
            self._build_fast()
            f = self._fast
        W1 = self.pool[f['idx1']].view(-1, self.in_dim)
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

    # ---- growth planning -------------------------------------------------

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

    def plan_spawn(self, c, calib_x):
        norms = [path_forward(self.pool, d, calib_x).norm().item()
                 for d in self.descs[c]]
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

    def cleanup(self, calib_x):
        dup = []
        for c in range(self.num_classes):
            if c in self.frozen_classes:
                continue
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
        calib = calib_x
        dead = []
        for c in range(self.num_classes):
            if c in self.frozen_classes:
                continue
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
        return len(dup), len(dead)

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

    # ---- maintenance: measure, grow, clean --------------------------------

    def maintain(self, loader, epoch, calib_x, growth_last_epoch=8,
                 verbose=True):
        """Measure per-class loss/acc/usage; run the growth ladder.

        Returns True if any growth happened. `calib_x` is a small
        representative batch used by spawn selection and dead-neuron
        compensation (the research drivers used the first 512-1024
        test inputs).
        """
        self.eval()
        NC = self.num_classes
        loss_sum = [0.0] * NC
        n = [0] * NC
        conf = torch.zeros(NC, NC, dtype=torch.long)
        usage_wins = {c: [0] * len(self.descs[c]) for c in range(NC)}
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
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

        growth_allowed = epoch <= growth_last_epoch
        grew = False
        actions = {}

        if growth_allowed:
            # age-aware anti-hallucination: delete sub-paths that win
            # <usage_min of their class after grace_epochs
            sp_ops = []
            for c in range(NC):
                if c in self.frozen_classes:
                    continue
                K = len(self.descs[c])
                if K < 2:
                    continue
                for si in range(K - 1, 0, -1):
                    age = epoch - self.birth[c][si]
                    if age < self.grace_epochs:
                        continue
                    if self.usage[c][si] < self.usage_min:
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
                if verbose:
                    print(f"  [anti-hallucination] pruned {len(sp_ops)} "
                          f"old-unused sub-path(s)")

            # prune silent neurons (norm < 0.05) before planning growth
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

            # the growth ladder: widen first; spawn on plateau/cap
            ops = []
            widen_budget = self.total_budget - self.pool.numel()
            spawn_budget = self.total_budget - self.pool.numel()
            avg_acc = sum(accs) / len(accs)
            order = sorted(range(NC), key=lambda c: -losses[c])
            for c in order:
                if c in self.frozen_classes:
                    continue
                hist = self.loss_history[c]
                hist.append(losses[c])
                hist[:] = hist[-4:]
                best_recent = min(hist)
                excess = best_recent - self.target_class_loss
                if excess <= 0:
                    self.last_action[c] = None
                    continue
                weak = accs[c] < avg_acc
                plateau = (len(hist) >= 2
                           and hist[-1] >= 0.98 * min(hist[:-1]))
                K = len(self.descs[c])
                k_req = max(1, int(np.ceil(excess * self.k_scale)))
                si = int(np.argmin(self.usage[c])) if K > 1 else 0
                d = self.descs[c][si]
                i0, co_out = d[0][4], d[1][3]
                per = i0 + 1 + co_out
                remaining = self.param_cap_per_subpath - path_params(d)
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
                    if (weak and K < self.k_max
                            and spawn_budget >= spawn_cost):
                        ops.extend(self.plan_spawn(c, calib_x))
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
                self.cleanup(calib_x)

        if verbose:
            print(f"\n--- Maintenance (Epoch {epoch}) "
                  f"[{'growth ACTIVE' if growth_allowed else 'FROZEN'}] ---")
            for c in range(NC):
                K = len(self.descs[c])
                structs = ' | '.join(describe_path(d) for d in self.descs[c])
                use = (' '.join(f'{u*100:.0f}%' for u in self.usage[c])
                       if K > 1 else '-')
                print(f"  P{c}: K={K} [{structs}] "
                      f"({self.class_params(c):,}p) loss {losses[c]:.3f} "
                      f"acc {accs[c]*100:.1f}% use[{use}] — "
                      f"{actions.get(c, 'no growth needed')}")
            print(f"  Pool: {self.pool.numel():,} | sub-paths: "
                  f"{sum(len(cd) for cd in self.descs)}")
            print("-" * 60)
        return grew
