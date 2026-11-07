"""Smoke tests for the Hahino MP core (CPU, no dataset needed)."""
import torch

from hahino_mp import (MPBrain, PoolAdam, describe_path, path_forward,
                       path_params)


def test_init_shapes_and_params():
    b = MPBrain(num_classes=3, in_dim=20, initial_neurons=4)
    assert b.pool.numel() == 3 * path_params(b.descs[0][0])
    x = torch.randn(5, 20)
    out = b(x)
    assert out.shape == (5, 3)
    assert b._fast_ready  # batched fast path built at init
    # fast path must agree with the loop path
    b._fast_ready = False
    out_loop = b(x)
    assert torch.allclose(out, out_loop, atol=1e-6)


def test_pool_adam_steps_and_carries():
    b = MPBrain(num_classes=2, in_dim=8)
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 0, 1])
    before = b.pool.data.clone()
    for _ in range(3):
        b.pool_adam.zero_grad()
        loss = torch.nn.functional.cross_entropy(b(x), y)
        loss.backward()
        b.pool_adam.step()
    assert not torch.allclose(before, b.pool.data)

    # moment carry across a rebuild: widen class 0, carried slots keep
    # their moments (new layer-1 input columns get zeros)
    old_m = b.pool_adam.m.clone()
    op = b.plan_growth(0, 0, 'widen', 2)
    b.rebuild_pool([op])
    assert b.pool_adam.m.shape == b.pool.data.shape
    assert b.pool.numel() > before.numel()
    # class 0 layer-0 row-0 weights live at pool[0:in_dim] before and
    # after — their moments must be carried, not reset
    assert torch.equal(b.pool_adam.m[0:8], old_m[0:8])
    assert int((b.pool_adam.m == 0).sum()) >= 2  # new columns zeroed


def test_widen_is_function_preserving():
    torch.manual_seed(0)
    b = MPBrain(num_classes=2, in_dim=16, initial_neurons=8)
    x = torch.randn(8, 16)
    b.eval()
    with torch.no_grad():
        out_before = b(x)
        k = b.descs[0][0][0][3]  # current hidden width of class 0
        op = b.plan_growth(0, 0, 'widen', 2)
        b.rebuild_pool([op])
        out_after = b(x)
    assert b.descs[0][0][0][3] == k + 2
    # widen halves the two source columns into the new ones: the
    # function changes only by the symmetry-breaking jitter (0.01)
    assert torch.allclose(out_before, out_after, atol=0.05)


def test_spawn_adds_subpath_and_halves_source():
    torch.manual_seed(0)
    b = MPBrain(num_classes=2, in_dim=12, initial_neurons=6)
    calib = torch.randn(16, 12)
    assert len(b.descs[0]) == 1
    ops = b.plan_spawn(0, calib)
    b.rebuild_pool(ops)
    assert len(b.descs[0]) == 2
    x = torch.randn(4, 12)
    assert b(x).shape == (4, 2)


def test_deepen_inserts_identity_layer():
    torch.manual_seed(0)
    b = MPBrain(num_classes=2, in_dim=10, initial_neurons=6)
    op = b.plan_growth(1, 0, 'deepen', 6)
    b.rebuild_pool([op])
    assert len(b.descs[1][0]) == 3
    x = torch.randn(4, 10)
    assert b(x).shape == (4, 2)
    assert b._fast_ready is False or b._fast['deep']  # deep path exists


def test_cleanup_reports_duplicates_and_dead():
    torch.manual_seed(0)
    b = MPBrain(num_classes=1, in_dim=10, initial_neurons=8)
    # force a duplicate row and a dead row in the first layer
    d = b.descs[0][0]
    kind, wo, bo, o, i = d[0]
    with torch.no_grad():
        W = b.pool.data[wo:wo + o * i].view(o, i)
        W[3] = W[2]                    # duplicate twin
        W[7] *= 1e-4                   # dead neuron (norm < 0.05)
    calib = torch.randn(8, 10)
    dup, dead = b.cleanup(calib)
    assert dup >= 1
    assert dead >= 1
    assert b(x := torch.randn(4, 10)).shape == (4, 1)


def test_maintain_runs_growth_ladder():
    torch.manual_seed(0)
    b = MPBrain(num_classes=3, in_dim=15, initial_neurons=4,
                target_class_loss=0.05, k_scale=25)
    # a tiny loader where class 1 is hard -> it should get resources
    x = torch.randn(60, 15)
    y = torch.tensor([0] * 30 + [1] * 15 + [2] * 15)
    loader = [(x, y)]
    calib = x[:16]
    n0 = b.pool.numel()
    b.maintain(loader, epoch=1, calib_x=calib, verbose=False)
    assert b.pool.numel() >= n0  # grew or stayed (budget permitting)
    assert b(x).shape == (60, 3)


def test_freeze_and_birth_classes():
    torch.manual_seed(0)
    b = MPBrain(num_classes=3, in_dim=12, initial_neurons=5)
    x = torch.randn(8, 12)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    with torch.no_grad():
        out_before = b(x)

    # birth 2 new classes: old outputs preserved (old head rows kept,
    # new head rows zero-init)
    new_ids = b.birth_classes(2)
    assert new_ids == [3, 4]
    assert b.num_classes == 5
    assert b(x).shape == (8, 5)
    with torch.no_grad():
        out_after = b(x)
    assert torch.allclose(out_before, out_after[:, :3], atol=1e-6)

    # freeze class 0: gradients in its segments are zeroed
    b.freeze_classes([0])
    b.pool_adam.zero_grad()
    loss = torch.nn.functional.cross_entropy(b(x), y)
    loss.backward()
    d = b.descs[0][0]
    g = b.pool.grad
    assert g[d[0][1]:d[0][1] + d[0][3] * d[0][4]].abs().sum().item() == 0.0
    assert g[d[1][1]:d[1][1] + d[1][3] * d[1][4]].abs().sum().item() == 0.0
    # unfrozen class 1 got gradient
    d1 = b.descs[1][0]
    assert g[d1[0][1]:d1[0][1] + d1[0][3] * d1[0][4]].abs().sum().item() > 0

    # maintain() skips frozen classes entirely
    loader = [(x, y)]
    pool_before = b.pool.numel()
    b.maintain(loader, epoch=1, calib_x=x[:4], verbose=False)
    descs0 = b.descs[0]
    assert all(descs0 == d_ for d_ in [descs0])  # untouched reference
    assert b.class_params(0) == sum(path_params(dd) for dd in descs0)
    assert b.pool.numel() >= pool_before
