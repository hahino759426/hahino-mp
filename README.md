# Hahino MP — the core multi-path neural network

The self-structuring multi-path (MP) neural network at the heart of
the **Hahino Brain** research project, extracted as a standalone,
dependency-light package (PyTorch + NumPy only).

A multi-path brain grows its own width, spawns its own sub-paths,
prunes its own hallucinations — and every structural decision is made
by the system's own loss signal, not by hand.

## The architecture in one paragraph

Every class owns **K emergent sub-paths** (shallow networks
`in_dim → k → num_classes`) carved as `(offset, shape)` views out of
one **flat parameter pool**. There is **no router**: sub-path outputs
combine by **MAX within class (winner-take-all), SUM across classes**.
Widths are *init values, not settings* — under sustained class loss a
sub-path **widens** (function-preservingly: the new neurons fork from
the highest-norm sources with their outgoing weights halved), on a
plateau or parameter cap a below-average class **spawns** a new
sub-path (the source is halved so both must work), silent neurons are
**pruned** with bias compensation, duplicate twins are cleaned after
every rebuild, and a sub-path winning <1% of its class's samples
after a grace period is deleted (age-aware anti-hallucination). Every
topology change is a **function-preserving pool rebuild** that carries
the Adam moments over to the new layout (`PoolAdam.carry_to`).

## Why it exists (the measured record)

This is the machinery behind Papers 2–3 of the Hahino Brain series
(all numbers from committed logs, seeds fixed):

- **Loss-driven growth = resource allocation**: per-class parameter
  spread (3,165 → 15,910) tracked class difficulty, replicated 3×.
- **Function-preserving growth kills loss spikes**: post-growth loss
  delta literally 0.0.
- **The router family is a failure mode**: any learned per-path or
  per-domain weighting (attention router, domain gate) caused
  starvation; deleting the router entirely (this architecture)
  reached the champion results: **97.68% MNIST @ 50,185 params**,
  letters **91.53%**, unified three-brain stack **94.59% joint @
  34,155 average active params** — widths the system grew itself
  (stem D=56 arrived five independent ways).
- **Minimum support pruning**: a path firing on <1% of samples is
  deleted — only general functions survive.
- The same structure later ported to language (14-channel topic
  experts with soft blending) is the LM layer of the project's
  real-text stack; the pure-MP finding there was that the growth
  ladder is *mechanism*, and patience — not capacity — is usually
  the ceiling.

## Install & run

```bash
pip install torch numpy torchvision   # torchvision only for the demo
python -m pytest tests/               # CPU smoke tests, no dataset
python examples/mnist_mp.py           # full MNIST run (auto-downloads)
python examples/mnist_mp.py --epochs 3 --subset 20000   # quick CPU pass
```

## Usage

```python
from hahino_mp import MPBrain
import torch

brain = MPBrain(num_classes=10, in_dim=784)     # 10 x (784->4->10)
logits = brain(x)                                # WTA/SUM readout

# ... train with brain.pool_adam (zero_grad/step like any optimizer) ...

brain.maintain(val_loader, epoch, calib_x)       # measure + grow + clean
brain.pool.numel()                               # flat pool size
```

`calib_x` is a small representative batch (e.g. first 512–1024
validation inputs) used for spawn source selection and dead-neuron
bias compensation.

## Repository layout

```
hahino_mp/core.py       MPBrain, PoolAdam, path utils (the machinery)
examples/mnist_mp.py    MNIST training demo (the classic use)
tests/test_core.py      CPU smoke tests (no dataset needed)
reference/              the original research driver, unmodified
```

## Provenance & fidelity

`reference/hahino_v10_digits_mp.py` is the original research run
(2026-10-04, committed in the hahino-brain repo), included verbatim.
`hahino_mp/core.py` is that code with exactly three
de-globalizations — device follows the pool tensor instead of a
module-global `DEVICE`; the input dim is a constructor argument
instead of the hardcoded 784; the calibration batch is passed in
instead of read from a global test set. No learning logic, constant,
or growth rule was changed.

## Paper lineage

1. *Self-Structuring Modular Neural Network* (routing era + its
   refutation) · 2. *Multi-Path Architecture* (this code's paper:
   router-free WTA/SUM, the champion) · 3. *The Growth Architecture*
   (canonicalizing stem, expert birth, noise governor) ·
4. *The Language Arc* (sandbox LM; the two caps) ·
5. *The Real-Text Stack* (the full stack on Shakespeare; where this
   MP core became the LM's 14-channel expert layer and the finding
   that vision-style growth transfers nowhere — patience, not
   capacity, is the ceiling at these widths).

## License

MIT (see LICENSE).
