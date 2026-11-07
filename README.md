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
pip install torch numpy torchvision   # torchvision only for the demos
python -m pytest tests/               # CPU smoke tests, no dataset
python examples/mnist_mp.py           # single-task MNIST run
python examples/mnist_mp.py --epochs 3 --subset 20000   # quick CPU pass
python examples/continual_3dataset_mp.py  # digits -> letters -> fashion,
                                          # no replay (see below)
```

## Continual learning: 3 datasets, no catastrophic forgetting

`examples/continual_3dataset_mp.py` runs the project's
class-incremental protocol on one brain:

1. **Phase A** — train digits (MNIST). Report per-domain accuracy.
2. `freeze_classes(digits)` — from here the digits' parameters are
   **immutable**: a gradient mask on the flat pool (registered hook)
   plus growth-ladder exclusion in `maintain()`. Freeze sovereignty
   (Law 4) at class granularity.
3. **Phase B** — `birth_classes(26)` adds EMNIST-letter classes
   (orientation fix `transpose + flip` applied). Existing paths are
   re-headed to the new class count with the new classes' columns
   **zero-init** (the research-sanctioned class-growth init — it
   prevents leak into old columns), so old outputs are unchanged.
   Train letters only. **Re-measure digits** — retention, not hope.
4. **Phase C** — same for Fashion-MNIST (10 classes).

MEASURED ON THIS EXACT CODE (2026-11-07, RTX 5060, seeds 42, full
test sets — 10k MNIST + 20,800 EMNIST letters + 10k Fashion-MNIST):

| Stage | digits | letters | fashion | gate |
|---|---|---|---|---|
| after its own phase | 96.44% | 89.73% | 86.38% | 99.79% |
| after +26 letters | **96.44%** | 89.73% | — | — |
| after +10 fashion | **96.44%** | **89.73%** | 86.38% | — |

Assembled joint (gate routes raw 784 -> sovereign brain): **90.42%**
46-way; gate routing accuracy 99.4-100% per domain. Retention is
exact — frozen brains are byte-identical after later phases, so their
accuracy cannot move. Honest gap to the research champions
(93.25% 2-domain / letters 91.53%): this demo omits the shared
canonicalizing stem (the README's largest measured lever for
letters), the maturation pass (~+1.1 pts), and neuron-fork surgery.

The research record for this protocol family (committed logs, Paper 1
era): **digits retained >= 97%** (target met) after 26 letters were
added to a trained digit network; forgetting resistance was traced to
mechanism, not luck: the noswitch SUM readout has no router to
collapse, and "forgetting lives in the router, not the paths."

**Transfer + replay (the v08 research protocol, ported to this
core)** — `examples/transfer_replay_mp.py`: digits frozen, 26 letter
classes born, then every training step runs letters (36-way CE) and a
digits replay batch (36-way CE); the freeze hook keeps digits
immutable while letters paths learn calibrated digit-column
suppression from real digit data. Measured 2026-11-07:

3-domain measured table (2026-11-07, full test sets):

| | digits | letters | fashion |
|---|---|---|---|
| baseline | 96.44% | — | — |
| after letters | 97.74% (+1.30) | 89.27% | — |
| after fashion | **97.82% (+1.38)** | **90.24% (+0.97)** | 85.64% |

Every frozen domain IMPROVES as later domains are added — monotone,
no decay. The record for this protocol was "digits IMPROVED +0.10
through joint sum training" — the direction reproduces and the gain
is larger on this core (the replay-side calibration does real work;
frozen weights never move; the improvement is the new paths'
calibrated suppression learned from replay batches). Both continual
recipes in this repo are therefore measured, not assumed:
sovereign-brains+gate = exact retention; shared pool + replay =
retention WITH improvement.

The single-brain class-incremental variant is kept as
`examples/class_incremental_stress.py` — a MEASURED NEGATIVE: one
shared pool cannot hold this (unmasked softmax: digits 96.8% -> 0.0%;
masked + frozen: still broken via OOD misfire and suppression
gradient on shared head rows). Domains must own their experts.

`MPBrain.freeze_classes()` is irrevocable within a run (mirroring the
research protocol where frozen experts are immutable); new capacity
arrives through separate brains (continual recipe) or
`birth_classes()` (single-task class growth).

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
hahino_mp/core.py                  MPBrain, PoolAdam, path utils, plus
                                   freeze sovereignty (Law 4) and
                                   continual class birth
examples/mnist_mp.py               single-task MNIST training demo
examples/continual_3dataset_mp.py  continual learning, the recorded way:
                                   3 sovereign brains + raw-pixel gate
examples/class_incremental_stress.py  measured negative: one shared
                                   pool cannot do continual learning
tests/test_core.py                 CPU smoke tests (no dataset needed)
reference/                         the original research driver, unmodified
```

## Provenance & fidelity

`reference/hahino_v10_digits_mp.py` is the original research run
(2026-10-04, committed in the hahino-brain repo), included verbatim.
`hahino_mp/core.py` is that code with exactly three
de-globalizations — device follows the pool tensor instead of a
module-global `DEVICE`; the input dim is a constructor argument
instead of the hardcoded 784; the calibration batch is passed in
instead of read from a global test set. No learning logic, constant,
or growth rule was changed. The continual-learning additions
(`freeze_classes`, `birth_classes`, the pool gradient hook) are the
class-level form of the same protocol the research drivers implemented
by hand (digits frozen while letters trained) — including its two
measured subtleties: frozen parameters are skipped by every
growth/cleanup path, and new classes' head rows for old classes are
zero-init so birth cannot leak into old columns.

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
