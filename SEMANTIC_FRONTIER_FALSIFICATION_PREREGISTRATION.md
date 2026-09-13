# Semantic Frontier Falsification — preregistration v1

## Purpose

This experiment is designed to stop exploratory drift and test one narrow claim from the earlier semantic-frontier runs.

The earlier runs established that a learned binary code can approach the exact present-task semantic partition, and suggested that representations retaining extra distinctions may preserve more information for unseen tasks. The next experiment tests that suggestion directly.

## Frozen primary hypothesis

Let `S` be the exact semantic state induced by the four current tasks:

- `bit_0`
- `bit_1`
- `xor_2_3`
- `majority`

Let `Z` be the learned binary bottleneck code.

For representations that are already approximately sufficient for the current tasks, larger excess retained information

`H(Z|S)`

should be associated with lower probability-weighted future semantic uncertainty

`U = E_T[H(Y_T|Z)]`

over the fixed unseen-task distribution already defined in `semantic_frontier_v2.py`.

This is the proposed **generality premium**.

## What is held fixed

- world: all 4096 twelve-bit states
- current task family: the four tasks above
- training states: 1024
- optimizer/model family/hyperparameters: unchanged from semantic-frontier v2
- training budget: 3000 optimization steps
- evaluation interval: 100 steps
- future task bank and task probabilities: unchanged from v2

## What is varied

- bottleneck capacity: 4, 5, 6 bits
- random seed: 0 through 9

The theoretical fixed-width minimum for the current task family is 4 bits.

## Checkpoint selection — frozen before running

For each run, a checkpoint is **qualified** only if both:

- `H(S|Z) <= 0.05` bits
- joint exhaustive current-task accuracy >= 0.99

Among qualified checkpoints, choose the checkpoint with the smallest `H(S|Z)`; ties are broken by larger joint exhaustive accuracy and then earlier step.

If a run has no qualified checkpoint, it is recorded as unqualified and is not silently replaced by a weaker criterion.

## Primary comparison

The primary paired comparison is 4-bit versus 6-bit bottlenecks for the same random seed.

### Manipulation check

The test is interpretable only if the 6-bit condition actually retains more excess information. Require:

- at least 5 qualified paired seeds; and
- median `[H(Z|S)_6bit - H(Z|S)_4bit] >= 0.05` bits.

If this fails, the primary result is **inconclusive — manipulation failed**, not support and not falsification.

### Primary outcome

For every qualified paired seed compute:

`delta_U = U_6bit - U_4bit`.

Negative `delta_U` means the larger representation preserved more information useful for unseen tasks.

Also compute Spearman rank correlation across all qualified 4/5/6-bit runs between `H(Z|S)` and `U`.

## Decision rule

Provided the manipulation check passes:

**Supported in this toy benchmark** only if all three hold:

1. at least 8 of 10 paired seeds have `delta_U < 0` (or all but at most 2 if fewer than 10 pairs qualify);
2. median `delta_U <= -0.01` bits; and
3. Spearman correlation between `H(Z|S)` and `U` is <= -0.30.

**Falsified in this toy benchmark** if any of these strong contrary conditions holds:

- no more than half of qualified pairs have `delta_U < 0`; or
- median `delta_U >= 0`; or
- Spearman correlation is >= 0.

Anything between the support and falsification regions is reported as **inconclusive**.

## Important interpretation limits

A supported result does not prove a universal law of semantic information or deep learning. It supports the generality-premium hypothesis only in this finite controlled benchmark.

A falsified result is useful: it means our current proposed quantity `H(Z|S)` is not sufficient, under this intervention, to explain future-task generality.

The synthetic future-task probabilities are fixed in advance and should not be interpreted as a model of real human task frequencies.

## No post-hoc target changes

After the run, the primary decision above is reported as written. Additional plots, alternative thresholds, grokking analyses, or new task distributions may be exploratory follow-ups, but they must not replace the preregistered primary result.