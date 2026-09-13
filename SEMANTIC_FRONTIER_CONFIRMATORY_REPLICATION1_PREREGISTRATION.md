# Semantic Frontier Confirmatory Replication 1 — preregistration

## Purpose

This is an independent confirmatory replication of the preregistered semantic-frontier falsification experiment. It is intentionally separate from the original experiment so the original result and decision remain unchanged.

The original experiment ended `INCONCLUSIVE_MANIPULATION_FAILED` because only 4 qualified 4-bit/6-bit seed pairs were available, below the preregistered minimum of 5. This replication uses fresh random seeds only.

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

## Frozen design

Held fixed from the original preregistration:

- world: all 4096 twelve-bit states
- current task family: the four tasks above
- training states: 1024
- optimizer/model family/hyperparameters: unchanged from semantic-frontier v2
- training budget: 3000 optimization steps
- evaluation interval: 100 steps
- future task bank and task probabilities: unchanged from v2
- bottleneck capacities: 4, 5, 6 bits
- qualification thresholds
- checkpoint-selection rule
- manipulation threshold
- support/falsification thresholds

The only planned change is the independent seed set:

- fresh seeds: 10 through 39 inclusive

Seeds 0 through 9 from the original experiment are not reused in the confirmatory decision.

## Checkpoint selection

For each run, a checkpoint is **qualified** only if both:

- `H(S|Z) <= 0.05` bits
- joint exhaustive current-task accuracy >= 0.99

Among qualified checkpoints, choose the checkpoint with the smallest `H(S|Z)`; ties are broken by larger joint exhaustive accuracy and then earlier step.

If a run has no qualified checkpoint, it remains unqualified.

## Primary comparison

The primary paired comparison remains 4-bit versus 6-bit bottlenecks for the same seed.

For every qualified paired seed:

- `delta_excess = H(Z|S)_6bit - H(Z|S)_4bit`
- `delta_U = U_6bit - U_4bit`

Negative `delta_U` means the larger representation preserved more information useful for unseen tasks.

Also compute Spearman rank correlation across all qualified 4/5/6-bit runs between `H(Z|S)` and `U`.

## Manipulation check — unchanged

The test is interpretable only if:

- at least 5 qualified paired seeds exist; and
- median `delta_excess >= 0.05` bits.

If this fails, the replication decision is `INCONCLUSIVE_MANIPULATION_FAILED`.

## Decision rule — unchanged

Provided the manipulation check passes:

**Supported in this toy benchmark** only if all three hold:

1. all but at most 2 qualified paired seeds have `delta_U < 0`;
2. median `delta_U <= -0.01` bits; and
3. Spearman correlation between `H(Z|S)` and `U` is <= -0.30.

**Falsified in this toy benchmark** if any strong contrary condition holds:

- no more than half of qualified pairs have `delta_U < 0`; or
- median `delta_U >= 0`; or
- Spearman correlation is >= 0.

Anything between the support and falsification regions is `INCONCLUSIVE`.

## Interpretation

This replication is confirmatory for this finite toy benchmark only. A supported result does not establish a universal law of semantic information or deep learning. A falsified result rejects this proposed `H(Z|S)` generality-premium relationship under the frozen benchmark conditions.

No thresholds, metrics, task probabilities, capacities, or checkpoint rules will be changed after seeing the replication outputs. Exploratory follow-ups must be labeled separately.