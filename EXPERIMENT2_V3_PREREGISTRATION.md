# Experiment 2 v3 — Shared-Regime Zero-Shot Prognosis

Created after Experiment 2 v2 failed its repaired preflight. No v2 A/B/C/D predictor evaluation was performed.

## What v2 falsified

The stronger assumption that one fine-grained event ontology containing divergence can be naturally instantiated across all cheap domains was not supported. Divergence remained modular-Transformer-only after the single allowed v2 configuration repair.

v3 does not rewrite v2. It asks a narrower fresh question.

## Claim under test

Task-agnostic normalized training telemetry can forecast **shared future training regimes that empirically exist across heterogeneous domains**, zero-shot on an unseen task/architecture, and outperform strong raw telemetry.

The shared outcomes are frozen as:

- `no_event` — right-censored healthy completion
- `stagnation`
- `overfit_or_memorization`

No `divergence` or `delayed_improvement` is part of the v3 target ontology.

## Why these three

The v2 repaired seed-0 development preflight showed:

- `no_event`: modular, Fashion-MNIST, CIFAR-10, GRU
- `stagnation`: Fashion-MNIST, CIFAR-10, GRU
- `overfit_or_memorization`: modular, CIFAR-10, GRU

Thus every v3 target outcome is represented in at least three heterogeneous domains, and for every leave-one-domain-out fold each class remains represented in at least two training domains whenever it appears in the held-out domain.

## Fresh corpus

The v2 seed-0 runs were used only to select recipes and are excluded from v3.

v3 uses fresh seeds:

`100, 101, 102, 103, 104, 105`

No seed 0-3 data is reused.

## Frozen outcome-generation recipes

The recipe name is metadata only and is never a predictor feature or target. Actual completed trajectories determine labels.

### no_event recipes
- modular_transformer: `healthy`
- fashion_mnist_mlp: `healthy`
- cifar10_cnn: `healthy`
- synthetic_sequence_gru: `high_lr`

### stagnation recipes
- fashion_mnist_mlp: `low_lr`
- cifar10_cnn: `low_lr`
- synthetic_sequence_gru: `low_lr`

### overfit_or_memorization recipes
- modular_transformer: `small_train`
- cifar10_cnn: `small_train`
- synthetic_sequence_gru: `small_train`

This is 10 domain/recipe cells × 6 fresh seeds = **60 runs**.

No further recipe tuning is allowed after v3 begins.

## Corpus validity gate

The complete 60-run corpus is INVALID_DATA_GENERATION if any of the following occur:

1. any observed label outside the frozen three-class ontology;
2. any recipe cell matches its intended outcome on fewer than 4 of 6 seeds;
3. for any leave-one-domain-out fold, any class present in the held-out domain has fewer than 4 training-domain runs;
4. any held-out domain contains fewer than 2 observed outcome classes.

If invalid, v3 stops. There is no configuration-repair round.

## Predictor systems

Unchanged from v2:

- A — learning curves
- B — strong raw telemetry
- C — hand-designed normalized/canonical telemetry
- D — raw + normalized hybrid, diagnostic only

D cannot rescue the core C-vs-B verdict.

## Observation fractions

Unchanged:

- 10%
- 20%
- 30%
- 40%

Observed events leave the risk set once they occur. Healthy runs are right-censored.

## Evaluation

Leave one entire domain/architecture out.

Report:

- macro one-vs-rest AUROC
- whole-run bootstrap 95% CI
- multiclass Brier score
- ECE
- event-time MAE on observed events
- warning lead at run-level <=5% false-alert threshold
- in-domain grouped-CV diagnostic
- domain-identifiability diagnostic
- false-stop compute-utility diagnostic

## Core pass/kill gates

Frozen from v2:

### Transfer
- mean zero-shot macro AUROC(C) >= 0.80
- every held-out domain macro AUROC(C) >= 0.75

### Incremental value over raw telemetry
- mean AUROC(C-B) >= +0.05
- no held-out-domain C-B < -0.02
- and either:
  - >=10% relative Brier improvement, or
  - >=15% relative event-time-MAE improvement

Failure of either transfer or incremental-value gate => **KILL_CANONICAL_TRAINING_DYNAMICS_THESIS**.

No post-result feature additions, learned canonicalizer, threshold changes, or ontology changes are allowed inside v3.

## Interpretation

Passing v3 does not prove commercial value or universal training-state representation. It only earns Experiment 3 on a genuine small LM/SFT workload.

Failing v3 kills the current hand-designed canonical-training-dynamics product thesis.
