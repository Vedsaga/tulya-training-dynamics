# Experiment 2 v2 — Censored Cross-Domain Prognosis

Created after Experiment 2 v1 was aborted during run generation and before any predictor evaluation. See `EXPERIMENT2_V1_ABORT.md`.

## Claim

Unchanged from v1:

A task-independent representation of normalized training dynamics can forecast the type and time of future training-regime events on an unseen task/architecture, using only telemetry available up to the forecast time, and can outperform strong raw-telemetry / learning-curve baselines by enough to justify a real-LM Experiment 3.

## What changed from v1

### 1. Healthy completion is right-censored

`healthy_convergence` is removed as an event type.

A run that completes successfully without a defined pathology/transition is:

- event = `no_event`
- event_observed = false
- censor time = final observed training fraction

It remains in the risk set at 10%, 20%, 30%, and 40% as long as that prefix precedes its censor time.

### 2. Actual events

The event ontology remains:

- divergence
- stagnation
- overfit_or_memorization
- delayed_improvement

The generic labeler determines these from the completed trajectory. Intended configuration names are never predictor targets.

### 3. Timing evaluation respects censoring

Censored runs participate in event/no-event probability and calibration evaluation, but are excluded from event-time regression error because their true event time is unobserved.

### 4. Uncertainty is reported

Held-out-domain macro AUROC receives a 95% bootstrap confidence interval by resampling **whole runs**, not individual prefixes.

The original numerical core pass/kill thresholds are unchanged. Confidence intervals are reported to expose sampling uncertainty; they are not permission to move a threshold after seeing results.

### 5. Stronger adequacy check

Before blind evaluation:

- every domain must contain at least two outcome classes including `no_event`;
- at least three outcome classes must exist globally;
- every class evaluated in a held-out domain must have at least two completed training-domain runs available to the forecaster.

If not, the corpus is INVALID_DATA_GENERATION, not a hypothesis pass/fail.

## Observation prefixes

Unchanged:

- 10%
- 20%
- 30%
- 40%

No main-test forecast later than 40%.

## A/B/C systems

Unchanged from v1:

- A: learning curve
- B: strong raw telemetry
- C: task-agnostic normalized/canonical dynamics

Task-specific modular Fourier features remain forbidden from C.

## Core kill gates

Unchanged:

### Transfer
- mean leave-one-domain-out macro AUROC(C) >= 0.80
- every held-out domain macro AUROC(C) >= 0.75

### Incremental value
- mean AUROC(C-B) >= +0.05
- no held-out domain C-B < -0.02
- plus either >=10% relative Brier improvement or >=15% relative event-time-MAE improvement

Failure of either core transfer or incremental-value gate => **KILL_PRODUCT_DIRECTION**.

Other pre-registered engineering/calibration/lead-time gates remain as in v1.

## Dual-GPU execution

Independent training runs may execute concurrently in separate OS processes pinned one-per-visible-GPU.

This is explicitly **not a scientific change**. Each run retains the same model, dataset, seed, hyperparameters, telemetry, event labeler, and output schema it would have under serial execution.

## Fresh corpus

v2 uses a new output root. No v1 metrics, summaries, or partially completed runs may be reused.
