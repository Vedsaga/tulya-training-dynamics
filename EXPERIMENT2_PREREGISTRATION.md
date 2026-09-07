# Experiment 2 — Pre-registered Ruthless Falsification

Committed before any Experiment-2 training results are observed.

## Claim under test

A task-independent representation of normalized training dynamics can forecast the **type** and **time** of future training-regime events on an unseen task/architecture, using only telemetry available up to the forecast time, and can outperform strong raw-telemetry / learning-curve baselines by enough to justify eventual economic validation.

This experiment is allowed to kill the project.

## Domains

Four deliberately heterogeneous classification domains:

1. modular arithmetic — Transformer
2. Fashion-MNIST — MLP
3. CIFAR-10 — small CNN
4. synthetic sequence classification — GRU

The downstream forecaster is evaluated with **leave-one-domain-out** splits. No sample, seed, threshold, scaler, or model from the held-out domain may be used to fit or tune the forecaster.

The small synthetic GRU domain is intentionally cheap and heterogeneous. A genuine small language-model transfer test is reserved for Experiment 3 if this experiment passes.

## Training-run generation

For each domain, generate multiple seeds across configurations intended to produce heterogeneous outcomes:

- healthy convergence
- divergence / instability
- stagnation
- memorization / overfitting
- delayed improvement where naturally produced

Outcome labels are derived from the observed complete trajectory by the frozen generic event-labeling code; the intended configuration name is not the target label.

Target: at least 20 completed runs per domain. Preferred: 25+.

## Leakage rule

The prognostic model may use only `feature_*` columns available by the observation step.

`label_*`, validation/test metrics used to define future events, intended-regime names, final outcome, event time, domain identity, architecture identity, and seed are forbidden as predictor inputs.

Hyperparameters such as learning rate may be included only in the strong prior-art baseline if explicitly listed. The candidate canonical model must also be evaluated without domain or architecture identity.

## Observation times

Forecasts are produced from prefixes ending at normalized run fractions:

- 10%
- 20%
- 30%
- 40%

No forecast after 40% of the run counts toward the main pass/fail test.

## Systems compared

### A — Learning-curve baseline

- normalized step fraction
- training loss
- training accuracy
- learning rate

### B — Strong raw-telemetry baseline

A plus:

- raw weight norm / growth
- raw gradient norm / variance
- raw update norm
- activation mean / standard deviation / saturation
- simple prediction confidence / entropy where available

### C — Candidate canonical dynamics

Task-agnostic, dimensionless or normalized quantities only, including:

- update-to-weight ratio
- gradient-to-weight ratio
- relative/log weight contraction
- normalized update effective rank
- normalized spectral entropy
- normalized participation ratio
- update directional coherence
- update cosine to previous update
- gradient cosine to previous gradient
- normalized representation effective rank
- normalized representation spectral entropy
- normalized activation statistics
- slopes / changes of the above over the observed prefix

Task-specific mechanistic features (e.g. modular Fourier structure) are forbidden from C.

The same simple downstream model family and tuning procedure must be used for A, B, and C.

## Outcomes

The frozen event-labeling implementation maps each complete run to one primary event and event time using generic trajectory conditions. Candidate categories:

- healthy_convergence
- divergence
- stagnation
- overfit_or_memorization
- delayed_improvement

If a category is absent in a held-out fold, macro metrics are computed over categories represented in that fold and in forecaster training. The absence is reported, not silently imputed.

## Main evaluation

Across leave-one-domain-out folds:

1. multiclass event discrimination (one-vs-rest macro AUROC)
2. multiclass Brier score / probability quality
3. calibration (ECE)
4. normalized event-time MAE, conditional on event category being forecast correctly
5. warning lead time for harmful events at a threshold selected on training domains only
6. instrumentation overhead

Bootstrap confidence intervals must resample entire runs, not checkpoints.

## PASS / KILL thresholds

All conditions below are frozen before results.

### Transfer quality
- Mean leave-one-domain-out macro AUROC for C >= **0.80**
- Every held-out domain macro AUROC for C >= **0.75**

### Incremental value over strong prior art
C must satisfy both:
- mean macro-AUROC improvement over B >= **+0.05 absolute**
- C may not be worse than B by more than **0.02** on any held-out domain

And at least one:
- mean multiclass Brier score improves by >= **10% relative**, or
- event-time MAE improves by >= **15% relative**

### Calibration
- mean ECE <= **0.10**

### Time-to-event usefulness
- normalized event-time MAE <= **0.15** of total run budget

### Early-warning usefulness
For harmful events that occur after the 10% observation point:
- at <= **5% false-positive rate** (threshold chosen without held-out-domain data),
- median warning lead >= **10% of total run duration**

### Monitoring overhead
- telemetry collection overhead <= **2%** of wall-clock training time in this small-model experiment

Experiment 3, if reached, tightens the production-oriented overhead target to <=0.5%.

## Kill rule

If the candidate fails **any of the two core gates** below, stop the predictive-training-monitor product direction rather than inventing another metric:

1. transfer quality gate, or
2. incremental-value-over-B gate.

Failure of calibration/time/lead/overhead can justify one engineering correction only if the core gates pass and the correction does not change the feature hypothesis.

No threshold changes after results.

## What success would justify

Passing Experiment 2 does **not** prove commercial value or novelty.

It only justifies Experiment 3: freeze the predictor and test transfer to a genuine small language-model/SFT workload. Customer outreach becomes justified only after a real-LM transfer result plus an explicit compute-ROI calculation.
