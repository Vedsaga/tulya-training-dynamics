# Experiment 2 v3 — Final Result

Status: **CLOSED — KILL_CANONICAL_TRAINING_DYNAMICS_THESIS**

Experiment 2 v3 completed its fresh 60-run corpus and passed the preregistered corpus-validity gate before blind evaluation.

## Corpus

- 60 runs
- fresh seeds 100–105
- four domains / architectures
- frozen shared-regime ontology:
  - no_event
  - stagnation
  - overfit_or_memorization
- every frozen recipe cell matched its intended outcome on 6/6 seeds
- no v2 development runs were reused in the v3 scientific corpus

## Blind zero-shot result

Mean zero-shot macro AUROC:

| System | Mean AUROC |
|---|---:|
| A — learning curves | 0.8623 |
| B — raw telemetry | 0.8710 |
| C — canonical normalized telemetry | 0.8600 |
| D — raw + normalized diagnostic | 0.8749 |

Core C metrics:

- mean AUROC(C): 0.8600
- minimum held-out-domain AUROC(C): 0.7161
- mean AUROC(C-B): -0.0110
- worst held-out-domain AUROC(C-B): -0.2697
- relative Brier improvement vs B: +24.9%
- relative conditional event-time-MAE improvement vs B: +78.2%
- mean ECE(C): 0.2278
- maximum warning FPR(C): 1.0

Frozen core gates failed because:

- not every held-out domain reached AUROC >= 0.75;
- C did not beat B by the required mean +0.05;
- C was worse than B by much more than 0.02 on at least one held-out domain.

Therefore the preregistered verdict is:

**KILL_CANONICAL_TRAINING_DYNAMICS_THESIS**

## Diagnostic interpretation

The strongest diagnostic result is that the proposed normalized coordinates did not remove domain/architecture identity:

- domain-identification AUROC(B): 0.9998
- domain-identification AUROC(C): 1.0000
- domain-identification AUROC(D): 1.0000

Within-domain grouped AUROC was 1.0 for all systems, while zero-shot performance dropped materially. The v3 corpus therefore contains highly predictable regimes, but the hand-designed normalized representation did not expose the claimed domain-independent coordinate system.

The synthetic-GRU held-out fold was particularly damaging to the canonicalization hypothesis:

- B raw telemetry AUROC: 0.9858
- C canonical AUROC: 0.7161
- D hybrid AUROC: 0.9997

This is consistent with normalization discarding useful information rather than producing an approximately sufficient invariant representation.

## Evaluator audit caveats

A post-result audit found two evaluator details that should be recorded:

1. For held-out domains containing only a subset of training-domain event classes, the original macro-AUROC helper renormalized probability over test-present classes. An independent recomputation without that renormalization did not rescue the result; C became slightly worse (mean AUROC approximately 0.858 rather than 0.860, and mean C-B approximately -0.013 rather than -0.011).
2. Event-time MAE is conditional on correct event-type classification, so its large apparent improvement should not be interpreted as unconditional timing superiority.

Neither caveat changes the preregistered kill decision.

## Scientific conclusion

Supported:
- early trajectory information is predictive in these constructed regimes;
- raw telemetry can add useful information in some held-out domains;
- the induced within-domain regimes are highly separable.

Not supported:
- the specific hand-designed normalized/canonical telemetry representation provides a robust zero-shot advantage over strong raw telemetry;
- the proposed normalized coordinates erase domain identity;
- this canonicalization hypothesis has earned continuation to a real-LM product experiment.

This branch is closed. Any learned invariant representation, function-space representation, new normalization, new ontology, or new threshold is a **new hypothesis** and must begin with a fresh literature-boundary search and preregistration rather than being treated as a rescue of v3.
