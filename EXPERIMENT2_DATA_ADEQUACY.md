# Experiment 2 — Data Adequacy Addendum

Committed before any Experiment-2 run results are observed.

The main preregistered pass/kill thresholds remain unchanged.

## Adequacy requirement

Before blind forecaster evaluation, the generated run corpus must contain:

- at least **2 observed event categories in every domain**;
- at least **3 observed event categories globally**;
- at least **2 completed runs** for every event category used in a held-out-domain AUROC calculation.

If these conditions fail, Experiment 2 is marked **INVALID_DATA_GENERATION**, not pass or fail.

Exactly one repair is allowed, and only to the run-generation configurations (learning rate, regularization, data fraction, or budget) needed to produce heterogeneous outcomes. The telemetry feature definitions, event-labeling code, observation fractions, evaluator, baselines, and pass/kill thresholds may not be changed.

After such a repair, all Experiment-2 runs are regenerated from scratch before evaluation.
