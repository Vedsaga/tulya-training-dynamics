# Experiment 2 v1 — Aborted Before Evaluation

Date: 2026-09-07

Experiment 2 v1 was stopped during run generation, before any A/B/C forecaster was fitted or evaluated.

Observed run-generation output:

- modular_transformer / healthy / seed 0: `healthy_convergence @ 0.025`
- modular_transformer / healthy / seed 1: `healthy_convergence @ 0.025`
- seed 2 had begun but was interrupted.

## Reason for abort

The preregistered forecast observation fractions begin at 0.10. The v1 event ontology treated ordinary successful convergence as an event, and the first two healthy modular runs reached that event at 0.025.

The evaluator excludes prefixes at or after an event. Therefore these healthy runs would contribute no 10%/20%/30%/40% forecasting examples, creating a structural selection bias and making the survival/prognosis interpretation incoherent.

No predictor performance, normalized-vs-raw comparison, AUROC, Brier score, calibration result, or time-to-event result was observed before this abort.

## v2 repair

This is an ontology/evaluation repair, not a feature-hypothesis repair:

- ordinary healthy completion becomes **right-censored / no event**;
- actual pathologies/transitions retain event times;
- healthy censored runs remain valid negative/risk-set examples at all observation fractions through their censor time;
- timing regression/evaluation excludes censored outcomes;
- original A/B/C feature sets and core pass/kill thresholds remain unchanged.

The v2 run uses a fresh output directory and does not reuse v1 run artifacts.

## Engineering-only change

Kaggle exposes two T4 GPUs. v2 may run independent training runs concurrently in separate subprocesses pinned one-per-GPU. This changes wall-clock throughput only; it does not change models, seeds, telemetry, labels, or evaluator logic.
