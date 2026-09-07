# Experiment 2 v2 — Preflight Config Repair 1

Date: 2026-09-07

This is the single run-generation/config-only repair allowed before blind predictor evaluation.

No A/B/C/D forecaster has been fitted or evaluated.

## Observed seed-0 preflight

- CIFAR-10 CNN:
  - overfit_or_memorization: 1
  - stagnation: 4
  - no_event: 0
- Fashion-MNIST MLP:
  - no_event: 4
  - stagnation: 1
- Modular Transformer:
  - divergence: 1
  - no_event: 2
  - overfit_or_memorization: 2
- Synthetic GRU:
  - no_event: 1
  - overfit_or_memorization: 2
  - stagnation: 2

## Why the original preflight gate was insufficient

The v2 seed-0 preflight checked only that each domain had >=2 outcome classes.

That is not enough for leave-one-domain-out prognosis:

1. `divergence` appeared only in the modular domain. If modular were held out and this persisted, the training domains would contain no divergence examples.
2. CIFAR contained no `no_event` / right-censored healthy run. That makes held-out false-stop / healthy-vs-harmful behavior poorly identified for CIFAR.

## Frozen repair

Only inducing configuration parameters are changed:

- Fashion-MNIST high-LR is made more aggressive to create a cross-domain divergence example.
- Fashion-MNIST low-LR is reduced further to make stagnation more reliably distinct from healthy completion.
- CIFAR high-LR is made more aggressive to create another cross-domain divergence example.
- CIFAR healthy receives more training data and budget so the domain contains at least one genuine censored/healthy control.

No architecture, telemetry feature, event definition, A/B/C/D predictor, observation fraction, or pass/kill threshold changes.

## Strengthened preflight adequacy

Before the remaining 60 runs may start, seed-0 must now satisfy all of:

- every domain has >=2 outcome classes;
- every domain contains >=1 `no_event` run;
- every domain contains >=1 observed harmful event;
- every observed event class in seed-0 appears in at least two domains.

If this repaired preflight fails, no second config tuning round is allowed under Experiment 2 v2.

## Fresh output

All pre-repair v2 run artifacts are excluded. The repaired experiment uses a new output root:

`/kaggle/working/tulya_exp2_v2r1`
