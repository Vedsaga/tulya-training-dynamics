# Tulya Training Dynamics

Experimental tooling for forecasting training-regime changes from cheap, training-time internal signals.

The first experiment studies **grokking in modular addition**. The goal is not merely to reproduce grokking, but to ask whether metrics available *before* held-out accuracy moves can predict whether/when a run will later generalize.

## Research discipline

Metrics are separated into two groups:

- `feature_*`: available to an online monitor at that training step. These are legitimate forecasting features.
- `label_*`: held-out/test quantities used only retrospectively to define the future event. Do not feed these into an early-warning model.

The first version logs ordinary baselines plus representation and optimization-trajectory diagnostics:

- train loss/accuracy and generalization gap
- learning rate and throughput
- parameter norms and matrix spectral statistics
- gradient norm/RMS/variance and gradient-direction cosine
- optimizer update norm, update/weight ratio, and update-direction cosine
- rolling update-trajectory effective rank, spectral entropy, participation ratio, and directional coherence
- input/final-representation effective rank, spectral entropy, participation ratio, top singular-value energy
- prediction entropy, confidence, and classification margin
- attention entropy, peak attention, and head diversity
- CUDA memory use when available
- retrospective test loss/accuracy
- memorization and grokking event times in the run summary

Raw spectral values are also written to JSONL so later analysis is not limited to the scalar summaries chosen today.

## Kaggle quick start

In a Kaggle notebook with GPU enabled:

```python
!git clone https://github.com/Vedsaga/tulya-training-dynamics.git
%cd tulya-training-dynamics
```

Then:

```python
from kaggle_grokking_experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    output_dir="/kaggle/working/tulya_runs/seed_0",
    seed=0,
    max_steps=30_000,
    eval_every=100,
)

summary = run_experiment(cfg)
summary
```

Or from a notebook shell cell:

```bash
python kaggle_grokking_experiment.py \
  --output-dir /kaggle/working/tulya_runs/seed_0 \
  --seed 0 \
  --max-steps 30000 \
  --eval-every 100
```

Outputs:

```text
config.json
metrics.csv
spectra.jsonl
summary.json
checkpoints/        # only when checkpoint_every > 0
```

## First falsifiable hypothesis

At a time when two runs both have near-perfect training accuracy and poor held-out accuracy, internal compression/dynamics metrics should help distinguish:

1. a run that will later grok, from
2. a run that will remain memorized within the observation horizon.

A useful signal must work on **unseen runs/seeds/configurations** and beat simple baselines such as train loss, weight norm, and gradient norm.

## Notes

This repository is intentionally separate from `tulya-leaning-model`. That project studies learned storage geometry; this one studies learning/training dynamics.

The code is designed to run with the PyTorch installation already present in Kaggle images.