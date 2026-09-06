# Tulya Training Dynamics

Experimental tooling for forecasting training-regime changes from cheap, training-time internal signals.

The first experiment studies **grokking in modular addition**. The default setup follows the well-studied reference regime: addition mod 113, 30% train split, a one-layer 4-head Transformer with d_model=128 and d_mlp=512, ReLU, no LayerNorm or biases, full-batch AdamW, lr=1e-3, weight decay 1.0, betas=(0.9, 0.98), and 40,000 optimization steps. The goal is not merely to reproduce grokking, but to ask whether metrics available *before* held-out accuracy moves can predict whether/when a run will later generalize.

## Research discipline

Metrics are separated into two groups:

- `feature_*`: available to an online monitor at that training step. These are legitimate forecasting features.
- `label_*`: held-out/test quantities used only retrospectively to define the future event. Do not feed these into an early-warning model.

The first version logs ordinary baselines plus representation and optimization-trajectory diagnostics:

- train loss/accuracy; held-out generalization gap is logged only as a retrospective `label_*`
- learning rate and throughput
- parameter norms and matrix spectral statistics
- gradient norm/RMS/variance and gradient-direction cosine
- optimizer update norm, update/weight ratio, and update-direction cosine
- rolling update-trajectory effective rank, spectral entropy, participation ratio, and directional coherence
- input/final-representation effective rank, spectral entropy, participation ratio, top singular-value energy
- prediction entropy, confidence, and classification margin
- attention entropy, peak attention, and head diversity
- Fourier concentration of embedding/unembedding weights as a task-specific mechanistic baseline
- per-parameter weight/gradient norms in the JSONL diagnostics stream
- CUDA memory use when available
- retrospective test loss/accuracy
- memorization and grokking event times in the run summary

Raw spectral values are also written to JSONL so later analysis is not limited to the scalar summaries chosen today.

## Kaggle quick start

In a Kaggle notebook with GPU enabled, this repository is currently **private**. The safest quick path is to download only the experiment module rather than embedding a token in a Git remote URL. Add a Kaggle Secret named `GITHUB_TOKEN` containing a GitHub token that can read this repository, then run:

```python
from kaggle_secrets import UserSecretsClient
import requests

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
url = "https://api.github.com/repos/Vedsaga/tulya-training-dynamics/contents/kaggle_grokking_experiment.py?ref=main"
response = requests.get(
    url,
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
    },
    timeout=30,
)
response.raise_for_status()
open("/kaggle/working/kaggle_grokking_experiment.py", "wb").write(response.content)
```

If you later make the repository public, you can clone it normally or download the raw file without a token.

Then:

```python
from kaggle_grokking_experiment import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    output_dir="/kaggle/working/tulya_runs/seed_0",
    seed=0,
    max_steps=40_000,
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
checkpoints/        # step 0, every 1000 steps by default, and final state
```

## First falsifiable hypothesis

At a time when two runs both have near-perfect training accuracy and poor held-out accuracy, internal compression/dynamics metrics should help distinguish:

1. a run that will later grok, from
2. a run that will remain memorized within the observation horizon.

A useful signal must work on **unseen runs/seeds/configurations** and beat simple baselines such as train loss, weight norm, and gradient norm.

## Notes

This repository is intentionally separate from `tulya-leaning-model`. That project studies learned storage geometry; this one studies learning/training dynamics.

The code is designed to run with the PyTorch installation already present in Kaggle images.