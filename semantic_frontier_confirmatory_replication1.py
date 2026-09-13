"""Confirmatory replication 1 for the Tulya semantic-frontier falsification test.

This runner preserves the original preregistered hypothesis, qualification rules,
manipulation check, and support/falsification logic. Only the random seed set is new.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import semantic_frontier_v2 as sf

OUT = Path("/kaggle/working/tulya_semantic_confirmatory_replication1")
OUT.mkdir(parents=True, exist_ok=True)

PREREG = "SEMANTIC_FRONTIER_CONFIRMATORY_REPLICATION1_PREREGISTRATION.md"
TASKS = tuple(sf.NESTED[:4])
TASK_IDS = [sf.TASK_NAMES.index(t) for t in TASKS]
_, _, THEORY_BITS = sf.pstats(sf.Y_np[:, TASK_IDS])

CAPACITIES = [THEORY_BITS, THEORY_BITS + 1, THEORY_BITS + 2]
SEEDS = list(range(10, 40))
N_TRAIN = 1024
STEPS = 3000
EVAL_EVERY = 100

# Frozen from the first preregistration.
QUAL_HSZ = 0.05
QUAL_JOINT = 0.99
MIN_PAIRED = 5
MANIPULATION_MIN_BITS = 0.05
SUPPORT_MEDIAN_EFFECT = -0.01
SUPPORT_RHO = -0.30


def _select_checkpoint(hist: pd.DataFrame):
    q = hist[
        (hist.h_semantic_given_code <= QUAL_HSZ)
        & (hist.joint_all_state_accuracy >= QUAL_JOINT)
    ].copy()
    if len(q) == 0:
        return None

    q = q.sort_values(
        ["h_semantic_given_code", "joint_all_state_accuracy", "step"],
        ascending=[True, False, True],
    )
    return q.iloc[0]


def _best_unqualified_checkpoint(hist: pd.DataFrame):
    return hist.sort_values(
        ["h_semantic_given_code", "joint_all_state_accuracy", "step"],
        ascending=[True, False, True],
    ).iloc[0]


def _spearman(x, y):
    x = pd.Series(x).rank(method="average")
    y = pd.Series(y).rank(method="average")
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="pearson"))


def _row_from_checkpoint(bits, seed, chosen, qualified):
    return {
        "bits": int(bits),
        "seed": int(seed),
        "qualified": bool(qualified),
        "theory_bits": int(THEORY_BITS),
        "n_train_states": int(N_TRAIN),
        "step_budget": int(STEPS),
        "selected_step": int(chosen.step),
        "h_semantic_given_code": float(chosen.h_semantic_given_code),
        "h_code_given_semantic": float(chosen.h_code_given_semantic),
        "joint_all_state_accuracy": float(chosen.joint_all_state_accuracy),
        "mutual_information_semantic_code": float(chosen.mutual_information_semantic_code),
        "expected_future_semantic_uncertainty_bits": float(
            chosen.expected_future_semantic_uncertainty_bits
        ),
        "prior_weighted_future_unresolved_bits": float(
            chosen.prior_weighted_future_unresolved_bits
        ),
        "future_recoverable_count": int(chosen.future_recoverable_count),
    }


def run_replication():
    rows = []

    for bits in CAPACITIES:
        for seed in SEEDS:
            print(f"running replication1 bits={bits} seed={seed}")
            cfg = sf.Cfg(
                TASKS,
                bits,
                n_train_states=N_TRAIN,
                steps=STEPS,
                eval_every=EVAL_EVERY,
                seed=seed,
            )
            _, hist, _, _ = sf.run(cfg)
            chosen = _select_checkpoint(hist)
            qualified = chosen is not None
            if chosen is None:
                chosen = _best_unqualified_checkpoint(hist)
            rows.append(_row_from_checkpoint(bits, seed, chosen, qualified))

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "replication1_runs.csv", index=False)

    qualified = runs[runs.qualified].copy()
    counts = qualified.groupby("bits").size().to_dict()

    # Paired 4-bit versus 6-bit comparison on seeds where both qualified.
    wide = qualified.pivot(
        index="seed",
        columns="bits",
        values=[
            "h_code_given_semantic",
            "expected_future_semantic_uncertainty_bits",
        ],
    )

    pair_rows = []
    b0, b2 = THEORY_BITS, THEORY_BITS + 2

    for seed in SEEDS:
        try:
            h0 = float(wide.loc[seed, ("h_code_given_semantic", b0)])
            h2 = float(wide.loc[seed, ("h_code_given_semantic", b2)])
            u0 = float(
                wide.loc[
                    seed,
                    ("expected_future_semantic_uncertainty_bits", b0),
                ]
            )
            u2 = float(
                wide.loc[
                    seed,
                    ("expected_future_semantic_uncertainty_bits", b2),
                ]
            )
        except Exception:
            continue

        vals = [h0, h2, u0, u2]
        if any(np.isnan(v) for v in vals):
            continue

        pair_rows.append(
            {
                "seed": int(seed),
                "h_excess_4bit": h0,
                "h_excess_6bit": h2,
                "future_uncertainty_4bit": u0,
                "future_uncertainty_6bit": u2,
                "delta_excess_bits_6minus4": h2 - h0,
                "delta_future_uncertainty_6minus4": u2 - u0,
                "future_improved": bool(u2 < u0),
            }
        )

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "replication1_paired_4bit_vs_6bit.csv", index=False)

    rho = (
        _spearman(
            qualified.h_code_given_semantic,
            qualified.expected_future_semantic_uncertainty_bits,
        )
        if len(qualified)
        else float("nan")
    )

    n_pairs = len(pairs)
    median_excess = (
        float(pairs.delta_excess_bits_6minus4.median())
        if n_pairs
        else float("nan")
    )
    median_u = (
        float(pairs.delta_future_uncertainty_6minus4.median())
        if n_pairs
        else float("nan")
    )
    improved = int(pairs.future_improved.sum()) if n_pairs else 0

    manipulation_ok = bool(
        n_pairs >= MIN_PAIRED and median_excess >= MANIPULATION_MIN_BITS
    )

    if not manipulation_ok:
        decision = "INCONCLUSIVE_MANIPULATION_FAILED"
    else:
        # Exact same support/falsification logic as the first preregistered runner.
        required_support_count = max(0, n_pairs - 2)
        supported = (
            improved >= required_support_count
            and median_u <= SUPPORT_MEDIAN_EFFECT
            and np.isfinite(rho)
            and rho <= SUPPORT_RHO
        )
        falsified = (
            improved <= n_pairs / 2
            or median_u >= 0
            or (np.isfinite(rho) and rho >= 0)
        )

        if supported:
            decision = "SUPPORTED_IN_TOY_BENCHMARK"
        elif falsified:
            decision = "FALSIFIED_IN_TOY_BENCHMARK"
        else:
            decision = "INCONCLUSIVE"

    analysis = {
        "experiment": "confirmatory_replication_1",
        "preregistration": PREREG,
        "independent_of_original_seeds": True,
        "original_seed_range": [0, 9],
        "replication_seed_range": [10, 39],
        "tasks": list(TASKS),
        "theory_bits": int(THEORY_BITS),
        "capacities": [int(x) for x in CAPACITIES],
        "seeds": [int(x) for x in SEEDS],
        "n_train_states": int(N_TRAIN),
        "step_budget": int(STEPS),
        "qualification": {
            "h_semantic_given_code_max": QUAL_HSZ,
            "joint_all_state_accuracy_min": QUAL_JOINT,
        },
        "qualified_runs_per_capacity": {
            str(k): int(v) for k, v in counts.items()
        },
        "qualified_pair_count_4_vs_6": int(n_pairs),
        "median_delta_excess_bits_6minus4": median_excess,
        "median_delta_future_uncertainty_6minus4": median_u,
        "paired_future_improvement_count": int(improved),
        "required_support_improvement_count": int(max(0, n_pairs - 2)),
        "spearman_excess_vs_future_uncertainty": rho,
        "manipulation_check_passed": bool(manipulation_ok),
        "decision": decision,
    }

    (OUT / "replication1_analysis.json").write_text(
        json.dumps(analysis, indent=2)
    )
    print(json.dumps(analysis, indent=2))
    return runs, pairs, analysis


if __name__ == "__main__":
    run_replication()
