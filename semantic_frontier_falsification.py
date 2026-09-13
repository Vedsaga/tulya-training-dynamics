"""Preregistered falsification runner for the Tulya semantic frontier benchmark.

Primary question: among representations already sufficient for the current task family,
does retaining more excess information H(Z|S) reduce uncertainty about unseen future tasks?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import semantic_frontier_v2 as sf

OUT = Path("/kaggle/working/tulya_semantic_falsification")
OUT.mkdir(parents=True, exist_ok=True)

TASKS = tuple(sf.NESTED[:4])
_, _, THEORY_BITS = sf.pstats(sf.Y_np[:, [sf.TASK_NAMES.index(t) for t in TASKS]])
CAPACITIES = [THEORY_BITS, THEORY_BITS + 1, THEORY_BITS + 2]
SEEDS = list(range(10))
N_TRAIN = 1024
STEPS = 3000
EVAL_EVERY = 100
QUAL_HSZ = 0.05
QUAL_JOINT = 0.99
MIN_PAIRED = 5
MANIPULATION_MIN_BITS = 0.05
SUPPORT_MEDIAN_EFFECT = -0.01
SUPPORT_RHO = -0.30


def _select_checkpoint(hist: pd.DataFrame):
    q = hist[(hist.h_semantic_given_code <= QUAL_HSZ) &
             (hist.joint_all_state_accuracy >= QUAL_JOINT)].copy()
    if len(q) == 0:
        return None
    q = q.sort_values(
        ["h_semantic_given_code", "joint_all_state_accuracy", "step"],
        ascending=[True, False, True],
    )
    return q.iloc[0]


def _spearman(x, y):
    x = pd.Series(x).rank(method="average")
    y = pd.Series(y).rank(method="average")
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="pearson"))


def run_primary():
    rows = []
    for bits in CAPACITIES:
        for seed in SEEDS:
            print(f"running bits={bits} seed={seed}")
            cfg = sf.Cfg(
                TASKS,
                bits,
                n_train_states=N_TRAIN,
                steps=STEPS,
                eval_every=EVAL_EVERY,
                seed=seed,
            )
            _, hist, _, summary = sf.run(cfg)
            chosen = _select_checkpoint(hist)
            row = {
                "bits": bits,
                "seed": seed,
                "qualified": chosen is not None,
                "theory_bits": THEORY_BITS,
                "n_train_states": N_TRAIN,
                "step_budget": STEPS,
            }
            if chosen is None:
                best = hist.sort_values(
                    ["h_semantic_given_code", "joint_all_state_accuracy", "step"],
                    ascending=[True, False, True],
                ).iloc[0]
                row.update({
                    "selected_step": int(best.step),
                    "h_semantic_given_code": float(best.h_semantic_given_code),
                    "h_code_given_semantic": float(best.h_code_given_semantic),
                    "joint_all_state_accuracy": float(best.joint_all_state_accuracy),
                    "expected_future_semantic_uncertainty_bits": float(best.expected_future_semantic_uncertainty_bits),
                    "future_recoverable_count": int(best.future_recoverable_count),
                })
            else:
                row.update({
                    "selected_step": int(chosen.step),
                    "h_semantic_given_code": float(chosen.h_semantic_given_code),
                    "h_code_given_semantic": float(chosen.h_code_given_semantic),
                    "joint_all_state_accuracy": float(chosen.joint_all_state_accuracy),
                    "expected_future_semantic_uncertainty_bits": float(chosen.expected_future_semantic_uncertainty_bits),
                    "future_recoverable_count": int(chosen.future_recoverable_count),
                })
            rows.append(row)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "primary_runs.csv", index=False)

    qualified = runs[runs.qualified].copy()
    counts = qualified.groupby("bits").size().to_dict()

    wide = qualified.pivot(index="seed", columns="bits", values=[
        "h_code_given_semantic",
        "expected_future_semantic_uncertainty_bits",
    ])

    pair_rows = []
    b0, b2 = THEORY_BITS, THEORY_BITS + 2
    for seed in SEEDS:
        try:
            h0 = float(wide.loc[seed, ("h_code_given_semantic", b0)])
            h2 = float(wide.loc[seed, ("h_code_given_semantic", b2)])
            u0 = float(wide.loc[seed, ("expected_future_semantic_uncertainty_bits", b0)])
            u2 = float(wide.loc[seed, ("expected_future_semantic_uncertainty_bits", b2)])
        except Exception:
            continue
        if any(np.isnan(v) for v in [h0, h2, u0, u2]):
            continue
        pair_rows.append({
            "seed": seed,
            "delta_excess_bits_6minus4": h2 - h0,
            "delta_future_uncertainty_6minus4": u2 - u0,
            "future_improved": bool(u2 < u0),
        })

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "primary_paired_4bit_vs_6bit.csv", index=False)

    rho = _spearman(
        qualified.h_code_given_semantic,
        qualified.expected_future_semantic_uncertainty_bits,
    ) if len(qualified) else float("nan")

    n_pairs = len(pairs)
    median_excess = float(pairs.delta_excess_bits_6minus4.median()) if n_pairs else float("nan")
    median_u = float(pairs.delta_future_uncertainty_6minus4.median()) if n_pairs else float("nan")
    improved = int(pairs.future_improved.sum()) if n_pairs else 0

    manipulation_ok = bool(n_pairs >= MIN_PAIRED and median_excess >= MANIPULATION_MIN_BITS)

    if not manipulation_ok:
        decision = "INCONCLUSIVE_MANIPULATION_FAILED"
    else:
        required_support_count = max(0, n_pairs - 2)
        supported = (
            improved >= required_support_count and
            median_u <= SUPPORT_MEDIAN_EFFECT and
            np.isfinite(rho) and rho <= SUPPORT_RHO
        )
        falsified = (
            improved <= n_pairs / 2 or
            median_u >= 0 or
            (np.isfinite(rho) and rho >= 0)
        )
        if supported:
            decision = "SUPPORTED_IN_TOY_BENCHMARK"
        elif falsified:
            decision = "FALSIFIED_IN_TOY_BENCHMARK"
        else:
            decision = "INCONCLUSIVE"

    analysis = {
        "preregistration": "SEMANTIC_FRONTIER_FALSIFICATION_PREREGISTRATION.md",
        "tasks": list(TASKS),
        "theory_bits": THEORY_BITS,
        "capacities": CAPACITIES,
        "seeds": SEEDS,
        "n_train_states": N_TRAIN,
        "step_budget": STEPS,
        "qualification": {
            "h_semantic_given_code_max": QUAL_HSZ,
            "joint_all_state_accuracy_min": QUAL_JOINT,
        },
        "qualified_runs_per_capacity": {str(k): int(v) for k, v in counts.items()},
        "qualified_pair_count_4_vs_6": n_pairs,
        "median_delta_excess_bits_6minus4": median_excess,
        "median_delta_future_uncertainty_6minus4": median_u,
        "paired_future_improvement_count": improved,
        "spearman_excess_vs_future_uncertainty": rho,
        "manipulation_check_passed": manipulation_ok,
        "decision": decision,
    }
    (OUT / "primary_analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps(analysis, indent=2))
    return runs, pairs, analysis


if __name__ == "__main__":
    run_primary()
