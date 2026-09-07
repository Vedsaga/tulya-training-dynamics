# Experiment 2 v2 — Diagnostic Addendum

Committed before any v2 corpus results are observed.

These analyses are **diagnostic only**. They do not modify or rescue the core pass/kill gates in `EXPERIMENT2_V2_PREREGISTRATION.md`.

Their purpose is to distinguish different failure mechanisms if the core C-vs-B hypothesis fails.

## D — hybrid raw + canonical telemetry

In addition to the frozen A/B/C systems, evaluate:

- D = union(B, C)

D is not eligible for the core pass verdict.

Interpretation:

- C > B and D ≈ C: normalized coordinates are approximately sufficient.
- C > B but D > C: normalization adds transferable information but removes some useful absolute information.
- B > C and D ≈ B: normalization is mostly harmful/redundant.
- D > both B and C: raw scale and normalized dynamics contain complementary information.
- A ≈ B ≈ C ≈ D: little useful transferable telemetry signal at this level.

## In-domain vs zero-shot diagnostic

For A/B/C/D, estimate run-level grouped cross-validation AUROC **within each domain**.

Compare it with leave-one-domain-out AUROC.

- high in-domain + poor zero-shot => predictive structure exists but is domain-specific;
- poor both => telemetry/ontology is insufficient even before transfer;
- high both => true cross-domain transfer is plausible.

No in-domain result can rescue a failed zero-shot core gate.

## Domain-identifiability diagnostic

Using run-grouped cross-validation, predict the domain/architecture identity from B, C, and D prefix features.

This asks whether normalization actually removes nuisance domain information.

Desired qualitative pattern:

```
domain-AUC(C) < domain-AUC(B)
```

while future-event prediction improves.

Low domain identifiability alone is not success: a representation can erase both nuisance and useful signal.

## Stop-now compute-utility diagnostic

For each leave-one-domain-out fold, select the harmful-event warning threshold on training domains only, using **run-level maximum harmful probability** so the training false-run-alert rate is approximately <=5%.

On the held-out domain:

- true harmful alert at observation fraction `o`, event fraction `e`: gross avoided budget = `max(e-o, 0)`;
- false alert on a nonharmful/censored run at `o`: lost budget = `1-o`.

Report normalized net compute utility:

```
U(m) = sum(true_positive_saved_fraction)
       - m * sum(false_stop_remaining_fraction)
```

for false-stop cost multipliers:

- m = 1.0
- m = 1.5
- m = 2.0

Also report the break-even false-stop multiplier:

```
m* = TP_saved / FP_remaining
```

when FP_remaining > 0.

This is a synthetic compute-utility diagnostic, not a production ROI claim. Real customer incident rates and restart/opportunity costs remain external quantities.

## Interpretation discipline

These diagnostics answer *why* a result failed. They do not authorize changing the feature set, event ontology, thresholds, or model after observing v2.

A learned invariant/canonical representation remains a separate future hypothesis and is explicitly not introduced into Experiment 2 v2.
