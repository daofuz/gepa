# Active-50 critical-case composition sweep

Protocol: fixed 50 labels; MO=k, BC=47-k, NO=2, BW=1; k=1..46 in step 1; three seeds; frozen DistilBERT with 8 soft tokens and two correctness heads; inverse-expected-cost weighted BCE; no prior; validation-max-accuracy threshold; post-hoc held-out 200.

## Main findings

- Best exact point: MO=26, BC=21, accuracy=91.00% ± 0.41, Mini saving=19.83%, MO recall=93.75%.
- Best point retaining at least 20% Mini saving: MO=38, accuracy=90.67%, saving=25.00%.
- Low-critical proxy MO=1: accuracy=87.83%, saving=61.50%, MO recall=41.67%.
- Old enriched-recipe neighborhood MO=22: accuracy=90.00%, saving=47.17%, MO recall=79.17%.
- Maximum enrichment MO=46: accuracy=89.50%, saving=36.67%, MO recall=66.67%.
- Across all 135 adjacent seed-paired steps: 60 increased accuracy, 55 decreased it, and 20 were unchanged. A single extra MO example is therefore not reliably monotonic at n=50.
- Correlation across exact points: MO count vs accuracy r=0.280; MO count vs Mini saving r=-0.262; MO count vs MO recall r=0.205.

## Windowed trend

| MO range | Accuracy | Mini saving | MO recall | Accuracy slope (pp / +1 MO) |
|---:|---:|---:|---:|---:|
| 1-10 | 88.73% ± 0.52 | 44.58% | 58.75% | +0.097 |
| 11-20 | 89.02% ± 1.03 | 47.85% | 63.54% | -0.096 |
| 21-30 | 89.22% ± 0.19 | 47.42% | 64.38% | +0.019 |
| 31-39 | 89.81% ± 0.72 | 32.48% | 74.31% | +0.094 |
| 40-46 | 89.12% ± 1.29 | 40.10% | 61.31% | +0.161 |

Interpretation: critical-case enrichment improves the router most clearly when moving away from very sparse MO coverage, but the raw step-1 curve is noisy and not monotone. Moderate enrichment preserves BC coverage; extreme enrichment can make threshold calibration and seed choice dominate, and it does not deliver a consistent accuracy gain.

Caveat: exact MO counts are available only after human gold annotation. Run-both acquisition can select disagreement candidates before labeling, but this composition sweep is a post-annotation diagnostic, not an implementable pre-label oracle selector.

## Matched pre-label acquisition baseline

A separate six-run matched baseline compares uniformly random 50 candidates with 25 observable Nano/Mini disagreements plus 25 agreements. Neither acquisition rule inspects gold.

| Threshold policy | Selection | Accuracy | Mini saving | MO recall |
|---|---|---:|---:|---:|
| Validation-max-accuracy | Random 50 | 90.33% +/- 0.62 | 20.83% | 79.17% |
| Validation-max-accuracy | 25 disagree + 25 agree | 89.83% +/- 1.43 | 38.50% | 75.00% |
| Validation accuracy floor | Random 50 | 88.83% +/- 1.65 | 43.00% | 58.33% |
| Validation accuracy floor | 25 disagree + 25 agree | 89.00% +/- 1.08 | 47.00% | 60.42% |
| Direct uplift > 0 | Random 50 | 89.50% +/- 2.83 | 28.67% | 70.83% |
| Direct uplift > 0 | 25 disagree + 25 agree | 91.50% +/- 0.00 | 3.00% | 100.00% |

Under validation-max-accuracy, observable active acquisition does not improve mean accuracy over random at n=50 (-0.50 pp); it increases Mini saving by 17.67 pp but is less stable. Under the accuracy-floor policy it gives +0.17 pp accuracy and +4.00 pp saving. Direct uplift nearly collapses to all-Mini (3% saving), so it is not a meaningful routing win. Preliminary evidence therefore supports a cost--accuracy trade-off and critical-case coverage, not a calibration-independent accuracy gain.

The best five-point stable composition window is MO=35--39: 89.93% accuracy, 30.03% Mini saving, and 76.67% MO recall. All-Nano and all-Mini achieve 85.00% and 91.50%. MO=26 reaches 91.00% (0.50 pp below all-Mini) with 19.83% saving; MO=22 reaches 90.00% with 47.17% saving.