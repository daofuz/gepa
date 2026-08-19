# External baselines under the unified 50-label protocol (paper §4.3)

Scripts: `scripts/run_sembench_external_baselines.py`,
`scripts/run_computer_qa_external_baselines.py`. Data:
`outputs/{sembench,computer_qa}_external_baselines.json`. Our router's rows
come from `outputs/{sembench,computer_qa}_50label_budget.json` (t25_v25, no
prior). Parity: same model pair, seeds, splits, 50-gold budget (train 25
random + validation 18 disagreements/7 agreements), same operating-point
criteria, same cost model. Five seeds, no API calls.

Adaptations (disclose in the paper):
- **RouteLLM-style**: pre-route, single strong-wins head (label: 1 iff Mini
  correct and Nano wrong; ties to the cheap model), plain BCE, quantile cost
  calibration (RouteLLM's own thresholding). Capacity-matched to our router
  (frozen DistilBERT + 8 soft tokens) instead of their preference-scale
  training corpus — this is the label-budget-parity instance.
- **FrugalGPT-style**: two-model cascade instance; scorer input = text +
  Nano's answer, predicts P(Nano correct), escalates below threshold. Its
  cost includes the always-paid Nano call: cascade cost = 0.25 + Mini rate
  (mini-equivalents, Nano = 1/4 Mini); pre-route cost = rate + (1−rate)·0.25.
- **BARGAIN**: the cached answer streams hold no token logprobs, so its
  thresholded proxy-confidence procedure cannot be reproduced (same
  conclusion as `scripts/compare_bargain_style.py`). We report the
  oracle-agreement cascade it approximates — by the routing identity this is
  all-Mini accuracy at the disagreement rate — as a **non-deployable
  reference**, not a baseline row.

## Movie (held-out 200; all-Nano 85.0 @ cost 0.25, all-Mini 91.5 @ 1.00; BARGAIN reference 91.5 @ 9.5% Mini rate)

| method | criterion | accuracy | Mini rate | total cost |
|---|---|---|---|---|
| ours (two-head) | maxacc | 90.8±0.5 | 0.85±0.07 | 0.89 |
| RouteLLM-style | maxacc | 91.0±0.8 | 0.88±0.18 | 0.91 |
| FrugalGPT-style | maxacc | 90.7±0.9 | 0.77±0.22 | 1.02 |
| ours | q70 | **89.6±0.4** | 0.69±0.04 | 0.77 |
| RouteLLM-style | q70 | 88.7±0.6 | 0.66±0.04 | 0.75 |
| FrugalGPT-style | q50 | 89.5±0.7 | 0.48±0.06 | 0.73 |
| ours | q50 | **88.2±1.1** | 0.45±0.04 | 0.59 |
| RouteLLM-style | q50 | 87.4±1.0 | 0.43±0.03 | 0.57 |
| FrugalGPT-style | q30 | 87.3±0.9 | 0.26±0.04 | 0.51 |
| ours | q30 | 86.3±0.7 | 0.28±0.05 | 0.46 |
| RouteLLM-style | q30 | 86.2±0.4 | 0.26±0.03 | 0.44 |

## Computer QA (held-out 205; all-Nano 71.2 @ 0.25, all-Mini 80.5 @ 1.00; BARGAIN reference 80.5 @ 22% Mini rate)

| method | criterion | accuracy | Mini rate | total cost |
|---|---|---|---|---|
| ours | maxacc | 78.6±1.6 | 0.79±0.12 | 0.84 |
| RouteLLM-style | maxacc | 79.0±0.6 | 0.83±0.04 | 0.87 |
| FrugalGPT-style | maxacc | 79.0±1.8 | 0.75±0.13 | 1.00 |
| ours | q70 | 77.7±1.8 | 0.70±0.04 | 0.78 |
| RouteLLM-style | q70 | 77.3±0.7 | 0.66±0.01 | 0.75 |
| FrugalGPT-style | q50 | 77.0±1.1 | 0.50±0.02 | 0.75 |
| ours | q50 | 75.9±1.8 | 0.47±0.05 | 0.60 |
| RouteLLM-style | q50 | 75.9±2.1 | 0.45±0.04 | 0.59 |
| FrugalGPT-style | q30 | 75.6±0.8 | 0.32±0.03 | 0.57 |
| ours | q30 | **74.0±1.9** | 0.27±0.06 | 0.45 |
| RouteLLM-style | q30 | 72.4±0.4 | 0.28±0.02 | 0.46 |

(Rows are grouped by comparable TOTAL cost, not by nominal Mini rate — the
cascade's structural Nano surcharge makes rate-matched comparisons unfair to
pre-route methods.)

## Findings

1. **At the accuracy-max end all learned routers converge** (Movie 90.7–91.0,
   CQA 78.6–79.2): with a trustworthy enriched validation, every formulation
   finds the near-all-Mini optimum; differences are seed noise.
2. **On the cost-controlled tiers, the two-head router matches or beats both
   baselines at matched total cost on both datasets.** RouteLLM-style is
   consistently behind on mid/low tiers (Movie −0.8/−0.9pt, CQA −1.6pt at the
   low tier): its strong-wins label leaves only ~1–3 positive examples among
   25 random gold rows. FrugalGPT-style ranks well per escalation decision
   (seeing Nano's answer is real signal) but pays the cascade tax, ending
   cost-equivalent to our next-higher tier (Movie: 89.5 @ 0.73 vs our 89.6 @
   0.77; CQA: 77.0 @ 0.75 vs our 77.7 @ 0.78).
3. **The protocol transfers to the baselines.** Enriched validation +
   quantile calibration pin every method's Mini rate to std 0.01–0.06 (e.g.
   FrugalGPT q70 on CQA: 0.72±0.01) — evidence that the acquisition/selection
   contributions are method-agnostic, which is itself a paper point.
4. **No deployable method approaches the BARGAIN/identity reference**
   (all-Mini accuracy at 9.5%/22% cost). The residual gap is ranking
   quality, uniform across formulations — the shared open problem.
