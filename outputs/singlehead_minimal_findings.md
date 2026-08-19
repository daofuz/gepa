# Closing experiment: the fully minimal single-head protocol works

Scripts: `scripts/run_{sembench,computer_qa}_singlehead_minimal.py`. Data:
`outputs/{sembench,computer_qa}_singlehead_minimal.json`. Five seeds, no
API calls.

Protocol under test — everything the line converged to, trained from
scratch: **one head predicting P(Nano correct) on plain text (pre-route),
25 uniform random rows labeled by Nano gold checks only, fixed 8 epochs, no
validation set of any kind, threshold fixed as a pool-score quantile at the
chosen budget.** Prior variants: none / mean anchored to the pool Nano rate
/ mean anchored to the 25 training labels' own rate (fully self-contained).

## Results (quantile operating points, accuracy @ realized Mini rate)

| dataset | q30 | q50 | q70 |
|---|---|---|---|
| Movie (no prior) | 86.2±0.5 @0.30±0.05 | 88.1±0.6 @0.52±0.05 | 89.3±0.4 @0.69±0.04 |
| CQA (no prior) | 73.2±0.8 @0.31±0.02 | 76.5±1.2 @0.52±0.05 | 78.9±1.8 @0.73±0.04 |

Versus the heaviest earlier pipeline (50 gold labels, two heads, enriched
validation, probing) at the same tiers: Movie gives up **0.3pt at q70**
(89.3 vs 89.6) and nothing within noise elsewhere; CQA gives up nothing
(78.9 vs 77.7 at q70 — nominally higher). Realized Mini rates stay pinned
(std 0.02–0.06) with no validation anywhere.

Priors: small and mixed — the self-contained train-estimated anchor equals
the pool-measured one everywhere (the last protocol impurity can be
dropped), helps CQA's low tier (+1.2pt at q30), costs ~0.5pt at Movie's
q50. Optional; include for low-budget CQA-like deployments.

## The final recipe

1. Run Nano over traffic (default anyway).
2. Gold-check Nano's answer on 25 random rows.
3. Train the single soft-prompt head, fixed 8 epochs.
4. Threshold = the (1−B) quantile of scores on ~100–200 unlabeled rows.

Total supervision: **25 Nano-only gold checks.** No Mini calls before
serving, no validation, no probing, no enrichment, no threshold search.
The enriched-validation pipeline remains relevant only for the
validation-searched max-accuracy tier (Movie ~90.8–91.4 at 0.85–0.92 Mini
rate); every budget-controlled tier is served by the minimal recipe at
≤0.3pt cost.

Full arc for the paper: two heads → one (the Mini head is dead weight);
50 labels → 25 (validation's jobs externalized to fixed epochs and pool
quantiles); curated data → random (selection effects below seed noise);
searched threshold → fixed quantile rule (the transferable object is the
distribution position, not the value).
