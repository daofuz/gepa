# The validation set can shrink to zero

Script: `scripts/run_computer_qa_zero_validation.py`. Data:
`outputs/computer_qa_zero_validation.json`. Computer QA, held-out 205, five
seeds, no API calls. Closes the line of
`outputs/valsize_threshold_findings.md`: with the threshold settled by pool
quantiles, the validation set's only remaining job is epoch selection — is
even that worth labels?

Design: one full-50-train model per seed; epoch pickers compared paired at
pool-quantile thresholds (q30/50/70). Zero-extra-label pickers: fixed last
epoch, training rows reused as validation, 5-fold CV inside the training
rows (5 extra fold models per seed, labels 0), and a fully unlabeled
pool-routing-stability rule. References: val10/val20 (random labeled rows,
epoch-only) and an oracle epoch picked on the test set itself.

## Results (test accuracy, mean±std over 5 seeds)

| epoch picker | extra labels | q30 | q50 | q70 |
|---|---|---|---|---|
| last_epoch (fixed 8) | 0 | 73.66±1.02 | 75.32±1.53 | 77.56±0.93 |
| train_as_val | 0 | 74.05±1.21 | 75.61±1.66 | 77.37±0.66 |
| cv5 | 0 | 73.76±1.21 | 75.12±1.48 | 77.76±1.26 |
| stability (unlabeled) | 0 | 73.95±0.90 | 75.61±1.88 | 77.46±0.95 |
| val10 | 10 | 73.76±1.13 | 75.61±2.12 | 78.73±1.92 |
| val20 | 20 | 73.66±1.07 | 75.41±1.97 | 78.05±1.45 |
| oracle epoch | — | 75.32±0.85 | 77.37±2.01 | 78.93±1.61 |

train_maxacc (threshold searched on the training rows): 73.66±1.31 at a
realized rate of 0.21 — zero-label threshold *search* stays broken; only the
quantile rule survives without labels.

## Reading

- **At q30 and q50 every picker ties.** Spread 0.5pt against stds 0.9–2.1;
  labels buy nothing.
- **At q70 val10 is nominally +1.2 over fixed epochs** (78.73 vs 77.56), but
  the paired per-seed differences are −1.46/+2.44/0.00/+0.98/+3.90
  (mean +1.17, std 1.87): sign-flipping, within noise, and carried by two
  seeds. val20 shows no dose-response over val10 (78.05 < 78.73), which is
  what noise looks like, not what signal looks like.
- **Epoch choice barely matters at all.** The oracle — the best any selector
  could ever do — sits only 1.4–2.1pt above fixed epochs, and its chosen
  epochs (8/4/7/4/1) are scattered: the per-epoch differences it exploits
  are themselves mostly noise. A selection problem whose ceiling is 1–2pt
  cannot repay a labeling budget.
- The fancier zero-label pickers (cv5 at 6 models per seed, train-as-val,
  stability) all land on top of plain fixed epochs. There is nothing to
  select, so how cleverly one selects is irrelevant.

## Verdict

Train a fixed number of epochs, set the threshold from the unlabeled pool's
score quantile, and spend the entire label budget on training rows. The
validation set's two historical jobs are both gone: threshold selection was
settled by the quantile rule (valsize findings), and epoch selection is
worth at most ~1pt at one operating point — below seed noise and below the
cost of the 10–20 rows it would take. This is the same endpoint the
singlehead-minimal line reached (25 labels, no validation, fixed epochs);
the present experiment shows the two-head 50-label pipeline obeys it too.
