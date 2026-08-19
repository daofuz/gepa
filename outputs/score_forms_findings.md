# Score read-out forms: the formula is second-order; the Mini head is dead weight

Scripts: `scripts/run_{sembench,computer_qa}_score_forms.py`. Data:
`outputs/{sembench,computer_qa}_score_forms.json`. Minimal protocol (25
random gold rows, fixed 8 epochs, no validation), same trained heads
re-scored under five formulas, frontier accuracy over budgets 0.1–0.9,
five seeds, no API calls.

Mean frontier accuracy over the nine budgets (no-prior column):

| form | Movie | CQA |
|---|---|---|
| diff = q_mini − q_nano (baseline) | 88.22 | 76.42 |
| neg_nano = −q_nano | 88.54 | 76.61 |
| logit_diff | 88.23 | 76.20 |
| **product = q_mini·(1 − q_nano)** | 88.46 | **76.89** |
| rank_diff | 88.17 | 76.15 |

With the mean_prob prior all forms cluster within 0.3pt on both datasets.

Findings:

1. **The read-out formula is a second-order effect**: the full spread is
   0.3–0.7pt, far below the acquisition/operating-point effects measured
   earlier. Ranking quality is not bottlenecked by the formula.
2. **The Mini head contributes nothing**: ignoring it entirely (neg_nano)
   ties or slightly beats the difference everywhere, and with visibly
   smaller seed variance where it matters (CQA no-prior: std 0.9–1.1 vs
   1.7–2.8 for diff). At 25 labels the Mini head has no signal to learn
   (Mini is right ~91/80% of the time) and its output is noise inside the
   difference. **Architectural consequence: the two-head router can be
   reduced to a single P(Nano correct) head with no ranking cost — and its
   training labels then require only gold checks of Nano's answer**,
   converging with the Nano-wrong line (which additionally showed such
   labels arrive free from user feedback online).
3. `product` (the q_mini·(1−q_nano) form suggested by the earlier
   acquisition analysis) is the nominal no-prior winner (+0.3/+0.5pt) —
   real but small; a safe default if two heads are kept.
4. `logit_diff` and `rank_diff` offer no advantage.
