# Does annotating more disagreements help, and are free agreements useful?

Scripts: `scripts/run_sembench_disagree_scale_augment.py` (Movie),
`scripts/run_computer_qa_disagree_scale_augment.py` (Computer QA).
Data: `outputs/sembench_disagree_scale_augment.json`,
`outputs/computer_qa_disagree_scale_augment.json`. Five seeds
(505/606/707/808/909 — two more than the older three-seed protocol), standard
two-head soft-prompt router and training path (identical to the prior×data and
active-selection studies; `random50` and `d25_a25` runs reused from those
files, same sampling salts). No answer-model API calls.

Baselines (held-out): Movie all-Nano 85.0 / all-Mini 91.5 / oracle 93.0 at
8.0% Mini rate; Computer QA 71.2 / 80.5 / 83.9 at 12.7% Mini rate.

## Question 1: scale the number of gold-labeled disagreements

Training sets of D gold disagreements (Nano pred ≠ Mini pred) + 25 gold
agreements, with uniform-random controls at matched total label counts.
Operating point = validation-max-accuracy checkpoint + threshold.

**Movie** (pool holds 53 disagreements; accuracy mean±std over 5 seeds):

| labels | disagreement-selected | random control |
|---|---|---|
| 30 | d5: 89.8±1.4 / 89.9±1.1 | 88.4±2.0 / 88.1±1.1 |
| 40 | d15: **90.6±1.0** / 89.0±1.6 | **90.3±0.5** / 88.5±1.6 |
| 50 | d25: 89.3±1.4 / 89.8±2.0 | 90.0±1.5 / 89.2±1.8 |
| 60 | d35: 89.4±1.4 / 88.0±1.2 | 87.9±2.1 / 90.0±1.1 |
| 78 | d53(all): 87.7±2.4 / 88.5±1.1 | 87.8±2.3 / 88.7±2.2 |

(cells: no-prior / mean_prob@1 prior)

**Computer QA** (pool holds only 24–26 disagreements, so D=25 is exhaustion):
d5 77.6±2.7, d15 77.0±2.8, d25 76.8±1.7 (no prior; the prior column is
uniformly worse except random50+mean_prob@1 at **79.3±1.3**, replicating the
earlier three-seed 79.35±0.92). Random controls: random30 77.6±3.3, random40
78.1±2.4, random50 75.8±1.4. On the ranking frontier random30/random40 are the
best CQA configurations at most budgets.

**Findings.**
1. The benefit of annotated disagreements saturates by ~15–25 on both
   datasets; annotating every available disagreement *hurts* on Movie
   (87.7–88.5 at 78 labels, the worst column of the table).
2. At matched human-label budgets, disagreement-guided selection never
   separates from uniform random by more than the seed noise, on either
   dataset, at the operating point or on the frontier. The strongest
   configurations in both tables are random ones (Movie random40 no-prior
   90.3±0.5; CQA random50+mean_prob 79.3±1.3).
3. Five seeds correct the three-seed record: the Movie
   disagreements+prior recipe previously reported as 90.67±0.85 re-estimates
   to **89.8±2.0** (seeds 808/909 land near 88). All twenty Movie
   configurations sit inside 87.7–90.6 with Mini-rate std 0.04–0.31;
   operating-point instability, not selection policy, dominates the spread.

## Question 2: free agreement rows as a consistency constraint

On an agreement both models return the same answer, so the two correctness
heads must be equal on that row — a label-free fact. We add N unlabeled
agreement rows to the d25_a25 gold base with loss `(q_nano − q_mini)²`
(weights 1.0 and 0.3; gold of augmented rows is never read).

**Movie** (base 89.3±1.4 / 89.8±2.0): aug100/aug300/augall span 88.5–89.9
at weight 1.0 and 88.2–89.4 at weight 0.3 — inside the baseline noise band at
every size and both weights. The mechanism does engage (held-out score std
compresses from 0.10–0.14 to 0.03–0.04) but neither accuracy nor Mini-rate
stability (std 0.16–0.29, unchanged) improves. **Null result, robust to the
weight choice.**

**Computer QA** (base 76.8±1.7 / 75.6±3.2): full augmentation (~92 rows) gives
78.5±2.0 (weight 1.0) and 78.5±1.5 (weight 0.3) — a consistent +1.7pt at both
weights, and the frontier moves the same way (budget-0.3 accuracy 73.1→74.2 /
74.5; budget-0.5 75.5→76.7 / 75.8). Partial augmentation (50 rows) is mixed.
Standard deviations overlap at five seeds, so this is a **directional signal,
not a confirmed gain**; it is consistent with augmentation mattering where
labels are scarcest (the CQA pool is 143 rows against Movie's 690).

## Implications

- "Annotate more disagreements" is not a scaling story: ~25 disagreements is
  the ceiling of its usefulness on both datasets, and the selection policy
  itself buys nothing over random at matched budgets once seed variance is
  measured honestly. The acquisition contribution in the paper draft needs to
  be reframed accordingly: what limits these routers at 30–80 labels is
  operating-point stability, not which rows get labeled.
- The one lever that moved anything is *where the operating point lands*
  (prior on CQA random50, thresholding noise everywhere), which points the
  next experiment at threshold/checkpoint selection rather than data
  selection.
- CQA full-agreement augmentation is worth a confirmation run at 10+ seeds
  (still zero API cost) before it can be claimed; if it survives, the honest
  claim is "free agreements help when the labeled pool is tiny."
- Any paper table quoting three-seed means for the active-selection or
  prior×data studies should be re-checked against the five-seed values in
  these two JSONs before submission.
