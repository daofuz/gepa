# Cold start from raw data only: no gold, no historical runs, no known rule

Script: `scripts/sim_coldstart_stratified_probing.py`;
data `outputs/coldstart_stratified_probing.json`. Movie historical pool (812,
55 MO, 64 disagreements), 20 seeds, no API calls.

Everything is discovered inside the budget being spent. One Nano pass over the
pool is the only free step; strata come from Nano's own outputs; each Mini
probe returns one self-labeling bit (agree / disagree, no gold); allocation is
Thompson sampling on per-stratum Beta posteriors.

## Discovering the rule costs ~19 probes

Mini probes needed to collect the 25 disagreements that `disagree_oracle`
consumes, and MO harvested along the way:

| strategy | probes to 25 disagreements | MO @100 | @200 | @300 | @400 |
|---|---:|---:|---:|---:|---:|
| uniform random | 308.4 ± 28.3 (worst 364) | 6.8 | 13.0 | 20.0 | 27.5 |
| Thompson over Nano-class strata | **155.8 ± 29.1** (worst 204) | 12.8 | 27.1 | 41.9 | 54.1 |
| Thompson over class × length grid | 176.5 ± 28.0 (worst 244) | 12.5 | 25.8 | 37.5 | 51.2 |
| oracle (hot class known in advance) | 137.1 ± 24.6 (worst 180) | 15.8 | 30.0 | 45.1 | 54.1 |

Cold start pays 155.8 against the oracle's 137.1 -- about 19 probes, 14% -- and
is still 2.0x cheaper than uniform. By 400 probes the two are identical (54.1
MO). The rule is cheap to find because the contrast is 90x
(`P(disagree | Nano=NEGATIVE) = 17.8%` vs `POSITIVE = 0.2%`); a Beta posterior
separates arms like that in a couple of dozen pulls.

Finer strata are worse: the class × length grid costs 21 extra probes because
exploration scales with arm count and length carries no signal (disagreement by
length tercile: 6.4 / 8.9 / 6.4 / 9.9%). Keep the arm set small and only add a
stratum when there is a reason to expect contrast.

Arm choice is dataset-specific. On Computer QA the Nano-predicted letter gives
11 noisy arms spanning 7.1--28.0%; the clean 2-arm split is Nano
self-disagreement (CoT answer vs 1-token direct answer): 28.6% vs 9.6%.

## The identity that makes gold unnecessary

On an agreement Nano's answer *is* Mini's answer, so routing to Mini exactly on
the disagreements reproduces all-Mini's output everywhere. Perfect
disagreement routing therefore equals all-Mini accuracy at disagreement-rate
cost, exactly -- verified on both held-out sets:

| | all-Nano | all-Mini | perfect disagreement routing |
|---|---:|---:|---:|
| Movie (200) | 85.0% | 91.5% | **91.5% at 9.5% Mini rate** |
| Computer QA (205) | 71.2% | 80.5% | **80.5% at 22.0% Mini rate** |

The current best router (`disagree_oracle` + mean_prob prior) sits at 90.67%
with a **0.83 Mini rate**: 0.8 points below a gold-free ceiling at 8.7x the
serving cost. The two-head correctness formulation needs gold and is solving a
strictly harder problem (which model is right) than the routing decision
actually requires (do they differ).

## What this does not yet show

The ceiling assumes a perfect disagreement predictor. Disagreement is a 7.9%
rare event on Movie, so realized accuracy depends entirely on the precision and
recall of a text→disagreement model at serving time, which has not been
measured here. Training labels for it are free (run both models on the probed
pool and record agreement), so the whole pipeline can be gold-free, but the
realized operating point is an open experiment.
