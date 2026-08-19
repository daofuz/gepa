# Initial hidden-gold acquisition findings

## Protocol

- Pool: first 200 cached Computer Science QA pairs (142 BC, 19 MO, 8 NO, 31 BW).
- Selection never sees an unselected item's gold label.
- All methods acquire 100 final gold labels. Nano is run on all 200 items.
- Text selectors use a random 20-label seed and TF--IDF features from the question, options, source, Nano prediction, and Nano response.
- Disagreement selectors run Mini on a random probe of 120, 150, or 200 items, then prioritize observable Nano--Mini disagreements before acquiring 100 gold labels.
- Results average 50 paired random seeds.

## Main result

| Strategy | Gold labels | Mini calls | Usable balanced | MO recall | Total answer-cost saving |
|---|---:|---:|---:|---:|---:|
| Random | 100 | 100 | 49.4 | 53.1% | 39.6% |
| Seeded Nano-error selector | 100 | 100 | 53.9 | 53.5% | 37.8% |
| Disagreement, 120 probes | 100 | 120 | 54.4 | 63.4% | 31.8% |
| Disagreement, 150 probes | 100 | 150 | 60.6 | 76.3% | 20.0% |
| Disagreement, 200 probes | 100 | 200 | 73.7 | 100.0% | 0.0% |
| Gold-aware oracle | 100 | 100 | 100.0 | 100.0% | 37.6% |

The 20-label text/Nano selector has essentially random initial discrimination (Nano-error AUC 0.503; MO AUC 0.480). Relative to random, its 4.48-example gain in usable balance is mostly BW enrichment: MO recall changes by only +0.42 percentage point (95% paired CI, -4.22 to +5.06), while BW recall increases by +6.97 points.

Disagreement is the useful no-gold signal. With 150 Mini probes, it raises usable balanced data by 11.16 examples (95% CI, 9.31 to 13.01) and MO recall by 23.26 points (95% CI, 20.58 to 25.95) versus random. The tradeoff is explicit: it uses 50 additional Mini probes, reducing total answer-cost saving by 19.60 points. It still saves 50% of gold-label calls and 25% of Mini calls versus labeling and pairing all 200 candidates.

## Interpretation

The initially proposed learned Nano-error selector does not work well with the currently cached signals because the responses contain neither token log-probabilities nor repeated samples. It preferentially discovers BW rather than recoverable MO cases. A partial paired probe followed by disagreement selection does work, but it trades Mini acquisition cost for critical-case recall. The next experiment should test whether Nano log-probabilities or self-consistency can recover the same MO gain without the extra 50 Mini probes.
