# Outcome-only pairwise direct-router experiment

## Protocol

- Router input: semantic operator description and review text only.
- Supervision: routing outcomes derived from cached mini/nano correctness.
- No positive/negative auxiliary target and no mini/nano answer in the input.
- Risk-averse targets: mini for mini-only and both-wrong; nano for both-correct and nano-only.
- Model: four learned soft tokens, frozen DistilBERT, linear route head.
- Split per seed: 160 train, 40 validation, 82 test; seeds 505, 606, and 707.
- Threshold: maximum validation answer accuracy, ties use fewer mini calls.
- No OpenAI API calls.

## Results

| Pairwise scope | Weight | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---|---:|---:|---:|---:|---:|
| None (regret-CE baseline) | 0 | **93.50%** | 17.48% | **93.33%** | **100.00%** |
| All mini-target vs nano-target | 0.05 | 92.68% | 39.84% | 73.33% | 75.00% |
| All mini-target vs nano-target | 0.10 | 91.46% | 54.88% | 53.33% | 75.00% |
| All mini-target vs nano-target | 0.25 | 92.28% | 49.19% | 66.67% | 50.00% |
| All mini-target vs nano-target | 0.50 | 92.68% | 33.33% | 73.33% | 75.00% |
| All mini-target vs nano-target | 1.00 | 92.28% | 36.59% | 73.33% | 66.67% |
| Mini-only vs safe only | 0.05 | 92.28% | 44.31% | 73.33% | 75.00% |
| Mini-only vs safe only | 0.10 | 91.46% | 54.47% | 53.33% | 66.67% |
| Mini-only vs safe only | 0.25 | 91.87% | 44.72% | 66.67% | 50.00% |

## Conclusion

Pairwise ranking reduced mini usage, but it did so by pushing critical examples
below the route threshold.  Even a 0.05 coefficient reduced critical recall
from 93.33% to 73.33% and answer accuracy from 93.50% to 92.28-92.68%.
The failure was unstable across splits: the same coefficient produced 20%
critical recall on seed 505 and 100% on another seed.  OOF threshold calibration
was therefore not pursued because the underlying score ordering, rather than
the threshold estimate, was the limiting factor.

Under the outcome-only, direct pre-routing constraint, regret-weighted CE
remains the best accepted Movie baseline at 93.50% answer accuracy and 17.48%
mini saving.
