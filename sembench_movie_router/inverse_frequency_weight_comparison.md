# Inverse-frequency versus regret-weighted cross entropy

Fixed setup: 100 training labels with the 50 both-correct / 46 mini-only / 3
nano-only / 1 both-wrong mixture, three seeds, and the same held-out-200
evaluation.

The route targets contain 53 nano and 47 mini rows. Standard balanced class
weights are therefore:

- nano: 100 / (2 × 53) = 0.943396
- mini: 100 / (2 × 47) = 1.063830

## Results

| Soft tokens | Loss weighting | Accuracy | Mini saving | Critical recall | Both-wrong to mini |
|---:|---|---:|---:|---:|---:|
| 4 | Regret outcome weights | 90.17% | 38.00% | 77.08% | 66.67% |
| 4 | Inverse route frequency | 89.33% | 45.67% | 64.58% | 50.00% |
| 8 | Regret outcome weights | 90.83% | 21.83% | 89.58% | 71.43% |
| 8 | Inverse route frequency | 89.67% | 30.00% | 70.83% | 59.52% |
| — | All mini | 91.50% | 0.00% | 100.00% | 100.00% |
| — | All nano | 85.00% | 100.00% | 0.00% | 0.00% |

The extra savings from inverse-frequency weighting come with substantially
more critical misses. Because the route classes are already close to balanced,
the inverse-frequency weights are close to 1 and behave similarly to ordinary
cross entropy. They do not distinguish mini-only from both-wrong or
both-correct from nano-only.

The 8-token inverse-frequency configuration is particularly unstable:
accuracy standard deviation is 1.65 points and mini-saving standard deviation
is 23.94 points, versus 0.47 and 4.48 points with regret weighting.

The current best accuracy-first configuration remains 8 soft tokens with the
manual outcome-regret weights.
