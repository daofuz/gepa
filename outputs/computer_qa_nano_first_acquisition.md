# CS-QA Nano-first acquisition simulation

Full-pair reveals Mini for every pool item before selecting a balanced subset. Nano-first selects the same 50/50 Nano-correct/Nano-wrong subset from existing gold labels and Nano outputs, then reveals Mini only for selected items.

Pool: 200 questions; outcomes {'both_correct': 142, 'mini_only': 19, 'nano_only': 8, 'both_wrong': 31}; Nano cost $0.052393; Mini cost $0.194493.

| Final balanced data | Mini calls: full | Mini calls: Nano-first | Mini-call saving | Mini-$ saving | Total answer-cost saving | Random reveal balanced yield | Selected BC/MO/NO/BW |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 (10%) | 200 | 20 | 90.0% | 89.4% | 70.5% | 9.8 | 9.5/4.0/0.5/6.0 |
| 40 (20%) | 200 | 40 | 80.0% | 78.9% | 62.1% | 18.9 | 19.0/7.6/1.0/12.4 |
| 60 (30%) | 200 | 60 | 70.0% | 68.4% | 53.9% | 29.2 | 28.5/11.3/1.5/18.7 |
| 80 (40%) | 200 | 80 | 60.0% | 58.0% | 45.7% | 39.1 | 38.2/15.2/1.8/24.8 |
| 100 (50%) | 200 | 100 | 50.0% | 47.6% | 37.5% | 49.6 | 47.5/19.0/2.5/31.0 |

## Existing balanced-100 counterfactual

The existing reported split selects all 50 Nano-wrong questions and 50 Nano-correct questions from its 205-question pool. It could therefore have been selected before revealing Mini. Revealing Mini only for its 100 selected questions saves 51.22% of Mini calls, 48.33% of Mini dollars, and 38.05% of total Nano+Mini answer cost while preserving the exact reported training IDs and downstream result.

The two acquisition methods use identical selected IDs for a given seed, so their downstream router data and performance are identical by construction. Only the acquisition-time Mini cost differs. Random reveal is included to show how much balanced data is recoverable without Nano-correctness stratification.
