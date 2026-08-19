# CS-QA acquisition with initially hidden gold

Selection sees question/options/source and, where indicated, the Nano response. Gold and Mini outcomes are revealed only after selection.

Pool: 200 questions; outcomes {'both_correct': 142, 'mini_only': 19, 'nano_only': 8, 'both_wrong': 31}.

| Strategy | Final labels | Nano-wrong | Usable balanced | MO recall | BW recall | Initial AUC | Mini-$ saving | Total cost saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold_oracle | 100 | 50.0 | 100.0 | 100.0% | 100.0% | -- | 47.8% | 37.6% |
| random | 100 | 24.7 | 49.4 | 53.1% | 47.2% | -- | 50.3% | 39.6% |
| seed20_nano_critical_iterative_explore20 | 100 | 26.3 | 52.6 | 48.4% | 55.2% | 0.480 | 49.0% | 38.6% |
| seed20_nano_error_iterative | 100 | 26.7 | 53.3 | 52.4% | 53.9% | 0.503 | 47.9% | 37.7% |
| seed20_nano_error_iterative_explore20 | 100 | 26.9 | 53.9 | 53.5% | 54.1% | 0.503 | 48.0% | 37.8% |
| seed20_nano_error_one_shot | 100 | 25.8 | 51.7 | 47.6% | 54.2% | 0.503 | 48.7% | 38.4% |
| seed20_question_error_one_shot | 100 | 25.9 | 51.7 | 48.5% | 53.7% | 0.501 | 49.0% | 38.6% |
