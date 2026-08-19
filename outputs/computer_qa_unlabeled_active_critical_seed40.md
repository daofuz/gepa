# CS-QA acquisition with initially hidden gold

Selection sees question/options/source and, where indicated, the Nano response. Gold and Mini outcomes are revealed only after selection.

Pool: 200 questions; outcomes {'both_correct': 142, 'mini_only': 19, 'nano_only': 8, 'both_wrong': 31}.

| Strategy | Final labels | Nano-wrong | Usable balanced | MO recall | BW recall | Initial AUC | Mini-$ saving | Total cost saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold_oracle | 100 | 50.0 | 100.0 | 100.0% | 100.0% | -- | 47.8% | 37.6% |
| random | 100 | 24.7 | 49.4 | 53.1% | 47.2% | -- | 50.3% | 39.6% |
| seed40_nano_critical_iterative_explore20 | 100 | 26.6 | 53.2 | 51.9% | 54.0% | 0.475 | 48.6% | 38.3% |
| seed40_nano_error_iterative | 100 | 26.4 | 52.8 | 51.8% | 53.4% | 0.527 | 47.8% | 37.6% |
| seed40_nano_error_iterative_explore20 | 100 | 26.3 | 52.6 | 51.4% | 53.4% | 0.527 | 47.9% | 37.8% |
| seed40_nano_error_one_shot | 100 | 25.5 | 51.1 | 46.8% | 53.7% | 0.527 | 48.7% | 38.3% |
| seed40_question_error_one_shot | 100 | 25.6 | 51.3 | 45.9% | 54.6% | 0.525 | 49.2% | 38.8% |
