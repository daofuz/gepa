# CS-QA acquisition with initially hidden gold

Selection sees question/options/source and, where indicated, the Nano response. Gold and Mini outcomes are revealed only after selection.

Pool: 200 questions; outcomes {'both_correct': 142, 'mini_only': 19, 'nano_only': 8, 'both_wrong': 31}.

| Strategy | Final labels | Nano-wrong | Usable balanced | MO recall | BW recall | Initial AUC | Mini-$ saving | Total cost saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold_oracle | 80 | 40.0 | 80.0 | 80.9% | 79.4% | -- | 58.3% | 45.9% |
| gold_oracle | 100 | 50.0 | 100.0 | 100.0% | 100.0% | -- | 47.8% | 37.6% |
| random | 80 | 19.3 | 38.7 | 41.2% | 37.2% | -- | 60.0% | 47.3% |
| random | 100 | 24.7 | 49.4 | 53.1% | 47.2% | -- | 50.3% | 39.6% |
| seed20_nano_iterative | 80 | 21.5 | 42.9 | 41.7% | 43.7% | 0.527 | 57.9% | 45.6% |
| seed20_nano_iterative | 100 | 26.4 | 52.8 | 51.8% | 53.4% | 0.527 | 47.8% | 37.6% |
| seed20_nano_iterative_explore20 | 80 | 21.4 | 42.8 | 41.2% | 43.8% | 0.527 | 58.0% | 45.7% |
| seed20_nano_iterative_explore20 | 100 | 26.3 | 52.6 | 51.4% | 53.4% | 0.527 | 47.9% | 37.8% |
| seed20_nano_one_shot | 80 | 20.9 | 41.8 | 36.7% | 44.9% | 0.527 | 58.4% | 46.0% |
| seed20_nano_one_shot | 100 | 25.5 | 51.1 | 46.8% | 53.7% | 0.527 | 48.7% | 38.3% |
| seed20_question_one_shot | 80 | 20.4 | 40.8 | 35.7% | 44.0% | 0.525 | 59.1% | 46.5% |
| seed20_question_one_shot | 100 | 25.6 | 51.3 | 45.9% | 54.6% | 0.525 | 49.2% | 38.8% |
