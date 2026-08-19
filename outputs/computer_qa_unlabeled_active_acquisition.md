# CS-QA acquisition with initially hidden gold

Selection sees question/options/source and, where indicated, the Nano response. Gold and Mini outcomes are revealed only after selection.

Pool: 200 questions; outcomes {'both_correct': 142, 'mini_only': 19, 'nano_only': 8, 'both_wrong': 31}.

| Strategy | Final labels | Nano-wrong | Usable balanced | MO recall | BW recall | Initial AUC | Mini-$ saving | Total cost saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gold_oracle | 60 | 30.0 | 60.0 | 60.1% | 59.9% | -- | 68.6% | 54.0% |
| gold_oracle | 80 | 40.0 | 80.0 | 80.9% | 79.4% | -- | 58.3% | 45.9% |
| gold_oracle | 100 | 50.0 | 100.0 | 100.0% | 100.0% | -- | 47.8% | 37.6% |
| random | 60 | 14.3 | 28.6 | 30.6% | 27.4% | -- | 69.8% | 55.0% |
| random | 80 | 19.3 | 38.7 | 41.2% | 37.2% | -- | 60.0% | 47.3% |
| random | 100 | 24.7 | 49.4 | 53.1% | 47.2% | -- | 50.3% | 39.6% |
| seed20_nano_iterative | 60 | 16.3 | 32.6 | 32.0% | 33.0% | 0.503 | 68.0% | 53.6% |
| seed20_nano_iterative | 80 | 21.4 | 42.8 | 42.4% | 43.1% | 0.503 | 57.8% | 45.5% |
| seed20_nano_iterative | 100 | 26.7 | 53.3 | 52.4% | 53.9% | 0.503 | 47.9% | 37.7% |
| seed20_nano_iterative_explore20 | 60 | 16.6 | 33.2 | 33.1% | 33.4% | 0.503 | 68.2% | 53.7% |
| seed20_nano_iterative_explore20 | 80 | 21.5 | 43.0 | 42.4% | 43.3% | 0.503 | 58.2% | 45.8% |
| seed20_nano_iterative_explore20 | 100 | 26.9 | 53.9 | 53.5% | 54.1% | 0.503 | 48.0% | 37.8% |
| seed20_nano_one_shot | 60 | 16.2 | 32.4 | 28.8% | 34.6% | 0.503 | 68.7% | 54.1% |
| seed20_nano_one_shot | 80 | 21.1 | 42.2 | 37.8% | 44.9% | 0.503 | 58.6% | 46.2% |
| seed20_nano_one_shot | 100 | 25.8 | 51.7 | 47.6% | 54.2% | 0.503 | 48.7% | 38.4% |
| seed20_question_one_shot | 60 | 16.3 | 32.5 | 28.7% | 34.8% | 0.501 | 68.9% | 54.3% |
| seed20_question_one_shot | 80 | 20.7 | 41.4 | 37.3% | 43.9% | 0.501 | 58.9% | 46.4% |
| seed20_question_one_shot | 100 | 25.9 | 51.7 | 48.5% | 53.7% | 0.501 | 49.0% | 38.6% |
