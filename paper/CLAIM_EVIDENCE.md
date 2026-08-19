# Claim-to-evidence map

This map links the main paper claims to the current repository artifacts. Update both the paper and this map whenever a result table changes.

| Paper claim or table | Primary evidence | Notes |
|---|---|---|
| Movie scale and historical/fresh composition | `../sembench_movie_router/model_outputs.json`; `../sembench_movie_router/untouched_200_manifest.json`; `../sembench_movie_router/untouched_200_report.md` | 2,000 rows / 1,865 unique IDs; 812 historical + 200 locked fresh; four outcome counts |
| Disagreement-guided active acquisition | `../scripts/simulate_computer_qa_unlabeled_active_acquisition.py`; `../outputs/computer_qa_unlabeled_active_final100.json`; `../scripts/run_sembench_active_balanced_router.py`; `../sembench_movie_router/leakage_safe_active_direct_router.json` | Both candidate models screen the hidden-gold pool; selection prioritizes prediction disagreement before gold-label reveal. CS-QA uses 50 selection seeds; SemBench reports five leakage-safe resamples. |
| Cost-derived score implementation | `../optimize_ollama_router_gepa.py`; `../train_softprompt_router.py` | Score is `1 / realized_cost_usd` for a correct selected answer and 0 otherwise; expected-score soft-prompt loss |
| Legacy regret-CE definition | `../scripts/run_sembench_accuracy_targeted_router.py`; `../scripts/train_sembench_balanced_direct_router.py`; `../scripts/run_computer_qa_regret_softprompt.py` | Weight-normalized CE; Mini targets are MO/BW, Nano targets are BC/NO; weights are BC/MO/NO/BW = 0.5/6.0/1.5/1.8 |
| Historical soft-prompt main results | `../sembench_movie_router/prompt_optimization_sweep.json`; `../sembench_movie_router/residual_prompt_sweep.json`; `../sembench_movie_router/outcome_pairwise_router.json`; `../sembench_movie_router/accuracy_targeted_router_comparison.json`; `../sembench_movie_router/accuracy_targeted_summary.md` | Accuracy, Mini saving, critical recall, both-wrong rate, and answer-aware diagnostic |
| Training-sampling ablation | `../sembench_movie_router/random100_selection.json`; `../sembench_movie_router/natural_direct_router_100label_repeated.json`; `../sembench_movie_router/leakage_safe_strategy_summary.json` | Natural/random versus outcome-enriched sampling |
| Prompt-length and label-size grid | `../sembench_movie_router/size_prompt_grid.json`; `../sembench_movie_router/p16_matched_steps.json` | 50/100/200 labels; 2/4/8/16 soft tokens; matched-step check |
| Locked 200-row untouched result | `../sembench_movie_router/untouched_200_protocol.md`; `../sembench_movie_router/untouched_200_report.md`; `../sembench_movie_router/untouched_200_router_result.json` | Main fresh claim; protocol locked before answer-model calls |
| Fresh-set post-hoc token sweep | `../sembench_movie_router/untouched_200_train100_token_sweep.md`; `../sembench_movie_router/untouched_200_train100_result.json`; `../sembench_movie_router/untouched_200_train100_8token_result.json` | Must remain labeled post-hoc, never untouched |
| Two-head correctness baseline | `../sembench_movie_router/twohead_correctness_summary.md`; `../sembench_movie_router/twohead_correctness_router.json` | Direct correctness prediction collapses to all-Mini |
| Random residual fresh baseline | `../sembench_movie_router/random100_heldout200.json` | Three-seed post-hoc diagnostic |
| Movie hard-token GEPA | `../sembench_movie_router/gepa_qwen35_2b_router_result.json`; `../sembench_movie_router/gepa_qwen35_2b_run100/candidates.json` | All-Nano collapse and discrete-prompt failure mode |
| Movie scored GEPA | `../sembench_movie_router/gepa_qwen35_2b_scored_result.json` | Thresholded JSON candidate; historical and fresh metrics |
| QA answer-model outputs and fixed split | `../computer science_result_mini.json`; `../computer science_result_nano.json`; `../routing_split_computer_qa_regret_train165_val40_no_test.json` | 410 questions; 165/40/205 split; all-Mini/all-Nano/oracle |
| QA soft prompt and OOF calibration | `../outputs/computer_qa_regret_softprompt_bert_base_train165_val40.json`; `../outputs/computer_qa_regret_bert_base_oof5_ensemble.json`; `../outputs/computer_qa_regret_bert_base_oof5_cost_curve.json` | Exact all-Mini collapse and realized cost curve |
| QA local hard router | `../outputs/computer_qa_qwen35_2b_local_router.json` | 80.00% accuracy and 4.26% cost saving |
| QA GEPA | `../outputs/computer_qa_gepa_qwen35_2b_test.json`; `../outputs/computer_qa_gepa_qwen35_2b_miniseed_test.json` | Default all-Nano collapse; Mini-seeded candidate |

## Paper locations

- Historical Movie: first main table in `sections/04b_main_results.tex`.
- Fresh Movie: locked/post-hoc table in the same file.
- Discrete versus continuous: GEPA discussion and table in the same file.
- Sampling, loss, and size/prompt ablations: `sections/04c_ablations.tex`.
- QA transfer and cost: QA table in `sections/04c_ablations.tex`.

All multi-seed values should use the three-seed population mean and standard deviation. Single-seed GEPA values must remain explicitly labeled.
