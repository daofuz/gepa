$ErrorActionPreference = "Stop"

python scripts\make_half1_trainval_sweep_splits.py

$sizes = @(50, 100, 150, 200)

foreach ($size in $sizes) {
  python optimize_ollama_router_gepa.py `
    --hybrid-easy-router `
    --router-model qwen3.5:latest `
    --reflection-provider openai `
    --reflection-model gpt-4.1 `
    --split-file "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json" `
    --score-mode risk_averse_utility `
    --router-output-format token `
    --max-metric-calls 120 `
    --reflection-minibatch-size 4 `
    --router-num-predict 32 `
    --reflection-num-predict 2048 `
    --output "gepa_router_prompt_result_hybrid_half1_trainval${size}_openai_reflect.json" `
    --run-dir "gepa_router_run_hybrid_half1_trainval${size}_openai_reflect"

  python evaluate_router_testdata.py `
    --gepa-result "gepa_router_prompt_result_hybrid_half1_trainval${size}_openai_reflect.json" `
    --split-file "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json" `
    --router-num-predict 32 `
    --trace-cache "routing_testdata_router_traces_hybrid_half1_trainval${size}_openai.jsonl" `
    --output-json "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openai.json" `
    --output-csv "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openai.csv" `
    --limit 0
}

python scripts\summarize_half1_trainval_sweep_results.py
