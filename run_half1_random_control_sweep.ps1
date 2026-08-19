$ErrorActionPreference = "Stop"

function Run-Python313 {
  py -3.13 @args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed: py -3.13 $args"
  }
}

Run-Python313 scripts\make_half1_random_control_splits.py

$sizes = @(50, 100, 150, 200)

foreach ($size in $sizes) {
  Run-Python313 optimize_ollama_router_gepa.py `
    --hybrid-easy-router `
    --router-model qwen3.5:latest `
    --reflection-provider openai `
    --reflection-model gpt-4.1 `
    --split-file "routing_split_half1_random_control_trainval${size}_seed0.json" `
    --score-mode risk_averse_utility `
    --router-output-format token `
    --max-metric-calls 120 `
    --reflection-minibatch-size 4 `
    --router-num-predict 32 `
    --reflection-num-predict 2048 `
    --output "gepa_router_prompt_result_hybrid_half1_random_control_trainval${size}_openai_reflect.json" `
    --run-dir "gepa_router_run_hybrid_half1_random_control_trainval${size}_openai_reflect"

  Run-Python313 evaluate_router_testdata.py `
    --gepa-result "gepa_router_prompt_result_hybrid_half1_random_control_trainval${size}_openai_reflect.json" `
    --split-file "routing_split_half1_random_control_trainval${size}_seed0.json" `
    --router-num-predict 32 `
    --trace-cache "routing_testdata_router_traces_hybrid_half1_random_control_trainval${size}_openai.jsonl" `
    --output-json "routing_cost_comparison_testdata_hybrid_half1_random_control_trainval${size}_openai.json" `
    --output-csv "routing_cost_comparison_testdata_hybrid_half1_random_control_trainval${size}_openai.csv" `
    --limit 0
}
