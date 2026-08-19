$ErrorActionPreference = "Stop"

py -3.13 scripts\make_half1_bucket_ablation_splits.py
if ($LASTEXITCODE -ne 0) { throw "split generation failed" }

$scenarios = @(
  "both_correct_heavy",
  "both_wrong_heavy",
  "balancedish",
  "critical_both_wrong",
  "critical_both_correct"
)

foreach ($scenario in $scenarios) {
  py -3.13 optimize_ollama_router_gepa.py `
    --hybrid-easy-router `
    --router-model qwen3.5:latest `
    --reflection-provider openai `
    --reflection-model gpt-4.1 `
    --split-file "routing_split_half1_ablation50_${scenario}_seed0.json" `
    --score-mode risk_averse_utility `
    --router-output-format token `
    --max-metric-calls 120 `
    --reflection-minibatch-size 4 `
    --router-num-predict 32 `
    --reflection-num-predict 2048 `
    --output "gepa_router_prompt_result_hybrid_half1_ablation50_${scenario}_openai_reflect.json" `
    --run-dir "gepa_router_run_hybrid_half1_ablation50_${scenario}_openai_reflect"
  if ($LASTEXITCODE -ne 0) { throw "GEPA failed for $scenario" }

  py -3.13 evaluate_router_testdata.py `
    --gepa-result "gepa_router_prompt_result_hybrid_half1_ablation50_${scenario}_openai_reflect.json" `
    --split-file "routing_split_half1_ablation50_${scenario}_seed0.json" `
    --router-num-predict 32 `
    --trace-cache "routing_testdata_router_traces_hybrid_half1_ablation50_${scenario}_openai.jsonl" `
    --output-json "routing_cost_comparison_testdata_hybrid_half1_ablation50_${scenario}_openai.json" `
    --output-csv "routing_cost_comparison_testdata_hybrid_half1_ablation50_${scenario}_openai.csv" `
    --limit 0
  if ($LASTEXITCODE -ne 0) { throw "evaluation failed for $scenario" }
}
