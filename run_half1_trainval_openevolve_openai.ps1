$ErrorActionPreference = "Stop"

python scripts\make_half1_trainval_sweep_splits.py

$size = 50
$iterations = if ($env:OPENEVOLVE_ITERATIONS) { $env:OPENEVOLVE_ITERATIONS } else { "10" }
$outputDir = "openevolve_router_run_hybrid_half1_trainval${size}_openai${iterations}"
$tracker = "openevolve_router_best_hybrid_half1_trainval${size}_openai${iterations}.json"
$result = "gepa_router_prompt_result_hybrid_half1_trainval${size}_openevolve_openai${iterations}.json"
$traceCache = "routing_testdata_router_traces_hybrid_half1_trainval${size}_openevolve_openai${iterations}.jsonl"
$evalJson = "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openevolve_openai${iterations}.json"
$evalCsv = "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openevolve_openai${iterations}.csv"

$env:PYTHONUTF8 = "1"
if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
      $name = $matches[1].Trim()
      $value = $matches[2].Trim().Trim('"').Trim("'")
      Set-Item -Path "env:$name" -Value $value
    }
  }
}
$env:OPENEVOLVE_ROUTER_SPLIT_FILE = "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json"
$env:OPENEVOLVE_ROUTER_EVAL_SET = "trainval"
$env:OPENEVOLVE_ROUTER_TRACKER = $tracker
$env:OPENEVOLVE_ROUTER_MODEL = "qwen3.5:latest"
$env:OPENEVOLVE_ROUTER_SCORE_MODE = "risk_averse_utility"
$env:OPENEVOLVE_ROUTER_OUTPUT_FORMAT = "token"
$env:OPENEVOLVE_ROUTER_NUM_PREDICT = "32"

python -m openevolve.cli `
  openevolve_router_initial.py `
  openevolve_router_evaluator.py `
  --config openevolve_router_config.yaml `
  --output $outputDir `
  --iterations $iterations `
  --api-base https://api.openai.com/v1 `
  --primary-model gpt-4.1

python scripts\materialize_openevolve_router_result.py `
  --tracker $tracker `
  --split-file "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json" `
  --output $result

python evaluate_router_testdata.py `
  --gepa-result $result `
  --split-file "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json" `
  --trace-cache $traceCache `
  --output-json $evalJson `
  --output-csv $evalCsv
