$ErrorActionPreference = "Stop"

python scripts\make_half1_trainval_sweep_splits.py

$size = 50
$outputDir = "openevolve_router_run_hybrid_half1_trainval${size}"
$tracker = "openevolve_router_best_hybrid_half1_trainval${size}.json"

$env:PYTHONUTF8 = "1"
$env:OPENAI_API_KEY = if ($env:OPENAI_API_KEY) { $env:OPENAI_API_KEY } else { "ollama" }
$env:OPENEVOLVE_ROUTER_SPLIT_FILE = "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json"
$env:OPENEVOLVE_ROUTER_EVAL_SET = "trainval"
$env:OPENEVOLVE_ROUTER_TRACKER = $tracker
$env:OPENEVOLVE_ROUTER_MODEL = "qwen3.5:latest"
$env:OPENEVOLVE_ROUTER_SCORE_MODE = "risk_averse_utility"
$env:OPENEVOLVE_ROUTER_OUTPUT_FORMAT = "token"
$env:OPENEVOLVE_ROUTER_NUM_PREDICT = "32"

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('openevolve') else 1)"
if ($LASTEXITCODE -ne 0) {
  python -m pip install openevolve
}

python -m openevolve.cli `
  openevolve_router_initial.py `
  openevolve_router_evaluator.py `
  --config openevolve_router_config.yaml `
  --output $outputDir `
  --iterations 30

python scripts\materialize_openevolve_router_result.py `
  --tracker $tracker `
  --split-file "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json" `
  --output "gepa_router_prompt_result_hybrid_half1_trainval${size}_openevolve.json"

python evaluate_router_testdata.py `
  --gepa-result "gepa_router_prompt_result_hybrid_half1_trainval${size}_openevolve.json" `
  --split-file "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json" `
  --trace-cache "routing_testdata_router_traces_hybrid_half1_trainval${size}_openevolve.jsonl" `
  --output-json "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openevolve.json" `
  --output-csv "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_openevolve.csv"




