$ErrorActionPreference = "Stop"

python scripts\make_half1_trainval_sweep_splits.py

$sizes = @(50)
$seedResults = @(
  "gepa_router_prompt_result_hybrid_half1_trainval50_openai_reflect.json",
  "gepa_router_prompt_result_hybrid_half1_trainval100_openai_reflect.json",
  "gepa_router_prompt_result_hybrid_half1_ablation50_balancedish_openai_reflect.json",
  "gepa_router_prompt_result_hybrid_half1_ablation50_critical_both_wrong_openai_reflect.json"
)

foreach ($size in $sizes) {
  $output = "gepa_router_prompt_result_hybrid_half1_trainval${size}_adaevolve_ollama_reflect.json"
  $runDir = "gepa_router_run_hybrid_half1_trainval${size}_adaevolve_ollama_reflect"
  $traceCache = "routing_testdata_router_traces_hybrid_half1_trainval${size}_adaevolve_ollama.jsonl"
  $evalJson = "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_adaevolve_ollama.json"
  $evalCsv = "routing_cost_comparison_testdata_hybrid_half1_trainval${size}_adaevolve_ollama.csv"

  $cmd = @(
    "optimize_ollama_router_gepa.py",
    "--optimizer", "adaevolve",
    "--hybrid-easy-router",
    "--router-model", "qwen3.5:latest",
    "--reflection-provider", "ollama",
    "--reflection-model", "qwen3.5:latest",
    "--split-file", "routing_split_half1_sweep_trainval${size}_withcritical_seed0.json",
    "--score-mode", "risk_averse_utility",
    "--router-output-format", "token",
    "--max-metric-calls", "120",
    "--router-num-predict", "32",
    "--reflection-num-predict", "2048",
    "--adaevolve-eval-set", "val",
    "--adaevolve-islands", "4",
    "--output", $output,
    "--run-dir", $runDir
  )
  foreach ($seedResult in $seedResults) {
    if (Test-Path $seedResult) {
      $cmd += @("--adaevolve-seed-result", $seedResult)
    }
  }
  python @cmd

  python evaluate_router_testdata.py `
    --gepa-result $output `
