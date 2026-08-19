import json, pathlib
files = [
    'softprompt_router_balanced50_risk_best_expected_utility.json',
    'softprompt_router_balanced50_risk_best_expected_utility_test205.json',
    'softprompt_router_half1_trainval50.json',
    'softprompt_router_half1_trainval50_test_smoke.json',
    'softprompt_router_first205_random50_seed0_risk_expected_utility.json',
]
for f in files:
    j = json.loads(pathlib.Path(f).read_text(encoding='utf-8'))
    print('\n' + f)
    print('args:', j.get('args'))
    print('model_config:', j.get('model_config'))
    print('checkpoint:', j.get('checkpoint'))
    print('selected_count:', j.get('selected_count'))
    print('selected_outcome_counts:', j.get('selected_outcome_counts'))
