import json, pathlib
for fname in ['softprompt_router_balanced50_risk_best_expected_utility_test205.json','softprompt_router_half1_trainval50_test_smoke.json','softprompt_router_first205_random50_seed0_risk_expected_utility.json','softprompt_router_balanced50_risk_best_expected_utility.json']:
    j=json.loads(pathlib.Path(fname).read_text(encoding='utf-8'))
    print('\n', fname)
    if 'test_threshold_summaries' in j and j['test_threshold_summaries']:
        summaries=j['test_threshold_summaries']
        print('threshold_count', len(summaries))
        # Handle list or dict
        if isinstance(summaries, dict):
            items=list(summaries.items())[:10]
        else:
            items=list(enumerate(summaries[:10]))
        for k,v in items[:8]:
            if isinstance(v, dict):
                print(k, {kk:v.get(kk) for kk in ['threshold','final_accuracy','cost_saving_rate_vs_all_mini','nano_routes','mini_routes','mean_score']})
    s=j.get('test_summary')
    if s: print('test', {k:s.get(k) for k in ['final_accuracy','cost_saving_rate_vs_all_mini','nano_routes','mini_routes','mean_score','failure_counts']})
