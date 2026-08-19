import json, pathlib
for p in sorted(pathlib.Path('.').glob('softprompt_router*.json')):
    j=json.loads(p.read_text(encoding='utf-8'))
    print('\n' + p.name)
    print('keys:', sorted(j.keys())[:30])
    for key in ['split_file','test_count','score_mode','decision_threshold','checkpoint']:
        if key in j: print(key, j[key])
    for sk in ['train_summary','val_summary','test_summary','summary']:
        s=j.get(sk)
        if isinstance(s, dict):
            pick={k:s.get(k) for k in ['count','final_accuracy','cost_saving_rate_vs_all_mini','nano_routes','mini_routes','mean_score','failure_counts','target_counts'] if k in s}
            print(sk, pick)
