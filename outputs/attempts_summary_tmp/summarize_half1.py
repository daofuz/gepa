import json, pathlib, csv, re
root=pathlib.Path('.')
sizes=[25,50,75,100,125,150,200]
print('SWEEP_OUTCOME_COUNTS')
for size in sizes:
    p=root/f'gepa_router_prompt_result_hybrid_half1_trainval{size}_openai_reflect.json'
    if not p.exists(): continue
    j=json.loads(p.read_text(encoding='utf-8'))
    split=json.loads((root/j['split_file']).read_text(encoding='utf-8'))
    counts=j.get('selected_outcome_counts',{})
    total=sum(counts.values())
    train=j.get('train_outcome_counts',{})
    val=j.get('val_outcome_counts',{})
    print(size, 'train/val', split.get('train_count'), split.get('val_count'), 'selected', counts, 'pct', {k: round(v/total*100,1) for k,v in counts.items()}, 'train', train, 'val', val)
print('\nRANDOM_CONTROL_OUTCOME_COUNTS')
for size in [50,100,150,200]:
    p=root/f'gepa_router_prompt_result_hybrid_half1_random_control_trainval{size}_openai_reflect.json'
    if not p.exists(): continue
    j=json.loads(p.read_text(encoding='utf-8'))
    counts=j.get('selected_outcome_counts',{})
    total=sum(counts.values())
    print(size, counts, {k: round(v/total*100,1) for k,v in counts.items()})
print('\nABLATION_COUNTS')
for p in sorted(root.glob('gepa_router_prompt_result_hybrid_half1_ablation50_*openai_reflect*.json')):
    j=json.loads(p.read_text(encoding='utf-8'))
    counts=j.get('selected_outcome_counts',{})
    if not counts: continue
    total=sum(counts.values())
    print(p.name, counts, {k: round(v/total*100,1) for k,v in counts.items()})
print('\nSWEEP_TEST_METRICS')
for size in sizes:
    c=root/f'routing_cost_comparison_testdata_hybrid_half1_trainval{size}_openai.csv'
    if not c.exists(): continue
    rows=list(csv.DictReader(c.open()))
    b=next(r for r in rows if r['scenario']=='gepa_best_on_testdata')
    print(size, 'acc', round(float(b['accuracy'])*100,1), 'savings', round(float(b['cost_savings_vs_baseline'])*100,1), 'routes', f"{b['nano_routes']}/{b['mini_routes']}")
print('\nRANDOM_TEST_METRICS')
for size in [50,100,150,200]:
    c=root/f'routing_cost_comparison_testdata_hybrid_half1_random_control_trainval{size}_openai.csv'
    if not c.exists(): continue
    rows=list(csv.DictReader(c.open()))
    b=next(r for r in rows if r['scenario']=='gepa_best_on_testdata')
    print(size, 'acc', round(float(b['accuracy'])*100,1), 'savings', round(float(b['cost_savings_vs_baseline'])*100,1), 'routes', f"{b['nano_routes']}/{b['mini_routes']}")
# fixed test outcome counts from mini/nano result files and one split test ids
mini=json.loads((root/'computer science_result_mini.json').read_text(encoding='utf-8'))
nano=json.loads((root/'computer science_result_nano.json').read_text(encoding='utf-8'))
by_m={int(r['question_id']): r for r in mini}
by_n={int(r['question_id']): r for r in nano}
split=json.loads((root/'routing_split_half1_sweep_trainval50_withcritical_seed0.json').read_text(encoding='utf-8'))
counts={'mini_only':0,'nano_only':0,'both_correct':0,'both_wrong':0}
for qid in split['test_ids']:
    m=by_m[qid]; n=by_n[qid]
    mc=str(m.get('pred'))==str(m.get('answer'))
    nc=str(n.get('pred'))==str(n.get('answer'))
    if mc and nc: counts['both_correct']+=1
    elif mc and not nc: counts['mini_only']+=1
    elif nc and not mc: counts['nano_only']+=1
    else: counts['both_wrong']+=1
print('\nFIXED_TEST205_OUTCOMES', counts, {k: round(v/205*100,1) for k,v in counts.items()})
