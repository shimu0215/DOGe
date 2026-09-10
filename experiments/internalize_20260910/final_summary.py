"""Read-only aggregation; preserve incomplete runs and matched evaluation protocols."""
import contextlib,io,json,runpy,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'results/internalize'
sys.path.insert(0,str(root/'experiments/gate_audit_20260909'))
from corrected_numeric_audit import prediction,gold,paired
buf=io.StringIO()
with contextlib.redirect_stdout(buf):runpy.run_path(str(Path(__file__).with_name('progress.py')))
d=json.loads(buf.getvalue())
d['logs']={k:v for k,v in d['logs'].items() if any(x in k for x in ['svamp','recovery','weight_audit'])}
for comp in d.get('comparisons',{}).values():
    comp.pop('unparsed_examples',None)
    for m in comp.get('models',{}).values():
        for key in list(m):
            if key not in ['path','accuracy','sha256']:m.pop(key)
for name in ['fresh600_existing_neuron_student_queue','confirm025_queue','selected_head_recovery_queue','plain_weight_audit','existing_neuron_teacher_pairs','existing_neuron_selection']:
    p=out/(name+'.json')
    if p.exists():d[name]=json.loads(p.read_text())
score=lambda rows:[int(prediction(r['prediction'].replace(r'\,',' '))[0]==gold(r['ground_truth'])) for r in rows]
read=lambda p:json.loads(p.read_text())['content']
key=lambda rows:[(r['id'],r['prompt'],r['ground_truth']) for r in rows]
paths={
 'sft':[Path('/scratch/wzhao20/DOGe-official/outputs/qwen2_5_0p5b_instruct_sft_14b_cot_gsm1000_correctonly_20260908/sft_gsm_test200/gsm8k-results.json'),out/'fresh600_sft/gsm8k-results.json'],
 'clean_s10':[root/'results/repaired_clean_gsm200/gsm8k-results.json',out/'fresh600_clean_s10/gsm8k-results.json'],
 'clean_s11':[root/'results/repaired_clean_seed11_gsm200/gsm8k-results.json',out/'fresh600_clean_s11/gsm8k-results.json'],
 'clean_s12':[out/'clean_replica_s12_student200/gsm8k-results.json',out/'fresh600_clean_replica_s12/gsm8k-results.json'],
 'neuron_s10':[out/'existing_neuron_s10_student200/gsm8k-results.json',out/'fresh600_existing_neuron_s10/gsm8k-results.json']}
allrows={k:sum([read(p) for p in pp],[]) for k,pp in paths.items()}
assert all(len(rr)==400 and key(rr)==key(allrows['sft']) for rr in allrows.values())
scores={k:score(rr) for k,rr in allrows.items()}
d['gsm400_posthoc']={'note':'Descriptive pooled 0:200 and600:800; not a new confirmatory endpoint or independent training seed',
 'scores':{k:{'old200':sum(s[:200])/200,'fresh200':sum(s[200:])/200,'pooled400':sum(s)/400} for k,s in scores.items()},
 'neuron_vs_matched_clean':paired(scores['clean_s10'],scores['neuron_s10']),
 'neuron_vs_sft':paired(scores['sft'],scores['neuron_s10'])}
d['svamp_pilot']={}
for label in ['original_teacher','neuron_teacher','sft','clean_s10','neuron_s10']:
    folder=out/('svamp_pilot_'+label);p=folder/'summary.json'
    if not p.exists():continue
    summary=json.loads(p.read_text());modes={}
    for mode in summary['modes']:
        rr=[json.loads(line) for line in (folder/(mode+'.jsonl')).read_text().splitlines()]
        ss=score(rr);assert len(ss)==summary['n']
        modes[mode]={'n':len(ss),'accuracy':sum(ss)/len(ss)}
        baseline='original_teacher' if label=='neuron_teacher' else 'clean_s10' if label=='neuron_s10' else None
        if baseline:
            bp=out/('svamp_pilot_'+baseline)/(mode+'.jsonl')
            if bp.exists():
                br=[json.loads(line) for line in bp.read_text().splitlines()]
                if key(br)==key(rr):modes[mode]['vs_'+baseline]=paired(score(br),ss)
    d['svamp_pilot'][label]={'complete':summary.get('complete',False),'modes':modes,'dtype':summary['dtype']}
d['time']=time.time()
d['svamp_generated_counts']={folder.name:{p.stem:len(p.read_text().splitlines()) for p in folder.glob('*.jsonl')} for folder in out.glob('svamp_pilot_*') if folder.is_dir()}
(out/'final_summary.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d))
