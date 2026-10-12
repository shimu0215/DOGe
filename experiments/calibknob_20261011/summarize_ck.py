"""汇总 calibknob 各 checkpoint 评测. --one D --tag T: 写 D/result.json 并打印 CKRES 行; --all ROOT [ROOT..]: 打印总表(含参照)."""
import argparse, glob, json, os, re, sys
W = '/scratch/wzhao20/anti_opd_claude_20260914'; C = W + '/results/claude_20260914'
sys.path.insert(0, W + '/experiments/gate_audit_20260909')
from corrected_numeric_audit import prediction, gold

def gsm_acc(path, n=None):  # 与 audit_outputs.py 同判分(boxed 且数值相等)
    if not os.path.exists(path): return None
    rows = json.load(open(path))['content'][:n] if n else json.load(open(path))['content']
    ok = 0
    for c in rows:
        v, m = prediction((c['prediction'] or '').replace(r'\,', ' ')); ok += (m == 'boxed' and v == gold(c['ground_truth']))
    return round(100 * ok / len(rows), 2)

def plain(d, mode):
    f = f'{d}/tgsm/{mode}.jsonl'
    if not os.path.exists(f): return None
    r = [json.loads(x) for x in open(f)]
    return dict(acc=round(100 * sum(x['correct'] for x in r) / len(r), 1), cap=round(sum(x['hit_cap'] for x in r) / len(r), 2), tok=round(sum(x['tokens'] for x in r) / len(r)))

def mcq(f):
    if not os.path.exists(f): return None
    d = json.load(open(f)); return dict(acc=round(100 * d['accuracy'], 1), cap=round(d.get('cap_rate', 0), 2), tok=round(d.get('mean_tokens', 0)))

def one(d, tag):
    r = dict(tag=tag, t_gsm_greedy=plain(d, 'greedy'), t_gsm_raw=plain(d, 'raw'), t_csqa_greedy=mcq(f'{d}/tcsqa_greedy/predictions.json'),
             t_csqa_raw=mcq(f'{d}/tcsqa_raw/predictions.json'), stu_csqa=mcq(f'{C}/mcq_csqa_05/{tag}/predictions.json'),
             stu_gsm300=gsm_acc(f'{d}/gsm_stu/gsm8k-results.json'))
    json.dump(r, open(f'{d}/result.json', 'w'), indent=1); return r

def fmt(x):
    if x is None: return '-'
    return f"{x['acc']}(cap{x['cap']},{x['tok']}t)" if isinstance(x, dict) else str(x)

def coder(d, tag):
    r = dict(tag=tag, coder_gsm300=gsm_acc(f'{d}/coder_gsm/gsm8k-results.json'), coder_csqa=mcq(f'{C}/mcq_csqa_coder/ckc_{tag}/predictions.json'))
    json.dump(r, open(f'{d}/coder_result.json', 'w'), indent=1); return r

p = argparse.ArgumentParser(); p.add_argument('--one'); p.add_argument('--coder'); p.add_argument('--tag'); p.add_argument('--all', nargs='*'); a = p.parse_args()
if a.coder:
    r = coder(a.coder, a.tag); print('CODERRES', a.tag, ' '.join(f'{k}={fmt(v)}' for k, v in r.items() if k != 'tag'), flush=True)
if a.one:
    r = one(a.one, a.tag); print('CKRES', a.tag, ' '.join(f'{k}={fmt(v)}' for k, v in r.items() if k != 'tag'), flush=True)
if a.all is not None:
    print('参照: 学生 CSQA initial 43.7 / clean80 48.3(三种子 50.3); 学生 GSM300 clean60 =', gsm_acc(f'{C}/fine_raw_clean/s60/gsm8k-results.json', 300),
          '; 学生 GSM300 initial =', gsm_acc(f'{C}/ckeval_ref/raw_student_gsm300/gsm8k-results.json'))
    print('列: teacher GSM100 greedy | raw(T1) | CSQA100 greedy | raw(T1) || 学生 CSQA300(同域) | 学生 GSM300(跨域)')
    for root in a.all:
        ds = sorted(glob.glob(root + '/*/result.json'), key=lambda f: int(re.sub(r'\D', '', os.path.basename(os.path.dirname(f))) or 0))
        for f in ds:
            r = json.load(open(f)); print(os.path.basename(root), os.path.basename(os.path.dirname(f)), '|', fmt(r['t_gsm_greedy']), '|', fmt(r['t_gsm_raw']), '|',
                                          fmt(r['t_csqa_greedy']), '|', fmt(r['t_csqa_raw']), '||', fmt(r['stu_csqa']), '|', fmt(r['stu_gsm300']))
    print('\n跨学生 Coder-0.5B 参照: GSM300 clean60 =', gsm_acc(f'{C}/fine_coder_clean/s60/gsm8k-results.json', 300), '; GSM300 initial =',
          gsm_acc(f'{C}/ckeval_ref/coder_gsm300/gsm8k-results.json'), '; CSQA300 initial 31.0 / clean80 33.7')
    for f in sorted(glob.glob(C + '/ckeval_coder/*/coder_result.json')):
        r = json.load(open(f)); print('coder', r['tag'], '| GSM300', fmt(r['coder_gsm300']), '| CSQA300', fmt(r['coder_csqa']))
