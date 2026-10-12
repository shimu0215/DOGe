"""汇总跨域 GSM 学生长程曲线: 每个 gsmlong/<名>/s<step> 的 test[0:300] 准确率与截断率; 参照 fine_raw_clean 同 300 题(到 180 步)."""
import glob, json, os, sys
W = '/scratch/wzhao20/anti_opd_claude_20260914'; C = W + '/results/claude_20260914'
sys.path.insert(0, W + '/experiments/gate_audit_20260909')
from corrected_numeric_audit import prediction, gold

def score(path):
    if not os.path.exists(path): return None
    rows = json.load(open(path))['content'][:300]; ok = trunc = 0
    for c in rows:
        t = c['prediction'] or ''; v, m = prediction(t.replace(r'\,', ' ')); ok += (m == 'boxed' and v == gold(c['ground_truth']))
        trunc += ('\\boxed' not in t and len(t) > 2500)
    return round(100 * ok / len(rows), 1), round(100 * trunc / len(rows))

steps = list(range(20, 201, 20))
print('步数 ' + ' '.join(f'{s:>9}' for s in steps))
ref = [score(f'{C}/fine_raw_clean/s{s}/gsm8k-results.json') for s in steps]
print('clean_old ' + ' '.join(f'{r[0]:>9}' if r else f'{"-":>9}' for r in ref))
for d in sorted(glob.glob(C + '/gsmlong/*/')):
    r = [score(f'{d}s{s}/gsm8k-results.json') for s in steps]
    print(os.path.basename(d.rstrip('/')).ljust(9) + ' ' + ' '.join(f'{x[0]:>5}({x[1]:>2})' if x else f'{"-":>9}' for x in r))
print('括号内: 无 boxed 且长文本(>2500 字符)的近似截断率 %; 初始(未蒸馏)GSM300 = 41.3')
