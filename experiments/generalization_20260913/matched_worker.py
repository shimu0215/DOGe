"""Existing worker protocol with explicit matched stop-token evaluation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/cross_student_20260913'))
import common_round2_fixed as c
class Worker(c.Worker):
 def evaluate(self,phase,model,split='train',start=7000,count=100,destination=None):
  dest=destination or c.OUT/(self.tag+'_'+phase)
  self.run(phase,[c.PY,str(Path(__file__).with_name('evaluate_matched_stops.py')),'--model',str(model),'--output',str(dest),'--split',split,'--start',str(start),'--count',str(count)])
  d=c.read(dest/'gsm8k-results.json');assert d['generation']['stop_token_ids']==[151643,151645] and len(d['content'])==count
  return d,dest/'gsm8k-results.json'
