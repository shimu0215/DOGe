import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['digits','top32','uniform','prompt'],required=True)
a=p.parse_args();scripts=Path(__file__).resolve().parent
suite='teacheronly_teacher_suite.py' if a.variant in ('digits','top32') else 'teacher_alternative_suite.py'
for mode,n in [('greedy',200),('sampling',200),('raw',64)]:
    print(f'START {a.variant} teacher {mode}',flush=True)
    subprocess.run([sys.executable,str(scripts/suite),'--variant',a.variant,'--mode',mode,'--limit',str(n)],check=True)
print(f'ALL TEACHER CHECKS COMPLETE {a.variant}',flush=True)
