import runpy
from pathlib import Path
from teacher_alternatives import install
install()
runpy.run_path(str(Path(__file__).with_name('train_entry.py')),run_name='__main__')
