# Data Registry

Shared ledger for dataset lifecycle and storage locations.

## Rules

- Update this file whenever you collect, delete, move, discard, or invalidate a dataset group.
- Record the current status and the storage path.
- Append new records at the end rather than rewriting history.
- If a dataset changes state, add a new record that points to the same dataset group name.

## Status Values

- `collected`
- `active`
- `moved`
- `deleted`
- `discarded`
- `invalid`

## Entry Template

### DATA-YYYYMMDD-HHMMSS-short-slug
- date: 2026-03-30T00:00:00-04:00
- ai_name: codex-水跃鱼
- action: collected
- dataset_group: example-dataset-group
- status: active
- storage_path: /absolute/path/to/data
- related_job_id: none
- notes: One short sentence describing what changed and why.

## Records

### DATA-20260330-000000-bootstrap-example
- date: 2026-03-30T00:00:00-04:00
- ai_name: codex-水跃鱼
- action: collected
- dataset_group: example-dataset-group
- status: active
- storage_path: /Users/shimu/Downloads/DOGe-main/data/example-dataset-group
- related_job_id: none
- notes: Example entry showing how future AI sessions should record dataset lifecycle updates.

### DATA-20260330-173500-qwen3-14b-math-teacher-raw
- date: 2026-03-30T17:35:00-04:00
- ai_name: codex-木守宫
- action: collected
- dataset_group: agentdistill-qwen3-14b-math-python-only-seeds42-56-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test
- related_job_id: none
- notes: Verified on Hopper that seeds 42..56 all exist for Qwen3-14B and each file is near the expected 500 rows, ranging from 499 to 501.

### DATA-20260330-173510-qwen3-32b-math-teacher-raw
- date: 2026-03-30T17:35:10-04:00
- ai_name: codex-木守宫
- action: collected
- dataset_group: agentdistill-qwen3-32b-math-python-only-seeds42-56-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/math_500_20250414_test
- related_job_id: none
- notes: Verified on Hopper that seeds 42..56 all exist for Qwen3-32B and each file is near the expected 500 rows, ranging from 499 to 501.

### DATA-20260330-173520-qwen3-14b-gsmhard-teacher-raw
- date: 2026-03-30T17:35:20-04:00
- ai_name: codex-木守宫
- action: collected
- dataset_group: agentdistill-qwen3-14b-gsmhard-python-only-seeds42-56-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/gsm_hard_500_20250507_test
- related_job_id: none
- notes: Verified on Hopper that seeds 42..56 all exist for Qwen3-14B and every file has the expected 500 rows.

### DATA-20260330-173530-qwen3-32b-gsmhard-teacher-raw
- date: 2026-03-30T17:35:30-04:00
- ai_name: codex-木守宫
- action: collected
- dataset_group: agentdistill-qwen3-32b-gsmhard-python-only-seeds42-56-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/gsm_hard_500_20250507_test
- related_job_id: none
- notes: Verified on Hopper that seeds 42..56 all exist for Qwen3-32B and every file has the expected 500 rows.

### DATA-20260330-173540-qwen3-14b-aime-teacher-raw
- date: 2026-03-30T17:35:40-04:00
- ai_name: codex-木守宫
- action: collected
- dataset_group: agentdistill-qwen3-14b-aime-python-only-seeds42-56-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/aime_90_20250504_test
- related_job_id: none
- notes: Verified on Hopper that seeds 42..56 all exist for Qwen3-14B and every file has the expected 90 rows.

### DATA-20260330-173550-qwen3-32b-aime-teacher-raw
- date: 2026-03-30T17:35:50-04:00
- ai_name: codex-木守宫
- action: collected
- dataset_group: agentdistill-qwen3-32b-aime-python-only-seeds42-56-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/aime_90_20250504_test
- related_job_id: none
- notes: Verified on Hopper that seeds 42..56 all exist for Qwen3-32B and every file has the expected 90 rows.

### DATA-20260330-154100-32b-math-seed57-60
- date: 2026-03-30T15:41:00-04:00
- ai_name: claude-火稚鸡
- action: collected
- dataset_group: qwen3-32b-math500-python-only-teacher-raw
- status: active
- storage_path: /scratch/wzhao20/AKDA2-vjk/AgentDistill/logs/qa_results_python_only_teacher_vjk/math_500_20250414_test/
- related_job_id: 6637805 (gpu008)
- notes: Qwen3-32B teacher raw trajectories on MATH-500, seeds 57-60, 500/500 each; seed 60 final entry is a per_task_timeout error (last-question bug, now fixed at root level).

### DATA-20260330-172034-32b-gsmhard-seed57-60
- date: 2026-03-30T17:20:34-04:00
- ai_name: claude-火稚鸡
- action: collected
- dataset_group: qwen3-32b-gsmhard500-python-only-teacher-raw
- status: active
- storage_path: /scratch/wzhao20/AgentDistill/logs/qa_results_python_only_teacher/gsm_hard_500_20250507_test/
- related_job_id: 6637805 (gpu008)
- notes: Qwen3-32B on GSM-Hard-500, seeds 57-60 采集完成（500/500 each）；seeds 42-56 已存在于同路径；全量 seeds 42-60 均可用。
## 2026-03-31 - 32B lambda0.8 singletraj seed42 reset and recollect

- dataset_group: `agentdistill-qwen3-32b-math-python-only-singletraj-lambda0p8-seed42`
- status: `recollecting`
- recorded_by: `codex-木守宫`
- location:
  - raw: `/scratch/wzhao20/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda0p8/qa_results/math_500_20250414_test/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42.jsonl`
  - scored: `/scratch/wzhao20/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda0p8/qa_results/math_500_20250414_test/evaluations/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42_scored.jsonl`
  - filtered: `/scratch/wzhao20/AgentDistill/training_outputs/qwen3-32B/agent_baseline_2epochs_math32b_entropy_owntraj_ds_lambda0p8/qa_results/math_500_20250414_test/filtered_data/Qwen3-32B_temp=0.7_n=1_seed=42_type=agent_steps=5_python_only_python_only_seed42_filtered.jsonl`
- note:
  - prior singletraj `lambda0.8` scored data showed severe `correct_but_empty_log` contamination
  - user requested deletion of the existing raw/scored/filtered trio and a fresh recollection using the current code
  - old trio was removed on Hopper before relaunching collection on `6635429@gpu022`
