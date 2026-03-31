# Task Record Prompt

Use this exact prompt with any AI that submits work onto Hopper and should update the dashboard task log.

## Prompt

Append one new task record to `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/task_records.md`.

Rules:
- Keep the existing file content unchanged except for appending one new record at the end.
- Use exactly this markdown structure:
  - `### TASK-YYYYMMDD-HHMMSS-short-slug`
  - `- submitted_at: <ISO 8601 timestamp with timezone>`
  - `- task_name: <short task name>`
  - `- job_id: <hopper reservation job id>`
  - `- gpu_count: <integer gpu count used by this task>`
  - `- description: <one sentence description>`
- `job_id` must be the reservation job id that owns the GPUs, not the inner task name.
- `description` must be a single sentence.
- Do not rewrite older records.
- If the job id is unknown, stop and ask for it instead of guessing.

## Example

### TASK-20260324-213000-math-eval-seed42
- submitted_at: 2026-03-24T21:30:00-04:00
- task_name: Math eval seed42
- job_id: 6610054
- gpu_count: 1
- description: Runs a single-seed MATH evaluation on the current checkpoint under the active reservation.
