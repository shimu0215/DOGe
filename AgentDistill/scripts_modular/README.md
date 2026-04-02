# Modular AgentDistill Scripts

This folder contains the new minimal Hopper-facing scripts.

Design:

- `common.sh`
  Shared environment setup, path helpers, resource cleanup, and merge helpers.
- `collect_unit.sh`
  Collect one model/dataset/seed unit. It only checks completeness, not quality.
- `check_quality.sh`
  Inspect raw collection quality on demand.
- `collect_batch.sh`
  Loop over seeds and call `collect_unit.sh`.

These scripts assume:

- code root: `/scratch/wzhao20/AKDA2/AgentDistill`
- conda env: `/scratch/wzhao20/conda_envs/AKDA1`

Both can be overridden with environment variables or flags.
