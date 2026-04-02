# Workbench Hub

This workbench now keeps only a minimal shared system.

## Required Rules For Other AI

1. Before doing meaningful work here, choose a unique name in the format:
   `<ai-name>-<Hoenn Pokemon name>`
   Example: `codex-mudkip`
2. Register that name in `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
3. Start all user-facing replies with your registered name
4. Read and follow `/Users/shimu/Downloads/DOGe-main/workbench/CORE_INFO.md`
5. Follow the execution rule in the core info file:
   change code locally, push to the repo, pull on Hopper, then run on Hopper
6. Unless the user explicitly asks for it, never cancel a Hopper job; only kill processes inside the job when needed

## What Remains In Workbench

- `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
  Shared AI name registry.
- `/Users/shimu/Downloads/DOGe-main/workbench/CORE_INFO.md`
  Core shared information that all AI should follow.
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/`
  Local dashboard that pulls direct Hopper reservation, running-job, GPU, and SSH status information.
