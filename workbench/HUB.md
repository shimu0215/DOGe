# Workbench Hub

This folder is the shared coordination base for Codex, Claude Code, OpenClaw, and related GUI or terminal AI sessions.

## Identity Rule

- Pick an agent name using this format if you do not already have one:
  `<ai-name>-<Hoenn Pokemon name>`
  Example: `codex-mudkip`
- Start each user-facing reply with that name so the human can tell you are still following this shared context.
- Use the same name in handoffs and task records.
- Before doing real work in this directory, register yourself in `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
- Your registered name must be unique.
- Even if you are also Codex, Claude Code, or another tool already listed there, you must use a new name if you do not share the same memory/context as that prior session.

## Default Read Order

Read only these files first unless your task clearly needs more:

1. `/Users/shimu/Downloads/DOGe-main/workbench/current/active_tasks.md`
2. `/Users/shimu/Downloads/DOGe-main/workbench/current/latest_handoff.md`
3. `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/ssh_status.md`
4. `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
5. `/Users/shimu/Downloads/DOGe-main/workbench/DATA_REGISTRY.md` when your work touches datasets

After that, read only the specific knowledge, task, or dashboard files you need.

## Directory Map

- `/Users/shimu/Downloads/DOGe-main/workbench/AI_REGISTRY.md`
  Session registry for all AI workers that enter this workspace.
- `/Users/shimu/Downloads/DOGe-main/workbench/DATA_REGISTRY.md`
  Shared registry for collected, deleted, or discarded datasets and their storage locations.
- `/Users/shimu/Downloads/DOGe-main/workbench/current/`
  Current state only. Keep these files short.
- `/Users/shimu/Downloads/DOGe-main/workbench/knowledge/`
  Reusable experience and stable procedures.
- `/Users/shimu/Downloads/DOGe-main/workbench/tasks/`
  Task history and daily execution logs.
- `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/`
  Files that support the dashboard and shared Hopper status/task display.

## Repo Structure Rules

- The canonical local git root is:
  `/Users/shimu/Downloads/DOGe-main`
- Treat this root as the single repository for all active project code, including:
  - `/Users/shimu/Downloads/DOGe-main/AgentDistill`
  - `/Users/shimu/Downloads/DOGe-main/LlamaFactory-main`
  - `/Users/shimu/Downloads/DOGe-main/workbench`
- Do not treat `AgentDistill` or `LlamaFactory-main` as separate local git repositories anymore.
- When syncing by git, use the outer `AKDA2` repository only.
- Hopper-side canonical git working tree is:
  `/scratch/wzhao20/AKDA2`
- Hopper experiments should be launched from:
  `/scratch/wzhao20/AKDA2/AgentDistill`
- `/scratch/wzhao20/AgentDistill` is legacy runtime residue, not the preferred source tree for new git sync operations.

## Current Layer Rules

- `active_tasks.md`
  Use this to claim work, note status, blockers, and current owner.
- `latest_handoff.md`
  Use this for the single most important current handoff summary.
  Keep it short and overwrite or refresh it when the current direction changes.

Who updates the current layer:

- Whoever changes the current state must update it.
- Do not wait for another AI to clean it up later.

## Decision Logging Rules

GUI and terminal decisions are both manual, not automatic.

You must write a short summary to shared files when you do any of the following:

- change the plan
- choose or reject an approach
- discover a blocker
- get a meaningful result
- hand off work
- pause unfinished work
- submit or modify an important Hopper run

Where to write:

- If it changes the current direction: update `/Users/shimu/Downloads/DOGe-main/workbench/current/latest_handoff.md`
- If it is a concrete work event or result: append to the current daily task log in `/Users/shimu/Downloads/DOGe-main/workbench/tasks/`

Do not treat GUI chat history or terminal output as shared memory unless you wrote a summary to disk.

## Task Log Rules

- Default to reading only `/Users/shimu/Downloads/DOGe-main/workbench/tasks/LATEST.md`
- Read older task logs only if the latest files point you there
- Use one daily task log file per day, named like `YYYY-MM-DD.md`
- If today already has a task file, append to it
- Keep `LATEST.md` short and current

## Knowledge Rules

- Only store reusable experience in `/Users/shimu/Downloads/DOGe-main/workbench/knowledge/`
- Do not put one-off status there
- Add short summaries at the top of each knowledge file

## Data Registry Rules

- If you collect, delete, move, discard, or invalidate a dataset group, update:
  `/Users/shimu/Downloads/DOGe-main/workbench/DATA_REGISTRY.md`
- Record what happened, where the data lives or lived, and the current status.
- Do not assume a dataset is still valid unless the registry says so.
- When your task depends on data availability, read the registry before acting.

## Dashboard Rules

- If your work creates, updates, or uses a Hopper task under a reservation, you must read and follow:
  `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/task_record_prompt.md`
- Do not invent your own task record format
- The dashboard task log is:
  `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/task_records.md`
- The dashboard SSH state file is:
  `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/ssh_status.md`

## OpenClaw Coordination

- OpenClaw may wake other AI tools but cannot rely on GUI chat history
- Therefore any decision needed by another AI must be written into the shared files above

## Conflict Avoidance

- Before starting a concrete task, check `active_tasks.md`
- Claim the task there before working
- Do not intentionally do the same concrete task as another AI unless the file explicitly says parallel work is intended

## Minimum Update Requirement

Before you end your session after meaningful work, do all applicable items:

- make sure you are registered in `AI_REGISTRY.md`
- update `active_tasks.md`
- update `latest_handoff.md` if the current direction changed
- append to today’s task log
- update `DATA_REGISTRY.md` if you touched dataset lifecycle or storage
- update dashboard task records if Hopper reservation work was involved
