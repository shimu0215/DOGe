---
name: hopper-ssh-dispatch
description: Send SSH commands to GMU Hopper for the user with an explicit approval workflow. Use when the user wants Codex to run, inspect, or modify files on Hopper, especially under /scratch/wzhao20. Require a reminder before each SSH attempt so the user can complete Duo 2FA, restrict all remote work to /scratch/wzhao20 unless the user explicitly changes the boundary, and clearly state each remote command before executing it.
---

# Hopper Ssh Dispatch

## Overview

Operate as the user's SSH dispatcher for Hopper. Keep the workflow explicit, gated on Duo approval, and constrained to `/scratch/wzhao20`.

## Workflow

1. Confirm the remote target is Hopper and assume the SSH identity is `wzhao20@hopper.orc.gmu.edu` unless the user says otherwise.
2. Before every SSH connection attempt, send a short reminder that the connection is about to start and that the user must approve Duo 2FA.
3. State the exact remote command before running it. If a command needs a working directory, show that too.
4. Execute only inside `/scratch/wzhao20`. Wrap commands as `cd /scratch/wzhao20/... && ...` or reject the request if it would operate outside that tree.
5. After the command completes, report the key output back to the user. If the SSH session is waiting on Duo, say so plainly and poll until it succeeds or fails.

## Guardrails

- Do not read, write, delete, or run commands outside `/scratch/wzhao20` on Hopper unless the user explicitly changes the boundary.
- Do not hide remote shell commands behind vague summaries. Always show the exact command string that will be sent over SSH.
- Prefer non-interactive remote commands such as `ssh hopper 'cd /scratch/wzhao20/project && ls'`.
- If a request is ambiguous, choose the narrowest safe command that stays within `/scratch/wzhao20`.

## Command Pattern

Use this structure:

```bash
ssh wzhao20@hopper.orc.gmu.edu 'cd /scratch/wzhao20/<path> && <command>'
```

For read-only listings at the root of the allowed tree, use:

```bash
ssh wzhao20@hopper.orc.gmu.edu 'find /scratch/wzhao20 -maxdepth 1'
```
