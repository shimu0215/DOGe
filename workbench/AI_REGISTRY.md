# AI Registry

All AI sessions that start meaningful work in this workspace must register here first.

## Rules

- Use a unique agent name.
- Name format: `<ai-name>-<Hoenn Pokemon name>`
- Example: `codex-mudkip` or `claudecode-treecko`
- Record both the agent name and the tool/model family.
- If you are a new session without shared memory/context, you must use a new name even if you are also Codex or Claude Code again.
- Append new entries at the end. Do not rewrite older entries unless fixing an obvious typo.

## Entries

- Name: Codex-bootstrap
  AI type: Codex
  Interface: GUI/terminal bootstrap
  Started at: 2026-03-28T00:00:00-04:00
  Notes: Created the initial multi-AI shared workbench structure.

- Name: codex-水跃鱼
  AI type: Codex
  Interface: GUI
  Started at: 2026-03-28T00:00:00-04:00
  Notes: Current shared-context session for workbench coordination and dashboard design follow-up.

- Name: codex-木守宫
  AI type: Codex
  Interface: terminal session
  Started at: 2026-03-28T18:56:39-04:00
  Notes: Took over active workspace context, reviewed shared workbench protocol, and is focusing on Hopper 4xA100 32B fine-tuning launch/debug readiness.

- Name: claude-蕾可
  AI type: Claude (Sonnet 4.6, Cowork mode)
  Interface: GUI desktop (Cowork)
  Started at: 2026-03-28T00:00:00-04:00
  Notes: New session. Read full project context: DOGe and AgentDistill are user-provided reference code. Research goal is designing a new method to protect LLM IP against knowledge distillation via fine-tuning. Will assist with research design, code analysis, and experiment planning.

- Name: claude-火稚鸡
  AI type: Claude (Sonnet 4.6, Claude Code CLI)
  Interface: terminal / worktree vibrant-jackson
  Started at: 2026-03-29T00:00:00-04:00
  Notes: New session. Read DOGe codebase and workbench protocol. Ready to assist with code analysis, research, and experiment tasks.

- Name: claude-芽儿豆
  AI type: Claude (Sonnet 4.6, Claude Code CLI)
  Interface: terminal
  Started at: 2026-03-30T00:00:00-04:00
  Notes: New session. Read workbench protocol and current context. Listing Hopper /scratch/wzhao20/AgentDistill/ and recording results.

- Name: codex-拉鲁拉丝
  AI type: Codex
  Interface: desktop terminal
  Started at: 2026-04-02T13:30:04-04:00
  Notes: New session. Read workbench protocol and current repository context. Focusing on repository walkthrough, workbench compliance, and identifying any missing information needed to proceed safely.

- Name: claude-可多拉
  AI type: Claude (Sonnet 4.6, Claude Code CLI)
  Interface: terminal / worktree modest-hugle
  Started at: 2026-04-02T00:00:00-04:00
  Notes: New session. Read workbench protocol and CORE_INFO. Ready to assist with code changes, research, and experiment tasks following local-edit → push → Hopper-pull workflow.
