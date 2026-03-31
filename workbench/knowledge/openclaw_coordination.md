# OpenClaw Coordination Knowledge

Summary:
- OpenClaw can trigger CLI workflows but cannot depend on GUI chat history.
- Any handoff needed by OpenClaw must be written into the shared workbench files.
- Keep wake-up instructions short and point OpenClaw to the current layer first.

## Notes

- Default OpenClaw read order should be:
  1. `/Users/shimu/Downloads/DOGe-main/workbench/HUB.md`
  2. `/Users/shimu/Downloads/DOGe-main/workbench/current/active_tasks.md`
  3. `/Users/shimu/Downloads/DOGe-main/workbench/current/latest_handoff.md`
  4. `/Users/shimu/Downloads/DOGe-main/workbench/dashboard/ssh_status.md`
