# Hopper GPU Dashboard

This dashboard serves live Hopper GPU queue status from a local Python backend.

## What it does

- polls Hopper over SSH on a timer
- fetches Slurm partition and queue data directly instead of reading markdown files
- exposes a local `/api/status` endpoint for the frontend
- refreshes the browser view automatically while the server is running

## Run

```bash
cd /Users/shimu/Downloads/DOGe-main/workbench/dashboard
python3 app.py
```

Then open:

```text
http://127.0.0.1:8124
```

## Notes

- The backend uses `ssh hopper 'cd /scratch/wzhao20 && ...'`.
- If Hopper requires Duo, the first SSH connection may wait for approval.
- The frontend refreshes every 15 seconds.
- The backend refreshes Hopper data every 30 seconds.
