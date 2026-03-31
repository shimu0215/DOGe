#!/usr/bin/env python3
"""Local dashboard for Hopper experiment progress.

Run locally:
    python scripts/experiment_progress_dashboard.py

Then open:
    http://127.0.0.1:8787

The server queries Hopper on each refresh via `ssh hopper`, so an existing SSH
session is recommended to avoid repeated interactive prompts.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOST = "127.0.0.1"
PORT = 8787
REMOTE_ROOT = "/scratch/wzhao20/llama_factory"


CURRENT_TASKS = [
    {
        "id": "trl_qwen3_32b_teacher",
        "title": "TRL Qwen3-32B Teacher Paths",
        "kind": "collection",
        "description": "AgentDistill 500 GSM-hard, 5 samples, temperature 0.7, resume enabled.",
        "result_dir": f"{REMOTE_ROOT}/outputs/trl_results_aligned/trl_qwen3_32b_gsmhard500_python_5samples_teacher",
        "records_path": f"{REMOTE_ROOT}/outputs/trl_results_aligned/trl_qwen3_32b_gsmhard500_python_5samples_teacher/trl_generated_data/records.jsonl",
        "contexts_path": f"{REMOTE_ROOT}/outputs/trl_results_aligned/trl_qwen3_32b_gsmhard500_python_5samples_teacher/trl_generated_data/contexts.jsonl",
        "summary_path": f"{REMOTE_ROOT}/outputs/trl_results_aligned/trl_qwen3_32b_gsmhard500_python_5samples_teacher/trl_generated_data/summary.json",
        "log_path": f"{REMOTE_ROOT}/hpc-results/trl_qwen3_32b_gsmhard500_python_5samples_teacher_resume_6576598.log",
        "notes": "Will continue on the next active 4xA100 allocation.",
    },
    {
        "id": "student_base14",
        "title": "Student Eval Base14",
        "kind": "eval",
        "description": "Qwen3-8B base with qwen7_from_base14 adapter on new aligned chain.",
        "result_dir": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_base14_gsmhard500_newchain_rerun_6577625",
        "records_path": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_base14_gsmhard500_newchain_rerun_6577625/gsm_hard_agentdistill_aligned_records_part_000.jsonl",
        "grouped_path": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_base14_gsmhard500_newchain_rerun_6577625/gsm_hard_agentdistill_aligned_grouped_part_000.jsonl",
        "summary_path": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_base14_gsmhard500_newchain_rerun_6577625/gsm_hard_agentdistill_aligned_summary.json",
        "log_path": f"{REMOTE_ROOT}/hpc-results/qwen3_8b_students_gsmhard500_newchain_rerun_6577625.log",
        "notes": "Runs first on the active 2xA100 allocation.",
    },
    {
        "id": "student_ft14",
        "title": "Student Eval FT14",
        "kind": "eval",
        "description": "Qwen3-8B base with qwen7_from_ft14 adapter on new aligned chain.",
        "result_dir": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_ft14_gsmhard500_newchain_rerun_6577625",
        "records_path": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_ft14_gsmhard500_newchain_rerun_6577625/gsm_hard_agentdistill_aligned_records_part_000.jsonl",
        "grouped_path": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_ft14_gsmhard500_newchain_rerun_6577625/gsm_hard_agentdistill_aligned_grouped_part_000.jsonl",
        "summary_path": f"{REMOTE_ROOT}/outputs/offline_agent_alignment/qwen3_8b_student_ft14_gsmhard500_newchain_rerun_6577625/gsm_hard_agentdistill_aligned_summary.json",
        "log_path": f"{REMOTE_ROOT}/hpc-results/qwen3_8b_students_gsmhard500_newchain_rerun_6577625.log",
        "notes": "Starts after Base14 finishes in the same sequential launcher.",
    },
]


NEXT_TASKS = [
    {
        "title": "Resume Qwen3-32B on 4xA100",
        "wait_for": ["6577637", "6577638"],
        "rule": "Keep gpuq if both 18h 4xA100 reservations activate; otherwise use the first active one.",
    },
    {
        "title": "Continue FT14 after Base14",
        "wait_for": ["student_base14 completion"],
        "rule": "Same 2xA100 sequential launcher continues into FT14 automatically.",
    },
]


COMPLETED_REFERENCE = [
    {
        "title": "AgentDistill Native Qwen3-1.7B",
        "result": "0.59 on gsm_hard_500_20250507",
        "path": "/scratch/wzhao20/AgentDistill/logs/qa_results/vllm/Qwen_Qwen3-1.7B/gsm_hard_500_20250507_test/Qwen3-1.7B_temp=0.0_seed=42_type=agent_steps=5_duckduckgo.jsonl",
    },
    {
        "title": "TRL / Non-TRL Alignment Checks",
        "result": "Qwen2.5-1.5B matched around 0.1; Qwen3-1.7B matched around 0.5 to 0.5667 on 30-question sanity runs.",
        "path": "/scratch/wzhao20/llama_factory/outputs/trl_results_aligned",
    },
]


REMOTE_SCRIPT = textwrap.dedent(
    """
    import json
    import os
    import subprocess
    from pathlib import Path

    TASKS = json.loads(__TASKS_JSON__)
    QUEUE_JOB_IDS = json.loads(__QUEUE_JOB_IDS_JSON__)

    def run(cmd):
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return res.stdout

    def queue_snapshot(job_ids):
        if not job_ids:
            return {}
        fmt = "%i|%P|%j|%T|%M|%l|%R|%N"
        out = run(["squeue", "-h", "-o", fmt, "-j", ",".join(job_ids)])
        jobs = {}
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) >= 8:
                jobs[parts[0]] = {
                    "job_id": parts[0],
                    "partition": parts[1],
                    "name": parts[2],
                    "state": parts[3],
                    "time_used": parts[4],
                    "time_limit": parts[5],
                    "reason": parts[6],
                    "nodelist": parts[7],
                }
        return jobs

    def line_count(path: str):
        if not path:
            return None
        p = Path(path)
        if not p.exists() or p.is_dir():
            return None
        count = 0
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                count += 1
        return count

    def file_info(path: str):
        if not path:
            return None
        p = Path(path)
        if not p.exists() or p.is_dir():
            return None
        st = p.stat()
        return {
            "size": st.st_size,
            "mtime": st.st_mtime,
        }

    def tail(path: str, lines: int = 20):
        if not path:
            return None
        p = Path(path)
        if not p.exists() or p.is_dir():
            return None
        text = p.read_text(encoding="utf-8", errors="ignore")
        return "\\n".join(text.splitlines()[-lines:])

    def summary_data(path: str):
        if not path:
            return None
        p = Path(path)
        if not p.exists() or p.is_dir():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    gpu_2 = run([
        "squeue", "-h", "-o", "%i|%P|%j|%T|%M|%l|%R|%N",
        "-j", "6577625,6577626"
    ])
    gpu_4 = run([
        "squeue", "-h", "-o", "%i|%P|%j|%T|%M|%l|%R|%N",
        "-j", "6577637,6577638"
    ])

    dashboard = {
        "generated_at": __import__("time").time(),
        "jobs": queue_snapshot(QUEUE_JOB_IDS),
        "gpu_2_snapshot": gpu_2,
        "gpu_4_snapshot": gpu_4,
        "tasks": [],
        "alerts": [],
    }

    for task in TASKS:
        entry = dict(task)
        entry["records_count"] = line_count(task.get("records_path", ""))
        entry["contexts_count"] = line_count(task.get("contexts_path", ""))
        entry["grouped_count"] = line_count(task.get("grouped_path", ""))
        entry["records_info"] = file_info(task.get("records_path", ""))
        entry["summary"] = summary_data(task.get("summary_path", ""))
        entry["log_tail"] = tail(task.get("log_path", ""))
        dashboard["tasks"].append(entry)

        log_tail = entry["log_tail"] or ""
        if "Traceback" in log_tail or "ERROR" in log_tail or "Internal Server Error" in log_tail:
            dashboard["alerts"].append({
                "task_id": task["id"],
                "message": "Recent log contains an error marker.",
            })

    print(json.dumps(dashboard))
    """
).strip()


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Progress Dashboard</title>
  <style>
    :root {
      --bg: #f3efe4;
      --panel: rgba(255,255,255,0.78);
      --ink: #1c1b18;
      --muted: #60584d;
      --line: rgba(28,27,24,0.12);
      --accent: #0c6b58;
      --warn: #b65424;
      --bad: #a22727;
      --good: #20663f;
      --gold: #c8a552;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(200,165,82,0.25), transparent 28%),
        radial-gradient(circle at top right, rgba(12,107,88,0.18), transparent 32%),
        linear-gradient(180deg, #f8f4e9 0%, var(--bg) 48%, #ece4d5 100%);
      min-height: 100vh;
    }
    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }
    .hero {
      display: grid;
      gap: 12px;
      padding: 24px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(247,241,228,0.86));
      box-shadow: 0 18px 60px rgba(51, 41, 22, 0.08);
      border-radius: 24px;
    }
    h1 {
      margin: 0;
      font-size: clamp(30px, 5vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.03em;
    }
    .sub {
      color: var(--muted);
      font-size: 16px;
      max-width: 72ch;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      cursor: pointer;
      background: var(--ink);
      color: white;
    }
    .soft {
      background: rgba(28,27,24,0.08);
      color: var(--ink);
    }
    .grid {
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }
    .card {
      grid-column: span 12;
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(10px);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(51, 41, 22, 0.05);
    }
    .span-7 { grid-column: span 7; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-4 { grid-column: span 4; }
    .kicker {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }
    h2, h3 {
      margin: 0 0 10px;
      font-size: 24px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 14px;
      border-radius: 16px;
      background: rgba(28,27,24,0.04);
      border: 1px solid rgba(28,27,24,0.06);
    }
    .stat .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .stat .value {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(12,107,88,0.1);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }
    .pill.warn { background: rgba(182,84,36,0.12); color: var(--warn); }
    .pill.bad { background: rgba(162,39,39,0.12); color: var(--bad); }
    .task {
      padding: 16px;
      border-radius: 16px;
      border: 1px solid rgba(28,27,24,0.08);
      background: rgba(255,255,255,0.56);
      margin-top: 14px;
    }
    .task-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      flex-wrap: wrap;
    }
    .task p {
      margin: 8px 0 0;
      color: var(--muted);
    }
    .meta {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .meta div {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(28,27,24,0.04);
    }
    .meta .name {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 6px;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.45;
      padding: 14px;
      border-radius: 16px;
      background: #181713;
      color: #ece7dc;
      margin: 14px 0 0;
      max-height: 260px;
      overflow: auto;
    }
    ul {
      margin: 0;
      padding-left: 18px;
    }
    li + li { margin-top: 8px; }
    .foot {
      color: var(--muted);
      font-size: 13px;
      margin-top: 16px;
    }
    @media (max-width: 900px) {
      .span-7, .span-5, .span-6, .span-4 { grid-column: span 12; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="kicker">Local Dashboard</div>
      <h1>Hopper Experiment Progress</h1>
      <div class="sub">Tracks current reservations, running experiments, queued work, next planned tasks, and recent errors. Refreshes from Hopper through <code>ssh hopper</code>.</div>
      <div class="toolbar">
        <button id="refresh">Refresh Now</button>
        <button class="soft" id="toggle">Pause Auto Refresh</button>
        <span id="status" class="pill">Loading</span>
      </div>
    </section>

    <section class="grid">
      <div class="card span-7">
        <div class="kicker">Right Now</div>
        <h2>Running And Queued</h2>
        <div id="running"></div>
      </div>
      <div class="card span-5">
        <div class="kicker">Priority View</div>
        <h2>What We Are Doing</h2>
        <div id="current-tasks"></div>
      </div>

      <div class="card span-6">
        <div class="kicker">Next Up</div>
        <h2>Upcoming Tasks</h2>
        <div id="next-tasks"></div>
      </div>
      <div class="card span-6">
        <div class="kicker">Completed Context</div>
        <h2>Finished Reference Runs</h2>
        <div id="completed"></div>
      </div>

      <div class="card span-12">
        <div class="kicker">Attention</div>
        <h2>Alerts And Health</h2>
        <div id="alerts"></div>
      </div>
    </section>
  </div>

  <script>
    const currentTasks = __CURRENT_TASKS__;
    const nextTasks = __NEXT_TASKS__;
    const completedReference = __COMPLETED_REFERENCE__;
    let timer = null;
    let autoRefresh = true;

    function fmtTime(ts) {
      if (!ts) return "Unknown";
      return new Date(ts * 1000).toLocaleString();
    }

    function fmtCount(v) {
      if (v === null || v === undefined) return "Not started";
      return v.toLocaleString();
    }

    function pill(text, kind = "") {
      const cls = kind ? `pill ${kind}` : "pill";
      return `<span class="${cls}">${text}</span>`;
    }

    function renderJobs(data) {
      const jobs = Object.values(data.jobs || {});
      if (!jobs.length) {
        return `<p>No matching reservations are active or queued right now.</p>`;
      }
      return jobs.map((job) => `
        <div class="task">
          <div class="task-head">
            <div>
              <h3>${job.name}</h3>
              <p>Job ${job.job_id} on ${job.partition}</p>
            </div>
            ${pill(job.state, job.state.includes("RUN") ? "" : job.state.includes("PEND") || job.state.includes("PD") ? "warn" : "bad")}
          </div>
          <div class="meta">
            <div><span class="name">Time Used</span>${job.time_used || "0:00"}</div>
            <div><span class="name">Time Limit</span>${job.time_limit || "-"}</div>
            <div><span class="name">Node / Reason</span>${job.nodelist || job.reason || "-"}</div>
          </div>
        </div>
      `).join("");
    }

    function renderCurrentTasks(data) {
      return currentTasks.map((task) => {
        const live = (data.tasks || []).find((item) => item.id === task.id) || {};
        let state = "Waiting";
        let kind = "warn";
        if (live.summary) {
          state = "Completed";
          kind = "";
        } else if (live.records_count) {
          state = "Running";
          kind = "";
        }
        const summaryBits = [];
        if (live.records_count !== undefined) summaryBits.push(`records ${fmtCount(live.records_count)}`);
        if (live.contexts_count !== undefined) summaryBits.push(`contexts ${fmtCount(live.contexts_count)}`);
        if (live.grouped_count !== undefined && live.grouped_count !== null) summaryBits.push(`grouped ${fmtCount(live.grouped_count)}`);
        if (live.records_info && live.records_info.mtime) summaryBits.push(`updated ${fmtTime(live.records_info.mtime)}`);

        return `
          <div class="task">
            <div class="task-head">
              <div>
                <h3>${task.title}</h3>
                <p>${task.description}</p>
              </div>
              ${pill(state, kind)}
            </div>
            <div class="meta">
              <div><span class="name">Counts</span>${summaryBits.join(" · ") || "No output yet"}</div>
              <div><span class="name">Output Dir</span><code>${task.result_dir}</code></div>
              <div><span class="name">Notes</span>${task.notes || "-"}</div>
            </div>
            ${live.log_tail ? `<pre>${escapeHtml(live.log_tail)}</pre>` : ""}
          </div>
        `;
      }).join("");
    }

    function renderNextTasks() {
      return nextTasks.map((task) => `
        <div class="task">
          <div class="task-head">
            <div>
              <h3>${task.title}</h3>
              <p>${task.rule}</p>
            </div>
            ${pill("Queued Logic", "warn")}
          </div>
          <div class="meta">
            <div><span class="name">Waiting For</span>${task.wait_for.join(", ")}</div>
          </div>
        </div>
      `).join("");
    }

    function renderCompleted() {
      return completedReference.map((item) => `
        <div class="task">
          <div class="task-head">
            <div>
              <h3>${item.title}</h3>
              <p>${item.result}</p>
            </div>
            ${pill("Done")}
          </div>
          <div class="meta">
            <div><span class="name">Path</span><code>${item.path}</code></div>
          </div>
        </div>
      `).join("");
    }

    function renderAlerts(data) {
      const alerts = data.alerts || [];
      const blocks = [];
      if (alerts.length) {
        alerts.forEach((alert) => {
          blocks.push(`
            <div class="task">
              <div class="task-head">
                <div>
                  <h3>${alert.task_id}</h3>
                  <p>${alert.message}</p>
                </div>
                ${pill("Needs Attention", "bad")}
              </div>
            </div>
          `);
        });
      }
      blocks.push(`
        <div class="stats">
          <div class="stat">
            <div class="label">Last Refresh</div>
            <div class="value" style="font-size:20px">${new Date(data.generated_at * 1000).toLocaleTimeString()}</div>
          </div>
          <div class="stat">
            <div class="label">SSH Mode</div>
            <div class="value" style="font-size:20px">ssh hopper</div>
          </div>
          <div class="stat">
            <div class="label">Auto Refresh</div>
            <div class="value" style="font-size:20px">${autoRefresh ? "On" : "Paused"}</div>
          </div>
        </div>
      `);
      return blocks.join("");
    }

    function escapeHtml(text) {
      return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    async function loadData() {
      const status = document.getElementById("status");
      status.textContent = "Refreshing";
      status.className = "pill warn";
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        const data = await res.json();
        document.getElementById("running").innerHTML = renderJobs(data);
        document.getElementById("current-tasks").innerHTML = renderCurrentTasks(data);
        document.getElementById("next-tasks").innerHTML = renderNextTasks();
        document.getElementById("completed").innerHTML = renderCompleted();
        document.getElementById("alerts").innerHTML = renderAlerts(data);
        status.textContent = "Healthy";
        status.className = "pill";
      } catch (err) {
        status.textContent = "Refresh Failed";
        status.className = "pill bad";
        document.getElementById("alerts").innerHTML = `
          <div class="task">
            <div class="task-head">
              <div>
                <h3>Dashboard Refresh Error</h3>
                <p>${escapeHtml(String(err))}</p>
              </div>
              ${pill("Connection Issue", "bad")}
            </div>
          </div>
        `;
      }
    }

    function setAutoRefresh(enabled) {
      autoRefresh = enabled;
      const toggle = document.getElementById("toggle");
      toggle.textContent = enabled ? "Pause Auto Refresh" : "Resume Auto Refresh";
      if (timer) clearInterval(timer);
      if (enabled) timer = setInterval(loadData, 30000);
    }

    document.getElementById("refresh").addEventListener("click", loadData);
    document.getElementById("toggle").addEventListener("click", () => setAutoRefresh(!autoRefresh));
    setAutoRefresh(true);
    loadData();
  </script>
</body>
</html>
"""


@dataclass
class DashboardConfig:
    current_tasks: list[dict[str, Any]]
    next_tasks: list[dict[str, Any]]
    completed_reference: list[dict[str, Any]]
    queue_job_ids: list[str]


class DashboardHandler(BaseHTTPRequestHandler):
    config = DashboardConfig(
        current_tasks=CURRENT_TASKS,
        next_tasks=NEXT_TASKS,
        completed_reference=COMPLETED_REFERENCE,
        queue_job_ids=["6577625", "6577626", "6577637", "6577638"],
    )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._send_html()
            return
        if self.path == "/api/status":
            self._send_status()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dashboard] " + (fmt % args) + "\n")

    def _send_html(self) -> None:
        page = (
            HTML_PAGE.replace("__CURRENT_TASKS__", json.dumps(self.config.current_tasks))
            .replace("__NEXT_TASKS__", json.dumps(self.config.next_tasks))
            .replace("__COMPLETED_REFERENCE__", json.dumps(self.config.completed_reference))
        )
        data = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_status(self) -> None:
        payload = collect_status(self.config)
        data = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_ssh_json(config: DashboardConfig) -> dict[str, Any]:
    remote_script = REMOTE_SCRIPT.replace("__TASKS_JSON__", repr(json.dumps(config.current_tasks))).replace(
        "__QUEUE_JOB_IDS_JSON__", repr(json.dumps(config.queue_job_ids))
    )
    remote_cmd = f"python - <<'PY'\n{remote_script}\nPY"
    cmd = ["ssh", "hopper", remote_cmd]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        return {
            "generated_at": time.time(),
            "jobs": {},
            "tasks": [],
            "alerts": [{"task_id": "ssh", "message": (res.stderr or res.stdout or "ssh failed").strip()}],
            "error": "ssh_failed",
        }
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {
            "generated_at": time.time(),
            "jobs": {},
            "tasks": [],
            "alerts": [{"task_id": "ssh", "message": "Could not parse Hopper dashboard payload."}],
            "error": "json_parse_failed",
            "raw": res.stdout[-2000:],
        }


def collect_status(config: DashboardConfig) -> dict[str, Any]:
    data = run_ssh_json(config)
    if "generated_at" not in data:
        data["generated_at"] = time.time()
    return data


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Dashboard running at http://{HOST}:{PORT}")
    print("This page refreshes from Hopper via `ssh hopper`.")
    server.serve_forever()


if __name__ == "__main__":
    main()
