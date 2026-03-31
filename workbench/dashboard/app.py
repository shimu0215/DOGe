import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
SHARED_STATUS_MD = BASE_DIR / "ssh_status.md"
TASK_RECORDS_MD = BASE_DIR / "task_records.md"
HOST = "127.0.0.1"
PORT = 8124
REFRESH_INTERVAL_SECONDS = 30
GPU_DETAIL_TIMEOUT_SECONDS = 20
TASK_LOOKBACK_HOURS = 48
SSH_COMMAND = [
    "ssh",
    "hopper",
    (
        "cd /scratch/wzhao20 && "
        "python3 - <<'PY'\n"
        "import json\n"
        "import re\n"
        "import subprocess\n"
        "\n"
        "def run(cmd):\n"
        "    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)\n"
        "    return result.stdout.strip()\n"
        "\n"
        "hopper_user = run('whoami')\n"
        "squeue_lines = run(f\"squeue -h -u {hopper_user} -o '%i|%P|%j|%T|%R|%N|%L'\").splitlines()\n"
        "\n"
        "jobs = []\n"
        "for line in squeue_lines:\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    job_id, partition, name, state, reason, node_name, time_left = line.split('|', 6)\n"
        "    base_job_id = job_id.split('_', 1)[0]\n"
        "    detail_text = run(f\"scontrol show job {base_job_id}\")\n"
        "    req_match = re.search(r'ReqTRES=([^\\n]+)', detail_text)\n"
        "    alloc_match = re.search(r'AllocTRES=([^\\n]+)', detail_text)\n"
        "    node_match = re.search(r'NodeList=(\\S*)', detail_text)\n"
        "    start_match = re.search(r'StartTime=(\\S+)', detail_text)\n"
        "    end_match = re.search(r'EndTime=(\\S+)', detail_text)\n"
        "    runtime_match = re.search(r'RunTime=(\\S+)', detail_text)\n"
        "    command_match = re.search(r'Command=(\\S+)', detail_text)\n"
        "    jobs.append({\n"
        "        'job_id': job_id,\n"
        "        'partition': partition,\n"
        "        'name': name,\n"
        "        'state': state,\n"
        "        'reason': reason,\n"
        "        'node_name': node_name,\n"
        "        'time_left': time_left,\n"
        "        'req_tres': req_match.group(1).strip() if req_match else '',\n"
        "        'alloc_tres': alloc_match.group(1).strip() if alloc_match else '',\n"
        "        'node_list': node_match.group(1) if node_match else '',\n"
        "        'start_time': start_match.group(1) if start_match else '',\n"
        "        'end_time': end_match.group(1) if end_match else '',\n"
        "        'run_time': runtime_match.group(1) if runtime_match else '',\n"
        "        'command': command_match.group(1) if command_match else '',\n"
        "    })\n"
        "\n"
        "print(json.dumps({'jobs': jobs}))\n"
        "PY"
    ),
]


state_lock = threading.Lock()
dashboard_state = {
    "running_jobs": [],
    "pending_jobs": [],
    "task_records": [],
    "summary": {
        "running_gpu_total": 0,
        "pending_gpu_total": 0,
        "running_jobs": 0,
        "pending_jobs": 0,
        "task_records": 0,
    },
    "metadata": {
        "status": "starting",
        "duo_required": False,
        "status_message": "Waiting for first SSH refresh",
        "last_attempt_at": None,
        "last_success_at": None,
        "last_duo_resolved_at": None,
        "last_error": None,
        "refresh_interval_seconds": REFRESH_INTERVAL_SECONDS,
        "ssh_command": " ".join(SSH_COMMAND),
    },
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def format_markdown_timestamp(value):
    if not value:
        return "--"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_remaining_time(end_time):
    if not end_time or end_time in {"Unknown", "N/A", "(null)"}:
        return "--"
    try:
        end_dt = datetime.fromisoformat(end_time)
    except ValueError:
        return end_time
    if end_dt.tzinfo is None:
        now_dt = datetime.now()
    else:
        now_dt = datetime.now(end_dt.tzinfo)
    delta = end_dt - now_dt
    if delta.total_seconds() <= 0:
        return "00:00:00"
    total_seconds = int(delta.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def extract_gpu_count(tres_value):
    if not tres_value or tres_value == "N/A":
        return 0
    total = 0
    for chunk in tres_value.split(","):
        if "gpu" not in chunk:
            continue
        match = re.search(r"=(\d+)$", chunk.strip())
        if match:
            total += int(match.group(1))
    return total


def extract_specific_gpu_count(tres_value, marker):
    if not tres_value or tres_value == "N/A":
        return 0
    total = 0
    for chunk in tres_value.split(","):
        normalized = chunk.strip().lower()
        if marker not in normalized:
            continue
        match = re.search(r"=(\d+)$", chunk.strip())
        if match:
            total += int(match.group(1))
    return total


def count_job_gpus(partition, alloc_tres, req_tres):
    normalized = partition.lower()
    marker = None
    if "gpuq" in normalized:
        marker = "a100"
    elif "b200" in normalized:
        marker = "b200"
    elif "h100" in normalized:
        marker = "h100"

    if marker:
        alloc_specific = extract_specific_gpu_count(alloc_tres, marker)
        req_specific = extract_specific_gpu_count(req_tres, marker)
        if alloc_specific:
            return alloc_specific
        if req_specific:
            return req_specific

    return extract_gpu_count(alloc_tres) or extract_gpu_count(req_tres)


def looks_like_duo_issue(error_text):
    normalized = (error_text or "").lower()
    duo_signals = [
        "duo",
        "two-factor",
        "2fa",
        "verification",
        "passcode",
        "push",
        "timeout",
        "timed out",
    ]
    return any(signal in normalized for signal in duo_signals)


def detect_gpu_type(alloc_tres, req_tres, partition):
    combined = f"{alloc_tres},{req_tres}".lower()
    if "a100.80gb" in combined:
        return "A100 80GB"
    if "a100.40gb" in combined:
        return "A100 40GB"
    if "h100.80gb" in combined:
        return "H100 80GB"
    if "b200.180gb" in combined:
        return "B200 180GB"
    if "a100" in combined:
        return "A100"
    if "h100" in combined:
        return "H100"
    if "b200" in combined:
        return "B200"
    if "gpuq" in partition.lower():
        return "A100"
    return "GPU"


def fetch_job_gpu_details(node_name):
    if not node_name or node_name in {"(null)", "None"}:
        raise ValueError("No running node is available for this job yet.")

    remote_command = (
        "cd /scratch/wzhao20 && "
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {node_name} "
        "\"nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits\""
    )
    result = subprocess.run(
        ["ssh", "hopper", remote_command],
        cwd=BASE_DIR,
        text=True,
        capture_output=True,
        timeout=GPU_DETAIL_TIMEOUT_SECONDS,
        check=True,
    )

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpu_index, gpu_name, memory_used, memory_total, utilization = parts
        gpus.append(
            {
                "gpu_index": gpu_index,
                "gpu_name": gpu_name,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "utilization_gpu_percent": utilization,
            }
        )

    return {"node_name": node_name, "gpus": gpus}


def load_task_records():
    if not TASK_RECORDS_MD.exists():
        return []

    text = TASK_RECORDS_MD.read_text(encoding="utf-8")
    records = []
    blocks = re.split(r"^###\s+", text, flags=re.MULTILINE)
    cutoff = datetime.now(timezone.utc).timestamp() - TASK_LOOKBACK_HOURS * 3600

    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue
        record_id = lines[0].strip()
        content = "\n".join(lines[1:])

        def get_field(name):
            match = re.search(rf"^- {re.escape(name)}:\s*(.+)$", content, flags=re.MULTILINE)
            return match.group(1).strip() if match else ""

        submitted_at = get_field("submitted_at")
        submitted_dt = parse_iso_datetime(submitted_at)
        if not submitted_dt or submitted_dt.timestamp() < cutoff:
            continue

        records.append(
            {
                "record_id": record_id,
                "task_name": get_field("task_name"),
                "job_id": get_field("job_id"),
                "gpu_count": get_field("gpu_count"),
                "submitted_at": submitted_at,
                "description": get_field("description"),
            }
        )

    records.sort(key=lambda item: (item["job_id"], item["submitted_at"]), reverse=False)
    return records


def write_shared_status_markdown():
    metadata = dashboard_state["metadata"]
    running_jobs = dashboard_state["running_jobs"]
    pending_jobs = dashboard_state["pending_jobs"]
    summary = dashboard_state["summary"]

    lines = [
        "# SSH Status",
        "",
        "Shared dashboard status for other AI tools.",
        "",
        "## SSH Health",
        "",
        f"- Status: `{metadata.get('status', '--')}`",
        f"- Message: {metadata.get('status_message', '--')}",
        f"- Duo required: `{metadata.get('duo_required', False)}`",
        f"- Last SSH success: {format_markdown_timestamp(metadata.get('last_success_at'))}",
        f"- Last Duo resolved: {format_markdown_timestamp(metadata.get('last_duo_resolved_at'))}",
        f"- Last attempt: {format_markdown_timestamp(metadata.get('last_attempt_at'))}",
        f"- Last error: {metadata.get('last_error') or '--'}",
        "",
        "## Reservation Summary",
        "",
        f"- Running jobs: `{summary.get('running_jobs', 0)}`",
        f"- Running GPUs: `{summary.get('running_gpu_total', 0)}`",
        f"- Pending jobs: `{summary.get('pending_jobs', 0)}`",
        f"- Pending GPUs: `{summary.get('pending_gpu_total', 0)}`",
        "",
        "## Running Jobs",
        "",
    ]

    if running_jobs:
        for job in running_jobs:
            lines.extend(
                [
                    f"- `{job['job_id']}` | `{job['name']}` | {job['gpu_type']} | {job['allocated_gpus']} GPU(s)",
                    f"  start: {format_markdown_timestamp(job.get('start_time'))} | remaining: {job.get('remaining_time', '--')} | node: {job.get('node_name', '--')}",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Pending Jobs", ""])

    if pending_jobs:
        for job in pending_jobs:
            lines.extend(
                [
                    f"- `{job['job_id']}` | `{job['name']}` | {job['gpu_type']} | {job['requested_gpus']} GPU(s)",
                    f"  reservation time: {format_markdown_timestamp(job.get('start_time'))}",
                ]
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Recent Task Records", ""])
    task_records = dashboard_state.get("task_records", [])
    if task_records:
        for task in task_records:
            lines.extend(
                [
                    f"- `{task['record_id']}` | job `{task['job_id']}` | `{task['task_name']}` | {task['gpu_count']} GPU(s)",
                    f"  submitted: {format_markdown_timestamp(task.get('submitted_at'))} | description: {task.get('description') or '--'}",
                ]
            )
    else:
        lines.append("- None")

    SHARED_STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_snapshot(raw):
    running_jobs = []
    pending_jobs = []
    task_records = load_task_records()
    running_gpu_total = 0
    pending_gpu_total = 0

    for job in raw.get("jobs", []):
        partition = job["partition"].rstrip("*")
        gpu_count = count_job_gpus(
            partition,
            job.get("alloc_tres", ""),
            job.get("req_tres", ""),
        )
        if job["state"] == "PENDING":
            pending_jobs.append(
                {
                    "job_id": job["job_id"],
                    "name": job["name"],
                    "partition": partition,
                    "gpu_type": detect_gpu_type(job.get("alloc_tres", ""), job.get("req_tres", ""), partition),
                    "requested_gpus": gpu_count,
                    "reason": job["reason"],
                    "start_time": job["start_time"],
                    "command": job["command"],
                }
            )
            pending_gpu_total += gpu_count
        elif job["state"] == "RUNNING":
            running_jobs.append(
                {
                    "job_id": job["job_id"],
                    "name": job["name"],
                    "partition": partition,
                    "gpu_type": detect_gpu_type(job.get("alloc_tres", ""), job.get("req_tres", ""), partition),
                    "allocated_gpus": gpu_count,
                    "start_time": job["start_time"],
                    "remaining_time": format_remaining_time(job.get("end_time", "")),
                    "node_name": job.get("node_name", ""),
                }
            )
            running_gpu_total += gpu_count

    running_jobs.sort(key=lambda item: (item["partition"], item["name"]))
    pending_jobs.sort(key=lambda item: (item["partition"], item["name"]))

    return {
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "summary": {
            "running_gpu_total": running_gpu_total,
            "pending_gpu_total": pending_gpu_total,
            "running_jobs": len(running_jobs),
            "pending_jobs": len(pending_jobs),
            "task_records": len(task_records),
        },
        "task_records": task_records,
    }


def refresh_state():
    attempt_at = now_iso()
    try:
        result = subprocess.run(
            SSH_COMMAND,
            cwd=BASE_DIR,
            text=True,
            capture_output=True,
            timeout=25,
            check=True,
        )
        raw = json.loads(result.stdout)
        snapshot = build_snapshot(raw)
        with state_lock:
            previous_duo_required = dashboard_state["metadata"].get("duo_required", False)
            dashboard_state["running_jobs"] = snapshot["running_jobs"]
            dashboard_state["pending_jobs"] = snapshot["pending_jobs"]
            dashboard_state["task_records"] = snapshot["task_records"]
            dashboard_state["summary"] = snapshot["summary"]
            dashboard_state["metadata"].update(
                {
                    "status": "ok",
                    "duo_required": False,
                    "status_message": "SSH refresh is healthy",
                    "last_attempt_at": attempt_at,
                    "last_success_at": now_iso(),
                    "last_duo_resolved_at": now_iso() if previous_duo_required else dashboard_state["metadata"].get("last_duo_resolved_at"),
                    "last_error": None,
                }
            )
            write_shared_status_markdown()
    except subprocess.TimeoutExpired as exc:
        with state_lock:
            dashboard_state["metadata"].update(
                {
                    "status": "duo_required",
                    "duo_required": True,
                    "status_message": "SSH may be waiting for Duo approval",
                    "last_attempt_at": attempt_at,
                    "last_error": str(exc),
                }
            )
            write_shared_status_markdown()
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or exc.stdout or str(exc)).strip()
        duo_required = looks_like_duo_issue(error_text)
        with state_lock:
            dashboard_state["metadata"].update(
                {
                    "status": "duo_required" if duo_required else "error",
                    "duo_required": duo_required,
                    "status_message": "Please approve Duo for ssh hopper" if duo_required else "SSH refresh failed",
                    "last_attempt_at": attempt_at,
                    "last_error": error_text,
                }
            )
            write_shared_status_markdown()
    except Exception as exc:
        with state_lock:
            dashboard_state["metadata"].update(
                {
                    "status": "error",
                    "duo_required": looks_like_duo_issue(str(exc)),
                    "status_message": "Please approve Duo for ssh hopper" if looks_like_duo_issue(str(exc)) else "SSH refresh failed",
                    "last_attempt_at": attempt_at,
                    "last_error": str(exc),
                }
            )
            write_shared_status_markdown()


def polling_loop():
    while True:
        refresh_state()
        time.sleep(REFRESH_INTERVAL_SECONDS)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            params = parse_qs(parsed.query)
            if params.get("refresh") == ["1"]:
                refresh_state()
            with state_lock:
                payload = json.dumps(dashboard_state).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/job-details":
            params = parse_qs(parsed.query)
            job_id = params.get("job_id", [""])[0]
            with state_lock:
                job = next((item for item in dashboard_state["running_jobs"] if item["job_id"] == job_id), None)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Running job not found")
                return
            try:
                payload = json.dumps(fetch_job_gpu_details(job.get("node_name", ""))).encode("utf-8")
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_GATEWAY, str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()


def main():
    thread = threading.Thread(target=polling_loop, daemon=True)
    thread.start()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Serving dashboard at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
