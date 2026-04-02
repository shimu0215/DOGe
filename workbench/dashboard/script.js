const refreshButton = document.getElementById("refreshButton");
const jobDetailCache = new Map();
const openDetailRows = new Set();

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function formatTimestamp(value) {
  if (!value) return "--";
  if (value === "Unknown" || value === "N/A") return value;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderGpuDetails(detail) {
  if (!detail?.gpus?.length) {
    return '<div class="gpu-detail-empty">No GPU detail returned.</div>';
  }

  const cards = detail.gpus.map((gpu) => `
    <div class="gpu-detail-card">
      <strong>GPU ${escapeHtml(gpu.gpu_index)}</strong>
      <span>${escapeHtml(gpu.gpu_name)}</span>
      <span>Memory: ${escapeHtml(gpu.memory_used_mb)} / ${escapeHtml(gpu.memory_total_mb)} MB</span>
      <span>Util: ${escapeHtml(gpu.utilization_gpu_percent)}%</span>
    </div>
  `).join("");

  return `<div class="gpu-detail-grid">${cards}</div>`;
}

async function toggleJobDetails(jobId, button) {
  const detailRow = document.getElementById(`job-detail-${jobId}`);
  if (!detailRow) return;

  if (openDetailRows.has(jobId)) {
    openDetailRows.delete(jobId);
    detailRow.hidden = true;
    button.textContent = "Show";
    return;
  }

  openDetailRows.add(jobId);
  detailRow.hidden = false;
  button.textContent = "Hide";

  if (jobDetailCache.has(jobId)) {
    detailRow.querySelector(".gpu-detail-shell").innerHTML = renderGpuDetails(jobDetailCache.get(jobId));
    return;
  }

  detailRow.querySelector(".gpu-detail-shell").innerHTML = '<div class="gpu-detail-empty">Loading GPU detail...</div>';
  try {
    const response = await fetch(`/api/job-details?job_id=${encodeURIComponent(jobId)}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const detail = await response.json();
    jobDetailCache.set(jobId, detail);
    detailRow.querySelector(".gpu-detail-shell").innerHTML = renderGpuDetails(detail);
  } catch (error) {
    detailRow.querySelector(".gpu-detail-shell").innerHTML =
      `<div class="gpu-detail-empty">Failed to load GPU detail: ${escapeHtml(error.message)}</div>`;
  }
}

function renderRunningJobs(jobs) {
  const tbody = document.getElementById("runningTableBody");
  tbody.innerHTML = "";

  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="7">No running Hopper jobs for this user.</td></tr>';
    return;
  }

  jobs.forEach((job) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${job.job_id}</td>
      <td>${job.name}</td>
      <td>${formatTimestamp(job.start_time)}</td>
      <td>${job.remaining_time || "--"}</td>
      <td>${job.gpu_type || "--"}</td>
      <td>${job.allocated_gpus}</td>
      <td><button class="detail-button" data-job-id="${job.job_id}">Show</button></td>
    `;
    tbody.appendChild(row);

    const detailRow = document.createElement("tr");
    detailRow.id = `job-detail-${job.job_id}`;
    detailRow.className = "job-detail-row";
    detailRow.hidden = !openDetailRows.has(job.job_id);
    detailRow.innerHTML = `
      <td colspan="7">
        <div class="gpu-detail-shell">
          ${jobDetailCache.has(job.job_id) ? renderGpuDetails(jobDetailCache.get(job.job_id)) : '<div class="gpu-detail-empty">Click Show to fetch GPU memory and util.</div>'}
        </div>
      </td>
    `;
    tbody.appendChild(detailRow);
  });

  tbody.querySelectorAll(".detail-button").forEach((button) => {
    const jobId = button.dataset.jobId;
    button.textContent = openDetailRows.has(jobId) ? "Hide" : "Show";
    button.addEventListener("click", () => toggleJobDetails(jobId, button));
  });
}

function renderPendingJobs(jobs) {
  const tbody = document.getElementById("pendingTableBody");
  tbody.innerHTML = "";

  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="5">No pending Hopper jobs for this user.</td></tr>';
    return;
  }

  jobs.forEach((job) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${job.job_id}</td>
      <td>${job.name}</td>
      <td>${formatTimestamp(job.start_time)}</td>
      <td>${job.gpu_type || "--"}</td>
      <td>${job.requested_gpus}</td>
    `;
    tbody.appendChild(row);
  });
}

function renderStatus(payload) {
  const summary = payload.summary || {};
  const metadata = payload.metadata || {};
  const duoRequired = Boolean(metadata.duo_required);
  const banner = document.getElementById("duoBanner");

  setText("runningGpus", summary.running_gpu_total ?? "--");
  setText("queuedGpus", summary.pending_gpu_total ?? "--");
  setText("pendingJobs", summary.pending_jobs ?? "--");
  setText("lastRefresh", formatTimestamp(metadata.last_success_at || metadata.last_attempt_at));
  setText("pollingBadge", `Polling every ${metadata.refresh_interval_seconds ?? "--"} s`);
  setText("runningCount", `${(payload.running_jobs || []).length} jobs`);
  setText("pendingCount", `${(payload.pending_jobs || []).length} jobs`);
  setText("lastSuccessAt", formatTimestamp(metadata.last_success_at));
  setText("lastDuoResolvedAt", formatTimestamp(metadata.last_duo_resolved_at));
  setText("duoBannerTitle", duoRequired ? "Duo action needed" : "SSH status");
  setText("duoBannerText", metadata.status_message || "SSH refresh is healthy");
  banner.classList.toggle("duo-needed", duoRequired);
  banner.classList.toggle("status-ok", !duoRequired);

  setText("runningTrend", summary.running_gpu_total > 0 ? "Your reserved GPUs are active" : "No running reserved GPUs");
  setText("queueTrend", summary.pending_gpu_total > 0 ? "You still have GPUs waiting in queue" : "No queued GPU requests");
  setText("pendingTrend", summary.pending_jobs > 0 ? "Queued reservations detected" : "No queued jobs");
  setText("refreshStatus", metadata.status === "ok" ? "Latest refresh succeeded" : "Latest refresh needs attention");

  renderRunningJobs(payload.running_jobs || []);
  renderPendingJobs(payload.pending_jobs || []);
}

async function fetchStatus(force = false) {
  const response = await fetch(force ? "/api/status?refresh=1" : "/api/status", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const payload = await response.json();
  renderStatus(payload);
}

async function loadStatus(force = false) {
  refreshButton.disabled = true;
  try {
    await fetchStatus(force);
  } catch (error) {
    setText("runningTrend", "Unable to refresh running jobs");
    setText("queueTrend", error.message);
    setText("refreshStatus", "Unable to load API response");
    setText("duoBannerTitle", "SSH status");
    setText("duoBannerText", "Dashboard could not reach the local status API.");
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", () => loadStatus(true));

loadStatus(true);
setInterval(() => loadStatus(false), 15000);
