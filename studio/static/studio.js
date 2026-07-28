"use strict";

const query = new URLSearchParams(window.location.search);

const state = {
  presets: new Map(),
  protocols: new Map(),
  results: [],
  selected: null,
  job: null,
  preflight: null,
  advanced:
    query.get("advanced") === "1"
    ||
    localStorage.getItem("tesseract-advanced") === "true"
    || localStorage.getItem("tesseract-science") === "true",
  analysis: null,
  terminalSequence: 0,
  terminalJob: null,
  terminalHistory: [],
  terminalHistoryIndex: 0,
  completedJob: null,
  polling: false,
  toastTimer: null,
};

const elements = {
  canvas: document.getElementById("spacetimeCanvas"),
  presetGrid: document.getElementById("presetGrid"),
  presetDialog: document.getElementById("presetDialog"),
  dialogEyebrow: document.getElementById("dialogEyebrow"),
  dialogTitle: document.getElementById("dialogTitle"),
  dialogSummary: document.getElementById("dialogSummary"),
  dialogExpectation: document.getElementById("dialogExpectation"),
  dialogDuration: document.getElementById("dialogDuration"),
  dialogVram: document.getElementById("dialogVram"),
  dialogDisk: document.getElementById("dialogDisk"),
  dialogAdvanced: document.getElementById("dialogAdvanced"),
  startButton: document.getElementById("startButton"),
  resumeButton: document.getElementById("resumeButton"),
  advancedToggle: document.getElementById("advancedToggle"),
  systemBadge: document.getElementById("systemBadge"),
  preflightGrid: document.getElementById("preflightGrid"),
  runDeck: document.getElementById("runDeck"),
  runEyebrow: document.getElementById("runEyebrow"),
  runTitle: document.getElementById("runTitle"),
  runStateBadge: document.getElementById("runStateBadge"),
  progressBar: document.getElementById("progressBar"),
  progressLabel: document.getElementById("progressLabel"),
  stepLabel: document.getElementById("stepLabel"),
  etaLabel: document.getElementById("etaLabel"),
  statusMessage: document.getElementById("statusMessage"),
  elapsedValue: document.getElementById("elapsedValue"),
  simTimeValue: document.getElementById("simTimeValue"),
  checkpointValue: document.getElementById("checkpointValue"),
  cancelButton: document.getElementById("cancelButton"),
  openOutputButton: document.getElementById("openOutputButton"),
  collapseRunButton: document.getElementById("collapseRunButton"),
  eventLog: document.getElementById("eventLog"),
  eventCount: document.getElementById("eventCount"),
  viewportCaption: document.getElementById("viewportCaption"),
  viewportLabel: document.getElementById("viewportLabel"),
  viewportClock: document.getElementById("viewportClock"),
  infoDialog: document.getElementById("infoDialog"),
  exploreButton: document.getElementById("exploreButton"),
  demoButton: document.getElementById("demoButton"),
  footerMode: document.getElementById("footerMode"),
  toast: document.getElementById("toast"),
  laboratory: document.getElementById("laboratory"),
  labStatusTitle: document.getElementById("labStatusTitle"),
  labStatusDetail: document.getElementById("labStatusDetail"),
  protocolSelect: document.getElementById("protocolSelect"),
  protocolSummary: document.getElementById("protocolSummary"),
  parameterGrid: document.getElementById("parameterGrid"),
  qualificationNote: document.getElementById("qualificationNote"),
  commandPreview: document.getElementById("commandPreview"),
  copyCommandButton: document.getElementById("copyCommandButton"),
  labLaunchButton: document.getElementById("labLaunchButton"),
  labResetButton: document.getElementById("labResetButton"),
  labRunner: document.getElementById("labRunner"),
  labEstimate: document.getElementById("labEstimate"),
  labVram: document.getElementById("labVram"),
  intelligenceCore: document.getElementById("intelligenceCore"),
  labTerminalLog: document.getElementById("labTerminalLog"),
  terminalForm: document.getElementById("terminalForm"),
  terminalInput: document.getElementById("terminalInput"),
  terminalConnection: document.getElementById("terminalConnection"),
  resultSelect: document.getElementById("resultSelect"),
  refreshResultsButton: document.getElementById("refreshResultsButton"),
  copyJsonButton: document.getElementById("copyJsonButton"),
  evidenceEmpty: document.getElementById("evidenceEmpty"),
  evidenceDashboard: document.getElementById("evidenceDashboard"),
  evidenceSchema: document.getElementById("evidenceSchema"),
  evidenceTitle: document.getElementById("evidenceTitle"),
  evidencePath: document.getElementById("evidencePath"),
  qualificationSeal: document.getElementById("qualificationSeal"),
  evidenceKpis: document.getElementById("evidenceKpis"),
  gateScore: document.getElementById("gateScore"),
  qualificationGrid: document.getElementById("qualificationGrid"),
  provenanceList: document.getElementById("provenanceList"),
  seriesSelect: document.getElementById("seriesSelect"),
  metricSelect: document.getElementById("metricSelect"),
  evidenceChart: document.getElementById("evidenceChart"),
  chartEmpty: document.getElementById("chartEmpty"),
  seriesTable: document.getElementById("seriesTable"),
  rawJson: document.getElementById("rawJson"),
};

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => {
    elements.toast.classList.remove("visible");
  }, 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function formatDuration(seconds, compact = false) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return "—";
  }
  const rounded = Math.max(0, Math.round(seconds));
  const days = Math.floor(rounded / 86400);
  const hours = Math.floor((rounded % 86400) / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const secs = rounded % 60;
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${minutes}m`;
  if (compact) {
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function renderPresetCards() {
  const cards = [...state.presets.values()].map((preset) => {
    const resume = preset.resume_available
      ? '<span class="resume-chip">● CHECKPOINT READY</span>'
      : "";
    return `
      <article class="preset-card" role="button" tabindex="0"
        data-preset="${escapeHTML(preset.id)}"
        style="--card-accent:${escapeHTML(preset.accent)}">
        <div class="card-top">
          <span class="eyebrow">${escapeHTML(preset.eyebrow)}</span>
          <span class="tier-chip">${escapeHTML(preset.tier)}</span>
        </div>
        <h3>${escapeHTML(preset.title)}</h3>
        <p>${escapeHTML(preset.summary)}</p>
        <div class="card-meta">
          <span><i></i>${escapeHTML(preset.duration_label)}</span>
          <span>${preset.minimum_vram_gib || 0} GiB VRAM</span>
        </div>
        ${resume}
      </article>
    `;
  });
  elements.presetGrid.innerHTML = cards.join("");
  for (const card of elements.presetGrid.querySelectorAll(".preset-card")) {
    const activate = () => openPreset(card.dataset.preset);
    card.addEventListener("click", activate);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  }
}

function openPreset(identifier) {
  const preset = state.presets.get(identifier);
  if (!preset) return;
  state.selected = preset;
  elements.dialogEyebrow.textContent = preset.eyebrow;
  elements.dialogEyebrow.style.color = preset.accent;
  elements.dialogTitle.textContent = preset.title;
  elements.dialogSummary.textContent = preset.summary;
  elements.dialogExpectation.textContent = preset.what_you_will_see;
  elements.dialogDuration.textContent = preset.duration_label;
  elements.dialogVram.textContent = `${preset.minimum_vram_gib || 0} GiB`;
  elements.dialogDisk.textContent = preset.disk_label;
  elements.dialogAdvanced.textContent = preset.total_steps
    ? `${preset.total_steps} locked evolution steps. The browser stops only the managed runner, and Resume is offered whenever its checkpoint is present.`
    : "The exact command is locked by the repository preset manifest.";
  elements.resumeButton.classList.toggle("hidden", !preset.resume_available);
  const active = state.job && ["starting", "running", "stopping"].includes(state.job.status);
  elements.startButton.disabled = active;
  elements.resumeButton.disabled = active;
  elements.presetDialog.showModal();
  visualizer.setMode(preset.visualizer, 0);
}

async function startSelected(resume = false) {
  if (!state.selected) return;
  elements.startButton.disabled = true;
  elements.resumeButton.disabled = true;
  try {
    const payload = await api("/api/jobs/start", {
      method: "POST",
      body: JSON.stringify({
        preset_id: state.selected.id,
        resume,
      }),
    });
    state.job = payload.job;
    elements.presetDialog.close();
    updateJob(payload.job);
    elements.runDeck.classList.remove("hidden");
    setTimeout(() => elements.runDeck.scrollIntoView({ behavior: "smooth", block: "center" }), 80);
  } catch (error) {
    showToast(error.message);
    elements.startButton.disabled = false;
    elements.resumeButton.disabled = false;
  }
}

function renderPreflight(preflight) {
  if (!preflight) return;
  state.preflight = preflight;
  elements.systemBadge.className = `system-badge ${preflight.ready ? "ready" : "blocked"}`;
  elements.systemBadge.innerHTML = `<span></span>${preflight.ready ? "System ready" : "Setup needed"}`;
  elements.preflightGrid.innerHTML = Object.values(preflight.checks)
    .map((check) => `
      <div class="preflight-card ${check.ok ? "ok" : "bad"}">
        <span class="preflight-icon">${check.ok ? "✓" : "!"}</span>
        <div>
          <strong>${escapeHTML(check.label)}</strong>
          <small title="${escapeHTML(check.detail)}">${escapeHTML(check.detail)}</small>
        </div>
      </div>
    `)
    .join("");
}

function renderEvents(events) {
  if (!events) return;
  const recent = events.slice(-80);
  elements.eventLog.innerHTML = recent
    .map((event) => {
      const time = new Date(event.time);
      const stamp = Number.isNaN(time.getTime())
        ? "--:--:--"
        : time.toLocaleTimeString([], { hour12: false });
      return `
        <div class="event-line ${escapeHTML(event.kind)}">
          <span>${stamp}</span>
          <span class="event-kind">${escapeHTML(event.kind)}</span>
          <span class="event-message">${escapeHTML(event.message)}</span>
        </div>
      `;
    })
    .join("");
  elements.eventCount.textContent = `${events.length} EVENT${events.length === 1 ? "" : "S"}`;
  elements.eventLog.scrollTop = elements.eventLog.scrollHeight;
}

function checkpointText(job) {
  if (!job || !job.preset_id) return "Watching";
  if (job.preset_id !== "puncture-production") {
    return job.status === "complete" ? "Results safe" : "Automatic";
  }
  const step = Number(job.current_step || 0);
  if (job.status === "complete") return "Final saved";
  if (job.status === "cancelled") return "Last saved";
  const next = Math.max(50, Math.ceil((step + 1) / 50) * 50);
  return `Step ${next}`;
}

function updateJob(job) {
  if (!job) return;
  state.job = job;
  elements.footerMode.textContent = job.demo ? "DEMO TELEMETRY" : "LIVE RUNNERS";
  if (!job.id) return;

  elements.runDeck.classList.remove("hidden");
  const preset = state.presets.get(job.preset_id);
  elements.runEyebrow.textContent = job.eyebrow || preset?.eyebrow || "EXPERIMENT";
  elements.runEyebrow.style.color = job.accent || preset?.accent || "#67e8f9";
  elements.runTitle.textContent = job.title || preset?.title || "Tesseract experiment";
  elements.runStateBadge.textContent = String(job.status).toUpperCase();
  elements.runStateBadge.className = `run-state ${job.status}`;
  elements.runDeck.classList.toggle("running", ["starting", "running", "stopping"].includes(job.status));

  const progress = Math.max(0, Math.min(1, Number(job.progress || 0)));
  elements.progressBar.style.width = `${(progress * 100).toFixed(1)}%`;
  elements.progressLabel.textContent = `${Math.round(progress * 100)}%`;
  elements.stepLabel.textContent = job.total_steps
    ? job.progress_source === "step"
      ? `Step ${job.current_step ?? 0} of ${job.total_steps}`
      : `Awaiting step telemetry · ${job.total_steps} planned`
    : ["estimated", "step_or_estimated"].includes(job.progress_source)
      ? "Estimated progress"
      : "Processing";
  elements.etaLabel.textContent = ["complete", "failed", "cancelled"].includes(job.status)
    ? job.status === "complete" ? "Complete" : "Stopped"
    : `ETA ${formatDuration(job.eta_seconds)}`;
  elements.statusMessage.textContent = job.message || "Simulation active.";
  elements.elapsedValue.textContent = formatDuration(job.elapsed_seconds, true);
  elements.simTimeValue.textContent = job.simulation_time === null || job.simulation_time === undefined
    ? "—"
    : Number(job.simulation_time).toFixed(5);
  elements.checkpointValue.textContent = checkpointText(job);

  const active = ["starting", "running", "stopping"].includes(job.status);
  elements.cancelButton.disabled = !active || job.status === "stopping";
  elements.openOutputButton.disabled = !job.output_directory;
  renderEvents(job.events || []);
  syncTerminal(job);

  const visualMode = job.visualizer || preset?.visualizer || "tesseract";
  visualizer.setMode(visualMode, progress);
  elements.viewportLabel.textContent = `${String(visualMode).toUpperCase()} VIEWPORT`;
  elements.viewportClock.textContent = active
    ? `T+ ${formatDuration(job.elapsed_seconds, true)}`
    : String(job.status).toUpperCase();
  elements.viewportCaption.innerHTML = `
    <span>${active ? "LIVE EXECUTION" : "EXPERIMENT STATE"}</span>
    <strong>${escapeHTML(job.message || preset?.title || "Tesseract")}</strong>
  `;

  const activeLabel = active
    ? `${String(job.status).toUpperCase()} · ${Math.round(progress * 100)}%`
    : String(job.status).toUpperCase();
  elements.labStatusTitle.textContent = activeLabel;
  elements.labStatusDetail.textContent = job.command_preview || job.message || "Validated commands only";
  elements.terminalConnection.textContent = active ? "LOCAL / STREAMING" : "LOCAL / CONNECTED";
  elements.labLaunchButton.disabled = active;

  if (job.status === "complete" && state.completedJob !== job.id) {
    state.completedJob = job.id;
    refreshPresets();
    refreshResults(job.id);
  }
}

async function refreshPresets() {
  try {
    const payload = await api("/api/presets");
    state.presets = new Map(payload.presets.map((preset) => [preset.id, preset]));
    renderPresetCards();
  } catch (error) {
    elements.presetGrid.innerHTML = `<p>Could not load presets: ${escapeHTML(error.message)}</p>`;
  }
}

async function pollStatus() {
  if (state.polling) return;
  state.polling = true;
  try {
    const payload = await api("/api/status");
    renderPreflight(payload.preflight);
    updateJob(payload.job);
  } catch (error) {
    elements.systemBadge.className = "system-badge blocked";
    elements.systemBadge.innerHTML = "<span></span>Studio offline";
  } finally {
    state.polling = false;
  }
}

async function cancelRun() {
  if (!state.job) return;
  if (!confirm("Stop this managed experiment and its child processes?")) return;
  try {
    const payload = await api("/api/jobs/cancel", {
      method: "POST",
      body: "{}",
    });
    updateJob(payload.job);
  } catch (error) {
    showToast(error.message);
  }
}

async function openOutput() {
  try {
    const payload = await api("/api/jobs/open-output", {
      method: "POST",
      body: "{}",
    });
    showToast(`Opened ${payload.path}`);
  } catch (error) {
    showToast(error.message);
  }
}

function setAdvancedMode(enabled) {
  state.advanced = enabled;
  document.body.classList.toggle("advanced-mode", enabled);
  elements.advancedToggle.classList.toggle("active", enabled);
  localStorage.setItem("tesseract-advanced", String(enabled));
  localStorage.removeItem("tesseract-science");
  if (enabled) {
    refreshProtocols();
    refreshResults();
  }
}

function activateLabTab(identifier) {
  for (const tab of document.querySelectorAll("[data-lab-tab]")) {
    tab.classList.toggle("active", tab.dataset.labTab === identifier);
  }
  for (const panel of document.querySelectorAll("[data-lab-panel]")) {
    panel.classList.toggle("active", panel.dataset.labPanel === identifier);
  }
  if (identifier === "results") refreshResults();
}

function currentProtocol() {
  return state.protocols.get(elements.protocolSelect.value);
}

function inputForParameter(parameter) {
  const identifier = `parameter-${parameter.key}`;
  if (parameter.type === "boolean") {
    return `
      <label class="parameter-field boolean-field lab-field" for="${identifier}">
        <span>
          ${escapeHTML(parameter.label)}
          <small>${escapeHTML(parameter.help)}</small>
        </span>
        <input id="${identifier}" data-parameter="${escapeHTML(parameter.key)}"
          type="checkbox" ${parameter.default ? "checked" : ""}>
      </label>
    `;
  }
  if (parameter.type === "choice") {
    const options = parameter.choices.map((choice) => `
      <option value="${escapeHTML(choice)}"
        ${choice === parameter.default ? "selected" : ""}>${escapeHTML(choice)}</option>
    `).join("");
    return `
      <label class="parameter-field lab-field" for="${identifier}">
        <span>${escapeHTML(parameter.label)}</span>
        <select id="${identifier}" data-parameter="${escapeHTML(parameter.key)}">${options}</select>
        <small>${escapeHTML(parameter.help)}</small>
      </label>
    `;
  }
  const numeric = ["integer", "float"].includes(parameter.type);
  const attributes = numeric
    ? `type="number" min="${parameter.minimum ?? ""}" max="${parameter.maximum ?? ""}" step="${parameter.step ?? "any"}"`
    : 'type="text"';
  return `
    <label class="parameter-field lab-field" for="${identifier}">
      <span>${escapeHTML(parameter.label)}</span>
      <input id="${identifier}" data-parameter="${escapeHTML(parameter.key)}"
        ${attributes} value="${escapeHTML(parameter.default)}">
      <small>${escapeHTML(parameter.help)}</small>
    </label>
  `;
}

function renderProtocol(identifier) {
  const protocol = state.protocols.get(identifier);
  if (!protocol) return;
  elements.protocolSelect.value = protocol.id;
  elements.protocolSummary.textContent = protocol.summary;
  elements.qualificationNote.textContent = protocol.qualification_note;
  elements.parameterGrid.innerHTML = protocol.parameters
    .map(inputForParameter)
    .join("");
  for (const input of elements.parameterGrid.querySelectorAll("[data-parameter]")) {
    input.addEventListener("input", updateLabPreview);
    input.addEventListener("change", updateLabPreview);
  }
  elements.labRunner.textContent = protocol.runner;
  elements.labVram.textContent = `${protocol.minimum_vram_gib || 0} GiB`;
  elements.intelligenceCore.textContent = protocol.visualizer === "puncture"
    ? "MOTS"
    : protocol.visualizer === "phase"
      ? "Λφ"
      : "GPU";
  updateLabPreview();
}

function collectLabParameters() {
  const protocol = currentProtocol();
  if (!protocol) return {};
  const values = {};
  for (const parameter of protocol.parameters) {
    const input = document.querySelector(`[data-parameter="${CSS.escape(parameter.key)}"]`);
    if (!input) continue;
    values[parameter.key] = parameter.type === "boolean" ? input.checked : input.value;
  }
  return values;
}

function labCommand(protocol, values) {
  const parameters = protocol.parameters.map((parameter) => {
    const value = values[parameter.key];
    return `${parameter.key}=${String(value).includes(" ") ? `"${value}"` : value}`;
  });
  return `run ${protocol.id} ${parameters.join(" ")}`;
}

function updateLabPreview() {
  const protocol = currentProtocol();
  if (!protocol) return;
  const values = collectLabParameters();
  elements.commandPreview.textContent = labCommand(protocol, values);
  elements.labEstimate.textContent = `Baseline ${formatDuration(protocol.estimated_seconds)}`;
}

async function refreshProtocols() {
  try {
    const payload = await api("/api/protocols");
    const prior = elements.protocolSelect.value;
    state.protocols = new Map(
      payload.protocols.map((protocol) => [protocol.id, protocol]),
    );
    elements.protocolSelect.innerHTML = payload.protocols
      .map((protocol) => `
        <option value="${escapeHTML(protocol.id)}">${escapeHTML(protocol.title)}</option>
      `)
      .join("");
    const selected = state.protocols.has(prior)
      ? prior
      : payload.protocols[0]?.id;
    if (selected) renderProtocol(selected);
  } catch (error) {
    appendTerminal("error", error.message);
  }
}

async function launchAdvanced() {
  const protocol = currentProtocol();
  if (!protocol) return;
  const parameters = collectLabParameters();
  if (
    protocol.id === "puncture"
    && String(parameters.plan_only).toLowerCase() === "false"
    && !confirm(
      "Launch a full puncture evolution? This can occupy the GPU for many hours. "
      + "The plan-only switch is currently disabled.",
    )
  ) {
    return;
  }
  elements.labLaunchButton.disabled = true;
  try {
    const payload = await api("/api/jobs/start-advanced", {
      method: "POST",
      body: JSON.stringify({ protocol_id: protocol.id, parameters }),
    });
    updateJob(payload.job);
    appendTerminal("system", `Dispatched ${payload.job.command_preview}`);
    activateLabTab("terminal");
    elements.laboratory.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    appendTerminal("error", error.message);
    showToast(error.message);
  } finally {
    elements.labLaunchButton.disabled = false;
  }
}

function resetProtocol() {
  const protocol = currentProtocol();
  if (protocol) renderProtocol(protocol.id);
}

function appendTerminal(kind, text, label = null) {
  const row = document.createElement("div");
  row.className = `terminal-row ${kind}`;
  const channel = document.createElement("span");
  channel.textContent = label || kind;
  const code = document.createElement("code");
  code.textContent = text;
  row.append(channel, code);
  elements.labTerminalLog.append(row);
  while (elements.labTerminalLog.children.length > 260) {
    elements.labTerminalLog.firstElementChild?.remove();
  }
  elements.labTerminalLog.scrollTop = elements.labTerminalLog.scrollHeight;
}

function syncTerminal(job) {
  if (!job?.id) return;
  if (state.terminalJob !== job.id) {
    state.terminalJob = job.id;
    state.terminalSequence = 0;
    appendTerminal("system", `${job.title} attached`, "session");
    if (job.command_preview) {
      appendTerminal("command", job.command_preview, "runner");
    }
  }
  for (const event of job.events || []) {
    if (event.sequence <= state.terminalSequence) continue;
    appendTerminal(event.kind || "output", event.message, event.kind);
    state.terminalSequence = Math.max(state.terminalSequence, event.sequence);
  }
}

async function executeTerminal(command) {
  const line = command.trim();
  if (!line) return;
  state.terminalHistory.push(line);
  state.terminalHistoryIndex = state.terminalHistory.length;
  appendTerminal("command", line, "λ");
  try {
    const payload = await api("/api/console", {
      method: "POST",
      body: JSON.stringify({ command: line }),
    });
    if (payload.action === "clear") {
      elements.labTerminalLog.innerHTML = "";
      return;
    }
    for (const entry of payload.lines || []) {
      appendTerminal(entry.kind || "output", entry.text);
    }
    if (payload.job) updateJob(payload.job);
    if (payload.action === "results") {
      await refreshResults();
    }
  } catch (error) {
    appendTerminal("error", error.message);
  }
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  const absolute = Math.abs(number);
  if ((absolute > 0 && absolute < 0.001) || absolute >= 1e6) {
    return number.toExponential(3);
  }
  return number.toLocaleString(undefined, { maximumSignificantDigits: 6 });
}

async function refreshResults(selectJob = null) {
  try {
    const payload = await api("/api/results");
    state.results = payload.results;
    const current = elements.resultSelect.value;
    const groups = {
      studio_run: "LABORATORY RUNS",
      qualified_reference: "TRACKED QUALIFICATION EVIDENCE",
    };
    elements.resultSelect.innerHTML = '<option value="">Choose a completed result</option>';
    for (const [source, label] of Object.entries(groups)) {
      const rows = payload.results.filter((row) => row.source === source);
      if (!rows.length) continue;
      const group = document.createElement("optgroup");
      group.label = label;
      for (const row of rows) {
        const option = document.createElement("option");
        option.value = row.id;
        const marker = row.qualified === true ? "✓ " : row.qualified === false ? "✕ " : "";
        option.textContent = `${marker}${row.name}`;
        option.title = row.relative_path;
        group.append(option);
      }
      elements.resultSelect.append(group);
    }
    let target = current;
    if (selectJob) {
      const match = payload.results.find(
        (row) => row.source === "studio_run"
          && row.relative_path.startsWith(`${selectJob}/`),
      );
      if (match) target = match.id;
    }
    if (!target && query.get("result") === "latest") {
      target = payload.results.find(
        (row) => row.source === "studio_run",
      )?.id;
    }
    if (!target && query.get("result") && query.get("result") !== "latest") {
      const requested = query.get("result");
      target = payload.results.find(
        (row) => row.id === requested || row.file === requested,
      )?.id;
    }
    if (target && payload.results.some((row) => row.id === target)) {
      elements.resultSelect.value = target;
      await loadResult(target);
    }
  } catch (error) {
    showToast(`Evidence index: ${error.message}`);
  }
}

async function loadResult(identifier) {
  if (!identifier) {
    state.analysis = null;
    elements.evidenceEmpty.classList.remove("hidden");
    elements.evidenceDashboard.classList.add("hidden");
    elements.copyJsonButton.disabled = true;
    return;
  }
  try {
    const payload = await api(`/api/results/${encodeURIComponent(identifier)}`);
    state.analysis = payload.analysis;
    renderEvidence(payload.analysis);
  } catch (error) {
    showToast(error.message);
  }
}

function renderEvidence(analysis) {
  elements.evidenceEmpty.classList.add("hidden");
  elements.evidenceDashboard.classList.remove("hidden");
  elements.copyJsonButton.disabled = false;
  elements.evidenceSchema.textContent = String(
    analysis.provenance.schema || analysis.provenance.format || "JSON EVIDENCE",
  ).toUpperCase();
  elements.evidenceTitle.textContent = analysis.title;
  elements.evidencePath.textContent = analysis.provenance.file || "";
  const qualified = analysis.qualified;
  elements.qualificationSeal.className = `qualification-seal ${
    qualified === true ? "passed" : qualified === false ? "failed" : ""
  }`;
  elements.qualificationSeal.innerHTML = `
    <strong>${qualified === true ? "PASS" : qualified === false ? "FAIL" : "N/A"}</strong>
    <span>QUALIFICATION</span>
  `;

  elements.evidenceKpis.innerHTML = analysis.metrics.slice(0, 8)
    .map((metric) => `
      <div class="evidence-kpi" title="${escapeHTML(metric.path)}">
        <span>${escapeHTML(metric.label)}</span>
        <strong>${escapeHTML(formatMetric(metric.value))}</strong>
      </div>
    `)
    .join("");

  const qualification = analysis.qualification;
  elements.gateScore.textContent = qualification.total
    ? `${qualification.passed}/${qualification.total}`
    : "NO GATES";
  elements.qualificationGrid.innerHTML = qualification.rows
    .map((row) => `
      <div class="gate-row ${row.passed ? "passed" : "failed"}">
        <i>${row.passed ? "✓" : "×"}</i>
        <div title="${escapeHTML(row.path)}">
          <strong>${escapeHTML(row.label)}</strong>
          <small>${escapeHTML(row.path)}</small>
        </div>
      </div>
    `)
    .join("");

  elements.provenanceList.innerHTML = Object.entries(analysis.provenance)
    .map(([key, value]) => `
      <div title="${escapeHTML(String(value))}">
        <dt>${escapeHTML(key.replaceAll("_", " "))}</dt>
        <dd>${escapeHTML(String(value))}</dd>
      </div>
    `)
    .join("");

  elements.rawJson.textContent = JSON.stringify(analysis.raw, null, 2);
  renderSeriesControls(analysis.series);
}

function renderSeriesControls(series) {
  elements.seriesSelect.innerHTML = series.length
    ? series.map((item, index) => `
        <option value="${index}">${escapeHTML(item.label)} · ${escapeHTML(item.axis)}</option>
      `).join("")
    : '<option value="">No numeric ladder</option>';
  renderSelectedSeries();
}

function renderSelectedSeries() {
  const series = state.analysis?.series || [];
  const selected = series[Number(elements.seriesSelect.value || 0)];
  if (!selected) {
    elements.metricSelect.innerHTML = '<option value="">No metric</option>';
    elements.chartEmpty.classList.remove("hidden");
    elements.evidenceChart.classList.add("hidden");
    elements.seriesTable.innerHTML = "";
    return;
  }
  const previous = elements.metricSelect.value;
  elements.metricSelect.innerHTML = selected.fields
    .map((field) => `<option value="${escapeHTML(field)}">${escapeHTML(field.replaceAll("_", " "))}</option>`)
    .join("");
  if (selected.fields.includes(previous)) elements.metricSelect.value = previous;
  elements.chartEmpty.classList.add("hidden");
  elements.evidenceChart.classList.remove("hidden");
  renderSeriesTable(selected);
  evidenceChart.draw(selected, elements.metricSelect.value || selected.fields[0]);
}

function renderSeriesTable(series) {
  const fields = [series.axis, ...series.fields];
  elements.seriesTable.innerHTML = `
    <thead><tr>${fields.map((field) => `<th>${escapeHTML(field)}</th>`).join("")}</tr></thead>
    <tbody>
      ${series.rows.map((row) => `
        <tr>${fields.map((field) => `<td>${escapeHTML(formatMetric(row[field]))}</td>`).join("")}</tr>
      `).join("")}
    </tbody>
  `;
}

class EvidenceChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.series = null;
    this.metric = null;
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(canvas);
  }

  draw(series, metric) {
    this.series = series;
    this.metric = metric;
    this.render();
  }

  render() {
    if (!this.series || !this.metric || this.canvas.classList.contains("hidden")) return;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    const ctx = this.context;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#030914";
    ctx.fillRect(0, 0, width, height);
    const pad = { left: 70, right: 28, top: 30, bottom: 48 };
    const rows = this.series.rows;
    const xs = rows.map((row) => Number(row[this.series.axis]));
    const ys = rows.map((row) => Number(row[this.metric]));
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xSpan = xMax - xMin || 1;
    const ySpan = yMax - yMin || Math.max(Math.abs(yMax), 1);
    const projectX = (value) => pad.left
      + ((value - xMin) / xSpan) * (width - pad.left - pad.right);
    const projectY = (value) => height - pad.bottom
      - ((value - yMin) / ySpan) * (height - pad.top - pad.bottom);

    ctx.font = "9px ui-monospace, monospace";
    ctx.lineWidth = 1;
    for (let index = 0; index <= 5; index += 1) {
      const fraction = index / 5;
      const y = pad.top + fraction * (height - pad.top - pad.bottom);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.strokeStyle = "rgba(103,232,249,.09)";
      ctx.stroke();
      const value = yMax - fraction * ySpan;
      ctx.fillStyle = "rgba(151,180,191,.7)";
      ctx.textAlign = "right";
      ctx.fillText(formatMetric(value), pad.left - 9, y + 3);
    }
    ctx.beginPath();
    rows.forEach((row, index) => {
      const x = projectX(Number(row[this.series.axis]));
      const y = projectY(Number(row[this.metric]));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    const gradient = ctx.createLinearGradient(pad.left, 0, width - pad.right, 0);
    gradient.addColorStop(0, "#a78bfa");
    gradient.addColorStop(1, "#67e8f9");
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.stroke();
    rows.forEach((row) => {
      const x = projectX(Number(row[this.series.axis]));
      const y = projectY(Number(row[this.metric]));
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = "#d9fbff";
      ctx.shadowColor = "#67e8f9";
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(151,180,191,.75)";
      ctx.textAlign = "center";
      ctx.fillText(formatMetric(row[this.series.axis]), x, height - 22);
    });
    ctx.fillStyle = "#67e8f9";
    ctx.textAlign = "left";
    ctx.fillText(this.metric.replaceAll("_", " ").toUpperCase(), pad.left, 16);
    ctx.textAlign = "right";
    ctx.fillText(this.series.axis.toUpperCase(), width - pad.right, height - 7);
  }
}

const evidenceChart = new EvidenceChart(elements.evidenceChart);

class SpacetimeVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.mode = "tesseract";
    this.progress = 0;
    this.targetProgress = 0;
    this.time = 0;
    this.stars = Array.from({ length: 92 }, (_, index) => ({
      x: ((index * 73) % 101) / 101,
      y: ((index * 47 + 19) % 97) / 97,
      size: 0.45 + ((index * 13) % 11) / 12,
      phase: index * 0.71,
    }));
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    requestAnimationFrame((time) => this.frame(time));
  }

  setMode(mode, progress) {
    this.mode = mode || "tesseract";
    this.targetProgress = Number(progress || 0);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(1, Math.round(rect.width * ratio));
    this.canvas.height = Math.max(1, Math.round(rect.height * ratio));
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
  }

  frame(timestamp) {
    const next = timestamp / 1000;
    const delta = Math.min(0.05, next - this.time || 0);
    this.time = next;
    this.progress += (this.targetProgress - this.progress) * Math.min(1, delta * 3);
    this.draw();
    requestAnimationFrame((time) => this.frame(time));
  }

  clear() {
    const ctx = this.context;
    ctx.clearRect(0, 0, this.width, this.height);
    const background = ctx.createRadialGradient(
      this.width * 0.5,
      this.height * 0.5,
      10,
      this.width * 0.5,
      this.height * 0.5,
      Math.max(this.width, this.height) * 0.7,
    );
    background.addColorStop(0, "rgba(15, 39, 67, 0.32)");
    background.addColorStop(0.52, "rgba(6, 16, 33, 0.18)");
    background.addColorStop(1, "rgba(2, 6, 15, 0.96)");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, this.width, this.height);
    for (const star of this.stars) {
      const alpha = 0.12 + 0.18 * (0.5 + 0.5 * Math.sin(this.time * 0.7 + star.phase));
      ctx.fillStyle = `rgba(179,231,243,${alpha})`;
      ctx.fillRect(star.x * this.width, star.y * this.height, star.size, star.size);
    }
  }

  drawGrid(strength = 1, bend = 0) {
    const ctx = this.context;
    const horizon = this.height * 0.48;
    const bottom = this.height * 0.94;
    ctx.save();
    ctx.lineWidth = 0.65;
    for (let index = -12; index <= 12; index += 1) {
      const x0 = this.width * 0.5 + index * this.width * 0.045;
      ctx.beginPath();
      for (let sample = 0; sample <= 30; sample += 1) {
        const fraction = sample / 30;
        const y = horizon + (bottom - horizon) * fraction;
        const spread = 0.25 + 0.95 * fraction;
        const bendWave = bend * Math.sin(fraction * Math.PI) * Math.exp(-Math.abs(index) * 0.13);
        const x = this.width * 0.5 + (x0 - this.width * 0.5) * spread + bendWave;
        if (sample === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `rgba(63,180,205,${0.055 * strength})`;
      ctx.stroke();
    }
    for (let row = 0; row <= 14; row += 1) {
      const fraction = row / 14;
      const y = horizon + (bottom - horizon) * Math.pow(fraction, 1.72);
      ctx.beginPath();
      ctx.moveTo(this.width * 0.06, y);
      ctx.lineTo(this.width * 0.94, y);
      ctx.strokeStyle = `rgba(63,180,205,${(0.035 + fraction * 0.07) * strength})`;
      ctx.stroke();
    }
    ctx.restore();
  }

  rotate4(point, a, b, angle) {
    const value = [...point];
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    value[a] = cosine * point[a] - sine * point[b];
    value[b] = sine * point[a] + cosine * point[b];
    return value;
  }

  drawTesseract() {
    const ctx = this.context;
    const points = [];
    for (let index = 0; index < 16; index += 1) {
      let point = [
        index & 1 ? 1 : -1,
        index & 2 ? 1 : -1,
        index & 4 ? 1 : -1,
        index & 8 ? 1 : -1,
      ];
      point = this.rotate4(point, 0, 3, this.time * 0.31);
      point = this.rotate4(point, 1, 2, this.time * 0.23 + 0.6);
      point = this.rotate4(point, 2, 3, this.time * 0.17);
      const fourScale = 1.7 / (3.2 - point[3]);
      const x3 = point[0] * fourScale;
      const y3 = point[1] * fourScale;
      const z3 = point[2] * fourScale;
      const depth = 2.8 - z3;
      points.push({
        x: this.width * 0.5 + (x3 / depth) * Math.min(this.width, this.height) * 1.45,
        y: this.height * 0.49 + (y3 / depth) * Math.min(this.width, this.height) * 1.45,
        z: z3,
        w: point[3],
      });
    }
    const edges = [];
    for (let left = 0; left < 16; left += 1) {
      for (const bit of [1, 2, 4, 8]) {
        const right = left ^ bit;
        if (left < right) edges.push([left, right]);
      }
    }
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const [left, right] of edges) {
      const a = points[left];
      const b = points[right];
      const alpha = 0.16 + 0.28 * ((a.z + b.z + 2) / 4);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = left & 8
        ? `rgba(167,139,250,${alpha})`
        : `rgba(103,232,249,${alpha})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    for (const point of points) {
      const radius = 1.2 + (point.z + 1) * 0.5;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = point.w > 0 ? "rgba(103,232,249,.86)" : "rgba(244,114,208,.72)";
      ctx.shadowColor = point.w > 0 ? "#67e8f9" : "#f472d0";
      ctx.shadowBlur = 12;
      ctx.fill();
    }
    ctx.restore();
  }

  drawPuncture() {
    const ctx = this.context;
    const progress = this.progress;
    this.drawGrid(1.2, Math.sin(this.time * 0.3) * 9);
    const centerY = this.height * 0.48;
    const separation = this.width * (0.15 - progress * 0.045);
    const centers = [this.width * 0.5 - separation, this.width * 0.5 + separation];
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const [index, centerX] of centers.entries()) {
      const glow = ctx.createRadialGradient(centerX, centerY, 1, centerX, centerY, 76);
      glow.addColorStop(0, "rgba(0,0,0,1)");
      glow.addColorStop(0.18, "rgba(0,0,0,.98)");
      glow.addColorStop(0.25, index ? "rgba(244,114,208,.42)" : "rgba(103,232,249,.42)");
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = glow;
      ctx.fillRect(centerX - 80, centerY - 80, 160, 160);
      ctx.beginPath();
      ctx.arc(centerX, centerY, 18, 0, Math.PI * 2);
      ctx.fillStyle = "#000";
      ctx.shadowColor = index ? "#f472d0" : "#67e8f9";
      ctx.shadowBlur = 23;
      ctx.fill();
      for (let ring = 0; ring < 3; ring += 1) {
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, 27 + ring * 10, 12 + ring * 5, 0, 0, Math.PI * 2);
        ctx.strokeStyle = index
          ? `rgba(244,114,208,${0.3 - ring * 0.07})`
          : `rgba(103,232,249,${0.3 - ring * 0.07})`;
        ctx.stroke();
      }
    }
    if (progress > 0.68) {
      const birth = Math.min(1, (progress - 0.68) / 0.18);
      ctx.beginPath();
      ctx.ellipse(this.width * 0.5, centerY, 35 + birth * separation * 1.55, 18 + birth * 43, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(94,230,168,${0.7 * birth})`;
      ctx.lineWidth = 2;
      ctx.shadowColor = "#5ee6a8";
      ctx.shadowBlur = 17;
      ctx.stroke();
      if (progress > 0.8) {
        ctx.beginPath();
        ctx.ellipse(this.width * 0.5, centerY, 24 + birth * separation, 13 + birth * 25, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(251,113,133,${0.64 * birth})`;
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  drawPhase() {
    const ctx = this.context;
    this.drawGrid(0.7, 0);
    const centerY = this.height * 0.5;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let layer = 0; layer < 8; layer += 1) {
      ctx.beginPath();
      const offset = layer * 12;
      for (let x = 0; x <= this.width; x += 5) {
        const leftWave = Math.sin(x * 0.025 - this.time * 2.2 + layer * 0.4);
        const rightWave = Math.sin((this.width - x) * 0.022 - this.time * 1.8 + layer * 0.5);
        const mix = Math.min(1, this.progress * 1.7);
        const y = centerY + (leftWave * (1 - x / this.width) + rightWave * (x / this.width)) * (34 + offset * 0.4) * (0.45 + 0.55 * mix) + offset - 42;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = layer % 2
        ? `rgba(244,114,208,${0.14 + layer * 0.015})`
        : `rgba(103,232,249,${0.16 + layer * 0.015})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
    const boundaryX = this.width * (0.5 + Math.sin(this.time * 0.55) * 0.035);
    const boundary = ctx.createLinearGradient(boundaryX - 20, 0, boundaryX + 20, 0);
    boundary.addColorStop(0, "rgba(244,114,208,0)");
    boundary.addColorStop(0.5, `rgba(251,191,36,${0.18 + this.progress * 0.34})`);
    boundary.addColorStop(1, "rgba(103,232,249,0)");
    ctx.fillStyle = boundary;
    ctx.fillRect(boundaryX - 35, 65, 70, this.height - 120);
    ctx.restore();
  }

  drawSpacetime() {
    const ctx = this.context;
    this.drawGrid(1, Math.sin(this.time * 0.7) * 5);
    ctx.save();
    ctx.translate(this.width * 0.5, this.height * 0.46);
    ctx.rotate(this.time * 0.07);
    for (let ring = 0; ring < 13; ring += 1) {
      const radius = 25 + ring * 12;
      ctx.beginPath();
      ctx.ellipse(0, 0, radius * 1.45, radius * 0.65, ring * 0.08 + this.time * 0.03, 0, Math.PI * 2);
      ctx.strokeStyle = ring % 3 === 0
        ? `rgba(167,139,250,${0.28 - ring * 0.012})`
        : `rgba(103,232,249,${0.2 - ring * 0.009})`;
      ctx.stroke();
    }
    ctx.restore();
  }

  draw() {
    this.clear();
    if (this.mode === "puncture") this.drawPuncture();
    else if (this.mode === "phase") this.drawPhase();
    else if (this.mode === "spacetime") this.drawSpacetime();
    else {
      this.drawGrid(0.7, Math.sin(this.time * 0.4) * 4);
      this.drawTesseract();
    }
  }
}

const visualizer = new SpacetimeVisualizer(elements.canvas);

elements.startButton.addEventListener("click", () => startSelected(false));
elements.resumeButton.addEventListener("click", () => startSelected(true));
elements.cancelButton.addEventListener("click", cancelRun);
elements.openOutputButton.addEventListener("click", openOutput);
elements.collapseRunButton.addEventListener("click", () => {
  elements.runDeck.classList.add("hidden");
});
elements.advancedToggle.addEventListener("click", () => {
  setAdvancedMode(!state.advanced);
  if (state.advanced) {
    elements.laboratory.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
elements.exploreButton.addEventListener("click", () => {
  document.getElementById("experiments").scrollIntoView({ behavior: "smooth" });
});
elements.demoButton.addEventListener("click", () => elements.infoDialog.showModal());

for (const tab of document.querySelectorAll("[data-lab-tab]")) {
  tab.addEventListener("click", () => activateLabTab(tab.dataset.labTab));
}
elements.protocolSelect.addEventListener("change", () => {
  renderProtocol(elements.protocolSelect.value);
});
elements.labLaunchButton.addEventListener("click", launchAdvanced);
elements.labResetButton.addEventListener("click", resetProtocol);
elements.copyCommandButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.commandPreview.textContent || "");
  showToast("Command copied");
});
elements.terminalForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const command = elements.terminalInput.value;
  elements.terminalInput.value = "";
  executeTerminal(command);
});
elements.terminalInput.addEventListener("keydown", (event) => {
  if (!["ArrowUp", "ArrowDown"].includes(event.key)) return;
  event.preventDefault();
  const offset = event.key === "ArrowUp" ? -1 : 1;
  state.terminalHistoryIndex = Math.max(
    0,
    Math.min(
      state.terminalHistory.length,
      state.terminalHistoryIndex + offset,
    ),
  );
  elements.terminalInput.value =
    state.terminalHistory[state.terminalHistoryIndex] || "";
});
for (const shortcut of document.querySelectorAll("[data-command]")) {
  shortcut.addEventListener("click", () => executeTerminal(shortcut.dataset.command));
}
elements.resultSelect.addEventListener("change", () => {
  loadResult(elements.resultSelect.value);
});
elements.refreshResultsButton.addEventListener("click", () => refreshResults());
elements.copyJsonButton.addEventListener("click", async () => {
  if (!state.analysis) return;
  await navigator.clipboard.writeText(JSON.stringify(state.analysis.raw, null, 2));
  showToast("JSON evidence copied");
});
elements.seriesSelect.addEventListener("change", renderSelectedSeries);
elements.metricSelect.addEventListener("change", renderSelectedSeries);

setAdvancedMode(state.advanced);
document.body.classList.toggle(
  "lab-focus",
  state.advanced && query.get("focus") === "lab",
);
if (["control", "terminal", "results"].includes(query.get("lab"))) {
  activateLabTab(query.get("lab"));
}
if (state.advanced && window.location.hash === "#laboratory") {
  setTimeout(() => {
    elements.laboratory.scrollIntoView({ block: "start" });
  }, 120);
}
refreshPresets();
pollStatus();
setInterval(pollStatus, 1000);
