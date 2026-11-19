const apiRoot = "/api/research";
const historyStorageKey = "researchDesk.history";
const activeThreadKey = "researchDesk.activeThread";
const pollDelay = 1100;
const maxPollDelay = 10000;

const $ = (selector) => document.querySelector(selector);
const form = $("#researchForm");
const questionInput = $("#questionInput");
const startButton = $("#startButton");
const formError = $("#formError");
let activeRun = null;
let pollTimer = null;
let pollFailures = 0;
let pollErrorMessage = "";
let pendingNoteAction = false;
let activeView = "new";
let historyRecords = loadHistory();

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function setVisible(element, visible) {
  element.classList.toggle("hidden", !visible);
}

function setFormError(message) {
  formError.textContent = message || "";
  setVisible(formError, Boolean(message));
}

function showHistoryNotice(message) {
  const notice = $("#historyNotice");
  notice.textContent = message || "";
  setVisible(notice, Boolean(message));
}

function loadHistory() {
  try {
    const saved = localStorage.getItem(historyStorageKey);
    if (!saved) return [];
    const records = JSON.parse(saved);
    if (!Array.isArray(records) || records.some((item) => (
      !item || typeof item.thread_id !== "string" || typeof item.question !== "string"
    ))) {
      showHistoryNotice("Saved run history is invalid and could not be loaded.");
      return [];
    }
    return records.slice(0, 12);
  } catch {
    showHistoryNotice("Saved run history is unavailable in this browser.");
    return [];
  }
}

function persistHistory() {
  try {
    localStorage.setItem(historyStorageKey, JSON.stringify(historyRecords));
    showHistoryNotice("");
  } catch {
    showHistoryNotice("Could not save run history in this browser.");
  }
}

function renderHistory() {
  const list = $("#researchHistory");
  list.replaceChildren();
  $("#historyTotal").textContent = String(historyRecords.length);
  setVisible($("#historyEmpty"), historyRecords.length === 0);

  historyRecords.forEach((record) => {
    const button = makeElement("button", "history-run");
    button.type = "button";
    button.classList.toggle("active", record.thread_id === activeRun?.thread_id);
    button.setAttribute("aria-label", `Open research run: ${record.question}`);
    button.append(makeElement("strong", "", record.question));
    button.append(makeElement("small", "", formatRunStatus(record.status)));
    button.addEventListener("click", () => openSavedRun(record.thread_id));
    list.append(button);
  });
}

function updateHistory(run) {
  if (!run?.thread_id || !run.question) return;
  const status = run.run_status || run.workflow_status || "unknown";
  const index = historyRecords.findIndex((record) => record.thread_id === run.thread_id);
  const existing = index >= 0 ? historyRecords[index] : null;
  if (existing && existing.question === run.question && existing.status === status) return;
  const record = {
    thread_id: run.thread_id,
    question: run.question,
    status,
    updated_at: new Date().toISOString(),
  };
  historyRecords = historyRecords.filter((item) => item.thread_id !== run.thread_id);
  historyRecords.unshift(record);
  historyRecords = historyRecords.slice(0, 12);
  persistHistory();
}

function formatRunStatus(status) {
  const labels = {
    running: "In progress",
    waiting_for_approval: "Ready for review",
    completed: "Approved",
    report_rejected: "Rejected",
    revision_limit_reached: "Needs review",
    failed: "Failed",
  };
  return labels[status] || "Research run";
}

function saveActiveThread(threadId) {
  try {
    if (threadId) sessionStorage.setItem(activeThreadKey, threadId);
    else sessionStorage.removeItem(activeThreadKey);
  } catch {
    showHistoryNotice("This browser could not save the active run for reload.");
  }
}

function switchView(view, options = {}) {
  const titles = {
    new: "New Research",
    progress: "Research Progress",
    evidence: "Evidence Library",
    report: "Final Report & Approval",
  };
  activeView = titles[view] ? view : "new";
  document.querySelectorAll("[data-screen]").forEach((screen) => {
    const selected = screen.dataset.screen === activeView;
    setVisible(screen, selected);
    screen.classList.toggle("active", selected);
  });
  document.querySelectorAll(".nav-item[data-view]").forEach((button) => {
    const selected = button.dataset.view === activeView;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#pageTitle").textContent = titles[activeView];
  if (options.focusQuestion) questionInput.focus();
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail || payload.error_message || "The request could not be completed.";
    throw new Error(detail);
  }
  return payload;
}

function beginPolling() {
  beginPollingAfter(pollDelay);
}

function beginPollingAfter(delay) {
  window.clearTimeout(pollTimer);
  if (!activeRun || activeRun.run_status !== "running") return;
  pollTimer = window.setTimeout(refreshRun, delay);
}

async function refreshRun() {
  if (!activeRun?.thread_id) return;
  try {
    const previousStatus = activeRun.run_status;
    activeRun = await apiRequest(`${apiRoot}/${encodeURIComponent(activeRun.thread_id)}`);
    pollFailures = 0;
    pollErrorMessage = "";
    renderRun(activeRun);
    if (previousStatus === "running" && activeRun.pending_approval) switchView("report");
    beginPolling();
  } catch (error) {
    pollErrorMessage = error.message || "A connection error interrupted this update.";
    showRunError(error.message);
    if (activeRun.run_status === "running") {
      pollFailures = Math.min(pollFailures + 1, 5);
      beginPollingAfter(Math.min(pollDelay * (2 ** pollFailures), maxPollDelay));
    }
  }
}

async function openSavedRun(threadId) {
  window.clearTimeout(pollTimer);
  try {
    activeRun = await apiRequest(`${apiRoot}/${encodeURIComponent(threadId)}`);
    pollFailures = 0;
    pollErrorMessage = "";
    saveActiveThread(threadId);
    renderRun(activeRun);
    const reportReady = Boolean(
      activeRun.pending_approval || activeRun.report_draft || activeRun.final_report,
    );
    switchView(reportReady ? "report" : "progress");
    beginPolling();
  } catch (error) {
    showRunError(error.message);
  }
}

async function startResearch(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question.length < 10) {
    setFormError("Enter a research question with at least 10 characters.");
    questionInput.focus();
    return;
  }
  setFormError("");
  pollFailures = 0;
  pollErrorMessage = "";
  startButton.disabled = true;
  startButton.textContent = "Starting research…";
  try {
    activeRun = await apiRequest(apiRoot, { method: "POST", body: JSON.stringify({ question }) });
    saveActiveThread(activeRun.thread_id);
    updateHistory(activeRun);
    renderRun(activeRun);
    switchView("progress");
    beginPolling();
  } catch (error) {
    setFormError(error.message);
  } finally {
    startButton.disabled = false;
    startButton.replaceChildren(document.createTextNode("Start research "));
    startButton.append(makeElement("span", "", "→"));
    startButton.lastElementChild.setAttribute("aria-hidden", "true");
  }
}

function renderRun(run) {
  activeRun = run;
  $("#runQuestion").textContent = run.question || "Research run";
  $("#threadId").textContent = run.thread_id || "";
  setVisible($("#runContext"), Boolean(run.thread_id));
  renderStatus(run);
  renderPlan(run.plan);
  renderWorkers(run);
  renderEvidence(run);
  renderFactChecks(run.fact_check_results || []);
  renderReport(run);
  renderCritic(run.critic_evaluation);
  renderRevisionHistory(run.revision_history || []);
  renderReviewerFeedback(run.revision_feedback || []);
  renderApproval(run);
  setVisible($("#errorBanner"), Boolean(run.error_message));
  $("#errorBanner").textContent = run.error_message || pollErrorMessage;
  setVisible($("#errorBanner"), Boolean(run.error_message || pollErrorMessage));
  updateHistory(run);
  renderHistory();
}

function renderStatus(run) {
  const statusLabels = {
    waiting_for_approval: "Ready for your review",
    completed: "Research approved",
    report_rejected: "Report rejected",
    revision_limit_reached: "Review required",
    failed: "Research stopped",
    running: run.stage || "Research in progress",
  };
  const descriptions = {
    waiting_for_approval: "The report is ready. Review it and choose an action.",
    completed: "The final report has been approved.",
    report_rejected: "This report was rejected.",
    revision_limit_reached: "The report needs human review before it can be approved.",
    failed: "The workflow stopped before completion.",
  };
  const title = statusLabels[run.run_status] || run.stage || "Research update";
  const isRunning = run.run_status === "running";
  const mark = $("#stateMark");
  mark.classList.toggle("waiting", run.run_status === "waiting_for_approval" || run.run_status === "revision_limit_reached");
  mark.classList.toggle("complete", run.run_status === "completed");
  mark.classList.toggle("failed", run.run_status === "failed" || run.run_status === "report_rejected");
  $("#stateTitle").textContent = title;
  $("#stateDescription").textContent = descriptions[run.run_status] || run.workflow_status || "Research is ready to start.";
  setVisible($("#stateSpinner"), isRunning);

  const chip = $("#runChip");
  const chipText = run.run_status === "running" ? run.stage || "In progress" : statusLabels[run.run_status] || run.stage || "Research";
  $("#runChipText").textContent = chipText;
  setVisible(chip, Boolean(run.thread_id));
  chip.classList.toggle("waiting", run.run_status === "waiting_for_approval");
  chip.classList.toggle("complete", run.run_status === "completed");
  chip.classList.toggle("failed", run.run_status === "failed" || run.run_status === "report_rejected");
}

function renderWorkers(run) {
  const workers = run.workers || [];
  const total = Math.max(workers.length, (run.tasks || []).length);
  const completed = workers.filter((worker) => worker.status === "completed").length;
  const percent = total ? Math.round((completed / total) * 100) : 0;
  $("#workerCount").textContent = `${total} ${total === 1 ? "worker" : "workers"}`;
  $("#progressCount").textContent = total
    ? `${completed} of ${total} completed`
    : "Waiting for a run";
  $("#overallProgress").setAttribute("aria-valuenow", String(percent));
  $("#overallProgressFill").style.width = `${percent}%`;
  const list = $("#workerList");
  list.replaceChildren();
  setVisible(list, workers.length > 0);
  setVisible($("#workersEmpty"), workers.length === 0);
  workers.forEach((worker, index) => {
    const row = makeElement("article", "worker-row");
    const name = makeElement("div", "worker-name");
    name.append(makeElement("span", "worker-number", String(index + 1).padStart(2, "0")));
    const label = makeElement("div");
    label.append(makeElement("strong", "", worker.title || worker.task_id || `Research task ${index + 1}`));
    label.append(makeElement(
      "small",
      "",
      worker.warning || (worker.status === "completed" ? "Sources collected" : "Research task"),
    ));
    name.append(label);
    const status = worker.status || "queued";
    const statusLabel = status === "completed" ? "Complete" : status === "running" ? "In progress" : "Queued";
    row.append(name, makeElement("span", `worker-state ${status}`, statusLabel));
    row.append(makeElement("span", "worker-findings", `${worker.finding_count || 0} findings`));
    list.append(row);
  });
}

function renderPlan(plan) {
  const panel = $("#researchPlan");
  const tasks = plan?.tasks || [];
  $("#planSummary").textContent = plan?.summary || "";
  const list = $("#planTasks");
  list.replaceChildren();
  tasks.forEach((task) => {
    const item = makeElement("li", "plan-task");
    item.append(makeElement("strong", "", task.title || task.task_id || "Research subtopic"));
    if (task.objective) item.append(makeElement("span", "", task.objective));
    if (task.expected_evidence) {
      item.append(makeElement("small", "", `Expected evidence: ${task.expected_evidence}`));
    }
    list.append(item);
  });
  setVisible(panel, Boolean(plan));
}

function safeHttpUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : null;
  } catch {
    return null;
  }
}

function renderEvidence(run) {
  const evidence = run.evidence || [];
  const evidenceList = $("#evidenceList");
  const citationList = $("#citationList");
  evidenceList.replaceChildren();
  citationList.replaceChildren();
  const formatScore = (value) => Number.isFinite(value) ? value.toFixed(2) : null;
  setVisible($("#evidenceEmpty"), evidence.length === 0);
  setVisible($("#citationsEmpty"), evidence.length === 0);
  $("#evidenceCount").textContent = `${evidence.length} ${evidence.length === 1 ? "finding" : "findings"}`;

  evidence.forEach((item) => {
    const titleText = item.claim || item.source_title || "Research finding";
    const status = ["verified", "supported", "conflicting", "unsupported", "unknown", "unverified"]
      .includes(item.verification_status) ? item.verification_status : "unverified";
    const retrievalLabels = {
      page_excerpt: "Retrieved page excerpt",
      search_snippet: "Search-result snippet",
      vector_excerpt: "Local vector excerpt",
    };
    const retrievalLabel = retrievalLabels[item.retrieval_method] || "Evidence excerpt";
    const card = makeElement("article", "evidence-card card");
    const heading = makeElement("div", "evidence-card-heading");
    heading.append(makeElement("h3", "", titleText));
    heading.append(makeElement("span", `evidence-badge ${status}`, status.replaceAll("_", " ")));
    card.append(heading);
    const sourceInfo = [
      item.source_title,
      item.publication_date,
      item.page_number ? `p. ${item.page_number}` : null,
      item.source_domain,
      retrievalLabel,
      item.retrieval_channels?.length
        ? `retrieved by ${item.retrieval_channels.join(" + ")}`
        : null,
      item.evidence_scope ? `scope: ${item.evidence_scope.replaceAll("_", " ")}` : null,
      item.evidence_basis ? `basis: ${item.evidence_basis.replaceAll("_", " ")}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    card.append(makeElement("div", "evidence-meta", sourceInfo || "Source details in citations"));
    const rankingInfo = [
      ["relevance", formatScore(item.relevance_score)],
      ["authority heuristic", formatScore(item.authority_score)],
      ["freshness heuristic", formatScore(item.freshness_score)],
    ]
      .filter(([, score]) => score !== null)
      .map(([label, score]) => `${label}: ${score}`)
      .join(" · ");
    if (rankingInfo) card.append(makeElement("div", "evidence-meta", `Ranking signals: ${rankingInfo}`));
    const details = document.createElement("details");
    details.append(makeElement("summary", "", `Read ${retrievalLabel.toLowerCase()}`));
    details.append(makeElement("p", "evidence-support", item.support || "No excerpt was provided."));
    card.append(details);
    if (item.notes) card.append(makeElement("p", "evidence-note", item.notes));
    const evidenceUrl = safeHttpUrl(item.source_url);
    if (evidenceUrl) {
      const sourceLink = makeElement("a", "citation-link", "Open source");
      sourceLink.href = evidenceUrl;
      sourceLink.target = "_blank";
      sourceLink.rel = "noopener noreferrer";
      card.append(sourceLink);
    }
    evidenceList.append(card);

    const citation = makeElement("article", "citation-card card");
    const citationHeading = makeElement("div", "citation-card-heading");
    citationHeading.append(makeElement("h3", "", item.source_title || item.claim || "Source"));
    citationHeading.append(makeElement("span", `evidence-badge ${status}`, status.replaceAll("_", " ")));
    citation.append(citationHeading);
    citation.append(makeElement(
      "div",
      "citation-meta",
      [
        item.publication_date,
        item.page_number ? `p. ${item.page_number}` : null,
        item.source_domain,
        retrievalLabel,
      ].filter(Boolean).join(" · ") || "Source citation",
    ));
    const href = safeHttpUrl(item.source_url);
    if (href) {
      const link = makeElement("a", "citation-link", item.source_url);
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      citation.append(link);
    } else {
      citation.append(makeElement("p", "citation-unavailable", "Source URL unavailable"));
    }
    citationList.append(citation);
  });
}

function renderFactChecks(results) {
  const list = $("#factCheckList");
  list.replaceChildren();
  results.forEach((result) => list.append(makeElement("li", "", result)));
  setVisible($("#factCheckPanel"), results.length > 0);
}

function renderMarkdown(target, markdown) {
  target.replaceChildren();
  const lines = String(markdown || "").split(/\r?\n/);
  let activeList = null;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      activeList = null;
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      activeList = null;
      target.append(makeElement(`h${heading[1].length}`, "", heading[2]));
      continue;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const ordered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || ordered) {
      const tag = bullet ? "ul" : "ol";
      if (!activeList || activeList.tagName.toLowerCase() !== tag) {
        activeList = document.createElement(tag);
        target.append(activeList);
      }
      activeList.append(makeElement("li", "", (bullet || ordered)[1]));
      continue;
    }
    activeList = null;
    target.append(makeElement("p", "", line));
  }
  if (!target.childElementCount) {
    target.append(makeElement("p", "report-empty", "The report will appear here after research is complete."));
  }
}

function renderReport(run) {
  const report = run.final_report || run.report_draft || "";
  const visible = Boolean(report || run.critic_evaluation || run.revision_history?.length);
  setVisible($("#reportWorkspace"), visible);
  setVisible($("#reportEmpty"), !visible);
  if (!visible) return;
  renderMarkdown($("#reportPaper"), report);
  $("#reportLabel").textContent = run.final_report ? "APPROVED" : run.final_approval_status === "rejected" ? "REJECTED" : "DRAFT";
}

function renderCritic(critic) {
  const scoreList = $("#scoreList");
  scoreList.replaceChildren();
  const score = critic || {};
  $("#overallScore").textContent = Number.isFinite(score.overall_score) ? score.overall_score.toFixed(2) : "—";
  const criticStatus = $("#criticStatus");
  criticStatus.textContent = critic ? (critic.passed ? "Passed" : "Needs work") : "Pending";
  criticStatus.classList.toggle("passed", Boolean(critic?.passed));
  const metrics = [
    ["Citation accuracy (claim support)", score.claim_audits?.length ? score.citation_accuracy : null],
    ["Completeness", score.research_completeness],
    ["Objectivity", score.objectivity],
  ];
  for (const [label, value] of metrics) {
    const numeric = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
    const row = makeElement("div", "score-row");
    row.append(makeElement("label", "", label));
    row.append(makeElement("strong", "", numeric === null ? "—" : numeric.toFixed(2)));
    const track = makeElement("div", "score-track");
    const fill = makeElement("div", "score-fill");
    fill.style.width = `${numeric === null ? 0 : numeric * 100}%`;
    track.append(fill);
    row.append(track);
    scoreList.append(row);
  }
  if (critic) {
    const audits = score.claim_audits || [];
    const supported = audits.filter((audit) => audit.status === "supported").length;
    const contradicted = audits.filter((audit) => audit.status === "contradicted").length;
    const unverified = audits.filter((audit) =>
      audit.status === "insufficient_retrieval" || audit.status === "insufficient_evidence"
    ).length;
    scoreList.append(makeElement(
      "p", "claim-audit-reason",
      audits.length
        ? `Audited claims: ${supported} supported, ${unverified} unverified, ${contradicted} contradicted.`
        : "Citation accuracy cannot be assessed: no claims were audited."
    ));
  }
  const issueContainer = $("#criticIssues");
  issueContainer.replaceChildren();
  (score.issues || []).forEach((issue) => issueContainer.append(makeElement("p", "", issue)));
  setVisible(issueContainer, Boolean(score.issues?.length));
  const auditContainer = $("#claimAuditList");
  auditContainer.replaceChildren();
  (score.claim_audits || []).forEach((audit) => {
    const item = makeElement("article", `claim-audit ${audit.status || "unknown"}`);
    const heading = makeElement("div", "claim-audit-heading");
    heading.append(makeElement("strong", "", audit.status?.replaceAll("_", " ") || "Unknown"));
    heading.append(makeElement("small", "", `Retrieval: ${audit.retrieval_completeness || "unknown"}`));
    if (audit.critical) heading.append(makeElement("small", "claim-audit-critical", "Critical"));
    item.append(heading);
    item.append(makeElement("p", "claim-audit-claim", audit.claim || ""));
    if (audit.excerpt) item.append(makeElement("blockquote", "", audit.excerpt));
    if (audit.reasoning) item.append(makeElement("small", "claim-audit-reason", audit.reasoning));
    if (audit.source_references?.length) {
      item.append(
        makeElement("small", "claim-audit-reason", `Source: ${audit.source_references.join("; ")}`),
      );
    }
    auditContainer.append(item);
  });
  setVisible(auditContainer, Boolean(score.claim_audits?.length));
}

function renderRevisionHistory(history) {
  $("#historyCount").textContent = String(history.length);
  const list = $("#historyList");
  list.replaceChildren();
  if (!history.length) {
    list.append(makeElement("p", "history-empty", "Review updates will appear here."));
    return;
  }
  history.forEach((entry) => {
    const item = makeElement("div", "history-item");
    item.append(makeElement("strong", "", `Revision ${entry.revision ?? 0} · ${Number(entry.overall_score || 0).toFixed(2)} overall`));
    const feedback = (entry.issues || []).join(" ") || (entry.passed ? "Critic review passed." : "Review recorded.");
    item.append(makeElement("small", "", feedback));
    list.append(item);
  });
}

function renderReviewerFeedback(feedback) {
  const list = $("#feedbackList");
  list.replaceChildren();
  feedback.forEach((note) => list.append(makeElement("li", "", note)));
  setVisible($("#reviewerFeedback"), feedback.length > 0);
}

function renderApproval(run) {
  const pending = run.pending_approval?.type === "final_approval";
  const canDecide = pending && run.run_status !== "running";
  const allowedActions = run.pending_approval?.allowed_actions || ["approve", "reject", "request_revision"];
  const canRequestRevision = canDecide && allowedActions.includes("request_revision");
  const manualRevisionAvailable = canRequestRevision
    && Number(run.revision_count || 0) >= Number(run.pending_approval?.max_revision_limit ?? 3);
  $("#approvalHeading").textContent = pending ? "Review the report" : "Review status";
  $("#approvalMessage").textContent = pending ? (run.pending_approval?.message || "") : "";
  setVisible($("#approvalMessage"), pending && Boolean(run.pending_approval?.message));
  $("#requestRevisionButton").textContent = manualRevisionAvailable
    ? "Request One Final Revision"
    : "Request Changes";
  const state = $("#approvalState");
  state.textContent = pending
    ? "Decision required"
    : run.run_status === "completed"
      ? "Approved"
      : run.run_status === "report_rejected"
        ? "Rejected"
        : run.run_status === "revision_limit_reached"
          ? "Human review needed"
          : run.run_status === "failed"
            ? "Workflow failed"
            : "Review in progress";
  state.classList.toggle("complete", run.run_status === "completed");
  state.classList.toggle("rejected", run.run_status === "report_rejected");
  state.classList.toggle("failed", run.run_status === "failed");
  document.querySelectorAll("#approvalPanel button").forEach((button) => {
    button.disabled = !canDecide;
  });
  setVisible($("#requestRevisionButton"), canRequestRevision);
  if (!canRequestRevision) pendingNoteAction = false;
  setVisible($("#finalDecisionActions"), canDecide && !pendingNoteAction);
  setVisible($("#finalNoteActions"), canRequestRevision && pendingNoteAction);
  setVisible($("#decisionNote"), canRequestRevision && pendingNoteAction);
}

function showNoteForm() {
  pendingNoteAction = true;
  $("#decisionNotes").value = "";
  $("#decisionNotes").setCustomValidity("");
  renderApproval(activeRun);
  $("#decisionNotes").focus();
}

function clearNoteForm() {
  pendingNoteAction = false;
  renderApproval(activeRun);
}

async function submitDecision(action) {
  if (!activeRun?.thread_id) return;
  const notes = $("#decisionNotes").value.trim();
  if (action === "request_revision" && !notes) {
    $("#decisionNotes").focus();
    $("#decisionNotes").setCustomValidity("Add a short note describing the requested changes.");
    $("#decisionNotes").reportValidity();
    return;
  }
  $("#decisionNotes").setCustomValidity("");
  try {
    document.querySelectorAll("#approvalPanel button").forEach((button) => { button.disabled = true; });
    activeRun = await apiRequest(`${apiRoot}/${encodeURIComponent(activeRun.thread_id)}/decision`, {
      method: "POST",
      body: JSON.stringify({ action, notes }),
    });
    pendingNoteAction = false;
    renderRun(activeRun);
    if (activeRun.run_status === "running") switchView("progress");
    beginPolling();
  } catch (error) {
    showRunError(error.message);
    renderApproval(activeRun);
  }
}

function showRunError(message) {
  $("#errorBanner").textContent = message || "A connection error interrupted this update.";
  setVisible($("#errorBanner"), true);
}

form.addEventListener("submit", startResearch);
questionInput.addEventListener("input", () => {
  $("#questionCount").textContent = `${questionInput.value.length.toLocaleString()} / 1,200`;
  setFormError("");
});
document.querySelectorAll(".nav-item[data-view]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view, { focusQuestion: button.dataset.view === "new" }));
});
document.querySelectorAll("[data-evidence-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    const selectedTab = button.dataset.evidenceTab;
    document.querySelectorAll("[data-evidence-tab]").forEach((tab) => {
      const selected = tab === button;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-selected", String(selected));
    });
    setVisible($("#evidencePanel"), selectedTab === "evidence");
    setVisible($("#citationsPanel"), selectedTab === "citations");
    $("#evidenceTabHeading").textContent = selectedTab === "evidence" ? "Evidence" : "Citations";
    $("#evidenceTabDescription").textContent = selectedTab === "evidence"
      ? "Findings gathered by the research workers."
      : "Source titles, links, and available publication details.";
    const count = (activeRun?.evidence || []).length;
    $("#evidenceCount").textContent = `${count} ${count === 1 ? selectedTab === "evidence" ? "finding" : "citation" : selectedTab === "evidence" ? "findings" : "citations"}`;
  });
});
$("#approveReportButton").addEventListener("click", () => submitDecision("approve"));
$("#rejectReportButton").addEventListener("click", () => submitDecision("reject"));
$("#requestRevisionButton").addEventListener("click", showNoteForm);
$("#submitReportRevisionButton").addEventListener("click", () => submitDecision("request_revision"));
$("[data-cancel-note]").addEventListener("click", clearNoteForm);
$("#decisionNotes").addEventListener("input", () => $("#decisionNotes").setCustomValidity(""));

renderHistory();
const savedThread = sessionStorage.getItem(activeThreadKey) || sessionStorage.getItem("northstar.activeThread");
if (savedThread) {
  activeRun = { thread_id: savedThread };
  openSavedRun(savedThread);
}
