from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

from langgraph.types import Command
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import ReportWriterTokenLimitError
from app.graph.builder import build_graph
from app.infrastructure.checkpointer import create_sqlite_checkpointer
from app.tools.vector_search import close_vector_search

logger = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class ResearchService:
    def __init__(self, checkpoint_path: str | None = None) -> None:
        self.checkpointer = create_sqlite_checkpointer(checkpoint_path or settings.checkpoint_db_path)
        self.graph = build_graph(checkpointer=self.checkpointer)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research-run")
        self._runs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = RLock()
        self._graph_lock = Lock()

    def start_research(self, question: str) -> dict[str, Any]:
        thread_id = uuid4().hex
        job = self._new_job(thread_id, question)
        with self._lock:
            self._runs[thread_id] = job
            self._futures[thread_id] = self.executor.submit(
                self._advance,
                thread_id,
                {
                    "thread_id": thread_id,
                    "research_question": question,
                    "revision_count": 0,
                    "max_revision_limit": settings.max_revision_attempts,
                    "manual_revision_count": 0,
                },
            )
        return self.get_run(thread_id)

    def decide(self, thread_id: str, action: str, notes: str = "") -> dict[str, Any]:
        self.get_run(thread_id)
        if action not in {"approve", "reject", "request_revision"}:
            raise ValueError("Unsupported approval action.")

        with self._lock:
            job = self._runs.get(thread_id)
            if job is None:
                raise KeyError(thread_id)
            if job.get("run_status") == "running":
                raise ValueError("This research run is already in progress.")
            approval = job.get("pending_approval")
            if not approval:
                raise ValueError("This research run is not waiting for an approval decision.")
            allowed_actions = approval.get("allowed_actions") or self._allowed_actions(
                job.get("revision_count", 0),
                settings.max_revision_attempts,
                job.get("manual_revision_count", 0),
            )
            if action not in allowed_actions:
                raise ValueError("This decision is unavailable at the current revision limit.")
            job["run_status"] = "running"
            job["stage"] = "Applying your decision"
            job["pending_approval"] = None
            self._futures[thread_id] = self.executor.submit(
                self._advance,
                thread_id,
                Command(resume={"action": action, "notes": notes}),
            )
            return self._response(job)

    def get_run(self, thread_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._runs.get(thread_id)
            if job is not None:
                return self._response(job)

        config = self._config(thread_id)
        with self._graph_lock:
            snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise KeyError(thread_id)

        job = self._new_job(thread_id, snapshot.values.get("research_question", ""))
        self._sync_from_state(job, snapshot.values)
        approval = self._snapshot_approval(snapshot)
        job["pending_approval"] = approval
        job["run_status"] = "waiting_for_approval" if approval else snapshot.values.get("workflow_status", "unknown")
        job["stage"] = self._approval_stage(approval) if approval else job["run_status"]
        with self._lock:
            self._runs[thread_id] = job
            return self._response(job)

    def wait_for_run(self, thread_id: str, timeout: float = 30) -> None:
        with self._lock:
            future = self._futures.get(thread_id)
        if future:
            future.result(timeout=timeout)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.checkpointer.conn.close()
        close_vector_search()

    def _advance(self, thread_id: str, input_value: Any) -> None:
        config = self._config(thread_id)
        try:
            with self._graph_lock:
                for event in self.graph.stream(input_value, config=config, stream_mode="updates"):
                    self._record_event(thread_id, event)
                snapshot = self.graph.get_state(config)
            with self._lock:
                job = self._runs[thread_id]
                self._sync_from_state(job, snapshot.values)
                approval = self._snapshot_approval(snapshot)
                job["pending_approval"] = approval
                if approval:
                    job["run_status"] = "waiting_for_approval"
                    job["stage"] = self._approval_stage(approval)
                else:
                    job["run_status"] = snapshot.values.get("workflow_status", "completed")
                    job["stage"] = job["run_status"]
        except Exception as exc:
            logger.exception("Research workflow failed for thread_id=%s.", thread_id)
            with self._lock:
                job = self._runs[thread_id]
                job["run_status"] = "failed"
                job["stage"] = "Workflow failed"
                job["error_message"] = (
                    str(exc)
                    if isinstance(exc, ReportWriterTokenLimitError)
                    else f"The workflow failed ({type(exc).__name__}). Check the server logs."
                )

    def _record_event(self, thread_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._runs[thread_id]
            if "__interrupt__" in event:
                interrupts = event["__interrupt__"]
                if not isinstance(interrupts, (list, tuple)):
                    interrupts = [interrupts]
                for item in interrupts:
                    approval = getattr(item, "value", item)
                    if isinstance(approval, dict):
                        job["pending_approval"] = self._approval_with_actions(
                            _json_value(approval),
                            job.get("revision_count", 0),
                            settings.max_revision_attempts,
                        )
                        job["run_status"] = "waiting_for_approval"
                        job["stage"] = self._approval_stage(approval)
                return

            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue
                job["stage"] = {
                    "planner": "Building research plan",
                    "research_worker": "Researching sources",
                    "aggregate_findings": "Organizing evidence",
                    "fact_checker": "Checking evidence",
                    "write_report": "Writing report",
                    "critic": "Evaluating report quality",
                    "report_revision": "Revising report",
                    "final_approval": "Final report review",
                    "save_final_report": "Saving approved report",
                }.get(node_name, node_name.replace("_", " ").title())
                job["workflow_status"] = update.get("workflow_status", job.get("workflow_status", "running"))
                if node_name == "planner":
                    job["plan"] = _json_value(update.get("research_plan"))
                    job["tasks"] = _json_value(update.get("tasks", []))
                    job["workers"] = [
                        {"task_id": task.get("task_id"), "title": task.get("title"), "status": "running", "finding_count": 0}
                        for task in job["tasks"]
                    ]
                if node_name == "research_worker":
                    for worker_update in update.get("worker_progress", []):
                        self._upsert_worker(job, worker_update)
                    job["evidence"] = _json_value(job.get("evidence", [])) + _json_value(update.get("findings", []))
                if "evidence" in update:
                    job["evidence"] = _json_value(update["evidence"])
                if "fact_check_results" in update:
                    job["fact_check_results"] = _json_value(update["fact_check_results"])
                if "report_draft" in update:
                    job["report_draft"] = update["report_draft"]
                if "critic_evaluation" in update:
                    job["critic_evaluation"] = _json_value(update["critic_evaluation"])
                if "revision_history" in update:
                    job["revision_history"] = _json_value(job.get("revision_history", [])) + _json_value(
                        update["revision_history"]
                    )
                if "revision_count" in update:
                    job["revision_count"] = update["revision_count"]
                if "manual_revision_count" in update:
                    job["manual_revision_count"] = update["manual_revision_count"]

    def _sync_from_state(self, job: dict[str, Any], values: dict[str, Any]) -> None:
        for source, target in (
            ("research_question", "question"),
            ("research_plan", "plan"),
            ("tasks", "tasks"),
            ("evidence", "evidence"),
            ("fact_check_results", "fact_check_results"),
            ("report_draft", "report_draft"),
            ("final_report", "final_report"),
            ("report_path", "report_path"),
            ("critic_evaluation", "critic_evaluation"),
            ("final_approval_status", "final_approval_status"),
            ("revision_history", "revision_history"),
            ("revision_feedback", "revision_feedback"),
            ("revision_count", "revision_count"),
            ("manual_revision_count", "manual_revision_count"),
            ("workflow_status", "workflow_status"),
            ("error", "error_message"),
        ):
            if source in values:
                job[target] = _json_value(values[source])
        for worker in values.get("worker_progress", []):
            self._upsert_worker(job, _json_value(worker))
        if not job.get("workers"):
            job["workers"] = [
                {"task_id": task.get("task_id"), "title": task.get("title"), "status": "queued", "finding_count": 0}
                for task in job.get("tasks", [])
            ]

    @staticmethod
    def _snapshot_approval(snapshot: Any) -> dict[str, Any] | None:
        values = getattr(snapshot, "values", {}) or {}
        revision_count = values.get("revision_count", 0)
        max_revision_limit = values.get("max_revision_limit", settings.max_revision_attempts)
        manual_revision_count = values.get("manual_revision_count", 0)
        for task in snapshot.tasks:
            for interruption in getattr(task, "interrupts", ()):
                value = getattr(interruption, "value", None)
                if isinstance(value, dict):
                    return ResearchService._approval_with_actions(
                        _json_value(value),
                        revision_count,
                        max_revision_limit,
                        manual_revision_count,
                    )
        return None

    @staticmethod
    def _allowed_actions(
        revision_count: int,
        max_revision_limit: int,
        manual_revision_count: int = 0,
    ) -> list[str]:
        actions = ["approve", "reject"]
        if revision_count < max_revision_limit or manual_revision_count == 0:
            actions.append("request_revision")
        return actions

    @staticmethod
    def _approval_with_actions(
        approval: dict[str, Any],
        revision_count: int,
        max_revision_limit: int,
        manual_revision_count: int = 0,
    ) -> dict[str, Any]:
        allowed_actions = approval.get("allowed_actions")
        if not isinstance(allowed_actions, list) or not all(
            isinstance(action, str) for action in allowed_actions
        ):
            allowed_actions = ResearchService._allowed_actions(
                revision_count,
                max_revision_limit,
                manual_revision_count,
            )
        return {**approval, "allowed_actions": allowed_actions}

    @staticmethod
    def _approval_stage(approval: dict[str, Any] | None) -> str:
        if not approval:
            return "Research"
        return "Final report review"

    @staticmethod
    def _upsert_worker(job: dict[str, Any], update: dict[str, Any]) -> None:
        workers = job.setdefault("workers", [])
        for index, worker in enumerate(workers):
            if worker.get("task_id") == update.get("task_id"):
                workers[index] = {**worker, **_json_value(update)}
                return
        workers.append(_json_value(update))

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _new_job(thread_id: str, question: str) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "question": question,
            "run_status": "running",
            "workflow_status": "starting",
            "stage": "Starting research",
            "pending_approval": None,
            "plan": None,
            "tasks": [],
            "workers": [],
            "evidence": [],
            "fact_check_results": [],
            "report_draft": "",
            "final_report": None,
            "report_path": None,
            "critic_evaluation": None,
            "final_approval_status": "pending",
            "revision_history": [],
            "revision_feedback": [],
            "revision_count": 0,
            "manual_revision_count": 0,
            "error_message": None,
        }

    @staticmethod
    def _response(job: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(job)