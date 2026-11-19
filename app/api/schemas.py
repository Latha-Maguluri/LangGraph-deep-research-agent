from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=10)


class ResearchResponse(BaseModel):
    thread_id: str
    question: str
    run_status: str
    workflow_status: str
    stage: str
    pending_approval: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    workers: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    fact_check_results: list[str] = Field(default_factory=list)
    report_draft: str = ""
    final_report: str | None = None
    report_path: str | None = None
    critic_evaluation: dict[str, Any] | None = None
    final_approval_status: str = "pending"
    revision_history: list[dict[str, Any]] = Field(default_factory=list)
    revision_feedback: list[str] = Field(default_factory=list)
    revision_count: int = 0
    manual_revision_count: int = 0
    error_message: str | None = None


class DecisionRequest(BaseModel):
    action: Literal["approve", "reject", "request_revision"]
    notes: str = Field(default="", max_length=4000)
