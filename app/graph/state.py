from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from app.models.research import CriticEvaluation, EvidenceRecord, ResearchPlan, ResearchTask


def append_list(values: list, update: list) -> list:
    return list(values) + list(update)


class ResearchState(TypedDict):
    thread_id: str
    research_question: str
    research_plan: ResearchPlan | None
    tasks: list[ResearchTask]
    findings: Annotated[list[EvidenceRecord], append_list]
    source_records: Annotated[list[dict[str, Any]], append_list]
    worker_progress: Annotated[list[dict[str, Any]], append_list]
    evidence: list[EvidenceRecord]
    writer_source_evidence: list[dict[str, Any]]
    writer_evidence_matrix: list[dict[str, Any]]
    fact_check_results: list[str]
    report_draft: str
    critic_evaluation: CriticEvaluation | None
    revision_history: Annotated[list[dict[str, Any]], append_list]
    revision_feedback: list[str]
    revision_count: int
    max_revision_limit: int
    manual_revision_count: int
    final_approval_status: Literal["pending", "approved", "rejected", "needs_human_review"]
    workflow_status: str
    error: str | None
    final_report: str | None
    report_path: str | None
