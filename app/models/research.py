from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ResearchTask(BaseModel):
    task_id: str
    title: str
    objective: str
    expected_evidence: str
    status: Literal["pending", "completed", "failed"] = "pending"


class ResearchPlan(BaseModel):
    summary: str
    tasks: list[ResearchTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("tasks")
    @classmethod
    def validate_task_count(cls, value: list[ResearchTask]) -> list[ResearchTask]:
        if not 3 <= len(value) <= 5:
            raise ValueError("A research plan must contain between 3 and 5 tasks.")
        return value


class EvidenceRecord(BaseModel):
    evidence_id: str
    task_id: str
    claim: str
    support: str
    source_id: str | None = None
    source_chunk_id: str | None = None
    retrieval_completeness: Literal["complete", "partial", "unknown"] = "unknown"
    source_url: str | None = None
    source_title: str | None = None
    publication_date: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_type: str = "web"
    retrieval_method: Literal["page_excerpt", "search_snippet", "vector_excerpt"] = "search_snippet"
    retrieval_channels: list[Literal["semantic", "keyword", "web"]] = Field(default_factory=list)
    source_domain: str | None = None
    evidence_scope: Literal["banking", "non_banking", "cross_industry", "unclear"] = "unclear"
    evidence_basis: Literal[
        "regulatory_official",
        "primary_empirical",
        "company_reported",
        "secondary_reporting",
        "vendor_marketing",
        "academic_review",
        "unknown",
    ] = "unknown"
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    authority_score: float | None = Field(default=None, ge=0, le=1)
    freshness_score: float | None = Field(default=None, ge=0, le=1)
    confidence: float = 0.0
    verification_status: Literal["verified", "unverified", "conflicting", "unsupported"] = "unverified"
    notes: str | None = None


class FactCheckResult(BaseModel):
    result_id: str
    task_id: str
    claim: str
    status: Literal["supported", "unsupported", "conflicting", "unknown"]
    reasoning: str
    source_urls: list[str] = Field(default_factory=list)


class ClaimAudit(BaseModel):
    claim_id: str | None = None
    claim: str
    critical: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal[
        "supported",
        "contradicted",
        "insufficient_evidence",
        "insufficient_retrieval",
    ]
    excerpt: str = ""
    reasoning: str
    retrieval_completeness: Literal["complete", "partial", "unknown"] = "unknown"
    source_references: list[str] = Field(default_factory=list)


class CriticEvaluation(BaseModel):
    overall_score: float = 0.0
    citation_accuracy: float = 0.0
    research_completeness: float = 0.0
    objectivity: float = 0.0
    passed: bool = False
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    claim_audits: list[ClaimAudit] = Field(default_factory=list)


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    methodology: str
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    markdown: str = ""


class HumanApprovalDecision(BaseModel):
    approved: bool
    notes: str = ""


class WorkflowResult(BaseModel):
    workflow_status: str
    final_report: str | None = None
    error_message: str | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
