from __future__ import annotations

from app.graph.state import ResearchState


def route_after_write(state: ResearchState) -> str:
    if state.get("workflow_status") == "report_revision_truncated":
        return "final_approval"
    return "critic"


def route_after_critic(state: ResearchState) -> str:
    evaluation = state.get("critic_evaluation")
    if evaluation and evaluation.passed:
        return "final_approval"
    if state.get("manual_revision_count", 0) > 0:
        return "final_approval"
    if state.get("revision_count", 0) >= state.get("max_revision_limit", 3):
        return "revision_limit_reached"
    return "report_revision"


def route_after_final_approval(state: ResearchState) -> str:
    if state.get("workflow_status") == "revision_requested":
        if state.get("manual_revision_count", 0) > 0:
            return "report_revision"
        if state.get("revision_count", 0) >= state.get("max_revision_limit", 3):
            return "revision_limit_reached"
        return "report_revision"
    if state.get("final_approval_status") == "approved":
        return "save_final_report"
    return "__end__"
