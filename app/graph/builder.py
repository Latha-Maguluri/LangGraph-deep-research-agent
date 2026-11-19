from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from app.graph.nodes import (
    aggregate_findings_node,
    critic_node,
    fact_checker_node,
    final_approval_node,
    planner_node,
    revision_limit_node,
    revise_report_node,
    research_worker_node,
    save_final_report_node,
    write_report_node,
)
from app.graph.routing import route_after_critic, route_after_final_approval, route_after_write
from app.graph.state import ResearchState


def dispatch_research_tasks(state: ResearchState):
    return [Send("research_worker", {"current_task": task}) for task in state.get("tasks", [])]


def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("research_worker", research_worker_node)
    builder.add_node("aggregate_findings", aggregate_findings_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("write_report", write_report_node)
    builder.add_node("critic", critic_node)
    builder.add_node("report_revision", revise_report_node)
    builder.add_node("final_approval", final_approval_node)
    builder.add_node("revision_limit_reached", revision_limit_node)
    builder.add_node("save_final_report", save_final_report_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_research_tasks)
    builder.add_edge("research_worker", "aggregate_findings")
    builder.add_edge("aggregate_findings", "fact_checker")
    builder.add_edge("fact_checker", "write_report")
    builder.add_conditional_edges(
        "write_report",
        route_after_write,
        {"critic": "critic", "final_approval": "final_approval"},
    )
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "final_approval": "final_approval",
            "report_revision": "report_revision",
            "revision_limit_reached": "revision_limit_reached",
        },
    ) 
    builder.add_edge("report_revision", "write_report")
    builder.add_edge("revision_limit_reached", "final_approval")
    builder.add_conditional_edges(
        "final_approval",
        route_after_final_approval,
        {
            "report_revision": "report_revision",
            "revision_limit_reached": "revision_limit_reached",
            "save_final_report": "save_final_report",
            END: END,
        },
    )
    builder.add_edge("save_final_report", END)

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())
