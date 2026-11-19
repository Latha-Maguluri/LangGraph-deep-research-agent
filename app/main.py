from __future__ import annotations

from app.graph.builder import build_graph


def run_research(question: str, thread_id: str = "demo-thread"):
    graph = build_graph()
    return graph.invoke(
        {
            "research_question": question,
            "revision_count": 0,
            "max_revision_limit": 3,
            "manual_revision_count": 0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )


if __name__ == "__main__":
    result = run_research(
        "Analyze whether an AI-powered customer support platform is a viable market entry for a mid-sized SaaS company in Europe."
    )
    print(result)
