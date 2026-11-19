from threading import Barrier, Event, Thread

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api.dependencies import get_research_service
from app.api.routes.research import router as research_router
from app.graph import builder as graph_builder, nodes
from app.graph.builder import build_graph, dispatch_research_tasks
from app.infrastructure.checkpointer import create_sqlite_checkpointer
from app.infrastructure.vector_store import VectorDocument
from app.models.research import CriticEvaluation, ResearchTask
from app.services.research_service import ResearchService
from fakes import DeterministicChatModel


def test_dispatch_research_tasks_creates_send_jobs() -> None:
    tasks = [
        ResearchTask(task_id="t1", title="Market", objective="Assess market size", expected_evidence="Market estimates"),
        ResearchTask(task_id="t2", title="Competition", objective="Assess competitors", expected_evidence="Competitor data"),
    ]
    jobs = dispatch_research_tasks({"tasks": tasks})
    assert len(jobs) == 2
    assert jobs[0].node == "research_worker"


def test_generated_plan_runs_parallel_workers_without_plan_approval(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: [{"title": query, "url": "https://example.org/source", "snippet": "Source excerpt."}],
    )
    vector_calls = []
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: (
            vector_calls.append(query)
            or [VectorDocument("vector-1", "Internal vector excerpt.", {"source_title": "Internal guide"})]
        ),
    )
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))
    graph = build_graph()
    config = {"configurable": {"thread_id": "parallel-research-test"}}
    initial = graph.invoke(
        {
            "thread_id": "a" * 32,
            "research_question": "Test question",
            "revision_count": 0,
            "max_revision_limit": 3,
        },
        config=config,
    )

    assert initial["__interrupt__"][0].value["type"] == "final_approval"
    assert len(vector_calls) == 5
    assert len(initial["evidence"]) == 2
    assert all(item.verification_status == "unverified" for item in initial["evidence"])
    assert {item.source_type for item in initial["evidence"]} == {"web", "vector"}
    result = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert len(result["findings"]) == 10
    assert result["workflow_status"] == "completed"
    assert (tmp_path / "outputs" / f'{"a" * 32}.md').read_text(encoding="utf-8") == result["final_report"]


def test_research_worker_keeps_local_evidence_when_web_search_fails(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())

    def failed_web_search(query, max_results):
        raise RuntimeError("Tavily temporarily unavailable")

    monkeypatch.setattr(nodes, "web_search", failed_web_search)
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: [
            VectorDocument(
                "local-evidence",
                "Local evidence about banking AI risk management.",
                {"source_title": "Banking AI risk report", "page_number": 4},
            )
        ],
    )

    result = nodes.research_worker_node(
        {
            "current_task": ResearchTask(
                task_id="task-web-failure",
                title="Risk management",
                objective="Assess banking AI risk management",
                expected_evidence="Risk controls and governance",
            )
        }
    )

    assert len(result["findings"]) == 1
    assert result["findings"][0].source_type == "vector"
    assert "Web search was unavailable" in result["findings"][0].notes
    assert "Web search was unavailable" in result["worker_progress"][0]["warning"]


def test_critic_revision_then_human_approval_resumes_and_saves_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    web_calls = []
    vector_calls = []
    monkeypatch.setattr(nodes, "web_search", lambda query, max_results: web_calls.append(query) or [])
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: (
            vector_calls.append(query)
            or [VectorDocument("vector-1", "Relevant internal evidence.", {"source_title": "Internal guide"})]
        ),
    )
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))
    evaluations = iter([False, True])

    def critic(state):
        passed = next(evaluations)
        evaluation = CriticEvaluation(
            overall_score=0.9 if passed else 0.4,
            citation_accuracy=0.9 if passed else 0.4,
            research_completeness=0.9 if passed else 0.4,
            objectivity=0.9 if passed else 0.4,
            passed=passed,
            issues=[] if passed else ["Add stronger evidence."],
            revision_instructions=[] if passed else ["Add stronger evidence."],
        )
        return {
            "critic_evaluation": evaluation,
            "revision_history": [{"revision": state["revision_count"], "passed": passed}],
            "revision_feedback": [] if passed else ["Add stronger evidence."],
            "workflow_status": "critic_reviewed",
        }

    monkeypatch.setattr(graph_builder, "critic_node", critic)
    graph = build_graph()
    config = {"configurable": {"thread_id": "critic-revision-test"}}
    initial = graph.invoke(
        {
            "thread_id": "b" * 32,
            "research_question": "Assess this research question.",
            "revision_count": 0,
            "max_revision_limit": 3,
        },
        config=config,
    )

    assert initial["revision_count"] == 1
    assert initial["critic_evaluation"].passed
    assert initial["__interrupt__"][0].value["type"] == "final_approval"
    assert len(web_calls) == 5
    assert len(vector_calls) == 5
    assert len(initial["findings"]) == 5

    resumed = graph.invoke(Command(resume={"action": "approve"}), config=config)

    assert resumed["workflow_status"] == "completed"
    assert resumed["final_approval_status"] == "approved"
    assert (tmp_path / "outputs" / f'{"b" * 32}.md').exists()


def test_truncated_revision_returns_previous_draft_and_review_for_approval(monkeypatch, tmp_path) -> None:
    class TruncatingWriter:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return type(
                    "Response",
                    (),
                    {"content": "# Previous complete report", "response_metadata": {"stop_reason": "end_turn"}},
                )()
            return type(
                "Response",
                (),
                {"content": "# Truncated revision", "response_metadata": {"stop_reason": "max_tokens"}},
            )()

    writer = TruncatingWriter()
    monkeypatch.setattr(
        nodes, "_create_model", lambda role: writer if role == "writer" else DeterministicChatModel()
    )
    monkeypatch.setattr(nodes, "web_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes, "vector_search", lambda query, max_results: [])
    critic_calls = []

    def critic(state):
        critic_calls.append(state["report_draft"])
        return {
            "critic_evaluation": CriticEvaluation(overall_score=0.7, passed=False),
            "revision_feedback": ["Check the missing source."],
            "revision_history": [{"revision": state["revision_count"], "overall_score": 0.7}],
            "workflow_status": "critic_reviewed",
        }

    monkeypatch.setattr(graph_builder, "critic_node", critic)
    checkpoint_path = str(tmp_path / "truncated-revision.sqlite")
    checkpointer = create_sqlite_checkpointer(checkpoint_path)
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "truncated-revision-test"}}
    result = graph.invoke(
        {
            "thread_id": "c" * 32,
            "research_question": "Assess the research question.",
            "revision_count": 0,
            "max_revision_limit": 3,
        },
        config=config,
    )

    approval = result["__interrupt__"][0].value
    assert writer.calls == 4
    assert critic_calls == ["# Previous complete report"]
    assert result["report_draft"] == "# Previous complete report"
    assert result["critic_evaluation"].overall_score == 0.7
    assert result["workflow_status"] == "report_revision_truncated"
    assert approval["type"] == "final_approval"
    assert "previous complete draft" in approval["message"]
    assert "feedback was not applied" in approval["message"]
    assert approval["report_draft"] == result["report_draft"]
    assert approval["critic_evaluation"]["overall_score"] == 0.7

    checkpointer.conn.close()
    checkpointer = create_sqlite_checkpointer(checkpoint_path)
    try:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = graph.get_state(config)
        assert snapshot.values["report_draft"] == "# Previous complete report"
        assert snapshot.tasks[0].interrupts[0].value["message"] == approval["message"]
        rejected = graph.invoke(Command(resume={"action": "reject"}), config=config)
        assert rejected["workflow_status"] == "report_rejected"
        assert rejected["report_draft"] == "# Previous complete report"
    finally:
        checkpointer.conn.close()


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_approval_status"),
    [
        ("approve", "completed", "approved"),
        ("reject", "report_rejected", "rejected"),
    ],
)
def test_critic_revision_limit_pauses_for_final_decision(
    monkeypatch,
    tmp_path,
    decision: str,
    expected_status: str,
    expected_approval_status: str,
) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(nodes, "web_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes, "vector_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))

    def failing_critic(state):
        evaluation = CriticEvaluation(
            overall_score=0.2,
            citation_accuracy=0.2,
            research_completeness=0.2,
            objectivity=0.2,
            passed=False,
            issues=["Insufficient evidence."],
            revision_instructions=["Expand evidence."],
        )
        return {
            "critic_evaluation": evaluation,
            "revision_history": [{"revision": state["revision_count"], "passed": False}],
            "revision_feedback": ["Expand evidence."],
            "workflow_status": "critic_reviewed",
        }

    monkeypatch.setattr(graph_builder, "critic_node", failing_critic)
    graph = build_graph()
    config = {"configurable": {"thread_id": "revision-limit-test"}}
    result = graph.invoke(
        {
            "thread_id": "c" * 32,
            "research_question": "Assess this research question.",
            "revision_count": 0,
            "max_revision_limit": 3,
        },
        config=config,
    )

    assert result["revision_count"] == 3
    assert result["workflow_status"] == "revision_limit_reached"
    assert result["final_approval_status"] == "needs_human_review"
    approval = result["__interrupt__"][0].value
    assert approval["type"] == "final_approval"
    assert approval["allowed_actions"] == ["approve", "reject", "request_revision"]

    completed = graph.invoke(Command(resume={"action": decision}), config=config)

    assert completed["workflow_status"] == expected_status
    assert completed["final_approval_status"] == expected_approval_status
    report_path = tmp_path / "outputs" / f'{"c" * 32}.md'
    assert report_path.exists() is (decision == "approve")


def test_revision_limit_rejects_second_human_revision_request(tmp_path) -> None:
    service = ResearchService(str(tmp_path / "revision-limit.sqlite"))
    thread_id = "e" * 32
    job = service._new_job(thread_id, "A final report")
    job["revision_count"] = 3
    job["manual_revision_count"] = 1
    job["run_status"] = "waiting_for_approval"
    job["pending_approval"] = {
        "type": "final_approval",
        "allowed_actions": ["approve", "reject"],
    }
    service._runs[thread_id] = job
    try:
        with pytest.raises(ValueError, match="unavailable at the current revision limit"):
            service.decide(thread_id, "request_revision", "One more change.")

        assert job["run_status"] == "waiting_for_approval"
        assert job["pending_approval"]["allowed_actions"] == ["approve", "reject"]
    finally:
        service.close()


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_approval_status"),
    [
        ("approve", "completed", "approved"),
        ("reject", "report_rejected", "rejected"),
    ],
)
def test_revision_limit_decision_survives_sqlite_restart(
    monkeypatch,
    tmp_path,
    decision: str,
    expected_status: str,
    expected_approval_status: str,
) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(nodes, "web_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes, "vector_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))

    def failing_critic(state):
        evaluation = CriticEvaluation(
            overall_score=0.2,
            citation_accuracy=0.2,
            research_completeness=0.2,
            objectivity=0.2,
            passed=False,
            issues=["Insufficient evidence."],
            revision_instructions=["Expand evidence."],
        )
        return {
            "critic_evaluation": evaluation,
            "revision_history": [{"revision": state["revision_count"], "passed": False}],
            "revision_feedback": ["Expand evidence."],
            "workflow_status": "critic_reviewed",
        }

    monkeypatch.setattr(graph_builder, "critic_node", failing_critic)
    checkpoint_path = str(tmp_path / "revision-limit.sqlite")
    service = ResearchService(checkpoint_path)
    try:
        started = service.start_research("Assess this banking AI research question.")
        thread_id = started["thread_id"]
        service.wait_for_run(thread_id)
        first_review = service.get_run(thread_id)

        assert first_review["revision_count"] == 3
        assert first_review["pending_approval"]["allowed_actions"] == [
            "approve",
            "reject",
            "request_revision",
        ]

        service.close()
        service = ResearchService(checkpoint_path)
        restored_review = service.get_run(thread_id)
        assert restored_review["pending_approval"]["allowed_actions"] == [
            "approve",
            "reject",
            "request_revision",
        ]

        service.decide(thread_id, decision)
        service.wait_for_run(thread_id)
        completed = service.get_run(thread_id)

        assert completed["workflow_status"] == expected_status
        assert completed["final_approval_status"] == expected_approval_status
        report_path = tmp_path / "outputs" / f"{thread_id}.md"
        assert report_path.exists() is (decision == "approve")
    finally:
        service.close()


def test_concurrent_human_decisions_only_resume_once(monkeypatch, tmp_path) -> None:
    service = ResearchService(str(tmp_path / "concurrent-decisions.sqlite"))
    thread_id = "f" * 32
    job = service._new_job(thread_id, "A final report")
    job["run_status"] = "waiting_for_approval"
    job["pending_approval"] = {
        "type": "final_approval",
        "allowed_actions": ["approve", "reject", "request_revision"],
    }
    service._runs[thread_id] = job
    barrier = Barrier(2)
    worker_started = Event()
    release_worker = Event()
    original_get_run = service.get_run
    results = []
    errors = []

    def synchronized_get_run(run_id: str) -> dict:
        result = original_get_run(run_id)
        barrier.wait(timeout=5)
        return result

    def hold_advance(_thread_id: str, _input_value) -> None:
        worker_started.set()
        release_worker.wait(timeout=5)

    def submit_decision(action: str) -> None:
        try:
            results.append(service.decide(thread_id, action))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(service, "get_run", synchronized_get_run)
    monkeypatch.setattr(service, "_advance", hold_advance)
    threads = [
        Thread(target=submit_decision, args=("approve",)),
        Thread(target=submit_decision, args=("reject",)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert worker_started.is_set()
        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert len(service._futures) == 1
    finally:
        release_worker.set()
        service.close()


def test_research_service_persists_review_and_revision_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: [{"title": query, "url": "https://example.org/source", "snippet": "Source excerpt."}],
    )
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: [VectorDocument("vector-1", "Internal excerpt.", {"source_title": "Internal guide"})],
    )
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))
    checkpoint_path = str(tmp_path / "research.sqlite")
    service = ResearchService(checkpoint_path)
    try:
        started = service.start_research("Evaluate this sample market opportunity for a SaaS business.")
        thread_id = started["thread_id"]
        service.wait_for_run(thread_id)
        assert service.get_run(thread_id)["pending_approval"]["type"] == "final_approval"

        service.close()
        service = ResearchService(checkpoint_path)
        final_review = service.get_run(thread_id)
        assert final_review["pending_approval"]["type"] == "final_approval"
        assert len(final_review["evidence"]) == 2
        assert all(item["verification_status"] == "unverified" for item in final_review["evidence"])

        service.decide(thread_id, "request_revision", "Clarify the assumptions.")
        service.wait_for_run(thread_id)
        revised = service.get_run(thread_id)
        assert revised["pending_approval"]["type"] == "final_approval"
        assert revised["revision_count"] == 4
        assert len(revised["revision_history"]) == 5
        assert not revised["critic_evaluation"]["passed"]

        service.decide(thread_id, "approve")
        service.wait_for_run(thread_id)
        completed = service.get_run(thread_id)
        assert completed["workflow_status"] == "completed"
        assert completed["final_report"]
        assert (tmp_path / "outputs" / f"{thread_id}.md").read_text(encoding="utf-8") == completed["final_report"]
    finally:
        service.close()


def test_human_revision_is_available_once_after_automatic_limit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(nodes, "web_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes, "vector_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))
    critic_results = iter([False, False, False, False, False])

    def critic(state):
        passed = next(critic_results)
        evaluation = CriticEvaluation(
            overall_score=0.9 if passed else 0.3,
            citation_accuracy=0.9 if passed else 0.3,
            research_completeness=0.9 if passed else 0.3,
            objectivity=0.9 if passed else 0.3,
            passed=passed,
            issues=[] if passed else ["Strengthen the evidence."],
            revision_instructions=[] if passed else ["Strengthen the evidence."],
        )
        return {
            "critic_evaluation": evaluation,
            "revision_history": [{"revision": state["revision_count"], "passed": passed}],
            "revision_feedback": [] if passed else ["Strengthen the evidence."],
            "workflow_status": "critic_reviewed",
        }

    monkeypatch.setattr(graph_builder, "critic_node", critic)
    service = ResearchService(str(tmp_path / "human-revision-budget.sqlite"))
    try:
        started = service.start_research("Assess this banking AI question.")
        thread_id = started["thread_id"]
        service.wait_for_run(thread_id)
        first_review = service.get_run(thread_id)

        assert first_review["revision_count"] == 3
        assert "request_revision" in first_review["pending_approval"]["allowed_actions"]

        service.decide(thread_id, "request_revision", "Clarify the limitations.")
        service.wait_for_run(thread_id)
        final_review = service.get_run(thread_id)

        assert final_review["revision_count"] == 4
        assert final_review["manual_revision_count"] == 1
        assert final_review["pending_approval"]["allowed_actions"] == ["approve", "reject"]
        service.close()
        service = ResearchService(str(tmp_path / "human-revision-budget.sqlite"))
        final_review = service.get_run(thread_id)
        assert final_review["manual_revision_count"] == 1
        assert final_review["pending_approval"]["allowed_actions"] == ["approve", "reject"]
        with pytest.raises(ValueError, match="unavailable at the current revision limit"):
            service.decide(thread_id, "request_revision", "Please revise again.")

        service.decide(thread_id, "approve")
        service.wait_for_run(thread_id)
        completed = service.get_run(thread_id)
        assert completed["workflow_status"] == "completed"
        assert (tmp_path / "outputs" / f"{thread_id}.md").exists()
    finally:
        service.close()


def test_research_service_logs_workflow_exception(monkeypatch, tmp_path, caplog) -> None:
    service = ResearchService(str(tmp_path / "failed-research.sqlite"))

    def fail_stream(*args, **kwargs):
        raise ValueError("Malformed critic JSON.")

    monkeypatch.setattr(service.graph, "stream", fail_stream)
    try:
        started = service.start_research("Assess the AI industry outlook.")
        service.wait_for_run(started["thread_id"])

        assert "Research workflow failed" in caplog.text
        assert "Malformed critic JSON." in caplog.text
    finally:
        service.close()


def test_research_service_surfaces_writer_token_limit(monkeypatch, tmp_path) -> None:
    from app.core.exceptions import ReportWriterTokenLimitError

    service = ResearchService(str(tmp_path / "writer-limit.sqlite"))

    def fail_stream(*args, **kwargs):
        raise ReportWriterTokenLimitError("The report writer reached the output token limit.")

    monkeypatch.setattr(service.graph, "stream", fail_stream)
    try:
        started = service.start_research("Assess the AI industry outlook.")
        service.wait_for_run(started["thread_id"])
        run = service.get_run(started["thread_id"])
        assert run["run_status"] == "failed"
        assert run["error_message"] == "The report writer reached the output token limit."
    finally:
        service.close()


def test_research_api_runs_plan_and_exposes_final_approval(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: [{"title": query, "url": "https://example.org/source", "snippet": "Source excerpt."}],
    )
    monkeypatch.setattr(nodes, "vector_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))
    service = ResearchService(str(tmp_path / "api-research.sqlite"))
    api = FastAPI()
    api.include_router(research_router, prefix="/api")
    api.dependency_overrides[get_research_service] = lambda: service
    try:
        with TestClient(api) as client:
            started = client.post(
                "/api/research",
                json={"question": "Assess a possible SaaS market entry."},
            )
            assert started.status_code == 202
            thread_id = started.json()["thread_id"]
            service.wait_for_run(thread_id)

            review = client.get(f"/api/research/{thread_id}")
            assert review.json()["pending_approval"]["type"] == "final_approval"
            assert len(review.json()["plan"]["tasks"]) == 5
            assert review.json()["fact_check_results"]

            resumed = client.post(
                f"/api/research/{thread_id}/decision",
                json={"action": "approve", "notes": ""},
            )
            assert resumed.status_code == 202
            service.wait_for_run(thread_id)
            status_response = client.get(f"/api/research/{thread_id}")
            assert status_response.json()["workflow_status"] == "completed"
            assert status_response.json()["report_path"] == str(tmp_path / "outputs" / f"{thread_id}.md")
    finally:
        api.dependency_overrides.clear()
        service.close()


def test_rejecting_final_report_ends_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: [{"title": query, "url": "https://example.org/source", "snippet": "Source excerpt."}],
    )
    monkeypatch.setattr(nodes, "vector_search", lambda query, max_results: [])
    monkeypatch.setattr(nodes.settings, "output_dir", str(tmp_path / "outputs"))
    service = ResearchService(str(tmp_path / "rejections.sqlite"))
    try:
        report_run = service.start_research("Evaluate a separate market opportunity for a SaaS business.")
        report_thread = report_run["thread_id"]
        service.wait_for_run(report_thread)
        service.decide(report_thread, "reject")
        service.wait_for_run(report_thread)
        rejected = service.get_run(report_thread)
        assert rejected["workflow_status"] == "report_rejected"
        assert rejected["final_approval_status"] == "rejected"
        assert rejected["final_report"] is None
        assert not list((tmp_path / "outputs").glob("*.md"))
    finally:
        service.close()


def test_server_registers_health_endpoint() -> None:
    from app.api.server import app

    assert "/api/health" in app.openapi()["paths"]


def test_server_serves_dashboard_assets_and_health(monkeypatch, tmp_path) -> None:
    from app.api.server import app
    from app.core.config import settings

    monkeypatch.setattr(settings, "checkpoint_db_path", str(tmp_path / "dashboard.sqlite"))
    get_research_service.cache_clear()
    with TestClient(app) as client:
        dashboard = client.get("/")
        script = client.get("/static/app.js")
        health = client.get("/api/health")

    assert dashboard.status_code == 200
    assert "Research Desk" in dashboard.text
    assert script.status_code == 200
    assert "beginPollingAfter" in script.text
    assert health.status_code == 200
