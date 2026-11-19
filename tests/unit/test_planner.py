import json

import pytest

from app.core.config import Settings
from app.graph import nodes
from app.graph.nodes import fact_checker_node, planner_node, research_worker_node, write_report_node
from app.infrastructure import llm_factory
from app.infrastructure.llm_factory import AnthropicChatModel, GeminiChatModel, LLMFactory
from app.infrastructure.vector_store import VectorDocument
from app.models.research import ClaimAudit, CriticEvaluation, EvidenceRecord, ResearchTask
from fakes import DeterministicChatModel


def _confirm_test_audits(prompt: str):
    checks = json.loads(prompt.split("Claim verdicts to confirm:\n", 1)[1])
    return type(
        "Response",
        (),
        {"content": json.dumps({"results": [
            {"claim_id": check["claim_id"], "verdict": "confirmed", "reasoning": "Supported by test evidence."}
            for check in checks
        ]})},
    )()


def test_settings_reads_provider_env_keys(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    settings = Settings()

    assert settings.anthropic_api_key == "test-anthropic-key"
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.tavily_api_key == "test-tavily-key"


def test_settings_defaults_to_role_specific_models(monkeypatch) -> None:
    for role in ("PLANNER", "RESEARCHER", "WRITER", "REVIEWER"):
        monkeypatch.delenv(f"{role}_LLM_PROVIDER", raising=False)
        monkeypatch.delenv(f"{role}_LLM_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.planner_llm_provider == "anthropic"
    assert settings.planner_llm_model == "claude-sonnet-5-5"
    assert settings.researcher_llm_provider == "gemini"
    assert settings.researcher_llm_model == "gemini-3.5-flash-lite"
    assert settings.writer_llm_provider == "anthropic"
    assert settings.writer_llm_model == "claude-sonnet-5-5"
    assert settings.reviewer_llm_provider == "anthropic"
    assert settings.reviewer_llm_model == "claude-sonnet-5-5"


def test_model_router_uses_role_specific_provider_and_model(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes.LLMFactory,
        "create_model",
        staticmethod(lambda provider, model: (provider, model)),
    )

    assert nodes._create_model("planner") == ("anthropic", "claude-sonnet-5-5")
    assert nodes._create_model("researcher") == ("gemini", "gemini-3.5-flash-lite")
    assert nodes._create_model("writer") == ("anthropic", "claude-sonnet-5-5")
    assert nodes._create_model("reviewer") == ("anthropic", "claude-sonnet-5-5")


def test_anthropic_model_skips_thinking_blocks() -> None:
    from types import SimpleNamespace

    model = AnthropicChatModel.__new__(AnthropicChatModel)
    model.model_name = "test-model"
    model.client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking="private reasoning"),
                    SimpleNamespace(type="text", text="report text"),
                ],
                stop_reason="end_turn",
            )
        )
    )

    response = model.invoke("test prompt")

    assert response.content == "report text"
    assert response.response_metadata["stop_reason"] == "end_turn"


def test_anthropic_model_reports_empty_response_metadata() -> None:
    from types import SimpleNamespace

    model = AnthropicChatModel.__new__(AnthropicChatModel)
    model.model_name = "test-model"
    model.client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                content=[SimpleNamespace(type="thinking", thinking="private reasoning")],
                stop_reason="max_tokens",
            )
        )
    )

    with pytest.raises(llm_factory.LLMOutputTokenLimitError) as error:
        model.invoke("test prompt")

    assert "stop_reason='max_tokens'" in str(error.value)
    assert "content_block_types=thinking" in str(error.value)
    assert "private reasoning" not in str(error.value)


def test_llm_factory_requires_anthropic_api_key(monkeypatch) -> None:
    monkeypatch.setattr(llm_factory.settings, "anthropic_api_key", "")

    try:
        LLMFactory.create_model("anthropic", "test-model")
    except RuntimeError as exc:
        assert "ANTHROPIC_API_KEY is required" in str(exc)
    else:
        raise AssertionError("Expected missing Anthropic credentials to raise RuntimeError.")


def test_llm_factory_requires_gemini_api_key(monkeypatch) -> None:
    monkeypatch.setattr(llm_factory.settings, "gemini_api_key", "")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is required"):
        LLMFactory.create_model("gemini", "test-model")


def test_json_retry_increases_token_budget_after_truncated_response() -> None:
    class TruncatedModel:
        calls = []

        def invoke(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if len(self.calls) == 1:
                return type(
                    "Response",
                    (),
                    {
                        "content": '{"score":"',
                        "response_metadata": {"stop_reason": "max_tokens"},
                    },
                )()
            return type("Response", (), {"content": '{"score":0.9}'})()

    model = TruncatedModel()
    result = nodes._invoke_json_with_retry(
        model,
        "Return a score.",
        {"score"},
        "test evaluator",
        max_tokens=1200,
        retry_max_tokens=2400,
    )

    assert result == {"score": 0.9}
    assert [call[1]["max_tokens"] for call in model.calls] == [1200, 2400]
    assert "hit its output token limit" in model.calls[1][0]


def test_gemini_model_uses_generate_content_and_maps_token_limit() -> None:
    from types import SimpleNamespace

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    captured = {}

    def generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            text="Gemini response",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
        )

    model = GeminiChatModel.__new__(GeminiChatModel)
    model.model_name = "test-model"
    model.genai = SimpleNamespace(
        types=SimpleNamespace(GenerateContentConfig=GenerateContentConfig)
    )
    model.client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )

    response = model.invoke("test prompt", max_tokens=73)

    assert response.content == "Gemini response"
    assert response.response_metadata["stop_reason"] == "max_tokens"
    assert captured["model"] == "test-model"
    assert captured["contents"] == "test prompt"
    assert captured["config"].kwargs == {"max_output_tokens": 73}


def test_gemini_empty_token_limited_output_is_explicit() -> None:
    from types import SimpleNamespace

    model = GeminiChatModel.__new__(GeminiChatModel)
    model.model_name = "test-model"
    model.genai = SimpleNamespace(
        types=SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    )
    model.client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text=None,
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
            )
        )
    )

    with pytest.raises(llm_factory.LLMOutputTokenLimitError):
        model.invoke("test prompt")


def test_planner_generates_valid_tasks(monkeypatch) -> None:
    class PlannerModel(DeterministicChatModel):
        prompt = ""

        def invoke(self, prompt, **kwargs):
            self.prompt = prompt
            return super().invoke(prompt, **kwargs)

    model = PlannerModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)
    state = {"research_question": "Is AI-powered support automation viable for a mid-sized SaaS company?"}
    result = planner_node(state)

    assert "research_plan" in result
    assert len(result["tasks"]) == 5
    assert all(task.title for task in result["tasks"])
    assert "numerator/denominator" in model.prompt
    assert "counterevidence" in model.prompt


def test_planner_parses_json_embedded_in_model_response(monkeypatch) -> None:
    class JsonResponseModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {
                    "content": (
                        'Plan follows:\n```json\n{"summary":"Test plan","tasks":['
                        '{"title":"Market","objective":"Assess demand","expected_evidence":"Market data"},'
                        '{"title":"Competition","objective":"Assess rivals","expected_evidence":"Competitor data"},'
                        '{"title":"Risks","objective":"Assess barriers","expected_evidence":"Risk sources"}'
                        ']}\n```'
                    )
                },
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: JsonResponseModel())
    result = planner_node({"research_question": "Test question"})

    assert result["research_plan"].summary == "Test plan"
    assert len(result["tasks"]) == 3


def test_planner_retries_invalid_json(monkeypatch) -> None:
    class CorrectingModel:
        calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            content = "not json" if self.calls == 1 else (
                '{"summary":"Test plan","tasks":['
                '{"title":"Market","objective":"Assess demand","expected_evidence":"Market data"},'
                '{"title":"Competition","objective":"Assess rivals","expected_evidence":"Competitor data"},'
                '{"title":"Risks","objective":"Assess barriers","expected_evidence":"Risk sources"}]}'
            )
            return type("Response", (), {"content": content})()

    model = CorrectingModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)

    result = planner_node({"research_question": "Test question"})

    assert model.calls == 2
    assert result["research_plan"].summary == "Test plan"


def test_planner_retries_task_count_outside_assignment_range(monkeypatch) -> None:
    class CorrectingModel:
        calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                tasks = [
                    '{"title":"Market","objective":"Assess demand","expected_evidence":"Market data"}',
                    '{"title":"Competition","objective":"Assess rivals","expected_evidence":"Competitor data"}',
                ]
            else:
                tasks = [
                    '{"title":"Market","objective":"Assess demand","expected_evidence":"Market data"}',
                    '{"title":"Competition","objective":"Assess rivals","expected_evidence":"Competitor data"}',
                    '{"title":"Risks","objective":"Assess barriers","expected_evidence":"Risk sources"}',
                ]
            content = '{"summary":"Test plan","tasks":[' + ",".join(tasks) + "]}"
            return type("Response", (), {"content": content})()

    model = CorrectingModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)

    result = planner_node({"research_question": "Test question"})

    assert model.calls == 2
    assert len(result["tasks"]) == 3


def test_research_worker_converts_search_results_to_unverified_evidence(monkeypatch) -> None:
    web_queries = []
    model = DeterministicChatModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: (
            web_queries.append((query, max_results))
            or [
                {
                    "title": "Market report",
                    "url": "https://example.org/report",
                    "snippet": "Market data.",
                    "raw_content": "Market demand grew in the latest reporting period.",
                    "source_domain": "example.org",
                    "relevance_score": 0.8,
                    "authority_score": 0.3,
                    "freshness_score": 0.7,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: [
            VectorDocument("doc-1", "Internal evidence.", {"source_title": "Internal guide"})
        ],
    )
    result = research_worker_node(
        {
            "current_task": {
                "task_id": "task-1",
                "objective": "Assess market demand",
                "expected_evidence": "Recent market estimates",
            }
        }
    )

    assert result["findings"][0].claim == "Market demand grew in the latest reporting period."
    assert result["findings"][0].verification_status == "unverified"
    assert result["findings"][0].source_url == "https://example.org/report"
    assert result["findings"][0].retrieval_method == "page_excerpt"
    assert result["findings"][0].support == "Market demand grew in the latest reporting period."
    assert result["findings"][0].source_domain == "example.org"
    assert result["findings"][0].confidence == 0.8
    assert result["findings"][0].evidence_scope == "banking"
    assert result["findings"][0].evidence_basis == "primary_empirical"
    assert result["source_records"][0]["source_id"] == "web:https://example.org/report"
    assert result["source_records"][0]["text"] == "Market demand grew in the latest reporting period."
    assert web_queries[0][1] == 4
    assert "Recent market estimates" in web_queries[0][0]
    assert result["findings"][1].source_type == "vector"
    assert result["findings"][1].source_title == "Internal guide"
    assert result["findings"][1].retrieval_method == "vector_excerpt"


def test_research_worker_searches_web_even_with_multiple_local_sources(monkeypatch) -> None:
    web_calls = []
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: (
            web_calls.append((query, max_results))
            or [
                {
                    "title": "Current source",
                    "url": "https://example.org/current",
                    "snippet": "Current evidence.",
                    "raw_content": "Current evidence provides an external comparison.",
                    "source_domain": "example.org",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: [
            VectorDocument("doc-1", "Local evidence one.", {"source_id": "source-1"}),
            VectorDocument("doc-2", "Local evidence two.", {"source_id": "source-2"}),
        ],
    )

    result = research_worker_node(
        {
            "current_task": {
                "task_id": "task-1",
                "objective": "Assess model risk",
                "expected_evidence": "Recent evidence",
            }
        }
    )

    assert web_calls and web_calls[0][1] == 4
    assert {item.source_type for item in result["findings"]} == {"web", "vector"}


def test_researcher_rejects_findings_with_unprovided_evidence_ids() -> None:
    with pytest.raises(ValueError, match="must reference an input evidence_id"):
        nodes._validate_research_data(
            {
                "findings": [
                    {
                        "evidence_id": "invented-id",
                        "key_finding": "Unsupported finding",
                        "confidence": 0.8,
                        "conflicting_evidence_ids": [],
                    }
                ]
            },
            {"provided-id"},
        )


def test_researcher_rejects_unknown_evidence_category() -> None:
    with pytest.raises(ValueError, match="evidence_basis"):
        nodes._validate_research_data(
            {
                "findings": [
                    {
                        "evidence_id": "provided-id",
                        "key_finding": "Supported finding",
                        "confidence": 0.8,
                        "conflicting_evidence_ids": [],
                        "evidence_scope": "banking",
                        "evidence_basis": "expert_opinion",
                    }
                ]
            },
            {"provided-id"},
        )


def test_writer_source_packet_extracts_source_passages_for_shared_use() -> None:
    from app.graph.evidence_audit import web_source_id

    url = "https://example.org/study"
    source_id = web_source_id(url)
    evidence = EvidenceRecord(
        evidence_id="ev-study",
        task_id="task-1",
        claim="Productivity improved among customer support agents.",
        support="A short retrieval excerpt.",
        source_id=source_id,
        source_url=url,
        source_title="Customer support study",
        source_type="web",
    )
    source_text = (
        "A randomized study at a software company measured customer support productivity. "
        "The study included 5,179 agents and found an average improvement in productivity."
    )

    packet = nodes._build_writer_source_evidence(
        "What evidence exists for employee productivity?",
        [evidence],
        [
            {
                "source_id": source_id,
                "task_id": "task-1",
                "source_url": url,
                "source_title": "Customer support study",
                "text": source_text,
            }
        ],
    )

    assert len(packet) == 1
    assert packet[0]["evidence_ids"] == ["ev-study"]
    assert "5,179 agents" in packet[0]["excerpts"][0]
    assert packet[0]["retrieval_completeness"] == "partial"


def test_researcher_retries_server_error_then_uses_reviewer_fallback(monkeypatch, caplog) -> None:
    from google.genai.errors import ServerError

    class UnavailableGemini:
        calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            raise ServerError(503, {"error": {"message": "temporary", "status": "UNAVAILABLE"}})

    gemini_model = UnavailableGemini()
    reviewer_model = DeterministicChatModel()
    models = {"researcher": gemini_model, "reviewer": reviewer_model}
    monkeypatch.setattr(nodes, "_create_model", lambda role: models[role])
    monkeypatch.setattr(nodes.settings, "researcher_llm_provider", "gemini")
    evidence = EvidenceRecord(
        evidence_id="evidence-1",
        task_id="task-1",
        claim="Source title",
        support="The retrieved source reports a measurable result.",
        source_url="https://example.org/report",
        source_title="Source report",
        source_type="web",
        retrieval_method="page_excerpt",
    )

    result = nodes._research_web_evidence(
        {
            "task_id": "task-1",
            "title": "Research topic",
            "objective": "Investigate the topic",
            "expected_evidence": "A measurable result",
        },
        [evidence],
    )

    assert gemini_model.calls == 2
    assert result[0].claim == evidence.support
    assert "configured reviewer model" in result[0].notes
    assert "retrying once" in caplog.text
    assert "using reviewer fallback" in caplog.text


def test_aggregation_keeps_distinct_excerpts_from_the_same_source() -> None:
    evidence = [
        EvidenceRecord(
            evidence_id="fy22",
            task_id="task-1",
            claim="NFAP data",
            support="FY2022 Infosys initial: 2,504; continuing: 5,688.",
            source_url="https://example.org/nfap",
        ),
        EvidenceRecord(
            evidence_id="fy23",
            task_id="task-2",
            claim="NFAP data",
            support="FY2023 Infosys initial: 2,171; continuing: 5,231.",
            source_url="https://example.org/nfap",
        ),
        EvidenceRecord(
            evidence_id="fy22-copy",
            task_id="task-3",
            claim="NFAP data",
            support="FY2022 Infosys initial: 2,504; continuing: 5,688.",
            source_url="https://example.org/nfap",
        ),
        EvidenceRecord(
            evidence_id="fy22-alt-rate",
            task_id="task-4",
            claim="NFAP data",
            support="FY2022 Infosys initial: 2,504; continuing: 5,688; denial rate: 41.3%.",
            source_url="https://example.org/nfap",
        ),
    ]

    unique = nodes._deduplicate_evidence(evidence)

    assert [item.evidence_id for item in unique] == ["fy22", "fy23", "fy22-alt-rate"]


def test_fact_checker_does_not_verify_search_snippets_or_local_excerpts(monkeypatch) -> None:
    class ClaimVerifier:
        def invoke(self, prompt, **kwargs):
            records = json.loads(prompt.split("Evidence records:\n", 1)[1])
            return type(
                "Response",
                (),
                {
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "evidence_id": item["evidence_id"],
                                    "status": "verified",
                                    "reasoning": "The model claimed verification.",
                                }
                                for item in records
                            ]
                        }
                    )
                },
            )()

    evidence = [
        EvidenceRecord(
            evidence_id="snippet",
            task_id="task-1",
            claim="Search result claim",
            support="A search result snippet.",
            retrieval_method="search_snippet",
        ),
        EvidenceRecord(
            evidence_id="local",
            task_id="task-1",
            claim="Local excerpt claim",
            support="A local vector excerpt.",
            source_type="vector",
            retrieval_method="vector_excerpt",
        ),
    ]
    monkeypatch.setattr(nodes, "_create_model", lambda role: ClaimVerifier())

    result = fact_checker_node({"evidence": evidence})

    assert [item.verification_status for item in result["evidence"]] == ["unverified", "unverified"]


def test_fact_checker_keeps_unverified_sources_unknown() -> None:
    result = fact_checker_node(
        {
            "evidence": [
                EvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-1",
                    claim="Market report",
                    support="Source excerpt.",
                    verification_status="unverified",
                )
            ]
        }
    )

    assert "task-1: unknown" in result["fact_check_results"][0]


def test_fact_checker_updates_evidence_from_llm_result(monkeypatch) -> None:
    class FactCheckingModel:
        def invoke(self, prompt, **kwargs):
            assert "Compare the supplied evidence records" in prompt
            assert "Lack of support is not proof of falsity." in prompt
            assert "Different periods or definitions are not" in prompt
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"results":[{"evidence_id":"evidence-1","status":"conflicting",'
                        '"reasoning":"The sources disagree."}]}'
                    )
                },
            )()

    evidence = EvidenceRecord(
        evidence_id="evidence-1",
        task_id="task-1",
        claim="Market report",
        support="Source excerpt.",
        verification_status="unverified",
    )
    monkeypatch.setattr(nodes, "_create_model", lambda role: FactCheckingModel())

    result = fact_checker_node({"evidence": [evidence]})

    assert evidence.verification_status == "unverified"
    assert result["evidence"][0].verification_status == "conflicting"
    assert result["evidence"][0].notes == "The sources disagree."
    assert result["fact_check_results"] == ["task-1: conflicting | Market report"]


def test_critic_cannot_pass_without_audited_claims(monkeypatch) -> None:
    class CriticModel:
        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                return type("Response", (), {"content": '{"claims":[]}'} )()
            assert "citation_accuracy" in prompt
            assert "research_completeness" in prompt
            assert "objectivity" in prompt
            assert "omitted" in prompt
            assert "denominators" in prompt
            assert "Do not call values contradictory" in prompt
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"citation_accuracy":0.91,"research_completeness":0.84,'
                        '"objectivity":0.88,"issues":[],"unsupported_claims":[],'
                        '"revision_instructions":[],"claim_audits":[]}'
                    )
                },
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: CriticModel())
    result = nodes.critic_node(
        {
            "research_question": "Assess the research question.",
            "report_draft": "A report with cited evidence.",
            "evidence": [],
            "revision_count": 0,
        }
    )

    evaluation = result["critic_evaluation"]
    assert not evaluation.passed
    assert evaluation.citation_accuracy == 0.0
    assert any("cannot be assessed" in issue for issue in evaluation.issues)
    assert evaluation.research_completeness == 0.84
    assert evaluation.objectivity == 0.88


def test_critic_returns_quote_validated_claim_audit(monkeypatch) -> None:
    from app.infrastructure.vector_store import VectorDocument

    evidence = EvidenceRecord(
        evidence_id="ev-1",
        task_id="task-1",
        claim="Telework productivity finding",
        support="A short retrieved excerpt.",
        source_id="report-1",
        source_chunk_id="chunk-4",
        source_url="https://example.org/report.pdf",
        source_title="Research report",
        source_type="vector",
        retrieval_method="vector_excerpt",
    )
    document = VectorDocument(
        "chunk-4",
        "Telework was associated with higher productivity.",
        {"page_number": 4, "chunk_index": 3},
    )

    class ClaimAuditCritic:
        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                claim = "Telework increased productivity. https://example.org/report.pdf"
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {"claim_id": "C1", "claim": claim, "critical": True}
                                ]
                            }
                        )
                    },
                )()
            if "Material claim batch " in prompt:
                assert '"audit_id": "ev-1"' in prompt
                assert '"page_number": 4' in prompt
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "claim_audits": [
                                    {
                                        "claim_id": "C1",
                                        "claim": "Telework increased productivity. https://example.org/report.pdf",
                                        "critical": True,
                                        "evidence_ids": ["ev-1"],
                                        "status": "supported",
                                        "excerpt": "Telework was associated with higher productivity.",
                                        "reasoning": "The retrieved passage supports the claim.",
                                    }
                                ]
                            }
                        )
                    },
                )()
            if "Claim verdicts to confirm:" in prompt:
                return _confirm_test_audits(prompt)
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"citation_accuracy":0.9,"research_completeness":0.9,'
                        '"objectivity":0.9,"issues":[],"unsupported_claims":[],'
                        '"revision_instructions":[],"claim_audits":[]}'
                    )
                },
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: ClaimAuditCritic())
    monkeypatch.setattr(nodes, "vector_source_documents", lambda source_id: [document])
    monkeypatch.setattr(
        nodes,
        "vector_search_source",
        lambda query, source_id, max_results: [document],
    )
    result = nodes.critic_node(
        {
            "research_question": "Assess telework productivity.",
            "report_draft": (
                "Telework increased productivity. https://example.org/report.pdf"
            ),
            "evidence": [evidence],
            "source_records": [],
            "revision_count": 0,
        }
    )

    audit = result["critic_evaluation"].claim_audits[0]
    assert audit.status == "supported"
    assert audit.evidence_ids == ["ev-1"]
    assert audit.retrieval_completeness == "complete"
    assert "https://example.org/report.pdf" in audit.source_references[0]
    assert result["critic_evaluation"].citation_accuracy == 1.0


def test_claim_audit_context_resolves_references_to_preceding_studies() -> None:
    report = (
        "Support agents handled 13.8% more inquiries per hour. "
        "A study of 5,000 agents found 15% higher productivity. "
        "These probably derive from the same study, so they are not independent."
    )
    context = nodes._claim_report_context(
        report, "These probably derive from the same study, so they are not independent."
    )

    assert "13.8% more inquiries" in context
    assert "5,000 agents" in context


def test_claim_confirmation_rejects_unrelated_quote_and_keeps_supported_quote() -> None:
    class ClaimConfirmationModel:
        def invoke(self, prompt, **kwargs):
            checks = json.loads(prompt.split("Claim verdicts to confirm:\n", 1)[1])
            assert "IT service desk calls fell 50%" in checks[0]["report_context"]
            return type(
                "Response",
                (),
                {"content": json.dumps({"results": [
                    {"claim_id": "C1", "verdict": "unconfirmed", "reasoning": "The quote concerns bank AI adoption."},
                    {"claim_id": "C2", "verdict": "confirmed", "reasoning": "The quote supports the claim."},
                ]})},
            )()

    claims = {
        "C1": {
            "report_context": "IT service desk calls fell 50%. It does not support a 55% HR/IT figure."
        },
        "C2": {"report_context": "The reported rate is 55%."},
    }
    audits = [
        ClaimAudit(
            claim_id="C1",
            claim="It does not support a 55% HR/IT figure.",
            critical=True,
            evidence_ids=["eba"],
            status="supported",
            excerpt="55% of banks use AI in consumer-facing processes.",
            reasoning="The passage includes a 55% figure.",
        ),
        ClaimAudit(
            claim_id="C2",
            claim="The reported rate is 55%.",
            evidence_ids=["eba"],
            status="supported",
            excerpt="55% of banks use AI in consumer-facing processes.",
            reasoning="The passage gives the rate.",
        ),
    ]
    context = [{
        "audit_id": "eba",
        "source_url": "https://example.org/eba",
        "retrieval_completeness": "partial",
        "excerpt": "55% of banks use AI in consumer-facing processes.",
    }]

    result = nodes._confirm_claim_audits(ClaimConfirmationModel(), audits, claims, context)

    assert [audit.status for audit in result] == ["insufficient_retrieval", "supported"]
    assert "bank AI adoption" in result[0].reasoning


def test_merged_audit_keeps_quote_with_its_own_source() -> None:
    claims = [{"claim_id": "C1", "claim": "A result.", "critical": True}]
    audits = [
        ClaimAudit(
            claim_id="C1", claim="A result.", critical=True, status="supported",
            excerpt="Supporting passage.", reasoning="Supported.",
            source_references=["https://example.org/support"],
        ),
        ClaimAudit(
            claim_id="C1", claim="A result.", critical=True, status="insufficient_retrieval",
            excerpt="", reasoning="This source has no relevant passage.",
            source_references=["https://example.org/unrelated"],
        ),
    ]

    merged = nodes._merge_material_claim_audits(claims, audits)[0]

    assert merged.status == "supported"
    assert merged.excerpt == "Supporting passage."
    assert merged.source_references == ["https://example.org/support"]
    assert merged.reasoning == "Supported."


def test_critic_citation_score_uses_audited_claims_not_model_score(monkeypatch) -> None:
    claims = [
        {"claim_id": "C1", "claim": "First result.", "critical": False},
        {"claim_id": "C2", "claim": "Second result.", "critical": False},
        {"claim_id": "C3", "claim": "Third result.", "critical": False},
    ]
    audits = [
        ClaimAudit(claim_id="C1", claim="First result.", status="supported", reasoning="Supported."),
        ClaimAudit(claim_id="C2", claim="Second result.", status="insufficient_retrieval", reasoning="Unverified."),
        ClaimAudit(claim_id="C3", claim="Third result.", status="contradicted", reasoning="Contradicted."),
    ]

    class ScoreModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {"content": json.dumps({
                    "citation_accuracy": 0.99,
                    "research_completeness": 0.8,
                    "objectivity": 0.8,
                    "issues": [],
                    "unsupported_claims": [],
                    "revision_instructions": [],
                    "claim_audits": [],
                })},
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: ScoreModel())
    monkeypatch.setattr(nodes, "_extract_material_claims", lambda model, question, report: claims)
    monkeypatch.setattr(nodes, "_merge_material_claim_audits", lambda claims, results: audits)
    result = nodes.critic_node({
        "research_question": "Assess the results.",
        "report_draft": "First result. Second result. Third result.",
        "evidence": [],
    })["critic_evaluation"]

    assert result.citation_accuracy == pytest.approx(1 / 3)
    assert result.overall_score == 0.64
    assert not result.passed


def test_critic_uses_context_and_rejects_an_unrelated_55_percent_quote(monkeypatch) -> None:
    source_url = "https://example.org/cosupport"
    first_claim = f"CoSupport reported IT service desk calls fell 50% ([source]({source_url}))."
    second_claim = "That source does not support a 55% HR/IT figure."
    source_text = (
        "IT service desk calls fell 50%. "
        "The source also says 55% of banks use AI in consumer-facing processes."
    )
    evidence = EvidenceRecord(
        evidence_id="ev-1",
        task_id="task-1",
        claim="Service desk and adoption",
        support=source_text,
        source_id=nodes.web_source_id(source_url),
        source_url=source_url,
        source_type="web",
        retrieval_method="page_excerpt",
    )

    class ReviewedCritic:
        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                content = {"claims": [
                    {"claim_id": "C1", "claim": first_claim, "critical": True},
                    {"claim_id": "C2", "claim": second_claim, "critical": True},
                ]}
            elif "Material claim batch " in prompt:
                claims_text = prompt.split("Material claim batch ", 1)[1].split(":\n", 1)[1]
                claim_batch, _ = json.JSONDecoder().raw_decode(claims_text)
                assert first_claim in claim_batch[1]["report_context"]
                batch = json.loads(prompt.rsplit("Source batch ", 1)[1].split(":\n", 1)[1])
                passage = next(item for item in batch if source_text in item.get("excerpt", ""))
                content = {"claim_audits": [
                    {
                        "claim_id": "C1", "evidence_ids": [passage["audit_id"]],
                        "status": "supported", "excerpt": "IT service desk calls fell 50%.",
                        "reasoning": "The excerpt states the figure.",
                    },
                    {
                        "claim_id": "C2", "evidence_ids": [passage["audit_id"]],
                        "status": "supported",
                        "excerpt": "55% of banks use AI in consumer-facing processes.",
                        "reasoning": "The excerpt contains 55%.",
                    },
                ]}
            elif "Claim verdicts to confirm:" in prompt:
                checks = json.loads(prompt.split("Claim verdicts to confirm:\n", 1)[1])
                assert first_claim in checks[1]["report_context"]
                content = {"results": [
                    {"claim_id": "C1", "verdict": "confirmed", "reasoning": "The service desk figure matches."},
                    {"claim_id": "C2", "verdict": "unconfirmed", "reasoning": "The 55% concerns bank AI adoption."},
                ]}
            else:
                content = {
                    "research_completeness": 0.8, "objectivity": 0.8, "issues": [],
                    "unsupported_claims": [], "revision_instructions": [], "claim_audits": [],
                }
            return type("Response", (), {"content": json.dumps(content)})()

    monkeypatch.setattr(nodes, "_create_model", lambda role: ReviewedCritic())
    evaluation = nodes.critic_node({
        "research_question": "Assess service desk results.",
        "report_draft": f"{first_claim} {second_claim}",
        "evidence": [evidence],
        "source_records": [{
            "source_id": evidence.source_id,
            "source_url": source_url,
            "text": source_text,
        }],
    })["critic_evaluation"]

    assert [audit.status for audit in evaluation.claim_audits] == [
        "supported", "insufficient_retrieval"
    ]
    assert evaluation.citation_accuracy == 0.5
    assert not evaluation.passed


def test_critic_blocks_pass_when_a_critical_claim_has_no_audit_evidence(monkeypatch) -> None:
    report_claim = "The rate increased by 15%."

    class UnresolvedCritic:
        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "claim_id": "C1",
                                        "claim": report_claim,
                                        "critical": True,
                                    }
                                ]
                            }
                        )
                    },
                )()
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"citation_accuracy":0.99,"research_completeness":0.99,'
                        '"objectivity":0.99,"issues":[],"unsupported_claims":[],'
                        '"revision_instructions":[],"claim_audits":[]}'
                    )
                },
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: UnresolvedCritic())
    result = nodes.critic_node(
        {
            "research_question": "Assess the measured rate.",
            "report_draft": report_claim,
            "evidence": [],
            "source_records": [],
        }
    )

    evaluation = result["critic_evaluation"]
    assert len(evaluation.claim_audits) == 1
    assert evaluation.claim_audits[0].status == "insufficient_retrieval"
    assert evaluation.claim_audits[0].critical
    assert not evaluation.passed
    assert any("Critical claim unresolved" in issue for issue in evaluation.issues)


def test_critic_normalizes_unknown_evidence_reference_and_blocks_critical_claim(monkeypatch) -> None:
    report_claim = "The rate increased by 15%. https://example.org/report"
    evidence = EvidenceRecord(
        evidence_id="ev-known",
        task_id="task-1",
        claim="Report result",
        support="The rate increased by 15%.",
        source_id="web:https://example.org/report",
        source_url="https://example.org/report",
        source_title="Report",
        source_type="web",
        retrieval_method="page_excerpt",
    )

    class InvalidReferenceCritic:
        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "claims": [
                                    {
                                        "claim_id": "C1",
                                        "claim": report_claim,
                                        "critical": True,
                                    }
                                ]
                            }
                        )
                    },
                )()
            if "Material claim batch " in prompt:
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps(
                            {
                                "claim_audits": [
                                    {
                                        "claim_id": "C1",
                                        "claim": report_claim,
                                        "critical": True,
                                        "evidence_ids": ["unknown-id"],
                                        "status": "supported",
                                        "excerpt": "The rate increased by 15%.",
                                        "reasoning": "This is not a supplied audit reference.",
                                    }
                                ]
                            }
                        )
                    },
                )()
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"citation_accuracy":0.99,"research_completeness":0.99,'
                        '"objectivity":0.99,"issues":[],"unsupported_claims":[],'
                        '"revision_instructions":[],"claim_audits":[]}'
                    )
                },
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: InvalidReferenceCritic())
    result = nodes.critic_node(
        {
            "research_question": "Assess the measured rate.",
            "report_draft": report_claim,
            "evidence": [evidence],
            "source_records": [
                {
                    "source_id": evidence.source_id,
                    "source_url": evidence.source_url,
                    "source_title": evidence.source_title,
                    "text": evidence.support,
                }
            ],
        }
    )

    audit = result["critic_evaluation"].claim_audits[0]
    assert audit.status == "insufficient_retrieval"
    assert audit.evidence_ids == []
    assert audit.critical
    assert not result["critic_evaluation"].passed


def test_critic_batches_more_than_twelve_sources_without_skipping_them(monkeypatch) -> None:
    from app.graph.evidence_audit import web_source_id

    class BatchedCritic:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt, **kwargs):
            self.prompts.append(prompt)
            if "Create a complete inventory of every material" in prompt:
                report = prompt.split("\nReport:\n", 1)[1]
                claims = [
                    {
                        "claim_id": f"C{claim_index}",
                        "claim": line,
                        "critical": True,
                    }
                    for claim_index, line in enumerate(
                        (line for line in report.splitlines() if line.startswith("Finding ")),
                        start=1,
                    )
                ]
                return type(
                    "Response",
                    (),
                    {"content": json.dumps({"claims": claims})},
                )()
            if "Material claim batch " in prompt:
                claim_part = prompt.split("Material claim batch ", 1)[1]
                claims_text = claim_part.split(":\n", 1)[1]
                claims, _ = json.JSONDecoder().raw_decode(claims_text)
                source_part = prompt.rsplit("Source batch ", 1)[1].split(":\n", 1)[1]
                context = json.loads(source_part)
                claim_audits = []
                for claim in claims:
                    index = int(claim["claim_id"][1:]) - 1
                    passage = next(
                        (
                            item
                            for item in context
                            if str(item.get("source_id", "")).endswith(f"source-{index}")
                            and item.get("excerpt")
                        ),
                        None,
                    )
                    if passage:
                        claim_audits.append(
                            {
                                "claim_id": claim["claim_id"],
                                "evidence_ids": [passage["audit_id"]],
                                "status": "supported",
                                "excerpt": passage["excerpt"],
                                "reasoning": "The supplied passage supports it.",
                            }
                        )
                return type(
                    "Response",
                    (),
                    {
                        "content": json.dumps({"claim_audits": claim_audits})
                    },
                )()
            if "Claim verdicts to confirm:" in prompt:
                return _confirm_test_audits(prompt)
            if "Audited material claims; every inventory item has an audit result:" in prompt:
                content = (
                    '{"citation_accuracy":0.9,"research_completeness":0.9,'
                    '"objectivity":0.9,"issues":[],"unsupported_claims":[],"revision_instructions":[],"claim_audits":[]}'
                )
                return type("Response", (), {"content": content})()
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"citation_accuracy":0.9,"research_completeness":0.9,'
                        '"objectivity":0.9,"issues":[],"unsupported_claims":[],'
                        '"revision_instructions":[],"claim_audits":[]}'
                    )
                },
            )()

    model = BatchedCritic()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)
    evidence = []
    source_records = []
    report_lines = []
    for index in range(13):
        source_url = f"https://example.org/source-{index}"
        source_id = web_source_id(source_url)
        support = f"Finding {index} is supported by this retrieved source excerpt."
        evidence.append(
            EvidenceRecord(
                evidence_id=f"ev-{index}",
                task_id="task-1",
                claim=f"Finding {index}",
                support=support,
                source_id=source_id,
                source_url=source_url,
                source_title=f"Source {index}",
                source_type="web",
                retrieval_method="page_excerpt",
            )
        )
        source_records.append(
            {
                "source_id": source_id,
                "source_url": source_url,
                "source_title": f"Source {index}",
                "text": support,
            }
        )
        report_lines.append(f"Finding {index}. {source_url}")

    result = nodes.critic_node(
        {
            "research_question": "Assess the supplied findings.",
            "report_draft": "\n\n".join(report_lines),
            "evidence": evidence,
            "source_records": source_records,
        }
    )

    batch_prompts = [prompt for prompt in model.prompts if "Material claim batch " in prompt]
    assert len(batch_prompts) == 8
    assert [
        "Source batch 1 of 2" in prompt
        for prompt in batch_prompts
    ] == [True, False] * 4
    assert [
        len(json.JSONDecoder().raw_decode(prompt.split("Material claim batch ", 1)[1].split(":\n", 1)[1])[0])
        for prompt in batch_prompts
    ] == [4, 4, 4, 4, 4, 4, 1, 1]
    assert len(result["critic_evaluation"].claim_audits) == 13
    assert all(audit.status == "supported" for audit in result["critic_evaluation"].claim_audits)
    assert result["critic_evaluation"].passed


def test_report_writer_receives_source_linked_claim_audit_excerpt(monkeypatch) -> None:
    class CapturingWriter:
        def invoke(self, prompt, **kwargs):
            assert "Retrieved source passage: Customer wait times fell from 48 to 12 hours." in prompt
            assert "Research report — p. 4 — https://example.org/report.pdf" in prompt
            assert "Do not reproduce reviewer notes" in prompt
            return type("Response", (), {"content": "# Revised report"})()

    monkeypatch.setattr(nodes, "_create_model", lambda role: CapturingWriter())
    evaluation = CriticEvaluation(
        claim_audits=[
            ClaimAudit(
                claim="Customer wait times improved.",
                evidence_ids=["chunk:chunk-4"],
                status="contradicted",
                excerpt="Customer wait times fell from 48 to 12 hours.",
                reasoning="The draft omitted the measured change.",
                retrieval_completeness="complete",
                source_references=[
                    "Research report — p. 4 — https://example.org/report.pdf"
                ],
            )
        ]
    )

    result = write_report_node(
        {
            "research_question": "Assess service outcomes.",
            "critic_evaluation": evaluation,
            "revision_feedback": ["Include the measured wait-time change."],
            "evidence": [],
        }
    )

    assert result["report_draft"] == "# Revised report"


def test_writer_evidence_matrix_preserves_task_source_and_conflict_metadata() -> None:
    record = EvidenceRecord(
        evidence_id="ev-1",
        task_id="task-1",
        claim="A measured outcome.",
        support="The outcome changed during the reporting period.",
        source_url="https://example.org/report",
        source_title="Outcome report",
        publication_date="2025",
        page_number=4,
        evidence_scope="banking",
        evidence_basis="primary_empirical",
        retrieval_method="page_excerpt",
        retrieval_channels=["web"],
        confidence=0.85,
        verification_status="conflicting",
        notes="Potential conflict with evidence IDs: ev-2.",
    )

    matrix = nodes._build_writer_evidence_matrix(
        [record],
        [ResearchTask(
            task_id="task-1",
            title="Customer outcomes",
            objective="Assess measured results",
            expected_evidence="Measured outcomes",
        )],
    )

    assert matrix[0]["task_title"] == "Customer outcomes"
    assert matrix[0]["evidence"][0]["source_title"] == "Outcome report"
    assert matrix[0]["evidence"][0]["page_number"] == 4
    assert matrix[0]["evidence"][0]["evidence_scope"] == "banking"
    assert matrix[0]["evidence"][0]["evidence_basis"] == "primary_empirical"
    assert matrix[0]["evidence"][0]["notes"].endswith("ev-2.")


def test_material_claim_inventory_cannot_omit_cited_or_numeric_claims() -> None:
    with pytest.raises(ValueError, match="omitted a cited or numeric"):
        nodes._validate_material_claims(
            {"claims": []},
            "The measured outcome improved by 15%.",
        )


def test_material_claim_inventory_does_not_count_another_sentences_number() -> None:
    with pytest.raises(ValueError, match="sentence 2"):
        nodes._validate_material_claims(
            {"claims": [
                {"claim_id": "C1", "claim": "Bank A reported 15%.", "critical": True}
            ]},
            "Bank A reported 15%.\n\nBank B reported 15%.",
        )


def test_material_claim_inventory_rejects_a_bare_number_as_the_claim() -> None:
    with pytest.raises(ValueError, match="sentence 1"):
        nodes._validate_material_claims(
            {"claims": [{"claim_id": "C1", "claim": "15%", "critical": False}]},
            "Bank A reduced costs by 15%.",
        )


def test_material_claim_inventory_recovers_omitted_sentences_after_retry(caplog) -> None:
    class IncompleteInventoryModel:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return type(
                "Response",
                (),
                {"content": json.dumps({"claims": [
                    {"claim_id": "C1", "claim": "The first result improved by 15%.", "critical": True}
                ]})},
            )()

    model = IncompleteInventoryModel()
    missing = "A second result fell by 8% ([report](https://example.org/report))."
    report = f"The first result improved by 15%.\n\n{missing}"

    claims = nodes._extract_material_claims(model, "Assess the results.", report)

    assert len(model.prompts) == 2
    assert "sentence 2" in model.prompts[1]
    assert claims == [
        {"claim_id": "C1", "claim": "The first result improved by 15%.", "critical": True},
        {"claim_id": "C2", "claim": missing, "critical": True},
    ]
    assert "added 1 omitted sentence" in caplog.text


def test_material_claim_inventory_uses_corrected_retry_without_fallback(caplog) -> None:
    class CorrectingInventoryModel:
        def __init__(self):
            self.attempts = 0

        def invoke(self, prompt, **kwargs):
            self.attempts += 1
            claims = []
            if self.attempts == 2:
                claims.append(
                    {"claim_id": "C1", "claim": "The result improved by 15%.", "critical": True}
                )
            return type("Response", (), {"content": json.dumps({"claims": claims})})()

    model = CorrectingInventoryModel()
    claims = nodes._extract_material_claims(
        model, "Assess the result.", "The result improved by 15%."
    )

    assert model.attempts == 2
    assert len(claims) == 1
    assert "added 1 omitted sentence" not in caplog.text


def test_material_claim_inventory_does_not_repair_invalid_claims() -> None:
    class InvalidInventoryModel:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {"content": json.dumps({"claims": [
                    {"claim_id": "C1", "claim": "Invented 15% claim.", "critical": True}
                ]})},
            )()

    with pytest.raises(ValueError, match="copied verbatim"):
        nodes._extract_material_claims(
            InvalidInventoryModel(), "Assess the results.", "The outcome improved by 15%."
        )


def test_critic_does_not_pass_recovered_claim_without_evidence(monkeypatch) -> None:
    class IncompleteInventoryCritic:
        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                content = '{"claims":[]}'
            else:
                content = (
                    '{"citation_accuracy":0.9,"research_completeness":0.9,'
                    '"objectivity":0.9,"issues":[],"unsupported_claims":[],'
                    '"revision_instructions":[],"claim_audits":[]}'
                )
            return type("Response", (), {"content": content})()

    monkeypatch.setattr(nodes, "_create_model", lambda role: IncompleteInventoryCritic())
    evaluation = nodes.critic_node({
        "research_question": "Assess the result.",
        "report_draft": "The result improved by 15%.",
        "evidence": [],
    })["critic_evaluation"]

    assert not evaluation.passed
    assert evaluation.claim_audits[0].claim == "The result improved by 15%."
    assert evaluation.claim_audits[0].critical
    assert evaluation.claim_audits[0].status == "insufficient_retrieval"


def test_material_claim_inventory_batches_long_reports_and_keeps_all_claims() -> None:
    class BatchedInventoryModel:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt, **kwargs):
            self.prompts.append(prompt)
            report_batch = prompt.split("\nReport:\n", 1)[1]
            claims = [
                {
                    "claim_id": f"C{index}",
                    "claim": sentence,
                    "critical": True,
                }
                for index, sentence in enumerate(report_batch.splitlines(), start=1)
            ]
            return type(
                "Response",
                (),
                {"content": json.dumps({"claims": claims})},
            )()

    report = "\n\n".join(
        f"Finding {index} is an independently verifiable claim."
        for index in range(45)
    )
    model = BatchedInventoryModel()

    claims = nodes._extract_material_claims(model, "Assess the findings.", report)

    assert len(model.prompts) == 6
    assert all(
        len(prompt.split("\nReport:\n", 1)[1].splitlines()) <= 8
        for prompt in model.prompts
    )
    assert len(claims) == 45
    assert [claim["claim_id"] for claim in claims] == [
        f"C{index}" for index in range(1, 46)
    ]
    assert claims[-1]["claim"] == "Finding 44 is an independently verifiable claim."


def test_critic_replaces_stale_revision_feedback_and_clears_it_on_pass(monkeypatch) -> None:
    class TwoPassCritic:
        responses = iter(
            [
                (
                    '{"citation_accuracy":0.4,"research_completeness":0.4,'
                    '"objectivity":0.4,"issues":["Add the missing measure."],'
                    '"unsupported_claims":[],"revision_instructions":["Use the measured values."],'
                    '"claim_audits":[]}'
                ),
                (
                    '{"citation_accuracy":0.9,"research_completeness":0.9,'
                    '"objectivity":0.9,"issues":[],"unsupported_claims":[],'
                    '"revision_instructions":[],"claim_audits":[]}'
                ),
            ]
        )

        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                return type("Response", (), {"content": '{"claims":[]}'} )()
            return type("Response", (), {"content": next(self.responses)})()

    monkeypatch.setattr(nodes, "_create_model", lambda role: TwoPassCritic())
    monkeypatch.setattr(
        nodes,
        "_extract_material_claims",
        lambda model, question, report: [
            {"claim_id": "C1", "claim": report, "critical": False}
        ],
    )
    monkeypatch.setattr(
        nodes,
        "_merge_material_claim_audits",
        lambda claims, audits: [
            ClaimAudit(
                claim_id="C1", claim=claims[0]["claim"], status="supported", reasoning="Test evidence."
            )
        ],
    )
    state = {
        "research_question": "Assess the research question.",
        "report_draft": "A report with cited evidence.",
        "evidence": [],
        "revision_feedback": ["Old instruction that no longer applies."],
        "revision_count": 1,
    }

    failed = nodes.critic_node(state)
    assert failed["revision_feedback"] == [
        "Use the measured values.",
        "Add the missing measure.",
    ]

    passed = nodes.critic_node(
        {
            **state,
            "report_draft": "Revised report.",
            "revision_feedback": failed["revision_feedback"],
            "revision_count": 2,
        }
    )
    assert passed["critic_evaluation"].passed
    assert passed["revision_feedback"] == []


def test_critic_retries_malformed_json_with_validation_error(monkeypatch) -> None:
    class CorrectingCriticModel:
        calls = 0

        def invoke(self, prompt, **kwargs):
            if "Create a complete inventory of every material" in prompt:
                return type("Response", (), {"content": '{"claims":[]}'} )()
            self.calls += 1
            if self.calls == 1:
                return type("Response", (), {"content": '{"citation_accuracy":0.8'})()
            assert "previous response was invalid" in prompt
            assert "Expecting" in prompt
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"citation_accuracy":0.9,"research_completeness":0.9,'
                        '"objectivity":0.9,"issues":[],"unsupported_claims":[],'
                        '"revision_instructions":[],"claim_audits":[]}'
                    )
                },
            )()

    model = CorrectingCriticModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)

    result = nodes.critic_node(
        {
            "research_question": "Assess the research question.",
            "report_draft": "A report with cited evidence.",
            "evidence": [],
            "revision_count": 0,
        }
    )

    assert model.calls == 2
    assert not result["critic_evaluation"].passed
    assert result["critic_evaluation"].citation_accuracy == 0.0


def test_report_writer_creates_markdown(monkeypatch) -> None:
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    state = {
        "research_question": "Should we enter a new AI workflow market?",
        "evidence": [],
    }
    result = write_report_node(state)
    assert "# Executive Research Report" in result["report_draft"]
    assert "Research Question" in result["report_draft"]


def test_report_writer_retries_truncated_output_with_source_context(monkeypatch) -> None:
    class TruncatedWriterModel:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if len(self.calls) == 1:
                return type(
                    "Response",
                    (),
                    {"content": "# Partial report", "response_metadata": {"stop_reason": "max_tokens"}},
                )()
            return type(
                "Response",
                (),
                {"content": "# Complete report", "response_metadata": {"stop_reason": "end_turn"}},
            )()

    model = TruncatedWriterModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)
    result = write_report_node(
        {
            "research_question": "Will the AI industry grow?",
            "revision_feedback": ["Explain the unresolved valuation figures."],
            "evidence": [
                EvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-1",
                    claim="AI revenue grew.",
                    support=(
                        ("Background context on reported labor statistics. " * 70)
                        + "Infosys FY2025: 74% denial rate (~7,800 petitions).\n"
                        + ("Additional supporting context. " * 100)
                        + "DO_NOT_INCLUDE_TAIL"
                    ),
                    source_title="Market report",
                    publication_date="2026-11-10",
                    source_url="https://example.org/report",
                    source_type="web",
                )
            ],
        }
    )

    assert len(model.calls) == 2
    assert [call[1]["max_tokens"] for call in model.calls] == [6000, 6000]
    assert "Market report" in model.calls[0][0]
    assert "2026-11-10" in model.calls[0][0]
    assert "Explain the unresolved valuation figures." in model.calls[0][0]
    assert "reached the output limit" in model.calls[1][0]
    assert "no more than 650 words" in model.calls[1][0]
    assert "never say a source lacks data" in model.calls[0][0]
    assert "bounded estimate" in model.calls[1][0]
    assert "cohort approval rate" in model.calls[0][0]
    assert "Infosys FY2025: 74% denial rate (~7,800 petitions)." in model.calls[0][0]
    assert "Infosys FY2025: 74% denial rate (~7,800 petitions)." in model.calls[1][0]
    assert "DO_NOT_INCLUDE_TAIL" not in model.calls[0][0]
    assert "DO_NOT_INCLUDE_TAIL" not in model.calls[1][0]
    assert result["report_draft"] == "# Complete report"


def test_report_writer_uses_brief_fallback_after_repeated_truncation(monkeypatch) -> None:
    class TruncatedWriterModel:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            if len(self.calls) < 3:
                stop_reason = "max_tokens"
                content = "# Truncated report"
            else:
                stop_reason = "end_turn"
                content = "# Brief report\n\n## Answer\nA bounded conclusion."
            return type(
                "Response",
                (),
                {
                    "content": content,
                    "response_metadata": {"stop_reason": stop_reason},
                },
            )()

    model = TruncatedWriterModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)

    result = write_report_node(
        {
            "research_question": "Assess the five-year outlook.",
            "evidence": [
                EvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-1",
                    claim="Market data",
                    support="FY2025 recorded 74% approval from 7,800 petitions.",
                    source_title="Market report",
                    source_url="https://example.org/report",
                )
            ],
        }
    )

    assert [call[1]["max_tokens"] for call in model.calls] == [6000, 6000, 6000]
    assert "no more than 250 words" in model.calls[2][0]
    assert "7,800 petitions" in model.calls[2][0]
    assert result["report_draft"].startswith("# Brief report")


def test_report_writer_bounds_evidence_and_feedback_in_final_fallback(monkeypatch) -> None:
    class TruncatedWriterModel:
        def __init__(self):
            self.calls = []

        def invoke(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            stop_reason = "max_tokens" if len(self.calls) < 3 else "end_turn"
            content = "# Truncated" if stop_reason == "max_tokens" else "# Bounded brief"
            return type(
                "Response",
                (),
                {
                    "content": content,
                    "response_metadata": {"stop_reason": stop_reason},
                },
            )()

    model = TruncatedWriterModel()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)
    evidence = [
        EvidenceRecord(
            evidence_id=f"evidence-{index}",
            task_id=f"task-{index % 4}",
            claim=f"Claim index {index}",
            support=f"Excerpt index {index}.",
            source_url=f"https://example.org/{index}",
        )
        for index in range(24)
    ]

    result = write_report_node(
        {
            "research_question": "Assess the research question.",
            "revision_feedback": [f"Reviewer note {index}" for index in range(21)],
            "evidence": evidence,
        }
    )

    brief_prompt = model.calls[2][0]
    assert brief_prompt.count("Evidence ID:") == 12
    assert "Excerpt index 11." in brief_prompt
    assert "Excerpt index 12." not in brief_prompt
    assert "Reviewer note 7" in brief_prompt
    assert "Reviewer note 8" not in brief_prompt
    assert "selected evidence is not exhaustive" in brief_prompt
    assert result["report_draft"] == "# Bounded brief"


def test_report_writer_retries_when_provider_returns_no_text_at_token_limit(monkeypatch) -> None:
    class EmptyTruncatedWriter:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise llm_factory.LLMOutputTokenLimitError("Token limit without text.")
            return type(
                "Response",
                (),
                {"content": "# Complete", "response_metadata": {"stop_reason": "end_turn"}},
            )()

    model = EmptyTruncatedWriter()
    monkeypatch.setattr(nodes, "_create_model", lambda role: model)

    result = write_report_node({"research_question": "Assess the results.", "evidence": []})

    assert model.calls == 2
    assert result["report_draft"] == "# Complete"


def test_report_writer_preserves_reviewed_draft_when_all_revisions_truncate(monkeypatch, caplog) -> None:
    class AlwaysTruncatedWriter:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {"content": "# Partial", "response_metadata": {"stop_reason": "max_tokens"}},
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: AlwaysTruncatedWriter())
    evaluation = CriticEvaluation(overall_score=0.7, passed=False)
    state = {
        "research_question": "Assess the results.",
        "evidence": [],
        "revision_count": 3,
        "report_draft": "# Previous complete report",
        "critic_evaluation": evaluation,
    }

    update = write_report_node(state)

    assert update["workflow_status"] == "report_revision_truncated"
    assert update["final_approval_status"] == "needs_human_review"
    assert "report_draft" not in update
    assert "critic_evaluation" not in update
    assert "retaining the previous draft and evaluation" in caplog.text


def test_report_writer_fails_explicitly_when_no_reviewed_draft_exists(monkeypatch) -> None:
    class AlwaysTruncatedWriter:
        def invoke(self, prompt, **kwargs):
            return type(
                "Response",
                (),
                {"content": "# Partial", "response_metadata": {"stop_reason": "max_tokens"}},
            )()

    monkeypatch.setattr(nodes, "_create_model", lambda role: AlwaysTruncatedWriter())
    with pytest.raises(RuntimeError, match="output token limit"):
        write_report_node({"research_question": "Assess the results.", "evidence": []})


def test_report_writer_does_not_hide_other_model_errors(monkeypatch) -> None:
    class FailedWriter:
        def invoke(self, prompt, **kwargs):
            raise RuntimeError("Provider unavailable.")

    monkeypatch.setattr(nodes, "_create_model", lambda role: FailedWriter())
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        write_report_node({"research_question": "Assess the results.", "evidence": []})
