from __future__ import annotations

from app.infrastructure.vector_store import VectorDocument
from app.tools import ingest_knowledge


def test_chunk_text_normalizes_and_overlaps_chunks() -> None:
    chunks = ingest_knowledge._chunk_text(
        "alpha beta gamma delta epsilon zeta ﬁ",
        chunk_size=16,
        overlap=5,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 16 for chunk in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()
    assert "".join(chunks).replace(" ", "").count("fi") == 1


def test_documents_from_pdf_keep_source_and_page_provenance(monkeypatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakeReader:
        def __init__(self, _pdf_bytes, strict=False) -> None:
            self.pages = [FakePage("Page one content."), FakePage("Page two content.")]

    monkeypatch.setattr(ingest_knowledge, "PdfReader", FakeReader)
    source = {
        "source_id": "test-report",
        "title": "Test report",
        "publisher": "Test publisher",
        "publication_date": "2025-01",
        "pdf_url": "https://example.org/report.pdf",
        "landing_page_url": "https://example.org/report",
        "focus": "Test source metadata",
    }

    documents = ingest_knowledge._documents_from_pdf(b"%PDF-test", source)

    assert [document.id for document in documents] == [
        "test-report-p1-c0",
        "test-report-p2-c0",
    ]
    assert documents[0].metadata["source_url"] == source["pdf_url"]
    assert documents[0].metadata["source_domain"] == "example.org"
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2
    assert all(document.metadata["domain"] == "generative_ai_banking" for document in documents)


def test_research_worker_uses_local_sources_before_web_fallback(monkeypatch) -> None:
    from app.graph import nodes
    from fakes import DeterministicChatModel

    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: [
            VectorDocument(
                "chunk-1",
                "Bank use case evidence.",
                {
                    "source_id": "report-1",
                    "source_title": "Banking AI report",
                    "source_url": "https://example.org/bank.pdf",
                    "page_number": 12,
                    "retrieval_distance": 0.3,
                },
            ),
            VectorDocument(
                "chunk-2",
                "Another independent report supports the banking use case.",
                {
                    "source_id": "report-2",
                    "source_title": "Financial services AI report",
                    "source_url": "https://example.net/finance.pdf",
                    "page_number": 8,
                    "retrieval_distance": 0.4,
                },
            ),
        ],
    )

    def unexpected_web_search(*_args, **_kwargs):
        raise AssertionError("Web search should not run when multiple local sources are relevant.")

    monkeypatch.setattr(nodes, "web_search", unexpected_web_search)
    result = nodes.research_worker_node(
        {
            "current_task": {
                "task_id": "task-1",
                "title": "Banking use case",
                "objective": "Assess generative AI use in banking",
                "expected_evidence": "Documented uses",
            }
        }
    )

    assert len(result["findings"]) == 2
    assert all(item.source_type == "vector" for item in result["findings"])
    assert {item.page_number for item in result["findings"]} == {8, 12}


def test_research_worker_falls_back_when_vector_matches_are_weak(monkeypatch) -> None:
    from app.graph import nodes
    from fakes import DeterministicChatModel

    web_queries = []
    monkeypatch.setattr(nodes, "_create_model", lambda role: DeterministicChatModel())
    monkeypatch.setattr(
        nodes,
        "vector_search",
        lambda query, max_results: [
            VectorDocument(
                "unrelated-1",
                "Unrelated local report.",
                {"source_id": "unrelated-1", "retrieval_distance": 0.9},
            ),
            VectorDocument(
                "unrelated-2",
                "Another unrelated local report.",
                {"source_id": "unrelated-2", "retrieval_distance": 0.95},
            ),
        ],
    )
    monkeypatch.setattr(
        nodes,
        "web_search",
        lambda query, max_results: (
            web_queries.append(query)
            or [
                {
                    "title": "Banking GenAI overview",
                    "url": "https://example.org/genai-banking",
                    "snippet": "Evidence about generative AI adoption by banks.",
                }
            ]
        ),
    )

    result = nodes.research_worker_node(
        {
            "current_task": {
                "task_id": "task-1",
                "title": "Banking GenAI use cases",
                "objective": "Assess generative AI use in banking",
                "expected_evidence": "Documented uses",
            }
        }
    )

    assert len(web_queries) == 1
    assert len(result["findings"]) == 1
    assert result["findings"][0].source_type == "web"
