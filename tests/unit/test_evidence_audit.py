from app.graph.evidence_audit import (
    build_audit_context,
    citation_coverage_issues,
    normalize_claim_audits,
    web_source_id,
)
from app.infrastructure.vector_store import ChromaVectorStore, VectorDocument
from app.models.research import EvidenceRecord


def test_audit_searches_and_groups_all_chunks_for_a_cited_local_source() -> None:
    evidence = [
        EvidenceRecord(
            evidence_id="ev-first",
            task_id="task-1",
            claim="Internal report",
            support="The report describes working arrangements.",
            source_id="report-1",
            source_chunk_id="chunk-1",
            source_type="vector",
            source_url="https://example.org/report.pdf",
        ),
        EvidenceRecord(
            evidence_id="ev-later",
            task_id="task-1",
            claim="Internal report",
            support="Telework was associated with higher productivity.",
            source_id="report-1",
            source_chunk_id="chunk-2",
            source_type="vector",
            source_url="https://example.org/report.pdf",
        ),
    ]
    later_passage = "Telework was associated with higher productivity."
    documents = [
        VectorDocument("chunk-1", "The report describes working arrangements.", {"page_number": 1}),
        VectorDocument(
            "chunk-2",
            ("An unrelated discussion. " * 90) + later_passage,
            {"page_number": 4},
        ),
    ]

    def search_source(_query, _source_id, max_results):
        assert max_results == 2
        return [documents[1]]

    context = build_audit_context(
        "Telework increased productivity. https://example.org/report.pdf",
        evidence,
        [],
        lambda _source_id: documents,
        search_source,
    )

    source_entries = [item for item in context if item["source_id"] == "report-1"]
    assert len([item for item in source_entries if item["audit_id"].startswith("source:")]) == 1
    assert any(item["audit_id"] == "ev-later" and item["page_number"] == 4 for item in source_entries)
    assert any(later_passage in item["excerpt"] for item in source_entries)
    assert source_entries[0]["searched_chunk_count"] == 2
    assert source_entries[0]["retrieval_completeness"] == "complete"


def test_report_acronym_and_page_citation_select_the_matching_source_page() -> None:
    source_id = "fsb-ai-financial-stability-2024"
    source_url = "https://www.fsb.org/uploads/P14112024.pdf"
    evidence = [
        EvidenceRecord(
            evidence_id="ev-fsb",
            task_id="task-1",
            claim="Financial Stability Board co-pilot finding",
            support="Co-pilot applications support human decisions.",
            source_id=source_id,
            source_chunk_id="fsb-p14-c0",
            source_url=source_url,
            source_title="The Financial Stability Implications of Artificial Intelligence",
            source_type="vector",
        )
    ]
    documents = [
        VectorDocument("fsb-p5-c0", "AI adoption data are limited.", {"page_number": 5}),
        VectorDocument(
            "fsb-p14-c0",
            "Many firms prefer co-pilot applications that support rather than replace human decisions.",
            {"page_number": 14, "chunk_index": 0},
        ),
        VectorDocument(
            "fsb-p23-c0",
            "Some financial institutions have unveiled LLM applications.",
            {"page_number": 23},
        ),
    ]
    semantic_queries = []

    def search_source(query, _source_id, max_results):
        semantic_queries.append(query)
        assert max_results == 2
        return [documents[2], documents[0]]

    context = build_audit_context(
        (
            "FSB p.14: Co-pilot applications support rather than replace human decisions.\n\n"
            "## Sources\n- FSB: https://www.fsb.org/uploads/P14112024.pdf"
        ),
        evidence,
        [],
        lambda _source_id: documents,
        search_source,
    )

    source_entries = [item for item in context if item["source_id"] == source_id]
    source_marker = next(item for item in source_entries if item["audit_id"] == f"source:{source_id}")
    assert source_marker["cited_pages"] == [14]
    assert source_marker["retrieval_completeness"] == "complete"
    assert any(item.get("page_number") == 14 for item in source_entries)
    assert semantic_queries
    assert all("Sources" not in query for query in semantic_queries)


def test_pdf_filename_digits_are_not_treated_as_cited_pages() -> None:
    source_id = "fsb-ai-financial-stability-2024"
    evidence = [
        EvidenceRecord(
            evidence_id="ev-fsb",
            task_id="task-1",
            claim="Financial Stability Board finding",
            support="Available excerpt.",
            source_id=source_id,
            source_url="https://www.fsb.org/uploads/P14112024.pdf",
            source_type="vector",
        )
    ]
    context = build_audit_context(
        "The report is available at https://www.fsb.org/uploads/P14112024.pdf.",
        evidence,
        [],
        lambda _source_id: [],
        lambda _query, _source_id, _max_results: [],
    )

    marker = next(item for item in context if item["audit_id"] == f"source:{source_id}")
    assert marker["cited_pages"] == []


def test_generic_payments_term_does_not_match_eba_source_alias() -> None:
    fsb_id = "fsb-ai-financial-stability-2024"
    eba_id = "eba-ai-banking-payments-2025"
    evidence = [
        EvidenceRecord(
            evidence_id="ev-fsb",
            task_id="task-1",
            claim="Financial Stability Board finding",
            support="Available excerpt.",
            source_id=fsb_id,
            source_url="https://www.fsb.org/report.pdf",
            source_type="vector",
        ),
        EvidenceRecord(
            evidence_id="ev-eba",
            task_id="task-1",
            claim="European Banking Authority finding",
            support="Available excerpt.",
            source_id=eba_id,
            source_url="https://www.eba.europa.eu/report.pdf",
            source_type="vector",
        ),
    ]
    context = build_audit_context(
        "FSB p.15 discusses payments services.",
        evidence,
        [],
        lambda _source_id: [],
        lambda _query, _source_id, _max_results: [],
    )

    markers = {item["source_id"]: item for item in context if item["audit_id"].startswith("source:")}
    assert markers[fsb_id]["cited_pages"] == [15]
    assert markers[eba_id]["cited_pages"] == []


def test_audit_sources_are_returned_in_batches_instead_of_skipped() -> None:
    evidence = [
        EvidenceRecord(
            evidence_id=f"ev-{index}",
            task_id="task-1",
            claim=f"Finding {index}",
            support=f"Evidence excerpt {index}.",
            source_id=f"source-{index}",
            source_type="vector",
        )
        for index in range(13)
    ]

    second_batch = build_audit_context(
        "A report with claims.",
        evidence,
        [],
        lambda _source_id: [],
        lambda _query, _source_id, _max_results: [],
        source_batch=1,
    )

    assert [item["audit_id"] for item in second_batch] == ["source:source-12", "ev-12"]
    assert all(not item["audit_id"].startswith("skipped:") for item in second_batch)


def test_citation_coverage_flags_urls_without_retrieved_excerpt() -> None:
    known_url = "https://example.org/evidence"
    unknown_url = "https://example.org/not-retrieved"
    evidence = [
        EvidenceRecord(
            evidence_id="ev-known",
            task_id="task-1",
            claim="Supported finding",
            support="Supplied excerpt.",
            source_url=known_url,
        )
    ]

    issues = citation_coverage_issues(
        (
            f"A supported claim. {known_url}\n\n"
            f"A claim with no gathered evidence. {unknown_url}\n\n"
            "## Sources\n- Historical source, https://example.org/unreferenced"
        ),
        evidence,
        [],
    )

    assert issues == [f"Cited URL has no retrieved evidence excerpt: {unknown_url}"]


def test_citation_coverage_flags_uncited_numeric_paragraphs() -> None:
    issues = citation_coverage_issues(
        "The productivity estimate increased by 14% in 2025.",
        [],
        [],
    )

    assert issues == [
        "Numeric claim has no inline source URL: The productivity estimate increased by 14% in 2025."
    ]


def test_missing_cited_page_is_flagged_as_incomplete_retrieval() -> None:
    source_id = "fsb-ai-financial-stability-2024"
    evidence = [
        EvidenceRecord(
            evidence_id="ev-fsb",
            task_id="task-1",
            claim="Financial Stability Board finding",
            support="Available excerpt.",
            source_id=source_id,
            source_url="https://www.fsb.org/uploads/P14112024.pdf",
            source_type="vector",
        )
    ]
    context = build_audit_context(
        "FSB p.14 discusses co-pilot applications.",
        evidence,
        [],
        lambda _source_id: [
            VectorDocument("fsb-p5-c0", "Adoption data are limited.", {"page_number": 5})
        ],
        lambda _query, _source_id, max_results: [],
    )

    marker = next(item for item in context if item["audit_id"] == f"source:{source_id}")
    assert marker["retrieval_completeness"] == "partial"
    assert "14" in marker["retrieval_issue"]


def test_partial_web_retrieval_cannot_be_called_insufficient_evidence() -> None:
    source_url = "https://example.org/telework"
    source_id = web_source_id(source_url)
    evidence = [
        EvidenceRecord(
            evidence_id="ev-web",
            task_id="task-1",
            claim="Telework productivity report",
            support="A search excerpt mentions telework.",
            source_id=source_id,
            retrieval_completeness="partial",
            source_url=source_url,
            source_type="web",
        )
    ]
    context = build_audit_context(
        f"Telework productivity is unclear. {source_url}",
        evidence,
        [
            {
                "source_id": source_id,
                "source_url": source_url,
                "source_title": "Telework report",
                "text": "The retrieved page excerpt discusses telework productivity.",
            }
        ],
        lambda _source_id: [],
        lambda _query, _source_id, max_results: [],
    )

    audit = normalize_claim_audits(
        [
            {
                "claim": "Telework productivity is unclear.",
                "evidence_ids": [f"source:{source_id}"],
                "status": "insufficient_evidence",
                "excerpt": "",
                "reasoning": "No support was found.",
            }
        ],
        context,
    )[0]

    assert audit.status == "insufficient_retrieval"
    assert audit.retrieval_completeness == "partial"


def test_complete_local_search_allows_absence_verdict_and_validates_quotes() -> None:
    context = [
        {
            "audit_id": "source:report-1",
            "source_id": "report-1",
            "retrieval_completeness": "complete",
            "excerpt": "",
        },
        {
            "audit_id": "chunk:chunk-4",
            "source_id": "report-1",
            "retrieval_completeness": "complete",
            "excerpt": "Telework was associated with higher productivity.",
        },
    ]
    audits = normalize_claim_audits(
        [
            {
                "claim": "The record does not discuss pay.",
                "evidence_ids": ["source:report-1"],
                "status": "insufficient_evidence",
                "excerpt": "",
                "reasoning": "The complete record was searched.",
            },
            {
                "claim": "Telework improved productivity.",
                "evidence_ids": ["chunk:chunk-4"],
                "status": "supported",
                "excerpt": "Telework was associated with higher productivity.",
                "reasoning": "The passage supports the claim.",
            },
            {
                "claim": "Telework increased productivity.",
                "evidence_ids": ["chunk:chunk-4"],
                "status": "supported",
                "excerpt": "Telework doubled productivity.",
                "reasoning": "This quote was not retrieved.",
            },
        ],
        context,
    )

    assert [audit.status for audit in audits] == [
        "insufficient_evidence",
        "supported",
        "insufficient_retrieval",
    ]
    assert audits[1].retrieval_completeness == "complete"


def test_supported_claim_cannot_use_quote_from_a_different_cited_source() -> None:
    context = [
        {
            "audit_id": "unrelated",
            "source_id": "eba",
            "source_url": "https://example.org/eba",
            "retrieval_completeness": "partial",
            "excerpt": "55% of banks use AI in consumer-facing processes.",
        }
    ]
    audit = normalize_claim_audits(
        [{
            "claim": "IT help desk calls fell 55% (https://example.org/cosupport).",
            "evidence_ids": ["unrelated"],
            "status": "supported",
            "excerpt": "55% of banks use AI in consumer-facing processes.",
            "reasoning": "Both figures are 55%.",
        }],
        context,
    )[0]

    assert audit.status == "insufficient_retrieval"
    assert "not from a source cited" in audit.reasoning


def test_quote_is_attributed_only_to_its_matching_source() -> None:
    context = [
        {
            "audit_id": "first",
            "source_url": "https://example.org/first",
            "retrieval_completeness": "partial",
            "excerpt": "A first passage.",
        },
        {
            "audit_id": "second",
            "source_url": "https://example.org/second",
            "retrieval_completeness": "partial",
            "excerpt": "A second passage.",
        },
    ]
    audit = normalize_claim_audits(
        [{
            "claim": "A finding.",
            "evidence_ids": ["first", "second"],
            "status": "supported",
            "excerpt": "A first passage.",
            "reasoning": "Both records were consulted.",
        }],
        context,
    )[0]

    assert audit.status == "supported"
    assert audit.evidence_ids == ["first", "second"]
    assert len(audit.source_references) == 1
    assert "https://example.org/first" in audit.source_references[0]


def test_chroma_source_search_limits_results_to_that_source() -> None:
    class FakeCollection:
        def get(self, where, include):
            documents = [
                ("chunk-2", "Second page.", {"source_id": "report-1", "page_number": 2}),
                ("chunk-1", "First page.", {"source_id": "report-1", "page_number": 1}),
            ]
            selected = [item for item in documents if item[2]["source_id"] == where["source_id"]]
            return {
                "ids": [item[0] for item in selected],
                "documents": [item[1] for item in selected] if "documents" in include else None,
                "metadatas": [item[2] for item in selected],
            }

        def query(self, query_texts, where, n_results):
            assert query_texts == ["telework productivity"]
            assert where == {"source_id": "report-1"}
            assert n_results == 2
            return {
                "ids": [["chunk-1", "chunk-2"]],
                "documents": [["First page.", "Second page."]],
                "metadatas": [
                    [{"source_id": "report-1"}, {"source_id": "report-1"}]
                ],
                "distances": [[0.1, 0.2]],
            }

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = FakeCollection()

    assert [document.id for document in store.get_source_documents("report-1")] == [
        "chunk-1",
        "chunk-2",
    ]
    assert [document.id for document in store.similarity_search_in_source(
        "telework productivity", "report-1", k=4
    )] == ["chunk-1", "chunk-2"]
