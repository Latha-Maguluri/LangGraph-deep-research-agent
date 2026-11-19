from __future__ import annotations

from datetime import UTC, datetime

from app.tools.web_search import _authority_score, _freshness_score, extract_relevant_passage, web_search


def test_extract_relevant_passage_uses_matching_page_text_and_fallback() -> None:
    page_text = (
        "General site navigation\n\n"
        "The report finds that clean energy investment rose substantially in 2024, "
        "with solar energy accounting for a large share of new capacity.\n\n"
        "Cookie settings"
    )

    passage, method = extract_relevant_passage(page_text, "solar energy investment outlook")

    assert method == "page_excerpt"
    assert "solar energy accounting" in passage
    assert "Cookie settings" not in passage

    fallback, fallback_method = extract_relevant_passage(
        "An unrelated page about gardening.",
        "solar energy investment",
        "Search result says solar energy investment increased.",
    )
    assert fallback_method == "search_snippet"
    assert "solar energy investment increased" in fallback


def test_extract_relevant_passage_preserves_table_rows() -> None:
    table = (
        "Infosys H-1B petition data\n"
        "| Fiscal year | Initial filings | Continuing filings |\n"
        "| FY2021 | 5,633 | 4,220 |\n"
        "| FY2022 | 7,694 | 5,018 |"
    )

    passage, method = extract_relevant_passage(
        table,
        "Infosys H-1B filings and approvals",
    )

    assert method == "page_excerpt"
    assert "| FY2021 | 5,633 | 4,220 |" in passage
    assert "| FY2022 | 7,694 | 5,018 |" in passage


def test_web_search_requests_page_content_and_prioritizes_diverse_sources(monkeypatch) -> None:
    import tavily

    captured: dict = {}

    class FakeTavilyClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        def search(self, **kwargs):
            captured["search_kwargs"] = kwargs
            return {
                "results": [
                    {
                        "title": "Official AI report",
                        "url": "https://www.gov.uk/reports/ai?utm_source=search#overview",
                        "content": "Search excerpt.",
                        "raw_content": "AI adoption and industry investment are measured in this official report.",
                        "published_date": "2025-01-01",
                        "score": 0.9,
                    },
                    {
                        "title": "Duplicate domain result",
                        "url": "https://data.gov.uk/reports/second",
                        "content": "Another excerpt.",
                        "raw_content": "Another AI adoption report.",
                        "published_date": "2025-01-01",
                        "score": 0.9,
                    },
                    {
                        "title": "Research paper",
                        "url": "https://arxiv.org/abs/2401.00001",
                        "content": "Research excerpt.",
                        "raw_content": "Research on AI adoption and industry investment.",
                        "published_date": "2025-01-01",
                        "score": 0.9,
                    },
                ]
            }

    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient)
    monkeypatch.setattr("app.tools.web_search.settings.tavily_api_key", "test-key")

    results = web_search("AI adoption and industry investment", max_results=2)

    assert captured["api_key"] == "test-key"
    assert captured["search_kwargs"]["search_depth"] == "advanced"
    assert captured["search_kwargs"]["include_raw_content"] == "markdown"
    assert captured["search_kwargs"]["max_results"] == 2
    assert len(results) == 2
    assert results[0]["url"] == "https://gov.uk/reports/ai"
    assert {item["source_domain"] for item in results} == {"gov.uk", "arxiv.org"}
    assert results[0]["relevance_score"] == 0.9
    assert results[0]["authority_score"] > results[1]["authority_score"]


def test_freshness_ranking_does_not_reward_future_publication_dates() -> None:
    score = _freshness_score(
        "2026-02-01",
        "AI industry outlook",
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert score == 0.0


def test_industry_research_publishers_receive_moderate_authority_signal() -> None:
    for domain in (
        "mckinsey.com",
        "deloitte.com",
        "pwc.com",
        "ibm.com",
        "weforum.org",
        "global.deloitte.com",
    ):
        assert _authority_score(domain) == 0.65
