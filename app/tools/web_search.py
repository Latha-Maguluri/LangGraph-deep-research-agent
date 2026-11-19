from __future__ import annotations

import html
import os
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.core.config import settings

_BOILERPLATE_LINES = {
    "accept cookies",
    "cookie settings",
    "home",
    "log in",
    "menu",
    "search",
    "sign in",
    "skip to main content",
    "subscribe",
}
_LOW_SIGNAL_DOMAINS = {"medium.com", "substack.com", "youtube.com", "youtu.be"}
_RESEARCH_DOMAINS = {
    "arxiv.org",
    "iea.org",
    "imf.org",
    "nber.org",
    "oecd.org",
    "pubmed.ncbi.nlm.nih.gov",
    "worldbank.org",
    "who.int",
}
_INDUSTRY_RESEARCH_DOMAINS = {
    "deloitte.com",
    "ibm.com",
    "mckinsey.com",
    "pwc.com",
    "weforum.org",
}
_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "among",
    "and",
    "are",
    "from",
    "have",
    "into",
    "more",
    "over",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "will",
    "with",
}


def _canonicalize_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        if parsed.scheme.lower() not in {"http", "https"} or not hostname:
            return ""
        if parsed.username or parsed.password:
            return ""
        port = parsed.port
    except ValueError:
        return ""

    host = hostname.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    netloc = host if port is None else f"{host}:{port}"
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid"}
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path.rstrip("/") or "/",
            urlencode(sorted(query)),
            "",
        )
    )


def _domain_for_url(url: str) -> str | None:
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return None
    return hostname.lower().removeprefix("www.") if hostname else None


def _diversity_key(domain: str) -> str:
    labels = domain.split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else domain


def _normalize_content(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    content = html.unescape(value)
    content = re.sub(r"(?is)<(script|style|noscript)\b[^>]*>.*?</\1\s*>", " ", content)
    content = re.sub(r"(?i)<br\s*/?>|</(?:p|div|li|h[1-6]|tr)\s*>", "\n", content)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", content)
    content = re.sub(r"\[([^\]]+)\]\((?:https?://)?[^)]*\)", r"\1", content)

    lines: list[str] = []
    seen_lines: set[str] = set()
    for raw_line in content.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        if line.casefold().strip("# ") in _BOILERPLATE_LINES:
            continue
        if line in seen_lines:
            continue
        seen_lines.add(line)
        lines.append(line)
    return "\n".join(lines).strip()


def _authority_score(domain: str | None) -> float:
    if not domain:
        return 0.0
    if any(
        domain == suffix.lstrip(".") or domain.endswith(suffix)
        for suffix in (".gov", ".gov.uk", ".gov.au", ".gc.ca", ".edu", ".edu.au", ".ac.uk", ".int")
    ):
        return 1.0
    if domain in _RESEARCH_DOMAINS or any(domain.endswith(f".{trusted}") for trusted in _RESEARCH_DOMAINS):
        return 0.85
    if domain in _INDUSTRY_RESEARCH_DOMAINS or any(
        domain.endswith(f".{trusted}") for trusted in _INDUSTRY_RESEARCH_DOMAINS
    ):
        return 0.65
    if domain in _LOW_SIGNAL_DOMAINS or any(
        domain.endswith(f".{low_signal}") for low_signal in _LOW_SIGNAL_DOMAINS
    ):
        return 0.05
    if domain.endswith(".org"):
        return 0.45
    return 0.3


def _freshness_score(published: Any, query: str, *, now: datetime | None = None) -> float:
    if not isinstance(published, str) or not published.strip():
        return 0.0
    if re.search(r"\b(?:historical|history|historically|(?:19|20)\d{2}|fy\d{4})\b", query, re.I):
        return 0.5
    try:
        published_at = datetime.fromisoformat(published.strip().replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    age_days = (reference - published_at.astimezone(UTC)).days
    if age_days < 0:
        return 0.0
    if age_days <= 365:
        return 1.0
    if age_days <= 3 * 365:
        return 0.7
    if age_days <= 5 * 365:
        return 0.4
    return 0.15


def _result_score(result: dict[str, Any], query: str) -> float:
    try:
        relevance = float(result.get("score", 0.5))
    except (TypeError, ValueError):
        relevance = 0.5
    relevance = min(1.0, max(0.0, relevance))
    authority = _authority_score(result.get("source_domain"))
    freshness = _freshness_score(result.get("published"), query)
    result["relevance_score"] = relevance
    result["authority_score"] = authority
    result["freshness_score"] = freshness
    return 0.7 * relevance + 0.2 * authority + 0.1 * freshness


def _rank_and_diversify(results: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for result in results:
        url = result["url"]
        dedupe_key = url.casefold() if url else result["title"].casefold()
        if dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)
        result["ranking_score"] = _result_score(result, query)
        unique.append(result)
    unique.sort(
        key=lambda item: (item["ranking_score"], item["relevance_score"]),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_urls: set[str] = set()
    selected_domains: set[str] = set()
    for result in unique:
        domain = result.get("source_domain") or ""
        diversity_key = _diversity_key(domain) if domain else ""
        if diversity_key in selected_domains:
            continue
        selected.append(result)
        selected_urls.add(result["url"])
        if diversity_key:
            selected_domains.add(diversity_key)
        if len(selected) == limit:
            return selected
    for result in unique:
        if result["url"] not in selected_urls:
            selected.append(result)
            if len(selected) == limit:
                break
    return selected


def extract_relevant_passage(
    content: Any,
    query: str,
    fallback: str = "",
    *,
    max_chars: int = 4500,
) -> tuple[str, str]:
    """Return query-relevant page text, or the search excerpt when extraction is unavailable."""
    normalized = _normalize_content(content)
    if not normalized:
        return _normalize_content(fallback), "search_snippet"

    terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", query.casefold())
        if term not in _STOPWORDS
    }
    if not terms:
        return _normalize_content(fallback), "search_snippet"

    blocks = [
        "\n".join(" ".join(line.split()) for line in block.splitlines() if line.strip())
        for block in re.split(r"\n\s*\n", normalized)
        if len(block.strip()) >= 40
    ]
    scored = [
        (sum(term in block.casefold() for term in terms), index, block)
        for index, block in enumerate(blocks)
    ]
    relevant = sorted(
        (item for item in scored if item[0] > 0),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    selected: list[tuple[int, str]] = []
    total_chars = 0
    for _, index, block in relevant:
        if total_chars + len(block) > max_chars:
            continue
        selected.append((index, block))
        total_chars += len(block)
        if total_chars >= max_chars * 0.8:
            break
    if not selected:
        return _normalize_content(fallback), "search_snippet"
    selected.sort(key=lambda item: item[0])
    return "\n\n[…]\n\n".join(block for _, block in selected), "page_excerpt"


def web_search(query: str, *, max_results: int = 3) -> list[dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY") or settings.tavily_api_key
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required for web research.")

    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tavily-python is required for web research.") from exc

    result_limit = min(10, max(1, max_results))
    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=result_limit,
        search_depth="advanced",
        include_raw_content="markdown",
    )
    search_results = response.get("results", []) if isinstance(response, dict) else []
    results: list[dict[str, Any]] = []
    for item in search_results:
        if not isinstance(item, dict):
            continue
        url = _canonicalize_url(item.get("url"))
        if not url:
            continue
        raw_content = _normalize_content(item.get("raw_content"))
        results.append(
            {
                "title": str(item.get("title") or "Untitled result").strip(),
                "url": url,
                "snippet": _normalize_content(item.get("content") or item.get("snippet")),
                "raw_content": raw_content[:100_000],
                "published": item.get("published_date") or item.get("published"),
                "source_domain": _domain_for_url(url),
                "score": item.get("score"),
            }
        )
    return _rank_and_diversify(results, query, result_limit)
