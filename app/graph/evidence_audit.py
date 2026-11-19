from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Callable
from urllib.parse import urldefrag, urlsplit, urlunsplit

from app.infrastructure.vector_store import VectorDocument
from app.models.research import ClaimAudit, EvidenceRecord

_URL_PATTERN = re.compile(r"https?://[^\s\])>\"']+", re.IGNORECASE)
_PAGE_PATTERN = re.compile(
    r"(?<!\w)(?:p{1,2}|pages?)\.?\s*(\d+)(?:\s*[-–]\s*(\d+))?",
    re.IGNORECASE,
)
_NUMERIC_CLAIM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|percent|hours?|days?|weeks?|months?|years?)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_STOPWORDS = {
    "about", "after", "also", "and", "are", "for", "from", "has", "have", "into",
    "its", "more", "our", "that", "the", "their", "there", "these", "this",
    "through", "was", "were", "which", "with", "would",
}
_ALIAS_STOPWORDS = {
    "ai", "artificial", "banking", "com", "eu", "financial", "genai", "gov", "home",
    "intelligence", "org", "payments", "report", "research", "services", "stability",
    "the", "www",
}
_MAX_PASSAGES_PER_SOURCE = 4
_MAX_PASSAGE_LENGTH = 1200
_MAX_QUERIES_PER_SOURCE = 4
_MAX_AUDIT_SOURCES = 12
logger = logging.getLogger(__name__)


def _url_key(url: str) -> str:
    value = urldefrag(url.strip().rstrip(".,;:")).url
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname.lower().removeprefix("www.")
        return urlunsplit((parsed.scheme.lower(), host, parsed.path.rstrip("/"), parsed.query, ""))
    except ValueError:
        return ""


def web_source_id(url: str) -> str:
    key = _url_key(url)
    return f"web:{key}" if key else ""


def _tokens(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
        if token.casefold() not in _STOPWORDS
    }
    if re.search(r"\bno\b", text, re.IGNORECASE):
        tokens.add("no")
    return tokens


def _report_paragraphs(report: str) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", report):
        heading = paragraph.splitlines()[0].strip()
        if re.fullmatch(
            r"(?:#{1,6}\s*)?(?:sources|references|bibliography)\s*:?",
            heading,
            re.IGNORECASE,
        ):
            break
        query = re.sub(r"\[([^\]]+)\]\((https?://[^)]*)\)", r"\1 \2", paragraph)
        query = re.sub(r"\[([^\]]+)\]\s*\(\s*\)", r"\1", query)
        query = re.sub(r"[#*`]", " ", query)
        query = " ".join(query.split())
        if query and not query.startswith("#"):
            paragraphs.append(query)
    return paragraphs


def _cited_pages(paragraph: str) -> set[int]:
    pages: set[int] = set()
    citation_text = _URL_PATTERN.sub(" ", paragraph)
    for start_text, end_text in _PAGE_PATTERN.findall(citation_text):
        start = int(start_text)
        end = int(end_text) if end_text else start
        if start < 1 or end > 10_000:
            continue
        if end - start > 20:
            pages.update((start, end))
        else:
            pages.update(range(start, end + 1))
    return pages


def _source_aliases(group: dict[str, Any]) -> set[str]:
    source_url = group.get("source_url", "")
    try:
        hostname = urlsplit(source_url).hostname or ""
    except ValueError:
        hostname = ""
    identity = f"{group['source_id']} {hostname}"
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(identity)
        if token.casefold() not in _ALIAS_STOPWORDS and not token.isdigit()
    }


def _fallback_queries(
    paragraphs: list[str],
    records: list[EvidenceRecord],
) -> list[str]:
    if not paragraphs:
        return []
    record_terms = _tokens(" ".join(f"{item.claim} {item.support}" for item in records))
    ranked = sorted(
        (
            (len(record_terms.intersection(_tokens(paragraph))), index, paragraph)
            for index, paragraph in enumerate(paragraphs)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    relevant = [paragraph for overlap, _, paragraph in ranked if overlap]
    return (relevant or paragraphs)[:_MAX_QUERIES_PER_SOURCE]


def _ranked_blocks(text: str, query: str) -> list[str]:
    query_tokens = _tokens(query)
    if not text.strip() or not query_tokens:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n|\r?\n", text) if block.strip()]
    ranked: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        pieces = [
            block[start : start + _MAX_PASSAGE_LENGTH].strip()
            for start in range(0, len(block), _MAX_PASSAGE_LENGTH)
        ]
        for piece in pieces:
            overlap = len(query_tokens.intersection(_tokens(piece)))
            if overlap:
                ranked.append((overlap, -index, piece))
    ranked.sort(reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, passage in ranked:
        normalized = " ".join(passage.casefold().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(passage)
        if len(selected) == _MAX_PASSAGES_PER_SOURCE:
            break
    return selected


def build_audit_context(
    report: str,
    evidence: list[EvidenceRecord],
    source_records: list[dict[str, Any]],
    get_source_documents: Callable[[str], list[VectorDocument]],
    search_source: Callable[..., list[VectorDocument]],
    *,
    source_batch: int = 0,
) -> list[dict[str, Any]]:
    if source_batch < 0:
        raise ValueError("The audit source batch cannot be negative.")
    groups: dict[str, dict[str, Any]] = {}
    url_to_source: dict[str, str] = {}
    report_paragraphs = _report_paragraphs(report)
    for item in evidence:
        source_id = item.source_id or (
            web_source_id(item.source_url or "")
            if item.source_type == "web"
            else ""
        )
        if not source_id:
            source_id = f"evidence:{item.evidence_id}"
        group = groups.setdefault(
            source_id,
            {
                "source_id": source_id,
                "source_type": item.source_type,
                "source_title": item.source_title or "",
                "source_url": item.source_url or "",
                "records": [],
                "source_text": "",
                "queries": [],
                "cited_pages": set(),
                "page_queries": defaultdict(list),
            },
        )
        group["records"].append(item)
        if item.source_title and not group["source_title"]:
            group["source_title"] = item.source_title
        if item.source_url:
            group["source_url"] = item.source_url
            url_to_source[_url_key(item.source_url)] = source_id

    for record in source_records:
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            continue
        group = groups.setdefault(
            source_id,
            {
                "source_id": source_id,
                "source_type": "web",
                "source_title": "",
                "source_url": "",
                "records": [],
                "source_text": "",
                "queries": [],
                "cited_pages": set(),
                "page_queries": defaultdict(list),
            },
        )
        source_text = str(record.get("text") or "")
        if len(source_text) > len(group["source_text"]):
            group["source_text"] = source_text
        group["source_title"] = str(record.get("source_title") or group["source_title"])
        group["source_url"] = str(record.get("source_url") or group["source_url"])
        if group["source_url"]:
            url_to_source[_url_key(group["source_url"])] = source_id

    for paragraph in report_paragraphs:
        urls = [_url_key(match.group()) for match in _URL_PATTERN.finditer(paragraph)]
        query = _URL_PATTERN.sub(" ", paragraph)
        query = re.sub(r"\[([^\]]+)\]\s*\(\s*\)", r"\1", query)
        query = " ".join(query.split())
        cited_source_ids = {
            source_id
            for url in urls
            if (source_id := url_to_source.get(url))
        }
        if not cited_source_ids:
            cited_source_ids = {
                source_id
                for source_id, group in groups.items()
                if any(
                    re.search(
                        rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                        paragraph,
                        re.IGNORECASE,
                    )
                    for alias in _source_aliases(group)
                )
            }
        for source_id in cited_source_ids:
            if query:
                groups[source_id]["queries"].append(query)
            pages = _cited_pages(paragraph)
            groups[source_id]["cited_pages"].update(pages)
            for page in pages:
                groups[source_id]["page_queries"][page].append(query)

    all_groups = list(groups.values())
    batch_start = source_batch * _MAX_AUDIT_SOURCES
    selected_groups = all_groups[batch_start : batch_start + _MAX_AUDIT_SOURCES]
    context: list[dict[str, Any]] = []
    for group in selected_groups:
        source_id = group["source_id"]
        source_type = group["source_type"]
        queries = list(dict.fromkeys(group["queries"]))[:_MAX_QUERIES_PER_SOURCE]
        if not queries:
            queries = _fallback_queries(report_paragraphs, group["records"])
        if not queries:
            queries = [report]
        completeness = "unknown"
        selected_passages: list[tuple[str, str, int | None, str | None]] = []
        retrieval_issue = ""
        documents: list[VectorDocument] = []

        if source_type == "vector" and source_id.startswith("evidence:"):
            retrieval_issue = "Stable source ID unavailable; full-record search could not run."
        elif source_type == "vector":
            try:
                documents = get_source_documents(source_id)
                if documents:
                    completeness = "complete"
                    document_by_id = {document.id: document for document in documents}
                    ranked_ids: list[str] = []
                    document_queries: dict[str, list[str]] = defaultdict(list)
                    semantic_search_failed = False
                    available_pages = {
                        document.metadata.get("page_number")
                        for document in documents
                        if isinstance(document.metadata.get("page_number"), int)
                    }
                    cited_pages = sorted(group["cited_pages"])
                    missing_pages = set(cited_pages) - available_pages
                    if missing_pages:
                        completeness = "partial"
                        page_issue = (
                            "Cited page(s) missing from indexed record: "
                            + ", ".join(str(page) for page in sorted(missing_pages))
                        )
                        retrieval_issue = "; ".join(
                            issue for issue in (retrieval_issue, page_issue) if issue
                        )
                    if len(cited_pages) > _MAX_PASSAGES_PER_SOURCE:
                        completeness = "partial"
                        page_issue = (
                            "Cited pages exceed the bounded passage budget; not every cited page "
                            "could be included."
                        )
                        retrieval_issue = "; ".join(
                            issue for issue in (retrieval_issue, page_issue) if issue
                        )
                    page_budget = max(
                        1,
                        _MAX_PASSAGES_PER_SOURCE // max(len(cited_pages), 1),
                    )
                    for page in cited_pages:
                        page_documents = [
                            document
                            for document in documents
                            if document.metadata.get("page_number") == page
                        ]
                        page_queries = (
                            list(dict.fromkeys(group["page_queries"].get(page, [])))
                            or queries
                        )
                        page_ranked = sorted(
                            page_documents,
                            key=lambda document: (
                                -max(
                                    (
                                        len(
                                            _tokens(page_query).intersection(
                                                _tokens(document.content)
                                            )
                                        )
                                        for page_query in page_queries
                                    ),
                                    default=0,
                                ),
                                document.metadata.get("chunk_index", 0) or 0,
                            ),
                        )
                        for document in page_ranked[:page_budget]:
                            ranked_ids.append(document.id)
                            document_queries[document.id].extend(page_queries)
                    for query in queries:
                        query_tokens = _tokens(query)
                        lexical = sorted(
                            (
                                (
                                    len(query_tokens.intersection(_tokens(document.content))),
                                    document.id,
                                )
                                for document in documents
                            ),
                            reverse=True,
                        )
                        ranked_ids.extend(
                            document_id
                            for overlap, document_id in lexical[:2]
                            if overlap
                        )
                        for overlap, document_id in lexical[:2]:
                            if overlap:
                                document_queries[document_id].append(query)
                        try:
                            for result in search_source(
                                query,
                                source_id,
                                max_results=2,
                            ):
                                ranked_ids.append(result.id)
                                document_queries[result.id].append(query)
                        except Exception as exc:
                            semantic_search_failed = True
                            logger.warning(
                                "Semantic audit search failed for source_id=%s (%s).",
                                source_id,
                                type(exc).__name__,
                            )
                            semantic_issue = (
                                f"Semantic search failed ({type(exc).__name__}); "
                                "exact-term scan completed."
                            )
                            retrieval_issue = "; ".join(
                                issue for issue in (retrieval_issue, semantic_issue) if issue
                            )
                    if semantic_search_failed:
                        completeness = "partial"
                    existing_chunk_ids = {
                        item.source_chunk_id: item.evidence_id
                        for item in group["records"]
                        if item.source_chunk_id
                    }
                    for document_id in dict.fromkeys(ranked_ids):
                        document = document_by_id.get(document_id)
                        if document is None:
                            continue
                        excerpts = [
                            passage
                            for query in dict.fromkeys(document_queries.get(document_id, []))
                            for passage in _ranked_blocks(document.content, query)[:1]
                        ]
                        audit_id = existing_chunk_ids.get(document.id, f"chunk:{document.id}")
                        selected_passages.append(
                            (
                                audit_id,
                                excerpts[0] if excerpts else document.content[:_MAX_PASSAGE_LENGTH],
                                document.metadata.get("page_number"),
                                document.id,
                            )
                        )
                        if len(selected_passages) == _MAX_PASSAGES_PER_SOURCE:
                            break
                else:
                    retrieval_issue = "No indexed chunks were returned for full-record search."
            except Exception as exc:
                logger.warning(
                    "Full local record audit failed for source_id=%s (%s).",
                    source_id,
                    type(exc).__name__,
                )
                retrieval_issue = f"Full local record search failed ({type(exc).__name__})."

        if source_type != "vector":
            completeness = "partial"
            for query in queries:
                for passage in _ranked_blocks(group["source_text"], query):
                    selected_passages.append((f"web:{source_id}:{len(selected_passages)}", passage, None, None))
                    if len(selected_passages) == _MAX_PASSAGES_PER_SOURCE:
                        break
                if len(selected_passages) == _MAX_PASSAGES_PER_SOURCE:
                    break

        context.append(
            {
                "audit_id": f"source:{source_id}",
                "source_id": source_id,
                "source_title": group["source_title"] or "Unknown",
                "source_url": group["source_url"] or "URL unavailable",
                "retrieval_completeness": completeness,
                "searched_chunk_count": len(documents) if source_type == "vector" else None,
                "cited_pages": sorted(group["cited_pages"]),
                "retrieval_issue": retrieval_issue,
                "excerpt": "",
            }
        )
        for audit_id, excerpt, page_number, chunk_id in selected_passages:
            context.append(
                {
                    "audit_id": audit_id,
                    "source_id": source_id,
                    "source_title": group["source_title"] or "Unknown",
                    "source_url": group["source_url"] or "URL unavailable",
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "retrieval_completeness": completeness,
                    "excerpt": excerpt[:_MAX_PASSAGE_LENGTH],
                }
            )

        if not selected_passages:
            for item in group["records"][:_MAX_PASSAGES_PER_SOURCE]:
                context.append(
                    {
                        "audit_id": item.evidence_id,
                        "source_id": source_id,
                        "source_title": item.source_title or group["source_title"] or "Unknown",
                        "source_url": item.source_url or group["source_url"] or "URL unavailable",
                        "page_number": item.page_number,
                        "chunk_id": item.source_chunk_id,
                        "retrieval_completeness": (
                            completeness
                            if completeness != "unknown"
                            else item.retrieval_completeness
                        ),
                        "excerpt": item.support[:_MAX_PASSAGE_LENGTH],
                    }
                )
    return context


def citation_coverage_issues(
    report: str,
    evidence: list[EvidenceRecord],
    source_records: list[dict[str, Any]],
) -> list[str]:
    cited_urls = {
        _url_key(match.group())
        for paragraph in _report_paragraphs(report)
        for match in _URL_PATTERN.finditer(paragraph)
    }
    gathered_urls = {
        _url_key(item.source_url)
        for item in evidence
        if item.source_url and item.support.strip()
    }
    gathered_urls.update(
        _url_key(str(item.get("source_url") or ""))
        for item in source_records
        if item.get("source_url") and str(item.get("text") or "").strip()
    )
    missing_urls = sorted(url for url in cited_urls - gathered_urls if url)
    issues = [
        f"Cited URL has no retrieved evidence excerpt: {url}"
        for url in missing_urls
    ]
    for paragraph in _report_paragraphs(report):
        citation_text = _URL_PATTERN.sub(" ", paragraph)
        citation_text = _PAGE_PATTERN.sub(" ", citation_text)
        if not _URL_PATTERN.search(paragraph) and _NUMERIC_CLAIM_PATTERN.search(citation_text):
            issues.append(
                "Numeric claim has no inline source URL: " + " ".join(paragraph.split())[:180]
            )
    return issues


def normalize_claim_audits(
    raw_audits: list[dict[str, Any]],
    audit_context: list[dict[str, Any]],
) -> list[ClaimAudit]:
    allowed = {item["audit_id"]: item for item in audit_context}
    normalized: list[ClaimAudit] = []
    for raw in raw_audits:
        if not isinstance(raw, dict):
            continue
        claim = raw.get("claim")
        claim_id = raw.get("claim_id")
        critical = raw.get("critical", False)
        status = raw.get("status")
        reasoning = raw.get("reasoning")
        evidence_ids = raw.get("evidence_ids", [])
        excerpt = raw.get("excerpt", "")
        if not isinstance(claim, str) or not claim.strip():
            continue
        if status not in {
            "supported",
            "contradicted",
            "insufficient_evidence",
            "insufficient_retrieval",
        }:
            status = "insufficient_retrieval"
        if not isinstance(evidence_ids, list) or not all(
            isinstance(item, str) for item in evidence_ids
        ):
            evidence_ids = []
        valid_ids = list(dict.fromkeys(item for item in evidence_ids if item in allowed))
        invalid_ids = set(evidence_ids) - allowed.keys()
        cited_sources = [allowed[item] for item in valid_ids]
        completeness_values = {item["retrieval_completeness"] for item in cited_sources}
        completeness = (
            "complete"
            if completeness_values == {"complete"}
            else "partial"
            if "partial" in completeness_values
            else "unknown"
        )
        if not valid_ids:
            completeness = "unknown"
        reference_items = cited_sources
        reason = reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else ""
        if invalid_ids:
            status = "insufficient_retrieval"
            reason = "The audit referenced unknown evidence IDs. " + reason
        elif status == "insufficient_evidence" and completeness != "complete":
            status = "insufficient_retrieval"
            reason = "The available source text was not complete enough to rule out support. " + reason
        elif status in {"supported", "contradicted"}:
            normalized_excerpt = " ".join(excerpt.split()) if isinstance(excerpt, str) else ""
            quoted_sources = [
                item for item in cited_sources
                if normalized_excerpt
                and normalized_excerpt in " ".join(str(item.get("excerpt") or "").split())
            ]
            claim_urls = {_url_key(match.group()) for match in _URL_PATTERN.finditer(claim)}
            if not quoted_sources:
                status = "insufficient_retrieval"
                reason = "The quoted passage could not be verified against the retrieved text. " + reason
            elif status == "supported" and claim_urls and not any(
                _url_key(str(item.get("source_url") or "")) in claim_urls
                for item in quoted_sources
            ):
                status = "insufficient_retrieval"
                reason = "The quoted passage is not from a source cited by the claim. " + reason
            if quoted_sources:
                reference_items = quoted_sources
        source_references = []
        for item in reference_items:
            parts = [str(item.get("source_title") or "Source title unavailable")]
            if item.get("page_number") is not None:
                parts.append(f"p. {item['page_number']}")
            source_url = item.get("source_url")
            if source_url:
                parts.append(str(source_url))
            reference = " — ".join(parts)
            if reference not in source_references:
                source_references.append(reference)
        normalized.append(
            ClaimAudit(
                claim_id=claim_id if isinstance(claim_id, str) else None,
                claim=claim.strip(),
                critical=critical if isinstance(critical, bool) else False,
                evidence_ids=valid_ids,
                status=status,
                excerpt=excerpt if isinstance(excerpt, str) else "",
                reasoning=reason or "No audit reasoning was provided.",
                retrieval_completeness=completeness,
                source_references=source_references,
            )
        )
    return normalized
