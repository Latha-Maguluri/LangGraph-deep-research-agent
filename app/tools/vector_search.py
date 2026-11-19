from __future__ import annotations

from threading import Lock

from app.core.config import settings
from app.infrastructure.vector_store import ChromaVectorStore, VectorDocument


class VectorSearchTool:
    def __init__(self, path: str | None = None) -> None:
        self.store = ChromaVectorStore(
            path or settings.vector_db_path,
            collection_name=settings.vector_db_collection_name,
        )

    def search(self, query: str, k: int = 3) -> list[VectorDocument]:
        candidate_count = max(k * 2, k)
        semantic = self.store.similarity_search(query, k=candidate_count)
        keyword = self.store.keyword_search(query, k=candidate_count)
        return _merge_hybrid_results(semantic, keyword, k)

    def source_documents(self, source_id: str) -> list[VectorDocument]:
        return self.store.get_source_documents(source_id)

    def search_source(self, query: str, source_id: str, k: int = 3) -> list[VectorDocument]:
        return self.store.similarity_search_in_source(query, source_id, k=k)

    def close(self) -> None:
        self.store.close()


def _merge_hybrid_results(
    semantic: list[VectorDocument],
    keyword: list[VectorDocument],
    limit: int,
) -> list[VectorDocument]:
    ranked: dict[str, tuple[VectorDocument, float, set[str]]] = {}
    for channel, results in (("semantic", semantic), ("keyword", keyword)):
        for rank, document in enumerate(results, start=1):
            prior = ranked.get(document.id)
            channels = set() if prior is None else prior[2]
            channels.add(channel)
            selected_document = (
                document
                if channel == "semantic" or prior is None
                else prior[0]
            )
            score = 0.0 if prior is None else prior[1]
            ranked[document.id] = (selected_document, score + 1 / (60 + rank), channels)

    ordered = sorted(
        ranked.values(),
        key=lambda entry: (
            -entry[1],
            -len(entry[2]),
            -float(entry[0].metadata.get("keyword_score", 0.0)),
            entry[0].id,
        ),
    )
    return [
        VectorDocument(
            document.id,
            document.content,
            {
                **document.metadata,
                "retrieval_channels": sorted(channels),
                "hybrid_search_score": score,
            },
        )
        for document, score, channels in ordered[:limit]
    ]


_vector_search_tools: dict[str, VectorSearchTool] = {}
_vector_search_tools_lock = Lock()


def vector_search(query: str, *, max_results: int = 3) -> list[VectorDocument]:
    return _get_vector_search_tool().search(query, k=max_results)


def vector_source_documents(source_id: str) -> list[VectorDocument]:
    return _get_vector_search_tool().source_documents(source_id)


def vector_search_source(
    query: str,
    source_id: str,
    *,
    max_results: int = 3,
) -> list[VectorDocument]:
    return _get_vector_search_tool().search_source(query, source_id, k=max_results)


def _get_vector_search_tool() -> VectorSearchTool:
    with _vector_search_tools_lock:
        tool = _vector_search_tools.get(settings.vector_db_path)
        if tool is None:
            tool = VectorSearchTool(settings.vector_db_path)
            _vector_search_tools[settings.vector_db_path] = tool
        return tool


def close_vector_search() -> None:
    with _vector_search_tools_lock:
        tools = list(_vector_search_tools.values())
        _vector_search_tools.clear()
    for tool in tools:
        tool.close()
