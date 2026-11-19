from app.infrastructure.vector_store import ChromaVectorStore, InMemoryVectorStore, VectorDocument
from app.tools.vector_search import VectorSearchTool


def test_keyword_search_ranks_exact_measure_and_acronym_matches() -> None:
    store = InMemoryVectorStore()
    store.add_documents(
        [
            VectorDocument("generic", "Productivity improved across the industry.", {}),
            VectorDocument("measure", "CSAT increased from 71% to 89% after the pilot.", {}),
        ]
    )

    results = store.keyword_search("CSAT 71% pilot", k=2)

    assert results[0].id == "measure"
    assert results[0].metadata["keyword_score"] == 1.0
    assert results[0].metadata["retrieval_channels"] == ["keyword"]


def test_chroma_keyword_search_reads_indexed_documents() -> None:
    class Collection:
        def get(self, *, include):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["generic", "exact"],
                "documents": ["General risk controls.", "GPAI adoption reached 55% of banks."],
                "metadatas": [{}, {"page_number": 3}],
            }

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.collection = Collection()

    results = store.keyword_search("GPAI 55% banks", k=1)

    assert results[0].id == "exact"
    assert results[0].metadata["page_number"] == 3


def test_vector_search_fuses_semantic_and_keyword_rankings() -> None:
    shared = VectorDocument("shared", "Exact supporting passage.", {})
    semantic_only = VectorDocument("semantic", "Conceptually similar passage.", {})
    keyword_only = VectorDocument(
        "keyword",
        "Exact text match.",
        {"keyword_score": 1.0, "retrieval_channels": ["keyword"]},
    )

    class Store:
        def similarity_search(self, query, k):
            return [shared, semantic_only][:k]

        def keyword_search(self, query, k):
            return [shared, keyword_only][:k]

        def close(self):
            pass

    tool = VectorSearchTool.__new__(VectorSearchTool)
    tool.store = Store()

    results = tool.search("query", k=2)

    assert [item.id for item in results] == ["shared", "keyword"]
    assert results[0].metadata["retrieval_channels"] == ["keyword", "semantic"]
