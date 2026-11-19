from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)
_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "being", "but",
    "can", "could", "did", "does", "for", "from", "had", "has", "have", "into",
    "its", "may", "might", "more", "most", "not", "our", "over", "should", "that",
    "the", "their", "there", "these", "this", "those", "through", "was", "were",
    "what", "when", "where", "which", "who", "will", "with", "would",
}


@dataclass
class VectorDocument:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.documents: list[VectorDocument] = []

    def add_documents(self, documents: list[VectorDocument]) -> None:
        self.documents.extend(documents)

    def similarity_search(self, query: str, k: int = 3) -> list[VectorDocument]:
        if not self.documents:
            return []
        return self.documents[:k]

    def keyword_search(self, query: str, k: int = 3) -> list[VectorDocument]:
        return _keyword_search(self.documents, query, k)


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str = "research_documents") -> None:
        database_path = Path(path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(database_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        self.client.close()

    def add_documents(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return
        self.collection.upsert(
            ids=[document.id for document in documents],
            documents=[document.content for document in documents],
            metadatas=[document.metadata for document in documents],
        )

    def similarity_search(self, query: str, k: int = 3) -> list[VectorDocument]:
        count = self.collection.count()
        if count == 0 or k <= 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(k, count))
        return self._documents_from_query(results)

    def keyword_search(self, query: str, k: int = 3) -> list[VectorDocument]:
        if not query.strip() or k <= 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"])
        documents = [
            VectorDocument(document_id, content or "", metadata or {})
            for document_id, content, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]
        return _keyword_search(documents, query, k)

    def get_source_documents(self, source_id: str) -> list[VectorDocument]:
        if not source_id:
            return []
        results = self.collection.get(
            where={"source_id": source_id},
            include=["documents", "metadatas"],
        )
        documents = [
            VectorDocument(
                id=document_id,
                content=content,
                metadata=metadata or {},
            )
            for document_id, content, metadata in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            )
        ]
        return sorted(
            documents,
            key=lambda document: (
                document.metadata.get("page_number", 0) or 0,
                document.metadata.get("chunk_index", 0) or 0,
                document.id,
            ),
        )

    def similarity_search_in_source(
        self,
        query: str,
        source_id: str,
        k: int = 3,
    ) -> list[VectorDocument]:
        if not query.strip() or not source_id or k <= 0:
            return []
        source_count = len(
            self.collection.get(
                where={"source_id": source_id},
                include=["metadatas"],
            )["ids"]
        )
        if source_count == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            where={"source_id": source_id},
            n_results=min(k, source_count),
        )
        return self._documents_from_query(results)

    @staticmethod
    def _documents_from_query(results: dict[str, Any]) -> list[VectorDocument]:
        ids = results["ids"][0]
        contents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]
        return [
            VectorDocument(
                id=document_id,
                content=content,
                metadata={
                    **(metadata or {}),
                    **({"retrieval_distance": distance} if distance is not None else {}),
                },
            )
            for document_id, content, metadata, distance in zip(
                ids,
                contents,
                metadatas,
                distances or [None] * len(ids),
            )
        ]


def _keyword_search(
    documents: list[VectorDocument],
    query: str,
    k: int,
) -> list[VectorDocument]:
    query_terms = {
        term.casefold()
        for term in _TOKEN_PATTERN.findall(query)
        if term.casefold() not in _STOPWORDS
    }
    if not documents or not query_terms or k <= 0:
        return []

    tokenized_documents = [
        [
            term.casefold()
            for term in _TOKEN_PATTERN.findall(document.content)
            if term.casefold() not in _STOPWORDS
        ]
        for document in documents
    ]
    document_frequencies = {
        term: sum(term in set(tokens) for tokens in tokenized_documents)
        for term in query_terms
    }
    average_length = max(
        1.0,
        sum(len(tokens) for tokens in tokenized_documents) / len(tokenized_documents),
    )
    scored: list[tuple[float, str, VectorDocument]] = []
    for document, tokens in zip(documents, tokenized_documents):
        term_frequencies = {term: tokens.count(term) for term in query_terms}
        score = 0.0
        for term, frequency in term_frequencies.items():
            if not frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalizer = 1.5 * (1 - 0.75 + 0.75 * len(tokens) / average_length)
            score += inverse_frequency * frequency * 2.5 / (frequency + length_normalizer)
        if score > 0:
            scored.append((score, document.id, document))

    scored.sort(key=lambda item: (-item[0], item[1]))
    maximum = scored[0][0] if scored else 1.0
    return [
        VectorDocument(
            document.id,
            document.content,
            {
                **document.metadata,
                "keyword_score": score / maximum,
                "retrieval_channels": ["keyword"],
            },
        )
        for score, _, document in scored[:k]
    ]
