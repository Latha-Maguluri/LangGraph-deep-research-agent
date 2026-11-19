from __future__ import annotations

import json
import logging
import re
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader

from app.core.config import settings
from app.infrastructure.vector_store import ChromaVectorStore, VectorDocument

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "data" / "knowledge_sources.json"
MAX_PDF_BYTES = 30_000_000
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than chunk size.")

    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _download_pdf(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "LangGraphResearchDemo/1.0"})
    with urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        pdf_bytes = response.read(MAX_PDF_BYTES + 1)

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES}-byte ingestion limit: {url}")
    if not pdf_bytes.startswith(b"%PDF-") or (
        content_type and "pdf" not in content_type and "octet-stream" not in content_type
    ):
        raise ValueError(f"Source did not return a PDF document: {url}")
    return pdf_bytes


def _documents_from_pdf(pdf_bytes: bytes, source: dict[str, Any]) -> list[VectorDocument]:
    source_id = source["source_id"]
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    documents: list[VectorDocument] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk_index, chunk in enumerate(_chunk_text(page_text)):
            documents.append(
                VectorDocument(
                    id=f"{source_id}-p{page_number}-c{chunk_index}",
                    content=chunk,
                    metadata={
                        "source_id": source_id,
                        "source_title": source["title"],
                        "publisher": source["publisher"],
                        "publication_date": source["publication_date"],
                        "source_url": source["pdf_url"],
                        "landing_page_url": source["landing_page_url"],
                        "source_domain": urlparse(source["landing_page_url"]).hostname,
                        "focus": source["focus"],
                        "domain": "generative_ai_banking",
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No extractable text found in PDF source: {source['title']}")
    return documents


def ingest_knowledge_sources() -> int:
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"The knowledge source manifest is empty or invalid: {SOURCE_MANIFEST}")

    documents: list[VectorDocument] = []
    for source in sources:
        pdf_bytes = _download_pdf(source["pdf_url"])
        source_documents = _documents_from_pdf(pdf_bytes, source)
        logger.info(
            "Prepared %d text chunks from %s.",
            len(source_documents),
            source["title"],
        )
        documents.extend(source_documents)

    store = ChromaVectorStore(
        settings.vector_db_path,
        collection_name=settings.vector_db_collection_name,
    )
    try:
        store.add_documents(documents)
    finally:
        store.close()

    logger.info(
        "Ingested %d chunks from %d curated reports into Chroma collection %s.",
        len(documents),
        len(sources),
        settings.vector_db_collection_name,
    )
    return len(documents)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    total = ingest_knowledge_sources()
    print(f"Ingested {total} chunks into {settings.vector_db_collection_name}.")
