from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

from pdfz.models import PDFDocument, PDFMetadata
from pdfz.pdf_utils import download_pdf, extract_page_range, get_total_pages
from pdfz.store import DocumentStore


class DuplicateDocumentError(Exception):
    """Raised when a PDF with the same content hash already exists."""

    def __init__(self, existing_doc: PDFDocument):
        self.existing_doc = existing_doc
        super().__init__(
            f"Document already ingested as '{existing_doc.metadata.title}' "
            f"(id: {existing_doc.id})"
        )


class ExtractionResult(BaseModel):
    """Structured output the LLM returns from analyzing the PDF."""

    title: str
    date_published: str | None = None  # ISO format or None
    authors: list[str] = Field(default_factory=list)
    toc: str = Field(
        description=(
            "The full table of contents as a markdown string. Use nested "
            "markdown lists with indentation to show hierarchy. Include "
            "page numbers where available, e.g. '- 1 Introduction (p. 7)'."
        )
    )
    contextual_summary: str = Field(
        description=(
            "A 2-4 sentence summary: what is this document, "
            "what topics does it cover, who is the intended audience."
        )
    )


EXTRACTION_SYSTEM_PROMPT = (
    "You are a document analysis assistant. You will receive the first "
    "pages of a PDF document. Extract the following structured information:\n\n"
    "1. **Title** of the document\n"
    "2. **Publication date** (ISO format YYYY-MM-DD if available, null otherwise)\n"
    "3. **Authors** (list of names)\n"
    "4. **Table of contents** as a markdown string. Transcribe the full TOC "
    "from the document using nested markdown lists. Include page numbers. "
    "Example format:\n"
    "   - 1 Introduction (p. 7)\n"
    "     - 1.1 Background (p. 8)\n"
    "     - 1.2 Overview (p. 10)\n"
    "   - 2 Methods (p. 15)\n"
    "   If there is no explicit TOC, infer one from section headings.\n"
    "5. **Contextual summary** - what this document is, what it covers, "
    "and who it is written for (2-4 sentences).\n"
)

_extraction_agent: Agent[None, ExtractionResult] | None = None


def _get_extraction_agent() -> Agent[None, ExtractionResult]:
    global _extraction_agent
    if _extraction_agent is None:
        _extraction_agent = Agent(
            "anthropic:claude-sonnet-4-5-20250929",
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
    return _extraction_agent


async def ingest_pdf(url: str, store: DocumentStore) -> PDFDocument:
    """Download a PDF, extract metadata via LLM, and store it.

    Raises:
        DuplicateDocumentError: If a PDF with the same content hash already exists.
    """
    pdf_bytes = await download_pdf(url)
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    existing = store.find_by_hash(content_hash)
    if existing is not None:
        raise DuplicateDocumentError(existing)

    total_pages = get_total_pages(pdf_bytes)

    max_pages = min(15, total_pages)
    first_pages_bytes = extract_page_range(pdf_bytes, 1, max_pages)

    result = await _get_extraction_agent().run(
        [
            BinaryContent(data=first_pages_bytes, media_type="application/pdf"),
            "Analyze this PDF document and extract structured metadata.",
        ]
    )

    extraction = result.output

    # Parse date if provided
    parsed_date = None
    if extraction.date_published:
        from datetime import date

        try:
            parsed_date = date.fromisoformat(extraction.date_published)
        except ValueError:
            pass

    doc = PDFDocument(
        content_hash=content_hash,
        metadata=PDFMetadata(
            title=extraction.title,
            date_published=parsed_date,
            authors=extraction.authors,
            source_url=url,
        ),
        toc=extraction.toc,
        contextual_summary=extraction.contextual_summary,
        source_url=url,
        total_pages=total_pages,
    )

    return store.add(doc)
