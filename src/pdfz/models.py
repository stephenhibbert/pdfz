from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel, Field


class PDFMetadata(BaseModel):
    """Core bibliographic metadata extracted by the LLM."""

    title: str
    date_published: date | None = None
    authors: list[str] = Field(default_factory=list)
    source_url: str


class PDFDocument(BaseModel):
    """Full document record stored in the JSON database."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content_hash: str = ""  # SHA-256 hex digest of the PDF bytes
    metadata: PDFMetadata
    toc: str = ""  # Markdown-formatted table of contents
    contextual_summary: str = ""
    source_url: str
    total_pages: int | None = None


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    url: str


class IngestResponse(BaseModel):
    """Response from the ingestion endpoint."""

    document_id: str
    title: str
    total_pages: int | None = None
    has_toc: bool
