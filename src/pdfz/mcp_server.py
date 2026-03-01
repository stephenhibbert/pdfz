from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(
    service_name="pdfz-mcp",
    environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

import io

from fastmcp import FastMCP
from pydantic_ai import Agent, BinaryContent
from pypdf import PdfReader

from pdfz import page_cache
from pdfz.pdf_utils import download_pdf, extract_page_range
from pdfz.store import DocumentStore

mcp = FastMCP(name="pdfz")
store = DocumentStore()

EXTRACTION_SYSTEM_PROMPT = (
    "You are a document content extractor. Extract all content from the "
    "provided PDF pages as clean, structured markdown. Preserve all headings, "
    "tables, lists, data points, and figures exactly as presented. "
    "Do not summarise, interpret, or omit any content."
)

_extraction_agent: Agent[None, str] | None = None


def _get_extraction_agent() -> Agent[None, str]:
    global _extraction_agent
    if _extraction_agent is None:
        _extraction_agent = Agent(
            "anthropic:claude-sonnet-4-5-20250929",
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
    return _extraction_agent


@mcp.tool()
def list_documents() -> str:
    """List all ingested PDF documents with their IDs, titles, and summaries.

    Start here to discover what documents are available. Each entry includes
    the document ID needed for subsequent tool calls, the title, page count,
    authors, and a summary to help you decide which document to explore.
    """
    docs = store.list_all()
    if not docs:
        return "No documents have been ingested yet."
    lines = []
    for doc in docs:
        lines.append(
            f"- **{doc.metadata.title}** (id: `{doc.id}`)\n"
            f"  Pages: {doc.total_pages} | "
            f"Authors: {', '.join(doc.metadata.authors) or 'Unknown'}\n"
            f"  Summary: {doc.contextual_summary}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def get_document_toc(document_id: str) -> str:
    """Get the table of contents for a specific document.

    Use this after list_documents to understand the document structure and
    identify which page ranges contain the information you need. The TOC
    includes section headings and page numbers to guide your retrieval.

    Args:
        document_id: The ID of the document (from list_documents).
    """
    doc = store.get(document_id)
    if doc is None:
        return f"Document {document_id} not found."
    if not doc.toc:
        return f"No table of contents available for '{doc.metadata.title}'."
    return f"# Table of Contents: {doc.metadata.title}\n\n{doc.toc}"


@mcp.tool()
async def find_pages(
    document_id: str,
    query: str,
) -> str:
    """Find which pages contain a specific term or phrase (like Ctrl+F in a PDF viewer).

    Performs a fast, case-insensitive text search across all pages and returns
    the matching page numbers with brief context snippets. Use the returned
    page numbers with extract_page_content to retrieve the full content.

    Args:
        document_id: The ID of the document to search.
        query: The term or phrase to search for.
    """
    doc = store.get(document_id)
    if doc is None:
        return f"Document {document_id} not found."

    pdf_bytes = await download_pdf(doc.source_url)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    query_lower = query.lower()
    results: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text_lower = text.lower()
        if query_lower not in text_lower:
            continue

        # Collect up to 3 context snippets per page
        snippets: list[str] = []
        idx = 0
        while len(snippets) < 3:
            pos = text_lower.find(query_lower, idx)
            if pos == -1:
                break
            start = max(0, pos - 60)
            end = min(len(text), pos + len(query) + 60)
            snippet = " ".join(text[start:end].split())
            snippets.append(f"  …{snippet}…")
            idx = pos + 1

        count = text_lower.count(query_lower)
        results.append(
            f"Page {page_num} ({count} occurrence{'s' if count != 1 else ''})\n"
            + "\n".join(snippets)
        )

    if not results:
        return f'No pages found containing "{query}".'

    header = f'Found "{query}" on {len(results)} page{"s" if len(results) != 1 else ""}:\n\n'
    return header + "\n\n".join(results)


@mcp.tool()
async def extract_page_content(
    document_id: str,
    page_start: int,
    page_end: int,
) -> str:
    """Extract the content of specific pages from a PDF document as markdown.

    Pages are extracted individually and cached — cached pages return in
    milliseconds. For the fastest first-response, call with a single page at
    a time (page_start == page_end); the agent can make multiple calls in
    sequence and benefit from caching on any revisited pages.

    Use the TOC or find_pages to identify which pages to retrieve, then call
    this tool to get the raw content. You reason over the returned markdown.

    Args:
        document_id: The ID of the document to search.
        page_start: First page number (1-indexed, inclusive).
        page_end: Last page number (1-indexed, inclusive). Max 10 page span.
    """
    doc = store.get(document_id)
    if doc is None:
        return f"Document {document_id} not found."

    if page_end - page_start > 9:
        return "Please limit page range to 10 pages at a time."
    if page_start < 1:
        return "page_start must be >= 1."
    if doc.total_pages and page_end > doc.total_pages:
        return f"Document only has {doc.total_pages} pages."

    pdf_bytes = await download_pdf(doc.source_url)
    parts: list[str] = []

    for page_num in range(page_start, page_end + 1):
        cached = page_cache.get(document_id, page_num)
        if cached:
            parts.append(f"## Page {page_num}\n\n{cached}")
            continue

        page_bytes = extract_page_range(pdf_bytes, page_num, page_num)
        result = await _get_extraction_agent().run(
            [
                BinaryContent(data=page_bytes, media_type="application/pdf"),
                f"Extract all content from page {page_num} of "
                f"'{doc.metadata.title}' as structured markdown.",
            ]
        )
        page_cache.put(document_id, page_num, result.output)
        parts.append(f"## Page {page_num}\n\n{result.output}")

    return "\n\n---\n\n".join(parts)


@mcp.tool()
async def extract_with_focus(
    document_id: str,
    page_start: int,
    page_end: int,
    focus: str,
) -> str:
    """Extract content from a page range with a specific extraction focus.

    ⚠ SLOW — sends all pages in a single LLM call to preserve cross-page
    context. Only use when that context is essential, e.g. footnotes that
    span pages, recurring notation, multi-page tables, or layout patterns
    that need to be understood together. For standard extraction use
    extract_page_content instead, which is cached and much faster.

    Args:
        document_id: The ID of the document.
        page_start: First page (1-indexed, inclusive).
        page_end: Last page (1-indexed, inclusive). Max 10 page span.
        focus: What to pay particular attention to during extraction, e.g.
               "cross-page footnotes", "mathematical notation in figures",
               "table structures that span multiple pages".
    """
    doc = store.get(document_id)
    if doc is None:
        return f"Document {document_id} not found."

    if page_end - page_start > 9:
        return "Please limit page range to 10 pages at a time."
    if page_start < 1:
        return "page_start must be >= 1."
    if doc.total_pages and page_end > doc.total_pages:
        return f"Document only has {doc.total_pages} pages."

    pdf_bytes = await download_pdf(doc.source_url)
    page_range_bytes = extract_page_range(pdf_bytes, page_start, page_end)

    result = await _get_extraction_agent().run(
        [
            BinaryContent(data=page_range_bytes, media_type="application/pdf"),
            f"Extract all content from pages {page_start}–{page_end} of "
            f"'{doc.metadata.title}' as structured markdown. "
            f"Pay particular attention to: {focus}.",
        ]
    )
    return result.output


def main():
    mcp.run()


if __name__ == "__main__":
    main()
