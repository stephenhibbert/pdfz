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

from fastmcp import FastMCP
from pydantic_ai import Agent, BinaryContent

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
async def extract_page_content(
    document_id: str,
    page_start: int,
    page_end: int,
) -> str:
    """Extract the content of specific pages from a PDF document as markdown.

    Use the TOC to identify relevant page ranges, then call this tool to
    retrieve the raw content. Returns clean markdown preserving all text,
    tables, and structure from the pages — you reason over the content.
    Make multiple calls for different sections as needed.

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
    page_range_bytes = extract_page_range(pdf_bytes, page_start, page_end)

    result = await _get_extraction_agent().run(
        [
            BinaryContent(data=page_range_bytes, media_type="application/pdf"),
            f"Extract all content from pages {page_start}-{page_end} of "
            f"'{doc.metadata.title}' as structured markdown.",
        ]
    )
    return result.output


def main():
    mcp.run()


if __name__ == "__main__":
    main()
