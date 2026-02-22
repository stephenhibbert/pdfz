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

QUERY_SYSTEM_PROMPT = (
    "You are a document research assistant. You will receive specific "
    "pages from a PDF document along with a question. Answer the question "
    "based solely on the content of the provided pages. If the answer is "
    "not found in the provided pages, say so clearly."
)

_query_agent: Agent[None, str] | None = None


def _get_query_agent() -> Agent[None, str]:
    global _query_agent
    if _query_agent is None:
        _query_agent = Agent(
            "anthropic:claude-sonnet-4-5-20250929",
            system_prompt=QUERY_SYSTEM_PROMPT,
        )
    return _query_agent


@mcp.tool()
def list_documents() -> str:
    """List all ingested PDF documents with their IDs, titles, and summaries.

    Call this first to see what documents are available for searching.
    """
    docs = store.list_all()
    if not docs:
        return "No documents have been ingested yet."
    lines = []
    for doc in docs:
        lines.append(
            f"- **{doc.metadata.title}** (id: {doc.id})\n"
            f"  Pages: {doc.total_pages} | "
            f"Authors: {', '.join(doc.metadata.authors) or 'Unknown'}\n"
            f"  Summary: {doc.contextual_summary}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def get_document_toc(document_id: str) -> str:
    """Get the table of contents for a specific document.

    Use this to understand the structure of a document and find
    which pages to search for specific topics.

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
async def search_document_pages(
    document_id: str,
    page_start: int,
    page_end: int,
    question: str,
) -> str:
    """Search specific pages of a PDF document by asking a question.

    Downloads the PDF, extracts the requested page range, sends it to an LLM
    along with the question, and returns the answer. Use the TOC to determine
    which pages to search.

    Args:
        document_id: The ID of the document to search.
        page_start: First page number (1-indexed, inclusive).
        page_end: Last page number (1-indexed, inclusive). Max 10 page span.
        question: The question to ask about the content of those pages.
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

    result = await _get_query_agent().run(
        [
            f"Question about pages {page_start}-{page_end} of "
            f"'{doc.metadata.title}':\n\n{question}",
            BinaryContent(data=page_range_bytes, media_type="application/pdf"),
        ]
    )
    return result.output


def main():
    mcp.run()


if __name__ == "__main__":
    main()
