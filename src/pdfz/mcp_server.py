"""PDFZ MCP server — thin client that calls the deployed PDFZ API."""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from fastmcp import FastMCP
import httpx

mcp = FastMCP(name="pdfz")

API_URL = os.environ.get("PDFZ_API_URL", "http://localhost:8000")
API_TOKEN = os.environ.get("PDFZ_API_TOKEN", "")


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, headers=_headers(), timeout=60.0)


@mcp.tool()
def list_documents() -> str:
    """List all ingested PDF documents with their IDs, titles, and summaries.

    Call this first to see what documents are available for searching.
    """
    with _client() as client:
        res = client.get("/documents")
        res.raise_for_status()
        docs = res.json()

    if not docs:
        return "No documents have been ingested yet."

    lines = []
    for doc in docs:
        meta = doc.get("metadata", {})
        authors = ", ".join(meta.get("authors", [])) or "Unknown"
        lines.append(
            f"- **{meta.get('title', 'Untitled')}** (id: {doc['id']})\n"
            f"  Pages: {doc.get('total_pages', '?')} | Authors: {authors}\n"
            f"  Summary: {doc.get('contextual_summary', '')}"
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
    with _client() as client:
        res = client.get(f"/documents/{document_id}")
        if res.status_code == 404:
            return f"Document {document_id} not found."
        res.raise_for_status()
        doc = res.json()

    toc = doc.get("toc", "")
    title = doc.get("metadata", {}).get("title", "Unknown")
    if not toc:
        return f"No table of contents available for '{title}'."
    return f"# Table of Contents: {title}\n\n{toc}"


@mcp.tool()
def search_document_pages(
    document_id: str,
    page_start: int,
    page_end: int,
    question: str,
) -> str:
    """Search specific pages of a PDF document by asking a question.

    The server downloads the PDF, extracts the requested page range, sends it
    to an LLM along with the question, and returns the answer. Use the TOC to
    determine which pages to search.

    Args:
        document_id: The ID of the document to search.
        page_start: First page number (1-indexed, inclusive).
        page_end: Last page number (1-indexed, inclusive). Max 10 page span.
        question: The question to ask about the content of those pages.
    """
    with _client() as client:
        res = client.post(
            "/api/search",
            json={
                "document_id": document_id,
                "page_start": page_start,
                "page_end": page_end,
                "question": question,
            },
            timeout=120.0,
        )
        if res.status_code == 404:
            return f"Document {document_id} not found."
        if res.status_code == 400:
            return res.json().get("detail", "Bad request")
        res.raise_for_status()
        return res.json().get("answer", "")


def main():
    mcp.run()


if __name__ == "__main__":
    main()
