from __future__ import annotations

import io

import httpx
from pypdf import PdfReader, PdfWriter


async def download_pdf(url: str) -> bytes:
    """Download PDF bytes from a URL."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def get_total_pages(pdf_bytes: bytes) -> int:
    """Return total page count from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)


def extract_page_range(pdf_bytes: bytes, start: int, end: int) -> bytes:
    """Extract pages [start, end] (1-indexed, inclusive) as new PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()

    start_idx = max(0, start - 1)
    end_idx = min(len(reader.pages), end)

    for i in range(start_idx, end_idx):
        writer.add_page(reader.pages[i])

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()
