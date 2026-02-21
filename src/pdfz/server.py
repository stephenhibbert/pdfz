from __future__ import annotations

from fastapi import FastAPI, HTTPException

from pdfz.ingest import DuplicateDocumentError, ingest_pdf
from pdfz.models import IngestRequest, IngestResponse, PDFDocument
from pdfz.store import DocumentStore

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="PDFZ", description="LLM-native PDF retrieval engine")
store = DocumentStore()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest a PDF from a URL."""
    try:
        doc = await ingest_pdf(request.url, store)
    except DuplicateDocumentError as e:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate document: already ingested as "
            f"'{e.existing_doc.metadata.title}' (id: {e.existing_doc.id})",
        )
    return IngestResponse(
        document_id=doc.id,
        title=doc.metadata.title,
        total_pages=doc.total_pages,
        has_toc=bool(doc.toc),
    )


@app.get("/documents", response_model=list[PDFDocument])
async def list_documents():
    """List all ingested documents."""
    return store.list_all()


@app.get("/documents/{document_id}", response_model=PDFDocument)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    doc = store.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
