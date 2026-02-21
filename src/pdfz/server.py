from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pdfz.eval_runner import get_latest_results, run_evals
from pdfz.ingest import DuplicateDocumentError, ingest_pdf
from pdfz.models import IngestRequest, IngestResponse, PDFDocument
from pdfz.store import DocumentStore

app = FastAPI(title="PDFZ", description="LLM-native PDF retrieval engine")
store = DocumentStore()
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


@app.post("/api/evals/run")
async def run_evals_endpoint(background_tasks: BackgroundTasks):
    """Trigger an eval run in the background."""
    current = get_latest_results()
    if current and current.get("status") == "running":
        raise HTTPException(status_code=409, detail="Eval run already in progress")
    background_tasks.add_task(run_evals)
    return {"status": "started"}


@app.get("/api/evals/latest")
async def get_evals_latest():
    """Get the latest eval run results."""
    results = get_latest_results()
    if results is None:
        return {"status": "none"}
    return results


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
