from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(
    service_name="pdfz-server",
    environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from pdfz.auth import API_TOKEN, APP_URL, SECRET_KEY, send_magic_link, verify_token
from pdfz.eval_runner import get_latest_results, run_evals
from pdfz.ingest import DuplicateDocumentError, ingest_pdf
from pdfz.models import IngestRequest, IngestResponse, PDFDocument
from pdfz.store import DocumentStore

app = FastAPI(title="PDFZ", description="LLM-native PDF retrieval engine")
logfire.instrument_fastapi(app)

# Auth enabled only in production (when RAILWAY_ENVIRONMENT is set)
_AUTH_ENABLED = bool(os.environ.get("RAILWAY_ENVIRONMENT"))


async def _require_auth(request: Request, call_next):
    """Redirect unauthenticated requests to the login page. Skipped in local dev."""
    if not _AUTH_ENABLED:
        return await call_next(request)
    path = request.url.path
    if path.startswith("/auth") or path == "/health":
        return await call_next(request)
    # Accept bearer token for API/MCP access
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer ") and auth_header[7:] == API_TOKEN:
        return await call_next(request)
    if not request.session.get("email"):
        return RedirectResponse(url="/auth/login", status_code=302)
    return await call_next(request)


app.add_middleware(BaseHTTPMiddleware, dispatch=_require_auth)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

store = DocumentStore()
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.get("/auth/login", response_class=HTMLResponse)
async def auth_login(request: Request, sent: str = ""):
    ctx: dict = {"request": request, "message": None, "message_type": ""}
    if sent:
        ctx["message"] = (
            "Check your inbox — a magic link is on its way. "
            "It expires in 1 hour."
        )
        ctx["message_type"] = "success"
    return templates.TemplateResponse("login.html", ctx)


@app.post("/auth/send", response_class=HTMLResponse)
async def auth_send(request: Request):
    form = await request.form()
    email = str(form.get("email", "")).strip().lower()
    # Always redirect to avoid email enumeration
    send_magic_link(email)
    return RedirectResponse(url="/auth/login?sent=1", status_code=303)


@app.get("/auth/verify")
async def auth_verify(request: Request, token: str = ""):
    email = verify_token(token)
    if not email:
        ctx = {
            "request": request,
            "message": "This link is invalid or has expired. Please request a new one.",
            "message_type": "error",
        }
        return templates.TemplateResponse("login.html", ctx)
    request.session["email"] = email
    return RedirectResponse(url="/", status_code=303)


@app.post("/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/auth/login", status_code=303)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_token": API_TOKEN,
        "app_url": APP_URL,
    })


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
