# PDFZ

LLM-native PDF retrieval engine. Instead of extracting text and relying on embeddings, PDFZ sends PDF pages directly to an LLM for both metadata extraction at ingest time and lossless question-answering at query time.

## How it works

1. **Ingest** - Give PDFZ a PDF URL. It downloads the file, sends the first 10 pages to Claude Sonnet, and extracts structured metadata: title, authors, date, table of contents, and a contextual summary. Duplicate PDFs are detected via SHA-256 content hash.

2. **Search** - Connect Claude Code (or any MCP client) to the PDFZ MCP server. Browse documents, inspect their table of contents, then ask questions about specific page ranges. Each query sends the actual PDF pages to the LLM - no lossy text extraction.

## Setup

```bash
# Install dependencies
uv sync

# Set your API key
export ANTHROPIC_API_KEY=sk-...
```

## Usage

### Ingestion server

Start the FastAPI server to ingest PDFs:

```bash
uv run pdfz-server
```

Ingest a PDF:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf"}'
```

List ingested documents:

```bash
curl http://localhost:8000/documents
```

### MCP server (Claude Code)

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "pdfz": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/pdfz", "pdfz-mcp"]
    }
  }
}
```

This exposes three tools to Claude Code:

| Tool | Description |
|------|-------------|
| `list_documents` | List all ingested PDFs with titles and summaries |
| `get_document_toc` | Get the table of contents for a document |
| `search_document_pages` | Ask a question about specific pages of a document |

### Evals

Run the retrieval evaluation suite:

```bash
uv sync --all-extras
uv run python evals/run_evals.py
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest a PDF from a URL. Body: `{"url": "..."}`. Returns 409 if duplicate. |
| `/documents` | GET | List all ingested documents |
| `/documents/{id}` | GET | Get a specific document by ID |

## Project structure

```
src/pdfz/
  models.py       - Pydantic data models
  pdf_utils.py    - PDF download and page extraction
  store.py        - JSON file-backed document store
  ingest.py       - LLM-powered metadata extraction pipeline
  server.py       - FastAPI ingestion server
  mcp_server.py   - MCP server for Claude Code
evals/
  run_evals.py    - Pydantic Evals retrieval suite
```
