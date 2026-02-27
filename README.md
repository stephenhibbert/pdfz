# PDFZ

LLM-native PDF retrieval engine. Instead of extracting text and relying on embeddings, PDFZ sends PDF pages directly to an LLM for metadata extraction at ingest time and clean markdown extraction at query time. The MCP server is a pure retrieval interface — your agent reasons over the returned content.

## How it works

1. **Ingest** - Give PDFZ a PDF URL. It downloads the file, sends the first 10 pages to Claude Sonnet, and extracts structured metadata: title, authors, date, table of contents, and a contextual summary. Duplicate PDFs are detected via SHA-256 content hash.

2. **Retrieve** - Connect Claude Code (or any MCP client) to the PDFZ MCP server. Follow a progressive disclosure pattern: list documents → inspect the table of contents → extract content from specific page ranges as markdown. Each extraction sends actual PDF pages to the LLM — no lossy text extraction, no embeddings.

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
| `extract_page_content` | Extract raw markdown content from specific pages of a document |

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

## Example

```commandline
❯ What is the ASL level of claude sonnet 4.6 according to the card? 

⏺ Web Search("Claude Sonnet 4.6 model card ASL level Anthropic 2026")
  ⎿  Did 1 search in 6s

⏺ Fetch(url: "https://anthropic.com/claude-sonnet-4-6-system-card", prompt: "What is the ASL (AI Safety Level) assigned to Claude
        Sonnet 4.6?")
  ⎿  Interrupted · What should Claude do instead?

❯ Use the search mcp 

∴ Thinking…

  The user wants me to use the search MCP tool. Let me check what MCP tools are available, specifically looking for a
  search-related one.

⏺ listMcpResources(List all MCP resources)
  ⎿  (No resources found)

⏺ pdfz - list_documents (MCP)
  ⎿  {
       "result": "- **System Card: Claude Sonnet 4.6** (id: 093b4f2f9273)\n  Pages: 134 | Authors: Unknown\n  Summary: This
     is a system card documenting the evaluation and safety assessment of Claude Sonnet 4.6, Anthropic's latest large
     language model released in February 2026. The document describes comprehensive capability evaluations (including
     coding, reasoning, mathematics, agentic behavior, and multilingual performance), safety evaluations, alignment
     assessments, and the rationale for releasing the model under the AI Safety Level 3 (ASL-3) Standard as defined by
     Anthropic's Responsible Scaling Policy. This document is intended for researchers, developers, policymakers, and other
     stakeholders interested in understanding the capabilities, safety profile, and deployment decisions for this AI model."
     }

⏺ pdfz - get_document_toc (MCP)(document_id: "093b4f2f9273")
  ⎿  {
       "result": "# Table of Contents: System Card: Claude Sonnet 4.6\n\n- Abstract (p. 2)\n- 1 Introduction (p. 7)\n  - 1.1
      Model training and characteristics (p. 8)\n    - 1.1.1 Training data and process (p. 8)\n    - 1.1.2 Thinking modes
     and the effort parameter (p. 8)\n    - 1.1.3 Crowd workers (p. 8)\n  - 1.2 Release decision process (p. 9)\n    - 1.2.1
      Overview (p. 9)\n    - 1.2.2 Iterative model evaluations (p. 9)\n    - 1.2.3 AI Safety Level determination process (p.
      10)\n    - 1.2.4 Sabotage risk assessment (p. 10)\n    - 1.2.5 Conclusions (p. 11)\n      - 1.2.5.1 On autonomy risks
     (p. 11)\n      - 1.2.5.2 On chemical, biological, radiological, and nuclear (CBRN) risks (p. 12)\n      - 1.2.5.3 On
     cyber risks (p. 12)\n- 2 Capabilities (p. 13)\n  - 2.1 Introduction and results summary (p. 13)\n  - 2.2 SWE-bench
     (Verified and Multilingual) (p. 15)\n  - 2.3 Terminal-Bench 2.0 (p. 15)\n  - 2.4 OpenRCA (p. 16)\n  - 2.5 τ²-bench (p.
     17)\n  - 2.6 OSWorld-Verified (p. 18)\n  - 2.7 ARC-AGI (p. 19)\n  - 2.8 GDPval-AA (p. 21)\n  - 2.9 GPQA Diamond (p.
     21)\n  - 2.10 AIME 2025 (p. 22)\n  - 2.11 MMMLU (p. 22)\n  - 2.12 Finance capabilities (p. 22)\n    - 2.12.1 Evaluation
      overview (p. 22)\n    - 2.12.2 Finance Agent (p. 23)\n    - 2.12.3 Real-World Finance (p. 24)\n    - 2.12.4
     Limitations and caveats (p. 25)\n  - 2.13 Vending-Bench 2 (p. 26)\n  - 2.14 MCP-Atlas (p. 27)\n  - 2.15 CyberGym (p.
     27)\n  - 2.16 Long context (p. 28)\n    - 2.16.1 OpenAI MRCR v2 (Multi Round Coreference Resolution) (p. 29)\n    -
     2.16.2 GraphWalks (p. 31)\n  - 2.17 Multimodal (p. 33)\n    - 2.17.1 LAB-Bench FigQA (p. 33)\n    - 2.17.2 MMMU-Pro (p.
      34)\n    - 2.17.3 CharXiv Reasoning (p. 35)\n  - 2.18 WebArena and WebArena-Verified (p. 36)\n    - 2.18.1 WebArena
     (p. 36)\n    - 2.18.2 WebArena-Verified (p. 37)\n  - 2.19 Multilingual performance (p. 40)\n    - 2.19.1 GMMLU results
     (p. 40)\n    - 2.19.2 MILU results (p. 42)\n    - 2.19.3 Findings (p. 43)\n  - 2.20 Agentic Search (p. 43)\n    -
     2.20.1 BrowseComp (p. 43)\n      - 2.20.1.1 BrowseComp (p. 43)\n      - 2.20.1.2 Test-time compute scaling on
     BrowseComp (p. 44)\n      - 2.20.1.3 Multi-agent BrowseComp (p. 44)\n    - 2.20.2 Humanity's Last Exam (p. 45)\n    -
     2.20.3 DeepSearchQA (p. 47)\n      - 2.20.3.1 DeepSearchQA with multi-agents (p. 47)\n  - 2.21 Healthcare and life
     sciences capabilities (p. 48)\n    - 2.21.1 Life sciences capabilities (p. 48)\n    - 2.21.2 MedCalc-Bench Verified (p.
      50)\n- 3 Safeguards and harmlessness (p. 52)\n  - 3.1 Single-turn evaluations (p. 52)\n    - 3.1.1 Violative request
     evaluations (p. 52)\n    - 3.1.2 Benign request evaluations (p. 53)\n    - 3.1.3 Experimental, higher-difficulty
     evaluations (p. 54)\n      - 3.1.3.1 Higher-difficulty violative request evaluations (p. 55)\n      - 3.1.3.2
     Higher-difficulty benign request evaluations (p. 55)\n  - 3.2 Ambiguous context evaluations (p. 56)\n  - 3.3 Multi-turn
      testing (p. 56)\n  - 3.4 User wellbeing evaluations (p. 58)\n    - 3.4.1 Child safety (p. 58)\n    - 3.4.2 Suicide and
      self-harm (p. 59)\n    - 3.4.3 Eating disorders (p. 62)\n  - 3.5 Bias evaluations (p. 63)\n    - 3.5.1 Political bias
     and evenhandedness (p. 63)\n    - 3.5.2 Bias Benchmark for Question Answering (p. 64)\n- 4 Alignment assessment (p.
     67)\n  - 4.1 Introduction and summary of findings (p. 67)\n  - 4.2 Reports and monitoring results from internal pilot
     use (p. 69)\n  - 4.3 Reward hacking and overly agentic actions (p. 70)\n    - 4.3.1 Overview (p. 70)\n    - 4.3.2
     Reward hacking in coding contexts (p. 70)\n    - 4.3.3 Overly agentic behavior in GUI computer use settings (p. 73)\n
     - 4.4 Training data review (p. 74)\n  - 4.5 Automated behavioral audit (p. 74)\n    - 4.5.1 Primary metrics and results
      (p. 75)\n      - 4.5.1.1 Discussion and observations (p. 80)\n    - 4.5.2 Pilot GUI computer-use investigations (p.
     84)\n    - 4.5.3 External comparisons with Petri (p. 84)\n  - 4.6 Additional behavioral testing (p. 85)\n    - 4.6.1
     Refusal to assist with AI safety R&D (p. 85)\n    - 4.6.2 Self-preference evaluation (p. 86)\n    - 4.6.3 Evidence from
      external testing with Andon Labs (p. 87)\n    - 4.6.4 Sandbagging assessment (p. 88)\n    - 4.6.5 Participation in
     junk science (p. 89)\n    - 4.6.6 Targeted sabotage capability evaluation (p. 90)\n  - 4.7 Model welfare (p. 91)\n- 5
     Agentic safety (p. 95)\n  - 5.1 Malicious use of agents (p. 95)\n    - 5.1.1 Agentic coding (p. 95)\n    - 5.1.2
     Malicious use of Claude Code (p. 95)\n    - 5.1.3 Malicious computer use (p. 96)\n  - 5.2 Prompt injection risk within
     agentic systems (p. 97)\n    - 5.2.1 External Agent Red Teaming benchmark for tool use (p. 97)\n    - 5.2.2 Robustness
     against adaptive attackers across surfaces (p. 98)\n      - 5.2.2.1 Coding (p. 98)\n      - 5.2.2.2 Computer use (p.
     99)\n      - 5.2.2.3 Browser use (p. 100)\n- 6 RSP evaluations (p. 102)\n  - 6.1 Preliminary assessment process (p.
     102)\n    - 6.1.1 Threat models and evaluation details (p. 102)\n    - 6.1.2 Result and determination (p. 102)\n  - 6.2
      CBRN evaluations (p. 102)\n    - 6.2.1 Biological risk evaluations: results (p. 103)\n      - 6.2.1.1 List of
     biological risk evaluations (p. 103)\n      - 6.2.1.2 ASL-3 evaluation results (p. 103)\n      - 6.2.1.3 ASL-4
     evaluation results (p. 104)\n      - 6.2.1.4 Safety Level determination (p. 104)\n    - 6.2.2 Biological risk
     evaluations: details (p. 104)\n      - 6.2.2.1 Long-form virology tasks (p. 104)\n      - 6.2.2.2 Multimodal virology
     (p. 105)\n      - 6.2.2.3 DNA Synthesis Screening Evasion (p. 106)\n      - 6.2.2.4 Creative Biology automated
     evaluations (p. 107)\n      - 6.2.2.5 Short-horizon computational biology tasks (p. 109)\n  - 6.3 Autonomy evaluations
     (p. 110)\n    - 6.3.1 AI R&D evaluations (p. 110)\n      - 6.3.1.1 List of AI R&D evaluations (p. 111)\n      - 6.3.1.2
      Evaluation results (p. 111)\n      - 6.3.1.3 Safety Level determination (p. 111)\n    - 6.3.2 SWE-bench Verified (hard
      subset) (p. 111)\n    - 6.3.3 Internal AI research evaluation suite 1 (p. 112)\n      - 6.3.3.1 Kernels task (p.
     112)\n      - 6.3.3.2 Time series forecasting (p. 114)\n      - 6.3.3.3 Text-based reinforcement learning task (p.
     115)\n      - 6.3.3.4 LLM training (p. 116)\n      - 6.3.3.5 Quadruped reinforcement learning (p. 117)\n      - 6.3.3.6
      Novel compiler (p. 118)\n  - 6.4 Cyber evaluations (p. 119)\n    - 6.4.1 List of cyber evaluations (p. 120)\n    -
     6.4.2 Web (p. 120)\n    - 6.4.3 Crypto (p. 121)\n    - 6.4.4 Pwn (p. 122)\n    - 6.4.5 Rev (p. 122)\n    - 6.4.6
     Network (p. 123)\n    - 6.4.7 Cybench (p. 124)\n  - 6.5 Third party assessments (p. 125)\n  - 6.6 Ongoing safety
     commitment (p. 125)\n- 7 Appendix (p. 126)\n  - 7.1 Additional automated behavioral audit figures (p. 126)\n  - 7.2
     Blocklist used for Humanity's Last Exam (p. 133)"
     }

⏺ pdfz - extract_page_content (MCP)(document_id: "093b4f2f9273", page_start: 9, page_end: 12)
  ⎿  {
       "result": "## 1.2.3 AI Safety Level Determination Process\n\n**Released under:** ASL-3 (AI Safety Level 3)\n\nClaude Sonnet 4.6 implemented ASL-3 protections based on the model's demonstrated capabilities. The model performed at or below the level of Claude Opus 4.6 on automated evaluations, and since Opus 4.6 was deployed with ASL-3 safeguards, Sonnet 4.6 does not push the capability frontier beyond it and is released under the same safety standard.\n\n| Model | Safety Level |\n|-------|-------------|\n| Claude Opus 4.6 | ASL-3 |\n| Claude Sonnet 4.6 | ASL-3 |"
     }

                                                                                               11:58 AM claude-sonnet-4-6
⏺ Claude Sonnet 4.6 is assigned ASL-3 (AI Safety Level 3).

  The system card notes that Sonnet 4.6 performs at or below Claude Opus 4.6 on automated evaluations, and since Opus 4.6 was
  already deployed with ASL-3 safeguards, Sonnet 4.6 does not push the capability frontier beyond it and is released under the
  same standard.

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all

```