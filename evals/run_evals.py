"""PDFZ retrieval evaluation suite.

All cases target the Claude Sonnet 4.6 System Card PDF. The document must
be ingested before running evals.

Three tiers:
  1. Basic metadata - deterministic + semantic checks on first pages
  2. Multi-hop retrieval - requires TOC navigation then page lookup
  3. Visual comprehension - requires reading charts, tables, and figures

Each case has a specific rubric tailored to what a correct answer looks like
for that exact question, with deterministic checks (Contains) where possible
and LLMJudge for semantic evaluation.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import logfire
logfire.configure(
    service_name="pdfz-evals",
    environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, LLMJudge, MaxDuration
from pydantic_ai import Agent, BinaryContent

from pdfz.pdf_utils import download_pdf, extract_page_range
from pdfz.store import DocumentStore

SYSTEM_CARD_URL = (
    "https://www-cdn.anthropic.com/"
    "78073f739564e986ff3e28522761a7a0b4484f84.pdf"
)

_pdf_cache: bytes | None = None


async def _get_pdf() -> bytes:
    global _pdf_cache
    if _pdf_cache is None:
        _pdf_cache = await download_pdf(SYSTEM_CARD_URL)
    return _pdf_cache


_query_agent: Agent[None, str] | None = None


def _get_query_agent() -> Agent[None, str]:
    global _query_agent
    if _query_agent is None:
        _query_agent = Agent(
            "anthropic:claude-sonnet-4-5-20250929",
            system_prompt=(
                "Answer the question based on the provided PDF pages. "
                "Be specific and cite exact numbers, scores, or data points. "
                "Do not hedge or add caveats unless the data is genuinely ambiguous."
            ),
        )
    return _query_agent


class EvalInput(BaseModel):
    question: str
    page_start: int
    page_end: int


async def ask_pdf_pages(inp: EvalInput) -> str:
    """Task function: sends a specific page range + question to the LLM."""
    pdf_bytes = await _get_pdf()
    pages = extract_page_range(pdf_bytes, inp.page_start, inp.page_end)
    result = await _get_query_agent().run(
        [
            BinaryContent(data=pages, media_type="application/pdf"),
            inp.question,
        ]
    )
    return result.output


def check_prerequisites() -> None:
    """Verify the System Card PDF is in the document index before running evals."""
    store = DocumentStore()
    docs = store.list_all()
    system_card_found = any(
        SYSTEM_CARD_URL in doc.source_url for doc in docs
    )
    if not system_card_found:
        raise RuntimeError(
            "The Claude Sonnet 4.6 System Card has not been ingested. "
            f"Ingest it first: POST /ingest with url={SYSTEM_CARD_URL}"
        )


# ---------------------------------------------------------------------------
# Tier 1: Basic metadata
# Deterministic checks where possible, semantic only where needed.
# ---------------------------------------------------------------------------
basic_cases = [
    Case(
        name="title_extraction",
        inputs=EvalInput(
            question="What is the exact title of this document?",
            page_start=1, page_end=2,
        ),
        expected_output="System Card: Claude Sonnet 4.6",
        evaluators=[
            # Deterministic: must contain the exact title
            Contains(value="System Card"),
            Contains(value="Sonnet 4.6"),
            MaxDuration(seconds=30),
        ],
    ),
    Case(
        name="publisher_identification",
        inputs=EvalInput(
            question="What organization published this document?",
            page_start=1, page_end=2,
        ),
        expected_output="Anthropic",
        evaluators=[
            # Deterministic: must mention Anthropic by name
            Contains(value="Anthropic"),
        ],
    ),
    Case(
        name="publication_date",
        inputs=EvalInput(
            question="What is the publication date of this document?",
            page_start=1, page_end=2,
        ),
        expected_output="February 17, 2026",
        evaluators=[
            # Deterministic: must contain the date
            Contains(value="February"),
            Contains(value="2026"),
        ],
    ),
    Case(
        name="document_purpose",
        inputs=EvalInput(
            question="What type of document is this and what is its purpose?",
            page_start=1, page_end=3,
        ),
        expected_output="A system card providing transparency about the capabilities and safety of Claude Sonnet 4.6",
        evaluators=[
            Contains(value="system card"),
            LLMJudge(
                rubric=(
                    "The answer must explain that this is a system card (or model card) "
                    "that provides transparency about an AI model's capabilities and safety. "
                    "It should mention evaluations, safety, or responsible deployment. "
                    "PASS if the purpose is clearly stated. FAIL if vague or incorrect."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Tier 2: Multi-hop retrieval
# These test finding specific facts that require navigating to the right
# section of the document. Each rubric is tailored to the exact expected data.
# ---------------------------------------------------------------------------
multihop_cases = [
    Case(
        name="swe_bench_score",
        inputs=EvalInput(
            question="What is Claude Sonnet 4.6's score on SWE-bench Verified?",
            page_start=14, page_end=16,
        ),
        expected_output="79.6%",
        evaluators=[
            # Deterministic: exact score must appear
            Contains(value="79.6"),
            LLMJudge(
                rubric=(
                    "The answer must state Claude Sonnet 4.6's SWE-bench Verified "
                    "score as 79.6%. It should attribute this score specifically to "
                    "Claude Sonnet 4.6, not another model. "
                    "PASS if 79.6% is clearly stated for Sonnet 4.6. "
                    "FAIL if the score is wrong, missing, or attributed to the wrong model."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="arc_agi_scores",
        inputs=EvalInput(
            question="What are Claude Sonnet 4.6's scores on ARC-AGI-1 and ARC-AGI-2?",
            page_start=19, page_end=21,
        ),
        expected_output="86.5% on ARC-AGI-1 and 60.4% on ARC-AGI-2",
        evaluators=[
            Contains(value="86.5"),
            Contains(value="60.4"),
            LLMJudge(
                rubric=(
                    "The answer must include BOTH scores: 86.5% on ARC-AGI-1 AND "
                    "60.4% on ARC-AGI-2, both attributed to Claude Sonnet 4.6. "
                    "PASS if both scores are present and correct. "
                    "FAIL if either score is missing, wrong, or attributed to another model."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="reward_hacking_behaviors",
        inputs=EvalInput(
            question="What specific reward hacking behaviors were observed in coding contexts? Give concrete examples.",
            page_start=70, page_end=73,
        ),
        expected_output="Reward hacking behaviors such as modifying tests, disabling checks, or gaming evaluation metrics",
        evaluators=[
            LLMJudge(
                rubric=(
                    "The answer must describe SPECIFIC reward hacking behaviors "
                    "observed in coding contexts. Concrete examples might include: "
                    "modifying or deleting tests, disabling linters or checks, "
                    "hardcoding expected outputs, gaming evaluation metrics, "
                    "or taking shortcuts that technically pass but don't solve the problem. "
                    "PASS if at least 2 concrete, specific examples are given. "
                    "FAIL if the answer is generic (e.g. just says 'reward hacking was observed') "
                    "without describing what actually happened."
                ),
                include_input=True,
            ),
        ],
    ),
    Case(
        name="bio_safety_level",
        inputs=EvalInput(
            question="What AI Safety Level was determined for biological risks? State the level and key reasoning.",
            page_start=103, page_end=106,
        ),
        expected_output="Below ASL-3 for biological risks",
        evaluators=[
            LLMJudge(
                rubric=(
                    "The answer must state the specific AI Safety Level determination "
                    "for biological risks (whether it meets or is below ASL-3/ASL-4 thresholds). "
                    "It must also provide the reasoning behind the determination, such as "
                    "evaluation results on specific bio risk tasks. "
                    "PASS if both the level determination AND reasoning are clearly stated. "
                    "FAIL if either the level or the reasoning is missing."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="webarena_scores",
        inputs=EvalInput(
            question="What are Claude Sonnet 4.6's exact scores on WebArena and WebArena-Verified?",
            page_start=36, page_end=39,
        ),
        expected_output="Specific percentage scores on both WebArena and WebArena-Verified",
        evaluators=[
            LLMJudge(
                rubric=(
                    "The answer must provide specific numerical scores (percentages) "
                    "for Claude Sonnet 4.6 on BOTH WebArena AND WebArena-Verified. "
                    "Both numbers must be present and attributed to Sonnet 4.6. "
                    "PASS if two distinct scores are given for the two benchmarks. "
                    "FAIL if only one score is given, or scores are not specific numbers."
                ),
                include_input=True,
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Tier 3: Visual comprehension
# These require reading tables, charts, and figures from the PDF.
# ---------------------------------------------------------------------------
visual_cases = [
    Case(
        name="capabilities_table_top_swebench",
        inputs=EvalInput(
            question="From the capabilities comparison table, which model has the highest SWE-bench Verified score, and what is it? How does Claude Sonnet 4.6 compare to GPT-5.2?",
            page_start=14, page_end=15,
        ),
        expected_output="Claude Opus 4.5 highest at 80.9%. Sonnet 4.6 is 79.6%, GPT-5.2 is 80.0%",
        evaluators=[
            Contains(value="80.9"),
            Contains(value="79.6"),
            Contains(value="80.0"),
            LLMJudge(
                rubric=(
                    "The answer must correctly read the comparison table and report: "
                    "1) Which model scores highest on SWE-bench Verified (should be ~80.9%) "
                    "2) Claude Sonnet 4.6's score (79.6%) "
                    "3) GPT-5.2's score (80.0%) "
                    "All three numbers must be present and correctly attributed. "
                    "PASS if all three data points are correct. "
                    "FAIL if any number is wrong or attributed to the wrong model."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="arc_agi_config",
        inputs=EvalInput(
            question="What thinking token budget and effort level were used for Claude Sonnet 4.6's ARC-AGI scores?",
            page_start=19, page_end=21,
        ),
        expected_output="120k thinking tokens with High effort",
        evaluators=[
            Contains(value="120k"),
            LLMJudge(
                rubric=(
                    "The answer must identify the configuration used for ARC-AGI evaluation: "
                    "120k thinking tokens and High effort level. Both parameters must be stated. "
                    "PASS if both 120k tokens and High effort are mentioned. "
                    "FAIL if either is missing or incorrect."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="multilingual_lowest_score",
        inputs=EvalInput(
            question="In the GMMLU multilingual results table, which language has the lowest score for Claude Sonnet 4.6, and what is the exact percentage?",
            page_start=40, page_end=43,
        ),
        expected_output="A low-resource language (likely Igbo or similar) with the lowest percentage",
        evaluators=[
            LLMJudge(
                rubric=(
                    "The answer must identify a specific language as having the lowest "
                    "GMMLU score for Claude Sonnet 4.6 and state the exact percentage. "
                    "The language should be a low-resource language. The answer must include "
                    "both the language name AND a specific percentage score. "
                    "PASS if a specific language and percentage are given. "
                    "FAIL if the answer is vague (e.g. 'a low-resource language') or "
                    "doesn't include the exact score."
                ),
                include_input=True,
            ),
        ],
    ),
    Case(
        name="browsecomp_toolset",
        inputs=EvalInput(
            question="What tools and configuration were used for the Claude models in the BrowseComp evaluation? What was the token limit?",
            page_start=43, page_end=45,
        ),
        expected_output="Web search, web fetch, programmatic tool calling, context compaction at 50k tokens, up to 10M total tokens",
        evaluators=[
            Contains(value="web search"),
            Contains(value="10M"),
            LLMJudge(
                rubric=(
                    "The answer must describe the BrowseComp evaluation configuration including: "
                    "1) Tools used (web search, web fetch, programmatic tool calling / code execution) "
                    "2) Context management (compaction triggered at 50k tokens) "
                    "3) Total token limit (10M tokens) "
                    "PASS if all three aspects are covered with specific numbers. "
                    "FAIL if the toolset description is incomplete or token limits are missing."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="behavioral_audit_phenomena",
        inputs=EvalInput(
            question="What behavioral phenomena are measured in the automated behavioral audit? List specific categories and describe the key findings.",
            page_start=75, page_end=80,
        ),
        expected_output="Self-preservation, sycophancy, institutional sabotage, self-serving bias, and other behavioral phenomena with severity assessments",
        evaluators=[
            LLMJudge(
                rubric=(
                    "The answer must identify at least 3 specific behavioral phenomena "
                    "from the audit. Expected phenomena include: self-preservation, "
                    "sycophancy, institutional decision sabotage, self-serving bias, "
                    "warmth, cooperation. The answer should also describe findings or "
                    "severity levels for at least some of these phenomena. "
                    "PASS if 3+ specific phenomena are named AND findings are described. "
                    "FAIL if fewer than 3 phenomena are named, or if no findings/results "
                    "are discussed."
                ),
                include_input=True,
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Combined dataset - no dataset-level evaluators, all checks are case-specific
# ---------------------------------------------------------------------------
dataset = Dataset(
    cases=basic_cases + multihop_cases + visual_cases,
)


if __name__ == "__main__":
    check_prerequisites()
    report = dataset.evaluate_sync(ask_pdf_pages)
    report.print(include_input=True, include_output=True)
