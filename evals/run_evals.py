"""PDFZ retrieval evaluation suite.

All cases target the Claude Sonnet 4.6 System Card PDF. The document must
be ingested before running evals.

Four tiers:
  1. Basic metadata - deterministic + semantic checks on first pages
  2. Multi-hop retrieval - requires TOC navigation then page lookup
  3. Visual comprehension - requires reading charts, tables, and figures
  4. Retrieval accuracy - agent selects pages from TOC, measures page recall

Each case has a specific rubric tailored to what a correct answer looks like
for that exact question, with deterministic checks (Contains) where possible
and LLMJudge for semantic evaluation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import logfire
logfire.configure(
    service_name="pdfz-evals",
    environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
)
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()

from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Contains,
    Evaluator,
    EvaluatorContext,
    EvaluationReason,
    LLMJudge,
    MaxDuration,
)
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


# ---------------------------------------------------------------------------
# Tier 4: Retrieval accuracy - models, agents, evaluators, task function
# ---------------------------------------------------------------------------

class RetrievalInput(BaseModel):
    question: str
    gold_pages: list[int] = Field(
        description="Pages that contain the answer (ground truth labels)."
    )


class RetrievalOutput(BaseModel):
    pages_fetched: list[int] = Field(
        description="Pages the retrieval agent chose from the TOC."
    )
    answer: str = Field(
        description="The query agent's answer from the fetched pages."
    )


class PageSelection(BaseModel):
    reasoning: str = Field(
        description="Brief explanation of why these pages were selected."
    )
    pages: list[int] = Field(
        description="Page numbers to fetch (1-indexed)."
    )


@dataclass
class PageRecall(Evaluator):
    """Measures what fraction of gold (answer-containing) pages were fetched."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        gold = set(ctx.inputs.gold_pages)
        fetched = set(ctx.output.pages_fetched)
        hits = gold & fetched
        recall = len(hits) / len(gold) if gold else 1.0
        return EvaluationReason(
            value=recall >= 0.5,
            reason=(
                f"Page recall: {recall:.0%} ({len(hits)}/{len(gold)}). "
                f"Gold: {sorted(gold)}, Fetched: {sorted(fetched)}"
            ),
        )


@dataclass
class AnswerContains(Evaluator):
    """Like Contains but checks the .answer field of RetrievalOutput."""

    value: str

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        found = self.value.lower() in ctx.output.answer.lower()
        return EvaluationReason(
            value=found,
            reason=f"{'Found' if found else 'Missing'}: '{self.value}'",
        )


_retrieval_agent: Agent[None, PageSelection] | None = None


def _get_retrieval_agent() -> Agent[None, PageSelection]:
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = Agent(
            "anthropic:claude-sonnet-4-5-20250929",
            output_type=PageSelection,
            system_prompt=(
                "You are a document retrieval assistant. Given a document's "
                "table of contents and a question, select the pages most likely "
                "to contain the answer. Return between 1 and 10 page numbers. "
                "Think about which section headings are most relevant to the "
                "question and include a small buffer of surrounding pages."
            ),
        )
    return _retrieval_agent


def _get_system_card_doc():
    """Find the System Card document in the store."""
    store = DocumentStore()
    for doc in store.list_all():
        if SYSTEM_CARD_URL in doc.source_url:
            return doc
    raise RuntimeError(
        "System Card not found in document store. "
        f"Ingest it first: POST /ingest with url={SYSTEM_CARD_URL}"
    )


async def retrieve_and_ask(inp: RetrievalInput) -> RetrievalOutput:
    """Task function: retrieval agent picks pages from TOC, then query agent answers."""
    doc = _get_system_card_doc()
    pdf_bytes = await _get_pdf()

    # Step 1: Ask retrieval agent to select pages from the TOC
    selection = await _get_retrieval_agent().run(
        f"Document: {doc.metadata.title} ({doc.total_pages} pages)\n\n"
        f"Table of Contents:\n{doc.toc}\n\n"
        f"Question: {inp.question}\n\n"
        f"Which pages should I read to answer this question?"
    )

    # Clamp pages to valid range
    max_page = doc.total_pages or 200
    pages = sorted(set(
        p for p in selection.output.pages
        if 1 <= p <= max_page
    ))
    if not pages:
        return RetrievalOutput(pages_fetched=[], answer="No pages selected.")

    # Step 2: Extract selected pages and ask the query agent
    page_bytes = extract_page_range(pdf_bytes, pages[0], pages[-1])
    result = await _get_query_agent().run(
        [
            BinaryContent(data=page_bytes, media_type="application/pdf"),
            inp.question,
        ]
    )

    return RetrievalOutput(pages_fetched=pages, answer=result.output)


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
# Tier 4: Retrieval accuracy
# The retrieval agent reads the TOC and picks pages. We measure whether
# the gold (answer-containing) pages were fetched, plus answer quality.
# ---------------------------------------------------------------------------
retrieval_cases = [
    Case(
        name="retrieve_swe_bench",
        inputs=RetrievalInput(
            question="What is Claude Sonnet 4.6's score on SWE-bench Verified?",
            gold_pages=[14, 15, 16],
        ),
        expected_output="79.6%",
        evaluators=[
            PageRecall(),
            AnswerContains(value="79.6"),
        ],
    ),
    Case(
        name="retrieve_arc_agi",
        inputs=RetrievalInput(
            question="What are Claude Sonnet 4.6's scores on ARC-AGI-1 and ARC-AGI-2?",
            gold_pages=[19, 20, 21],
        ),
        expected_output="86.5% on ARC-AGI-1 and 60.4% on ARC-AGI-2",
        evaluators=[
            PageRecall(),
            AnswerContains(value="86.5"),
            AnswerContains(value="60.4"),
        ],
    ),
    Case(
        name="retrieve_reward_hacking",
        inputs=RetrievalInput(
            question="What specific reward hacking behaviors were observed in coding contexts? Give concrete examples.",
            gold_pages=[70, 71, 72, 73],
        ),
        expected_output="Reward hacking behaviors such as modifying tests, disabling checks, or gaming evaluation metrics",
        evaluators=[
            PageRecall(),
            LLMJudge(
                rubric=(
                    "The answer must describe at least 2 SPECIFIC reward hacking behaviors "
                    "in coding contexts (e.g. modifying tests, disabling checks, hardcoding outputs). "
                    "PASS if concrete examples are given. FAIL if generic or missing."
                ),
                include_input=True,
            ),
        ],
    ),
    Case(
        name="retrieve_bio_safety",
        inputs=RetrievalInput(
            question="What AI Safety Level was determined for biological risks? State the level and key reasoning.",
            gold_pages=[103, 104, 105, 106],
        ),
        expected_output="Below ASL-3 for biological risks",
        evaluators=[
            PageRecall(),
            LLMJudge(
                rubric=(
                    "The answer must state the AI Safety Level determination for biological "
                    "risks and provide reasoning. PASS if both level and reasoning are stated. "
                    "FAIL if either is missing."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    ),
    Case(
        name="retrieve_browsecomp",
        inputs=RetrievalInput(
            question="What tools and configuration were used for the Claude models in the BrowseComp evaluation? What was the token limit?",
            gold_pages=[43, 44, 45],
        ),
        expected_output="Web search, web fetch, programmatic tool calling, context compaction at 50k tokens, up to 10M total tokens",
        evaluators=[
            PageRecall(),
            AnswerContains(value="10M"),
        ],
    ),
    Case(
        name="retrieve_behavioral_audit",
        inputs=RetrievalInput(
            question="What behavioral phenomena are measured in the automated behavioral audit? List specific categories and describe the key findings.",
            gold_pages=[75, 76, 77, 78, 79, 80],
        ),
        expected_output="Self-preservation, sycophancy, institutional sabotage, self-serving bias, and other behavioral phenomena with severity assessments",
        evaluators=[
            PageRecall(),
            LLMJudge(
                rubric=(
                    "The answer must name at least 3 specific behavioral phenomena from "
                    "the audit (e.g. self-preservation, sycophancy, sabotage, self-serving bias) "
                    "and describe findings. PASS if 3+ phenomena named with results. FAIL otherwise."
                ),
                include_input=True,
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset = Dataset(
    cases=retrieval_cases,
)


if __name__ == "__main__":
    check_prerequisites()
    report = dataset.evaluate_sync(retrieve_and_ask)
    report.print(include_input=True, include_output=True)
