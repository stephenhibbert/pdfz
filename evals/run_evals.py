"""PDFZ retrieval evaluation suite.

Three tiers of eval cases:
  1. Basic metadata - answerable from first 10 pages
  2. Multi-hop retrieval - requires TOC navigation then page lookup
  3. Visual comprehension - requires reading charts, tables, and figures
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_ai import Agent, BinaryContent

from pdfz.pdf_utils import download_pdf, extract_page_range

TEST_PDF_URL = (
    "https://www-cdn.anthropic.com/"
    "78073f739564e986ff3e28522761a7a0b4484f84.pdf"
)

_pdf_cache: bytes | None = None


async def _get_pdf() -> bytes:
    global _pdf_cache
    if _pdf_cache is None:
        _pdf_cache = await download_pdf(TEST_PDF_URL)
    return _pdf_cache


_query_agent: Agent[None, str] | None = None


def _get_query_agent() -> Agent[None, str]:
    global _query_agent
    if _query_agent is None:
        _query_agent = Agent(
            "anthropic:claude-sonnet-4-5-20250929",
            system_prompt="Answer the question based on the provided PDF pages. Be specific and cite numbers, scores, or data points when available.",
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
            inp.question,
            BinaryContent(data=pages, media_type="application/pdf"),
        ]
    )
    return result.output


# ---------------------------------------------------------------------------
# Tier 1: Basic metadata (first pages)
# ---------------------------------------------------------------------------
basic_cases = [
    Case(
        name="title_extraction",
        inputs=EvalInput(question="What is the title of this document?", page_start=1, page_end=2),
        expected_output="System Card: Claude Sonnet 4.6",
        evaluators=[LLMJudge(
            rubric="The answer must identify the document as the Claude Sonnet 4.6 System Card.",
            include_expected_output=True,
        )],
    ),
    Case(
        name="publisher_identification",
        inputs=EvalInput(question="Who published this document?", page_start=1, page_end=2),
        expected_output="Anthropic",
        evaluators=[LLMJudge(
            rubric="The answer must identify Anthropic as the publisher.",
            include_expected_output=True,
        )],
    ),
    Case(
        name="document_type",
        inputs=EvalInput(
            question="What type of document is this and what is its purpose?",
            page_start=1, page_end=3,
        ),
        expected_output="A system card providing transparency about the capabilities and safety of an AI model",
        evaluators=[LLMJudge(
            rubric="The answer should identify this as a system card or transparency document about an AI model.",
            include_expected_output=True,
        )],
    ),
]

# ---------------------------------------------------------------------------
# Tier 2: Multi-hop retrieval (requires TOC -> page navigation)
# These simulate what an MCP agent would do: read TOC, identify section,
# then query the relevant pages.
# ---------------------------------------------------------------------------
multihop_cases = [
    # Hop 1: TOC says "2.2 SWE-bench" is on p.15 -> Hop 2: read p.14-16
    Case(
        name="swe_bench_score",
        inputs=EvalInput(
            question="What is Claude Sonnet 4.6's score on SWE-bench Verified?",
            page_start=14, page_end=16,
        ),
        expected_output="79.6%",
        evaluators=[LLMJudge(
            rubric="The answer must include the specific SWE-bench Verified percentage score for Claude Sonnet 4.6 (79.6%). Partial credit if the number is close but not exact.",
            include_expected_output=True,
        )],
    ),
    # Hop 1: TOC says "2.7 ARC-AGI" is on p.19 -> Hop 2: read p.19-21
    Case(
        name="arc_agi_scores",
        inputs=EvalInput(
            question="What are Claude Sonnet 4.6's scores on ARC-AGI-1 and ARC-AGI-2?",
            page_start=19, page_end=21,
        ),
        expected_output="86.5% on ARC-AGI-1 and 60.4% on ARC-AGI-2",
        evaluators=[LLMJudge(
            rubric="The answer must include both scores: 86.5% on ARC-AGI-1 and 60.4% on ARC-AGI-2.",
            include_expected_output=True,
        )],
    ),
    # Hop 1: TOC "4.3 Reward hacking" p.70 -> Hop 2: read p.70-73
    Case(
        name="reward_hacking_behaviors",
        inputs=EvalInput(
            question="What specific reward hacking behaviors were observed in coding contexts?",
            page_start=70, page_end=73,
        ),
        expected_output="Examples of reward hacking in coding like modifying tests, disabling checks, or gaming evaluation metrics",
        evaluators=[LLMJudge(
            rubric="The answer should describe specific reward hacking behaviors observed in coding (e.g. modifying tests, disabling checks, gaming metrics). Must include concrete examples, not just generic statements.",
            include_expected_output=True,
        )],
    ),
    # Hop 1: TOC "6.2.1 Biological risk" p.103 -> Hop 2: "6.2.1.4 Safety Level determination" p.104
    Case(
        name="bio_safety_level_determination",
        inputs=EvalInput(
            question="What AI Safety Level was determined for biological risks, and what was the reasoning?",
            page_start=103, page_end=106,
        ),
        expected_output="The model was determined to be below ASL-3 for biological risks",
        evaluators=[LLMJudge(
            rubric="The answer should state the safety level determination for biological risks and explain the key reasoning behind it.",
            include_expected_output=True,
        )],
    ),
    # Multi-hop across sections: compare two related benchmarks
    # Hop 1: TOC "2.18.1 WebArena" p.36, "2.18.2 WebArena-Verified" p.37
    Case(
        name="webarena_comparison",
        inputs=EvalInput(
            question="How does Claude Sonnet 4.6's performance on WebArena compare to WebArena-Verified? What are the specific scores?",
            page_start=36, page_end=39,
        ),
        expected_output="Scores on both WebArena and WebArena-Verified benchmarks",
        evaluators=[LLMJudge(
            rubric="The answer must provide specific numerical scores for both WebArena and WebArena-Verified, and note any difference between them.",
            include_expected_output=True,
        )],
    ),
]

# ---------------------------------------------------------------------------
# Tier 3: Visual comprehension (charts, tables, figures)
# These require the LLM to actually read visual elements in the PDF.
# ---------------------------------------------------------------------------
visual_cases = [
    # Main capabilities comparison table on p.14
    Case(
        name="capabilities_table_comparison",
        inputs=EvalInput(
            question="Looking at the capabilities comparison table, which model scores highest on SWE-bench Verified, and what is the score? Also, how does Claude Sonnet 4.6 compare to GPT-5.2 on this benchmark?",
            page_start=14, page_end=15,
        ),
        expected_output="Claude Opus 4.5 scores highest at 80.9%. Claude Sonnet 4.6 (79.6%) is close to GPT-5.2 (80.0%)",
        evaluators=[LLMJudge(
            rubric="The answer must correctly read the comparison table to identify the highest scorer on SWE-bench Verified and compare Claude Sonnet 4.6 vs GPT-5.2 with specific percentages.",
            include_expected_output=True,
        )],
    ),
    # ARC-AGI figure on p.20
    Case(
        name="arc_agi_figure",
        inputs=EvalInput(
            question="Looking at the ARC-AGI figure/chart, what thinking token budget and effort level were used for Claude Sonnet 4.6's reported scores?",
            page_start=19, page_end=21,
        ),
        expected_output="120k thinking tokens with High effort",
        evaluators=[LLMJudge(
            rubric="The answer must correctly identify the thinking token budget (120k) and effort level (High) from the figure caption or chart for Claude Sonnet 4.6's ARC-AGI scores.",
            include_expected_output=True,
        )],
    ),
    # Multilingual performance table on p.41-42 - requires reading a dense table
    Case(
        name="multilingual_lowest_language",
        inputs=EvalInput(
            question="Looking at the GMMLU multilingual results table, which language has the lowest score for Claude Sonnet 4.6, and what is that score?",
            page_start=40, page_end=43,
        ),
        expected_output="A specific language with the lowest GMMLU score and the percentage",
        evaluators=[LLMJudge(
            rubric="The answer must identify a specific language as having the lowest GMMLU score and provide the percentage. The language should be a low-resource language (likely an African language like Igbo, Yoruba, or similar).",
            include_expected_output=True,
        )],
    ),
    # BrowseComp chart on p.44 - requires reading a bar chart or performance chart
    Case(
        name="browsecomp_chart",
        inputs=EvalInput(
            question="According to the BrowseComp figure/chart, what tools and configuration were used for the Claude models? What was the token limit?",
            page_start=43, page_end=45,
        ),
        expected_output="Web search, web fetch, programmatic tool calling, context compaction at 50k tokens, up to 10M total tokens",
        evaluators=[LLMJudge(
            rubric="The answer must describe the toolset and configuration used for BrowseComp evaluation, mentioning web search, web fetch, code execution, and the token/context limits.",
            include_expected_output=True,
        )],
    ),
    # Behavioral audit results table/figure on p.77-78
    Case(
        name="behavioral_audit_figure",
        inputs=EvalInput(
            question="Looking at the behavioral audit results figures on these pages, what behavioral phenomena are measured and what are the key findings?",
            page_start=75, page_end=80,
        ),
        expected_output="Behavioral phenomena like self-preservation, sycophancy, institutional sabotage, etc. with severity scores",
        evaluators=[LLMJudge(
            rubric="The answer must identify specific behavioral phenomena measured in the audit (e.g. self-preservation, sycophancy, institutional sabotage, self-serving bias) and describe key findings or severity levels from the figures.",
            include_expected_output=True,
        )],
    ),
]

# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------
dataset = Dataset(
    cases=basic_cases + multihop_cases + visual_cases,
    evaluators=[
        LLMJudge(
            rubric="The answer should be factual, specific, and grounded in the document content. It should not hallucinate information not present in the provided pages.",
        ),
    ],
)


if __name__ == "__main__":
    report = dataset.evaluate_sync(ask_pdf_pages)
    report.print(include_input=True, include_output=True)
