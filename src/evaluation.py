"""
evaluation.py
-------------
Evaluation framework for the insight pipeline.

Uses an LLM-as-judge approach to score insight quality on:
  - Groundedness: Are insights traceable to the transcript?
  - Specificity: Are insights concrete, not generic?
  - Actionability: Do recommendations lead to clear next steps?

Also includes retrieval precision scoring.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """You are evaluating the quality of AI-generated insights from a customer interview.

ORIGINAL TRANSCRIPT:
{transcript}

GENERATED INSIGHTS:
{insights}

Score each dimension from 1-5 and provide a brief justification.

Return ONLY a valid JSON object:
{{
  "groundedness": {{
    "score": <1-5>,
    "justification": "Is each insight traceable to something said in the transcript?"
  }},
  "specificity": {{
    "score": <1-5>,
    "justification": "Are insights specific to this customer, or generic platitudes?"
  }},
  "actionability": {{
    "score": <1-5>,
    "justification": "Do recommendations clearly suggest what a product team should do?"
  }},
  "coverage": {{
    "score": <1-5>,
    "justification": "Did the insights capture the most important things from the interview?"
  }},
  "overall_score": <average of the four scores, one decimal>,
  "summary": "One sentence overall assessment."
}}
"""


def evaluate_insights(
    transcript: str,
    insights: dict,
    backend: str = "openai",
    model: str | None = None,
) -> dict:
    """
    Use an LLM to evaluate the quality of generated insights.

    Args:
        transcript: Original interview transcript.
        insights: Dict output from insights.extract_themes().
        backend: LLM backend ("openai" or "gemini").
        model: Optional model override.

    Returns:
        Evaluation dict with scores and justifications.
    """
    from src.insights import _call_llm

    insights_str = json.dumps(insights, indent=2)
    prompt = EVALUATION_PROMPT.format(
        transcript=transcript,
        insights=insights_str,
    )

    logger.info("Running insight quality evaluation.")
    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Evaluation parse error: {e}")
        return {"error": str(e), "raw": raw}


def score_retrieval_hits(
    query: str,
    hits: list[dict],
    transcript: str,
    backend: str = "openai",
    model: str | None = None,
) -> dict:
    """
    Score the relevance of semantic search results.

    Args:
        query: The original search query.
        hits: List of retrieval hits from retrieval.semantic_search().
        transcript: Full transcript for context.
        backend: LLM backend.
        model: Optional model override.

    Returns:
        Dict with per-hit relevance scores and an average precision score.
    """
    from src.insights import _call_llm

    hits_str = "\n\n".join(
        f"[Chunk {i+1} | score={h['score']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )

    prompt = f"""You are evaluating semantic search quality for customer research.

Query: "{query}"

Retrieved chunks:
{hits_str}

For each chunk, rate its relevance to the query: 0 (irrelevant), 1 (partially relevant), 2 (highly relevant).

Return ONLY a JSON object:
{{
  "chunk_scores": [0, 1, 2, ...],
  "average_precision": <float 0-1>,
  "notes": "Brief comment on retrieval quality."
}}"""

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Parse failed", "raw": raw}


def _clean_json(raw: str) -> str:
    """Strip markdown fences from LLM response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()
