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
    "score": 5,
    "justification": "Is each insight traceable to something said in the transcript?"
  }},
  "specificity": {{
    "score": 4,
    "justification": "Are insights specific to this customer, or generic platitudes?"
  }},
  "actionability": {{
    "score": 3,
    "justification": "Do recommendations clearly suggest what a product team should do?"
  }},
  "coverage": {{
    "score": 5,
    "justification": "Did the insights capture the most important things from the interview?"
  }},
  "summary": "One sentence overall assessment."
}}
Note: All scores MUST be integers between 1 and 5.
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
    from src.insights import _call_llm, _attempt_fix_json

    insights_str = json.dumps(insights, indent=2)
    prompt = EVALUATION_PROMPT.format(
        transcript=transcript,
        insights=insights_str,
    )

    logger.info("Running insight quality evaluation.")
    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        fixed = _attempt_fix_json(cleaned)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            logger.error(f"Evaluation parse error: {e}\nRaw: {raw}")
            return {"error": str(e), "raw": raw}

    # Safely compute overall_score in Python rather than relying on LLM math
    scores = [v.get("score") for k, v in parsed.items() if isinstance(v, dict) and "score" in v]
    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    parsed["overall_score"] = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0.0

    return parsed


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
    from src.insights import _call_llm, _attempt_fix_json

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
  "chunk_scores": [2, 0, 1],
  "notes": "Brief comment on retrieval quality."
}}"""

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        fixed = _attempt_fix_json(cleaned)
        try:
            parsed = json.loads(fixed)
        except json.JSONDecodeError:
            return {"error": "Parse failed", "raw": raw}
            
    # Compute average precision in Python
    scores = parsed.get("chunk_scores", [])
    if scores and all(isinstance(s, (int, float)) for s in scores):
        parsed["average_precision"] = round(sum(scores) / (len(scores) * 2), 2) # max score is 2
        
    return parsed


def _clean_json(raw: str) -> str:
    """Extract JSON block even if hidden inside markdown with conversational text."""
    raw = raw.strip()
    match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()
