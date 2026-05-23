"""
insights.py
-----------
LLM-powered insight extraction from interview transcripts.
Supports OpenAI GPT and Google Gemini backends.
All prompts live in prompts.py for easy iteration.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

LLMBackend = str  # "openai" | "gemini"


def _clean_json_response(raw: str) -> str:
    """Strip markdown fences and whitespace from an LLM JSON response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_themes(
    transcript: str,
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> dict:
    """
    Send the full transcript to an LLM and extract structured insights.

    Args:
        transcript: Full transcript text.
        backend: "openai" or "gemini".
        model: Optional model override (e.g. "gpt-4o-mini", "gemini-1.5-flash").

    Returns:
        Parsed JSON dict with themes, pain_points, feature_requests,
        overall_sentiment, sentiment_summary, and recommendations.
    """
    from src.prompts import THEME_EXTRACTION_PROMPT

    prompt = THEME_EXTRACTION_PROMPT.format(transcript=transcript)
    logger.info(f"Extracting themes using {backend}.")

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
        raise ValueError(f"LLM returned invalid JSON: {e}")


def generate_evidence_insight(
    topic: str,
    quotes: list[str],
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> dict:
    """
    Generate a single evidence-backed insight card for a given topic.

    Args:
        topic: The insight topic (e.g. "subscription tracking frustration").
        quotes: List of supporting quotes from the transcript.
        backend: "openai" or "gemini".
        model: Optional model override.

    Returns:
        Dict with 'insight', 'evidence', 'recommendation', 'confidence'.
    """
    from src.prompts import EVIDENCE_INSIGHT_PROMPT

    quotes_str = "\n".join(f'- "{q}"' for q in quotes)
    prompt = EVIDENCE_INSIGHT_PROMPT.format(topic=topic, quotes=quotes_str)
    logger.info(f"Generating evidence insight for topic: '{topic}'")

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
        raise ValueError(f"LLM returned invalid JSON: {e}")


def extract_aggregate_insights(
    transcripts: list[dict],
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> dict:
    """
    Extract research findings across multiple interview transcripts.

    Args:
        transcripts: List of interview dicts containing metadata and text.
        backend: "openai" or "gemini".
        model: Optional model override.

    Returns:
        Parsed JSON dict with themes, pain_points, feature_requests, cross-session findings,
        trend_signals, recommendations, and research next steps.
    """
    from src.prompts import AGGREGATE_INSIGHT_PROMPT

    transcripts_str = "\n\n---\n\n".join(
        f"Interview: {t.get('name', 'unknown')}\n"
        f"Participant: {t.get('participant_name', 'unknown')}\n"
        f"Segment: {t.get('segment', 'unknown')}\n"
        f"Role: {t.get('role', 'unknown')}\n"
        f"Notes: {t.get('notes', '')}\n\n"
        f"{t.get('text', '')}"
        for t in transcripts
    )

    prompt = AGGREGATE_INSIGHT_PROMPT.format(transcripts=transcripts_str)
    logger.info(f"Extracting aggregate research insights using {backend}.")

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
        raise ValueError(f"LLM returned invalid JSON: {e}")


def answer_research_query(
    query: str,
    quotes: list[str],
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> dict:
    """
    Answer a research question using retrieved evidence quotes.

    Args:
        query: Natural language research question.
        quotes: Supporting quote list.
        backend: "openai" or "gemini".
        model: Optional model override.

    Returns:
        Parsed JSON dict with answer, supporting quotes, and recommended next steps.
    """
    from src.prompts import RESEARCH_ASSISTANT_PROMPT

    quotes_str = "\n".join(f'- "{q}"' for q in quotes)
    prompt = RESEARCH_ASSISTANT_PROMPT.format(query=query, quotes=quotes_str)
    logger.info(f"Answering research question: '{query}'")

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
        raise ValueError(f"LLM returned invalid JSON: {e}")


def _call_llm(
    prompt: str,
    backend: LLMBackend = "openai",
    model: str | None = None,
    max_tokens: int = 2048,
) -> str:
    """
    Dispatch a prompt to the selected LLM backend.

    Returns the raw text response string.
    """
    if backend == "openai":
        return _call_openai(prompt, model=model or "gpt-4o-mini", max_tokens=max_tokens)
    elif backend == "gemini":
        return _call_gemini(prompt, model=model or "models/gemini-2.5-flash", max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'. Use 'openai' or 'gemini'.")


def _call_openai(prompt: str, model: str, max_tokens: int) -> str:
    """Call OpenAI chat completion API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,  # low temp for consistent structured output
    )

    return response.choices[0].message.content


def _call_gemini(prompt: str, model: str, max_tokens: int) -> str:
    """Call Google Gemini API via the google-generativeai SDK."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. "
            "Run: pip install google-generativeai"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment.")

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)

    generation_config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
    )

    response = gemini_model.generate_content(
        prompt,
        generation_config=generation_config,
    )

    # The deprecated google-generativeai SDK may return a response object where
    # the quick accessor `.text` is unavailable for some model outputs.
    if hasattr(response, "text"):
        try:
            return response.text
        except ValueError:
            pass

    if getattr(response, "candidates", None):
        candidate = response.candidates[0]
        if getattr(candidate, "content", None):
            content = candidate.content
            if getattr(content, "text", None) is not None:
                return content.text
            if getattr(content, "parts", None):
                return "".join(
                    getattr(part, "text", "") for part in content.parts
                ).strip()

    raise ValueError("Gemini returned no text output.")
