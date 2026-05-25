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
    """Extract JSON block even if hidden inside markdown with conversational text."""
    raw = raw.strip()
    # Look for a markdown code block anywhere in the text
    match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


def _attempt_fix_json(cleaned: str) -> str:
    """Try to repair common JSON formatting issues from LLM output."""
    # If raw output contains leading text before JSON, strip to the first JSON opener.
    first_json = re.search(r"[\{\[]", cleaned)
    if first_json:
        cleaned = cleaned[first_json.start():]

    # Fallback heuristic: Fix Pythonic booleans/nulls often hallucinated by LLMs
    cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
    cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
    cleaned = re.sub(r'\bNone\b', 'null', cleaned)

    stack: list[str] = []
    in_string = False
    escape = False
    last_top_level_end: int | None = None
    fixed_chars: list[str] = []

    def _insert_comma_if_needed():
        # Look backwards to see if we missed a comma between properties
        for prev in reversed(fixed_chars):
            if not prev.isspace():
                # If the previous meaningful char is the end of a value (", }, ])
                # or an alphanumeric char (end of true, false, null, or a digit)
                if prev in '\"}]' or prev.isalnum():
                    fixed_chars.append(',')
                break

    for ch in cleaned:
        if escape:
            escape = False
            fixed_chars.append(ch)
            continue
        if ch == "\\":
            escape = True
            fixed_chars.append(ch)
            continue
        if ch == '"':
            if not in_string:
                _insert_comma_if_needed()
            in_string = not in_string
            fixed_chars.append(ch)
            continue
        if in_string:
            fixed_chars.append(ch)
            continue
            
        # --- Safe manipulation outside of strings ---
        if ch in '{[':
            _insert_comma_if_needed()

        if ch in '}]':
            # Strip trailing commas safely before a closing bracket
            while fixed_chars and fixed_chars[-1].isspace():
                fixed_chars.pop()
            if fixed_chars and fixed_chars[-1] == ',':
                fixed_chars.pop()
                
        fixed_chars.append(ch)

        if ch in '{[':
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
            if not stack:
                last_top_level_end = len(fixed_chars) - 1
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()
            if not stack:
                last_top_level_end = len(fixed_chars) - 1

    fixed = "".join(fixed_chars)

    if in_string:
        fixed += '"'

    # Strip trailing commas safely if we are about to forcibly close the structure
    if stack:
        fixed = fixed.rstrip(" \n\r\t,")
        while stack:
            opener = stack.pop()
            fixed += '}' if opener == '{' else ']'
    elif last_top_level_end is not None and last_top_level_end < len(fixed) - 1:
        fixed = fixed[: last_top_level_end + 1]

    return fixed


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
        fixed = _attempt_fix_json(cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
            logger.debug(f"Attempted fixed JSON:\n{fixed}")
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
        fixed = _attempt_fix_json(cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
            logger.debug(f"Attempted fixed JSON:\n{fixed}")
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

    raw = _call_llm(prompt, backend=backend, model=model, max_tokens=4096)
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        fixed = _attempt_fix_json(cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
            logger.debug(f"Attempted fixed JSON:\n{fixed}")
            raise ValueError(f"LLM returned invalid JSON: {e}")


def answer_research_query(
    query: str,
    quotes: list[str],
    chat_history: list[dict] | None = None,
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> dict:
    """
    Answer a research question using retrieved evidence quotes and chat history.

    Args:
        query: Natural language research question.
        quotes: Supporting quote list.
        chat_history: List of previous chat messages.
        backend: "openai" or "gemini".
        model: Optional model override.

    Returns:
        Parsed JSON dict with answer, supporting quotes, and recommended next steps.
    """
    from src.prompts import RESEARCH_ASSISTANT_PROMPT

    history_str = "No prior conversation."
    if chat_history:
        history_str = ""
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n\n"
        history_str = history_str.strip()

    quotes_str = "\n".join(f'- "{q}"' for q in quotes)
    prompt = RESEARCH_ASSISTANT_PROMPT.format(
        query=query, quotes=quotes_str, chat_history=history_str
    )
    logger.info(f"Answering research question: '{query}'")

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        fixed = _attempt_fix_json(cleaned)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            logger.error(f"JSON parse error: {e}\nRaw response:\n{raw}")
            logger.debug(f"Attempted fixed JSON:\n{fixed}")
            raise ValueError(f"LLM returned invalid JSON: {e}")


def expand_research_query(
    query: str,
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> list[str]:
    """
    Expand a single query into multiple alternative phrasings to improve retrieval recall.
    """
    from src.prompts import QUERY_EXPANSION_PROMPT

    prompt = QUERY_EXPANSION_PROMPT.format(query=query)
    logger.info(f"Expanding search query: '{query}'")

    raw = _call_llm(prompt, backend=backend, model=model)
    cleaned = _clean_json_response(raw)

    try:
        expansions = json.loads(cleaned)
    except json.JSONDecodeError:
        fixed = _attempt_fix_json(cleaned)
        try:
            expansions = json.loads(fixed)
        except json.JSONDecodeError:
            expansions = []
            
    if isinstance(expansions, list) and expansions:
        return [query] + [str(e) for e in expansions]
    return [query]


def format_transcript_dialogue(
    transcript: str,
    backend: LLMBackend = "openai",
    model: str | None = None,
) -> str:
    """
    Use an LLM to add speaker labels (diarization) to a raw text wall.
    """
    from src.prompts import DIARIZE_PROMPT

    prompt = DIARIZE_PROMPT.format(transcript=transcript)
    logger.info(f"Formatting transcript dialogue using {backend}.")

    # Increase max_tokens significantly as we are rewriting the entire transcript
    raw = _call_llm(prompt, backend=backend, model=model, max_tokens=8192)
    
    # Clean up accidental markdown formatting the LLM might have applied
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:text|markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    return cleaned.strip()


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
    elif backend == "ollama":
        return _call_ollama(prompt, model=model or "llama3", max_tokens=max_tokens)
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


def _call_ollama(prompt: str, model: str, max_tokens: int) -> str:
    """Call local Ollama using the OpenAI compatible endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    # Ollama's default local API endpoint
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama-local")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
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
