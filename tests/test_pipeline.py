"""
test_pipeline.py
----------------
Unit tests for the core pipeline components.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (
    load_transcript,
    chunk_transcript,
    _clean_text,
    _split_by_word_count,
    get_full_text,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT = """Interviewer: What frustrates you most about budgeting apps?

Customer: I often forget subscriptions. Notifications are noisy.
Sometimes syncing with bank accounts fails.

Interviewer: Have you tried any alternatives?

Customer: Yes. I tried Mint but it kept disconnecting. I gave up after two months.
The onboarding was fine but support was useless.

Interviewer: What would your ideal solution look like?

Customer: Smart subscription tracking. Alerts only when something unusual happens.
I want control over my categories and notifications."""

PLAIN_TRANSCRIPT = "This is a plain text transcript without any speaker labels. " * 20


# ─── Ingestion Tests ──────────────────────────────────────────────────────────

class TestLoadTranscript:
    def test_load_from_text(self):
        result = load_transcript(text=SAMPLE_TRANSCRIPT)
        assert "Customer" in result
        assert isinstance(result, str)

    def test_load_from_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TRANSCRIPT)
        result = load_transcript(file_path=str(f))
        assert "Interviewer" in result

    def test_raises_without_args(self):
        with pytest.raises(ValueError):
            load_transcript()

    def test_raises_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_transcript(file_path="/nonexistent/path.txt")

    def test_cleans_excessive_newlines(self):
        messy = "Line one\n\n\n\n\nLine two"
        result = _clean_text(messy)
        assert "\n\n\n" not in result

    def test_normalizes_crlf(self):
        crlf = "Line one\r\nLine two\r\n"
        result = _clean_text(crlf)
        assert "\r" not in result


class TestChunkTranscript:
    def test_returns_list_of_dicts(self):
        chunks = chunk_transcript(SAMPLE_TRANSCRIPT)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_chunk_has_required_keys(self):
        chunks = chunk_transcript(SAMPLE_TRANSCRIPT)
        for chunk in chunks:
            assert "id" in chunk
            assert "text" in chunk
            assert "word_count" in chunk

    def test_chunk_ids_are_unique(self):
        chunks = chunk_transcript(SAMPLE_TRANSCRIPT)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_speaker_based_chunking(self):
        """Transcripts with speaker labels should produce meaningful chunks."""
        chunks = chunk_transcript(SAMPLE_TRANSCRIPT, chunk_size=100)
        assert len(chunks) >= 1
        # Each chunk should contain some text
        for c in chunks:
            assert len(c["text"]) > 10

    def test_plain_text_chunking(self):
        """Plain text without speaker labels should still chunk cleanly."""
        chunks = chunk_transcript(PLAIN_TRANSCRIPT, chunk_size=50, overlap=10)
        assert len(chunks) >= 2
        for c in chunks:
            assert c["word_count"] <= 60  # some leeway

    def test_word_count_chunk_size(self):
        chunks = _split_by_word_count(PLAIN_TRANSCRIPT, chunk_size=20, overlap=5)
        for c in chunks[:-1]:  # last chunk may be shorter
            assert len(c.split()) <= 25  # small leeway

    def test_empty_chunks_excluded(self):
        transcript = "Customer: \n\nInterviewer: \n\n"
        chunks = chunk_transcript(transcript)
        for c in chunks:
            assert c["text"].strip() != ""

    def test_get_full_text_roundtrip(self):
        chunks = chunk_transcript(PLAIN_TRANSCRIPT, chunk_size=50)
        full = get_full_text(chunks)
        # All original words should appear somewhere in reconstructed text
        original_words = set(PLAIN_TRANSCRIPT.split())
        full_words = set(full.split())
        assert original_words.issubset(full_words)


# ─── Prompts Tests ────────────────────────────────────────────────────────────

class TestPrompts:
    def test_theme_prompt_has_placeholder(self):
        from src.prompts import THEME_EXTRACTION_PROMPT
        assert "{transcript}" in THEME_EXTRACTION_PROMPT

    def test_evidence_prompt_has_placeholders(self):
        from src.prompts import EVIDENCE_INSIGHT_PROMPT
        assert "{topic}" in EVIDENCE_INSIGHT_PROMPT
        assert "{quotes}" in EVIDENCE_INSIGHT_PROMPT

    def test_theme_prompt_formats(self):
        from src.prompts import THEME_EXTRACTION_PROMPT
        formatted = THEME_EXTRACTION_PROMPT.format(transcript="Test transcript.")
        assert "Test transcript." in formatted

    def test_evidence_prompt_formats(self):
        from src.prompts import EVIDENCE_INSIGHT_PROMPT
        formatted = EVIDENCE_INSIGHT_PROMPT.format(
            topic="subscription fatigue",
            quotes='"I forget my subscriptions."'
        )
        assert "subscription fatigue" in formatted

    def test_aggregate_prompt_has_placeholders(self):
        from src.prompts import AGGREGATE_INSIGHT_PROMPT
        assert "{transcripts}" in AGGREGATE_INSIGHT_PROMPT

    def test_research_assistant_prompt_has_placeholders(self):
        from src.prompts import RESEARCH_ASSISTANT_PROMPT
        assert "{query}" in RESEARCH_ASSISTANT_PROMPT
        assert "{quotes}" in RESEARCH_ASSISTANT_PROMPT

    def test_research_assistant_prompt_formats(self):
        from src.prompts import RESEARCH_ASSISTANT_PROMPT
        formatted = RESEARCH_ASSISTANT_PROMPT.format(
            query="What is the main pain point?",
            quotes='- "I hate syncing."'
        )
        assert "What is the main pain point?" in formatted
        assert "I hate syncing." in formatted


# ─── Retrieval Tests (no API key needed) ─────────────────────────────────────

class TestRetrievalHelpers:
    def test_extract_quotes_filters_short(self):
        from src.retrieval import extract_quotes_from_hits
        hits = [
            {
                "id": "chunk_000",
                "text": "Customer: Hi.\nInterviewer: Hello.\nCustomer: I hate subscription tracking because it never works right for me.",
                "score": 0.8,
            }
        ]
        quotes = extract_quotes_from_hits(hits, min_score=0.3)
        assert any("subscription" in q for q in quotes)

    def test_extract_quotes_filters_low_score(self):
        from src.retrieval import extract_quotes_from_hits
        hits = [
            {
                "id": "chunk_000",
                "text": "Customer: This is irrelevant content about subscriptions.",
                "score": 0.1,  # below min_score
            }
        ]
        quotes = extract_quotes_from_hits(hits, min_score=0.3)
        assert quotes == []

    def test_extract_quotes_deduplicates(self):
        from src.retrieval import extract_quotes_from_hits
        repeated_text = "Customer: I forget my subscriptions constantly and it drives me crazy."
        hits = [
            {"id": "chunk_000", "text": repeated_text, "score": 0.9},
            {"id": "chunk_001", "text": repeated_text, "score": 0.85},
        ]
        quotes = extract_quotes_from_hits(hits, min_score=0.3)
        assert len(quotes) == len(set(quotes))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
