"""
ingestion.py
------------
Handles loading, cleaning, and chunking interview transcripts.
Transcripts are split into semantically meaningful segments before embedding.
"""

import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_transcript(file_path: str | None = None, text: str | None = None) -> str:
    """
    Load a transcript from a file path or raw text string.

    Args:
        file_path: Path to a .txt or .md transcript file.
        text: Raw transcript string (e.g. from Streamlit file upload).

    Returns:
        Cleaned transcript string.
    """
    if text:
        logger.info("Loading transcript from raw text input.")
        return _clean_text(text)

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        logger.info(f"Loading transcript from file: {file_path}")
        return _clean_text(path.read_text(encoding="utf-8"))

    raise ValueError("Must provide either file_path or text.")


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_transcript(
    transcript: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    """
    Chunk a transcript into overlapping segments for embedding.

    Tries to split on speaker turns (e.g. "Interviewer:" / "Customer:") first.
    Falls back to word-count chunking if no speaker labels are detected.

    Args:
        transcript: Full transcript text.
        chunk_size: Target word count per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of dicts with keys: 'id', 'text', 'word_count'.
    """
    # Detect speaker-turn structure
    speaker_pattern = re.compile(
        r"^(Interviewer|Customer|Participant|Researcher|Host|Guest)\s*:", 
        re.MULTILINE | re.IGNORECASE
    )

    if speaker_pattern.search(transcript):
        logger.info("Speaker labels detected — chunking by turn.")
        raw_chunks = _split_by_speaker_turns(transcript, chunk_size)
    else:
        logger.info("No speaker labels — falling back to word-count chunking.")
        raw_chunks = _split_by_word_count(transcript, chunk_size, overlap)

    chunks = [
        {
            "id": f"chunk_{i:03d}",
            "text": chunk.strip(),
            "word_count": len(chunk.split()),
        }
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]

    logger.info(f"Created {len(chunks)} chunks from transcript.")
    return chunks


def _split_by_speaker_turns(transcript: str, chunk_size: int) -> list[str]:
    """
    Split on speaker labels and group adjacent turns into chunks
    that stay near chunk_size words.
    """
    # Split into individual turns
    turn_pattern = re.compile(
        r"(?=(?:Interviewer|Customer|Participant|Researcher|Host|Guest)\s*:)",
        re.IGNORECASE,
    )
    turns = [t.strip() for t in turn_pattern.split(transcript) if t.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for turn in turns:
        turn_words = len(turn.split())
        if current_words + turn_words > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(turn)
        current_words += turn_words

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_by_word_count(
    transcript: str, chunk_size: int, overlap: int
) -> list[str]:
    """Split plain text by word count with overlap."""
    words = transcript.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def get_full_text(chunks: list[dict]) -> str:
    """Reconstruct full text from a list of chunk dicts."""
    return "\n\n".join(c["text"] for c in chunks)
