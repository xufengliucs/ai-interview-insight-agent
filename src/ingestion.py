"""
ingestion.py
------------
Handles loading, cleaning, and chunking interview transcripts.
Transcripts are split into semantically meaningful segments before embedding.
"""

import re
import logging
import uuid
from pathlib import Path
from src.constants import SPEAKER_PATTERN, TURN_PATTERN

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
    # Detect speaker-turn structure using shared regex
    if SPEAKER_PATTERN.search(transcript):
        logger.info("Speaker labels detected — chunking by turn.")
        raw_chunks = _split_by_speaker_turns(transcript, chunk_size)
    else:
        logger.info("No speaker labels — falling back to word-count chunking.")
        raw_chunks = _split_by_word_count(transcript, chunk_size, overlap)

    chunks = []
    for chunk in raw_chunks:
        chunk_text = chunk.strip()
        if chunk_text:
            # Generate a globally unique ID for the chunk to prevent ChromaDB collisions
            chunk_id = uuid.uuid4().hex[:12]
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
            })

    logger.info(f"Created {len(chunks)} chunks from transcript.")
    return chunks


def _split_by_speaker_turns(transcript: str, chunk_size: int) -> list[str]:
    """
    Split on speaker labels and group adjacent turns into chunks
    that stay near chunk_size words.
    """
    # Split into individual turns using shared regex
    turns = [t.strip() for t in TURN_PATTERN.split(transcript) if t.strip()]

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
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap to prevent infinite loops.")

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


def chunks_from_interviews(entries: list[dict]) -> list[dict]:
    """Chunk all interview transcripts and attach retrieval metadata."""
    all_chunks: list[dict] = []
    for entry in entries:
        chunks = chunk_transcript(entry["text"])
        inv_name = entry.get("name") or "Unknown"
        part_name = entry.get("participant_name") or "Unknown"
        for c in chunks:
            c["interview_name"] = inv_name
            c["participant_name"] = part_name
        all_chunks.extend(chunks)
    return all_chunks


def transcribe_audio(audio_bytes: bytes, file_name: str, prompt: str | None = None) -> str:
    """
    Transcribe an audio file using OpenAI's Whisper API.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment. Cannot transcribe audio.")

    client = OpenAI(api_key=api_key)

    # OpenAI requires a file-like object with a valid filename extension
    import io
    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = file_name

    logger.info(f"Transcribing audio file: {file_name} with Whisper API.")
    
    kwargs = {"model": "whisper-1", "file": file_obj}
    if prompt and prompt.strip():
        kwargs["prompt"] = prompt.strip()
        
    response = client.audio.transcriptions.create(**kwargs)
    return response.text
