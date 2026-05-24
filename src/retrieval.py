"""
retrieval.py
------------
Handles embedding generation and semantic search over transcript chunks.
Uses ChromaDB as the vector store (runs fully in-memory — no server needed).
Supports OpenAI embeddings or a local sentence-transformers fallback.
"""

import hashlib
import logging
import os
from typing import Any, Literal

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["openai", "local"]

# Module-level cache so we don't re-initialize on every Streamlit re-run
_client: Any = None
_collection: Any = None
_current_collection_name: str | None = None
_current_chunks_hash: str | None = None
_current_embedding_backend: str | None = None
_embedding_function_cache: dict[str, object] = {}


def _check_chromadb_available() -> tuple[Any, Any]:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise ImportError(
            "chromadb and its embedding utilities are required for retrieval. "
            "Install with `pip install chromadb sentence-transformers` "
            "or switch to a lighter environment."
        ) from exc
    return chromadb, embedding_functions


def _get_embedding_function(backend: EmbeddingBackend):
    """
    Return a ChromaDB-compatible embedding function.

    - "openai": Uses text-embedding-3-small (requires OPENAI_API_KEY env var).
    - "local": Uses all-MiniLM-L6-v2 via sentence-transformers (free, no API key).
    """
    global _embedding_function_cache
    chromadb, embedding_functions = _check_chromadb_available()

    if backend in _embedding_function_cache:
        return _embedding_function_cache[backend]

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file "
                "or switch to the 'local' embedding backend."
            )
        logger.info("Using OpenAI text-embedding-3-small.")
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
    else:
        logger.info("Using local sentence-transformers (all-MiniLM-L6-v2).")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    _embedding_function_cache[backend] = ef
    return ef


def build_index(
    chunks: list[dict],
    collection_name: str = "interview",
    backend: EmbeddingBackend = "openai",
) -> Any:
    """
    Embed transcript chunks and store them in a ChromaDB in-memory collection.

    Args:
        chunks: List of chunk dicts from ingestion.chunk_transcript().
        collection_name: Name for the ChromaDB collection (use per-transcript names
                         to avoid stale data across uploads).
        backend: "openai" or "local".

    Returns:
        ChromaDB Collection ready for querying.
    """
    global _client, _collection, _current_collection_name, _current_chunks_hash, _current_embedding_backend

    if not chunks:
        raise ValueError("Cannot build an index from an empty chunk list.")

    content_hash = hashlib.sha256(
        "".join(f"{c['id']}:{c['text']}" for c in chunks).encode("utf-8")
    ).hexdigest()

    if (
        _collection is not None
        and _current_collection_name == collection_name
        and _current_embedding_backend == backend
        and _current_chunks_hash == content_hash
    ):
        logger.info(f"Reusing existing collection: {collection_name}")
        return _collection

    logger.info(f"Building index for collection: {collection_name}")
    chromadb, _ = _check_chromadb_available()
    ef = _get_embedding_function(backend)

    if _client is None:
        _client = chromadb.Client()  # ephemeral in-memory client

    # Drop old collection if it exists
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass

    _collection = _client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"word_count": c["word_count"]} for c in chunks]

    _collection.add(documents=documents, ids=ids, metadatas=metadatas)
    _current_collection_name = collection_name
    _current_chunks_hash = content_hash
    _current_embedding_backend = backend

    logger.info(f"Indexed {len(chunks)} chunks.")
    return _collection


def semantic_search(
    query: str,
    collection: Any,
    n_results: int = 5,
) -> list[dict]:
    """
    Run a semantic search over the indexed transcript chunks.

    Args:
        query: Natural language query string.
        collection: ChromaDB collection from build_index().
        n_results: Number of top results to return.

    Returns:
        List of result dicts with keys: 'text', 'score', 'id'.
    """
    logger.info(f"Searching for: '{query}' (top {n_results})")

    total_docs = collection.count()
    if total_docs == 0:
        logger.warning("Collection is empty; returning no search results.")
        return []

    n_results = max(1, min(n_results, total_docs))
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    hits = []
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc, dist, chunk_id in zip(docs, distances, ids):
        hits.append(
            {
                "id": chunk_id,
                "text": doc,
                # Cosine distance → similarity score (higher = more relevant)
                "score": round(1 - dist, 4),
            }
        )

    return hits


def extract_quotes_from_hits(hits: list[dict], min_score: float = 0.3) -> list[str]:
    """
    Extract customer-only quotes from search results.
    Filters out interviewer lines and low-relevance chunks.
    """
    import re

    quotes = []
    for hit in hits:
        if hit["score"] < min_score:
            continue
        # Extract customer lines specifically
        lines = hit["text"].split("\n")
        capturing = False
        for line in lines:
            if re.match(r"^Customer\s*:", line, re.IGNORECASE):
                capturing = True
                # Strip the "Customer:" prefix
                text = re.sub(r"^Customer\s*:\s*", "", line, flags=re.IGNORECASE).strip()
                if text:
                    quotes.append(text)
            elif re.match(r"^(Interviewer|Researcher|Host)\s*:", line, re.IGNORECASE):
                capturing = False
            elif capturing and line.strip():
                quotes.append(line.strip())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in quotes:
        if q not in seen and len(q) > 20:
            seen.add(q)
            unique.append(q)

    return unique
