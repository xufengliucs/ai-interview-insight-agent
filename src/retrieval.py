"""
retrieval.py
------------
Handles embedding generation and semantic search over transcript chunks.
Uses ChromaDB as the vector store (persistent SQLite under chroma_db/).
Supports OpenAI embeddings or a local sentence-transformers fallback.
"""

import hashlib
import logging
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal

logger = logging.getLogger(__name__)

EmbeddingBackend = Literal["openai", "local"]

# Optional loaders (e.g. Streamlit cache_resource wrappers from retrieval_streamlit.py)
_local_embedding_loader: Callable[[], Any] | None = None
_cross_encoder_loader: Callable[[], Any] | None = None

# Module-level cache so we don't re-initialize Chroma collections on every call
_client: Any = None
_collection: Any = None
_current_collection_name: str | None = None
_current_chunks_hash: str | None = None
_current_embedding_backend: str | None = None


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


@lru_cache(maxsize=1)
def _load_local_embedding_function() -> Any:
    """Load and cache the sentence-transformers embedding function."""
    chromadb, embedding_functions = _check_chromadb_available()
    logger.info("Loading local sentence-transformers (all-MiniLM-L6-v2)...")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=1)
def _load_cross_encoder() -> Any:
    """Load and cache the CrossEncoder model for reranking."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install with `pip install sentence-transformers`."
        ) from exc
    logger.info("Loading CrossEncoder (ms-marco-MiniLM-L-6-v2) for reranking...")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def install_model_loaders(
    *,
    local_embedding: Callable[[], Any] | None = None,
    cross_encoder: Callable[[], Any] | None = None,
) -> None:
    """Register optional model loaders (used by Streamlit UI for cached spinners)."""
    global _local_embedding_loader, _cross_encoder_loader
    if local_embedding is not None:
        _local_embedding_loader = local_embedding
    if cross_encoder is not None:
        _cross_encoder_loader = cross_encoder


def clear_model_loaders() -> None:
    """Reset to built-in lru_cache loaders (useful in tests)."""
    global _local_embedding_loader, _cross_encoder_loader
    _local_embedding_loader = None
    _cross_encoder_loader = None


def _resolve_local_embedding_function() -> Any:
    if _local_embedding_loader is not None:
        return _local_embedding_loader()
    return _load_local_embedding_function()


def _resolve_cross_encoder() -> Any:
    if _cross_encoder_loader is not None:
        return _cross_encoder_loader()
    return _load_cross_encoder()


def _get_embedding_function(backend: EmbeddingBackend):
    """
    Return a ChromaDB-compatible embedding function.

    - "openai": Uses text-embedding-3-small (requires OPENAI_API_KEY env var).
    - "local": Uses all-MiniLM-L6-v2 via sentence-transformers (free, no API key).
    """
    chromadb, embedding_functions = _check_chromadb_available()

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file "
                "or switch to the 'local' embedding backend."
            )
        logger.info("Using OpenAI text-embedding-3-small.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
    return _resolve_local_embedding_function()


def relevance_score(hit: dict) -> float:
    """
    Cosine-similarity score for thresholding (0–1), even when results were reranked.

    Prefer ``dense_score`` when present; otherwise use ``score``.
    """
    if "dense_score" in hit:
        return float(hit["dense_score"])
    return float(hit["score"])


def format_hit_score(hit: dict) -> str:
    """Human-readable score label for UI."""
    if "rerank_score" in hit:
        return (
            f"rerank {hit['rerank_score']:.3f} · dense {relevance_score(hit):.3f}"
        )
    return f"{hit['score']:.3f}"


def build_index(
    chunks: list[dict],
    collection_name: str = "interview",
    backend: EmbeddingBackend = "openai",
) -> Any:
    """
    Embed transcript chunks and store them in a ChromaDB collection.

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
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=db_path)

    try:
        _collection = _client.get_collection(name=collection_name, embedding_function=ef)
        if _collection.metadata and _collection.metadata.get("chunks_hash") == content_hash:
            logger.info(f"Loaded persisted collection from disk: {collection_name}")
            _current_collection_name = collection_name
            _current_chunks_hash = content_hash
            _current_embedding_backend = backend
            return _collection
    except Exception:
        pass

    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass

    _collection = _client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine", "chunks_hash": content_hash},
    )

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "word_count": c.get("word_count", 0),
            "interview_name": c.get("interview_name", "Unknown"),
            "participant_name": c.get("participant_name", "Unknown"),
        }
        for c in chunks
    ]

    _collection.add(documents=documents, ids=ids, metadatas=metadatas)
    _current_collection_name = collection_name
    _current_chunks_hash = content_hash
    _current_embedding_backend = backend

    logger.info(f"Indexed {len(chunks)} chunks.")
    return _collection


def semantic_search(
    query: str | list[str],
    collection: Any,
    n_results: int = 5,
    rerank: bool = False,
) -> list[dict]:
    """
    Run a semantic search over the indexed transcript chunks.

    Args:
        query: Natural language query string, or a list of expanded queries.
        collection: ChromaDB collection from build_index().
        n_results: Number of top results to return.
        rerank: If True, fetches more candidates and reranks with a CrossEncoder.

    Returns:
        List of result dicts with keys: id, text, score, metadata.
        When reranked, also includes dense_score (cosine) and rerank_score (cross-encoder).
        ``score`` stays as dense similarity for filtering; sort order uses rerank_score.
    """
    queries = [query] if isinstance(query, str) else query
    logger.info(f"Searching for: {queries} (top {n_results}, rerank={rerank})")

    total_docs = collection.count()
    if total_docs == 0:
        logger.warning("Collection is empty; returning no search results.")
        return []

    fetch_k = n_results * 4 if rerank else n_results
    n_results_per_query = max(1, min(fetch_k, total_docs))
    results = collection.query(
        query_texts=queries,
        n_results=n_results_per_query,
        include=["documents", "distances", "metadatas"],
    )

    unique_hits: dict[str, dict] = {}
    for doc_list, dist_list, id_list, meta_list in zip(
        results["documents"],
        results["distances"],
        results["ids"],
        results["metadatas"],
    ):
        for doc, dist, chunk_id, meta in zip(doc_list, dist_list, id_list, meta_list):
            dense = round(1 - dist, 4)
            if chunk_id not in unique_hits or unique_hits[chunk_id]["dense_score"] < dense:
                unique_hits[chunk_id] = {
                    "id": chunk_id,
                    "text": doc,
                    "score": dense,
                    "dense_score": dense,
                    "metadata": meta,
                }

    hits_list = list(unique_hits.values())

    if rerank and hits_list:
        try:
            ce_model = _resolve_cross_encoder()
            original_query = queries[0]
            pairs = [[original_query, hit["text"]] for hit in hits_list]
            logger.info(f"Reranking {len(pairs)} candidate chunks...")

            ce_scores = ce_model.predict(pairs)
            for hit, ce_score in zip(hits_list, ce_scores):
                hit["rerank_score"] = float(ce_score)
        except Exception as e:
            logger.error(f"Reranking failed, falling back to dense scores: {e}")

    if rerank and hits_list and hits_list[0].get("rerank_score") is not None:
        sorted_hits = sorted(
            hits_list,
            key=lambda x: x.get("rerank_score", relevance_score(x)),
            reverse=True,
        )
        final_n = n_results
    else:
        sorted_hits = sorted(hits_list, key=lambda x: relevance_score(x), reverse=True)
        final_n = min(n_results * len(queries), len(sorted_hits))

    return sorted_hits[:final_n]


def extract_quotes_from_hits(hits: list[dict], min_score: float = 0.3) -> list[dict]:
    """
    Extract customer-only quotes from search results.
    Filters out interviewer lines and low-relevance chunks.
    Uses dense/cosine similarity for thresholds, not cross-encoder scores.
    """
    import re
    from src.constants import INTERVIEWEE_PATTERN, INTERVIEWER_PATTERN

    quotes = []
    for hit in hits:
        if relevance_score(hit) < min_score:
            continue

        meta = hit.get("metadata", {})
        source = f"{meta.get('interview_name', 'Unknown')} ({meta.get('participant_name', 'Unknown')})"

        lines = hit["text"].split("\n")
        capturing = False
        for line in lines:
            if INTERVIEWEE_PATTERN.match(line):
                capturing = True
                text = re.sub(r"^[^:]+:\s*", "", line).strip()
                if text:
                    quotes.append({"text": text, "source": source})
            elif INTERVIEWER_PATTERN.match(line):
                capturing = False
            elif capturing and line.strip():
                quotes.append({"text": line.strip(), "source": source})

    seen = set()
    unique = []
    for q in quotes:
        if q["text"] not in seen and len(q["text"]) > 20:
            seen.add(q["text"])
            unique.append(q)

    return unique
