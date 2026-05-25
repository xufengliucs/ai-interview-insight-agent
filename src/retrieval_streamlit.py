"""
retrieval_streamlit.py
----------------------
Streamlit cache_resource wrappers for heavy retrieval models.
Keeps spinners in the UI layer while core logic stays in retrieval.py.
"""

from __future__ import annotations

import streamlit as st

from src.retrieval import (
    _load_cross_encoder,
    _load_local_embedding_function,
    install_model_loaders,
)


@st.cache_resource(show_spinner="Loading local embedding model...")
def cached_local_embedding_function():
    """Load local embeddings once per Streamlit server process."""
    return _load_local_embedding_function()


@st.cache_resource(show_spinner="Loading Cross-Encoder reranking model...")
def cached_cross_encoder():
    """Load reranker once per Streamlit server process."""
    return _load_cross_encoder()


def install_streamlit_model_caches() -> None:
    """Wire Streamlit-cached loaders into retrieval. Call once from app.py."""
    install_model_loaders(
        local_embedding=cached_local_embedding_function,
        cross_encoder=cached_cross_encoder,
    )
