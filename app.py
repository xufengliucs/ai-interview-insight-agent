"""
app.py
------
AI Customer Interview Insight Agent
Streamlit UI — upload a transcript and surface themes, quotes, and insights.
"""

import json
import logging
import os
import re
import html
import sys
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.ingestion import (
    load_transcript,
    chunk_transcript,
    get_full_text,
    chunks_from_interviews,
)
from src.retrieval import (
    build_index,
    semantic_search,
    extract_quotes_from_hits,
    format_hit_score,
)
from src.retrieval_streamlit import install_streamlit_model_caches
from src.insights import (
    extract_aggregate_insights,
    generate_evidence_insight,
    answer_research_query,
)
from src.evaluation import evaluate_insights

install_streamlit_model_caches()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Insight Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
  }

  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
  }

  .main-header h1 {
    font-size: 2.2rem;
    margin: 0 0 0.5rem 0;
    font-weight: 400;
    letter-spacing: -0.5px;
  }

  .main-header p {
    opacity: 0.65;
    font-size: 1rem;
    margin: 0;
  }

  .stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
  }

  .stat-number {
    font-size: 2rem;
    font-weight: 600;
    color: #0f172a;
    line-height: 1;
  }

  .stat-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .quote-card {
    background: #fefefe;
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }

  .quote-text {
    font-style: italic;
    color: #1e293b;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  .score-badge {
    font-size: 0.72rem;
    background: #eff6ff;
    color: #3b82f6;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
  }

  .theme-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.15s;
  }

  .theme-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }

  .badge-high { background: #fef2f2; color: #dc2626; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-medium { background: #fffbeb; color: #d97706; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-low { background: #f0fdf4; color: #16a34a; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-critical { background: #fef2f2; color: #dc2626; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-positive { background: #f0fdf4; color: #16a34a; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-negative { background: #fef2f2; color: #dc2626; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-neutral { background: #f1f5f9; color: #475569; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }
  .badge-mixed { background: #faf5ff; color: #7c3aed; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 500; }

  .section-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 2rem 0;
  }
  .insight-card {
    background: #1e293b;
    color: white;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  }
  .insight-card h4 {
    color: #94a3b8;
    font-size: 0.85rem;
    letter-spacing: 1px;
    margin-top: 0;
    margin-bottom: 0.75rem;
  }

</style>
""", unsafe_allow_html=True)


# ── App state management ─────────────────────────────────────────────────────

def load_app_state():
    state_file = Path(__file__).parent / "chroma_db" / "app_state.json"
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load app state: {e}")
    return None


def save_app_state():
    state_file = Path(__file__).parent / "chroma_db" / "app_state.json"
    state_file.parent.mkdir(exist_ok=True)
    state = {
        "project_entries": st.session_state.project_entries,
        "research_chat": st.session_state.research_chat,
        "aggregate_insights": st.session_state.aggregate_insights,
        "evaluation_data": st.session_state.evaluation_data,
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def init_session_state():
    defaults = {
        "project_entries": [],
        "all_chunks": [],
        "collection": None,
        "search_results": None,
        "assistant_answer": None,
        "research_chat": [],
        "evidence_insight": None,
        "aggregate_insights": None,
        "evaluation_data": None,
        "transcript": None,
        "pending_transcript": "",
        "pending_name": "",
        "pending_participant_name": "",
        "pending_segment": "",
        "pending_role": "",
        "pending_notes": "",
        "last_embedding_backend": None,
        "last_uploaded_file_id": None,
        "_state_loaded": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state._state_loaded:
        saved_state = load_app_state()
        if saved_state and saved_state.get("project_entries"):
            st.session_state.project_entries = saved_state.get("project_entries", [])
            st.session_state.research_chat = saved_state.get("research_chat", [])
            st.session_state.aggregate_insights = saved_state.get("aggregate_insights", None)
            st.session_state.evaluation_data = saved_state.get("evaluation_data", None)
            sync_project_chunks()
            if st.session_state.all_chunks:
                st.session_state._needs_collection_rebuild = True
        st.session_state._state_loaded = True


def sync_project_chunks() -> None:
    """Rebuild indexed chunks from project entries (single source of truth)."""
    st.session_state.all_chunks = chunks_from_interviews(st.session_state.project_entries)


def refresh_collection(backend: str):
    if st.session_state.all_chunks:
        st.session_state.collection = build_index(
            st.session_state.all_chunks,
            collection_name="interview",
            backend=backend,
        )
    else:
        st.session_state.collection = None


init_session_state()

openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    st.header("Settings")
    llm_backend = st.selectbox(
        "LLM backend", 
        ["openai", "gemini", "ollama"], 
        index=2 if not openai_key and not gemini_key else 0
    )
    embedding_backend = st.selectbox("Embedding backend", ["openai", "local"], index=1)
    n_results = st.slider("Search results", 2, 10, 5)
    st.markdown("---")
    st.markdown(
        "### No-key / local mode\n"
        "- Set `Embedding backend` to **local** for free, on-device embeddings.\n"
        "- Local embeddings do not require an API key.\n"
        "- LLM features need `OPENAI_API_KEY`, `GEMINI_API_KEY`, or a local **Ollama** server."
    )

    if embedding_backend == "openai" and not openai_key:
        st.warning(
            "OpenAI API key missing. Switch to **local** embeddings or set `OPENAI_API_KEY` in `.env`."
        )
    elif embedding_backend == "local":
        st.success(
            "Local embeddings enabled. You can search transcripts without an OpenAI key."
        )

    if llm_backend == "openai" and not openai_key:
        st.warning(
            "OpenAI key missing. Research assistant, insight extraction, and evaluation features will not work until `OPENAI_API_KEY` is available."
        )
    elif llm_backend == "gemini" and not gemini_key:
        st.warning(
            "Gemini key missing. Research assistant, insight extraction, and evaluation features will not work until `GEMINI_API_KEY` is available."
        )
    elif llm_backend == "ollama":
        st.info(
            "Using local Ollama. Ensure Ollama is running on your machine (e.g., `ollama run llama3`)."
        )
    else:
        st.info(f"LLM backend set to **{llm_backend}**.")

    st.markdown("---")
    st.markdown(
        "### Quick start\n"
        "1. Upload a transcript or click **Load demo project**.\n"
        "2. Choose `local` embeddings to try without a key.\n"
        "3. Use the Research tab for semantic search.\n"
        "4. Add an LLM key later for full insights and evaluation."
    )


if st.session_state.last_embedding_backend != embedding_backend:
    st.session_state.last_embedding_backend = embedding_backend
    try:
        if embedding_backend == "openai" and not openai_key:
            st.session_state.collection = None
        else:
                with st.spinner("Switching embedding backend and rebuilding index..."):
                    refresh_collection(embedding_backend)
    except Exception as exc:
        st.session_state.collection = None
        st.sidebar.error(f"Could not build embedders: {exc}")

if st.session_state.get("_needs_collection_rebuild") and st.session_state.all_chunks:
    st.session_state._needs_collection_rebuild = False
    try:
        if embedding_backend == "openai" and not openai_key:
            st.session_state.collection = None
        else:
            refresh_collection(embedding_backend)
    except Exception as exc:
        st.session_state.collection = None
        st.sidebar.error(f"Could not rebuild index from saved project: {exc}")


tabs = st.tabs([
    "Dashboard",
    "Interviews",
    "Insights",
    "Research",
    "Export",
])

with tabs[0]:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 00 · Project dashboard")

    interview_count = len(st.session_state.project_entries)
    chunk_count = len(st.session_state.all_chunks)
    segment_counts = Counter(
        entry.get("segment", "Unspecified") or "Unspecified"
        for entry in st.session_state.project_entries
    )
    llm_ready = (
        (llm_backend == "openai" and bool(openai_key))
        or (llm_backend == "gemini" and bool(gemini_key))
        or (llm_backend == "ollama")
    )

    cols = st.columns(4)
    cols[0].metric("Interviews", interview_count)
    cols[1].metric("Indexed chunks", chunk_count)
    cols[2].metric("Search results", n_results)
    cols[3].metric("LLM ready", "Yes" if llm_ready else "No")

    if interview_count:
        st.markdown("### Interview coverage")
        st.bar_chart(segment_counts)
    else:
        st.info("Upload at least one transcript to start building your research project.")

    st.markdown("---")
    st.markdown(
        "### Usage notes\n"
        "- Use the Interviews tab to add transcripts and metadata.\n"
        "- The Research tab surface quotes and evidence from all indexed interviews.\n"
        "- The Insights tab summarizes themes and recommendations across your project."
    )


def _load_demo_project():
    """Instantly inject 3 realistic mock interviews to let users test cross-session insights."""
    demo_interviews = [
        {
            "name": "Interview 1 - Pricing & Onboarding",
            "participant_name": "Alice",
            "segment": "SMB",
            "role": "CEO",
            "notes": "Struggled with initial setup.",
            "text": "Interviewer: How was your onboarding experience?\n\nCustomer: It was okay, but the pricing tiers were really confusing. I didn't know if I needed the Pro or Enterprise plan.\n\nInterviewer: What would make it better?\n\nCustomer: A clear comparison chart. Also, the invite system for my team was buggy. I had to resend invites three times."
        },
        {
            "name": "Interview 2 - Mobile App Sync",
            "participant_name": "Bob",
            "segment": "Enterprise",
            "role": "Field Sales",
            "notes": "Uses the app mostly on the go.",
            "text": "Interviewer: You mentioned using the mobile app heavily. How is that going?\n\nCustomer: The mobile app is fast, but the offline sync is a nightmare. I lose data when I'm in an elevator or subway.\n\nInterviewer: How about notifications?\n\nCustomer: Way too many. I get pinged for every little update. I need a daily digest option, otherwise it's just notification fatigue."
        },
        {
            "name": "Interview 3 - API & Integrations",
            "participant_name": "Charlie",
            "segment": "Mid-Market",
            "role": "Data Analyst",
            "notes": "Power user, wants API access.",
            "text": "Interviewer: What's the biggest missing feature for you right now?\n\nCustomer: Integrations, hands down. We use Slack and Jira, and right now I have to manually copy-paste data between them and your tool.\n\nInterviewer: Are you looking for a native integration or an API?\n\nCustomer: Native Jira sync would save me 5 hours a week. An open API would be a nice bonus for our engineering team."
        }
    ]

    st.session_state.project_entries = list(demo_interviews)
    sync_project_chunks()
        
    st.session_state.search_results = None
    st.session_state.research_chat = []
    st.session_state.assistant_answer = None
    st.session_state.evidence_insight = None
    st.session_state.aggregate_insights = None
    st.session_state.evaluation_data = None
    
    # Populate the intake form with the first interview so users can see what it looks like
    first = demo_interviews[0]
    st.session_state.pending_transcript = first["text"]
    st.session_state.pending_name = first["name"] + " (Draft Copy)"
    st.session_state.pending_participant_name = first["participant_name"]
    st.session_state.pending_segment = first["segment"]
    st.session_state.pending_role = first["role"]
    st.session_state.pending_notes = first["notes"]

    refresh_collection(st.session_state.last_embedding_backend or "local")
    save_app_state()
    st.rerun()


def _clear_intake_form():
    st.session_state.pending_transcript = ""
    st.session_state.pending_name = ""
    st.session_state.pending_participant_name = ""
    st.session_state.pending_segment = ""
    st.session_state.pending_role = ""
    st.session_state.pending_notes = ""


def _esc(text) -> str:
    """Escape user/LLM text before embedding in HTML."""
    return html.escape(str(text)) if text is not None else ""


def _safe_css_token(value: str, default: str = "neutral") -> str:
    """Restrict LLM-provided tokens used in CSS class names."""
    token = re.sub(r"[^a-z0-9_-]", "", str(value).lower())
    return token or default


def render_chat_markdown(content: str) -> None:
    """Render chat messages with Markdown but without raw HTML."""
    st.markdown(content, unsafe_allow_html=False)


def render_chat_quote(quote: str) -> None:
    st.markdown(f"- {_esc(quote)}", unsafe_allow_html=False)


def highlight_keywords(text: str, query: str) -> str:
    """Helper to safely highlight query terms in text using HTML <mark> tags."""
    if not text:
        return ""
    escaped_text = html.escape(text)
    if not query:
        return escaped_text
    
    # Extract words longer than 2 characters from the query
    # Safety: ensure the extracted query words are also escaped to prevent XSS
    words = [html.escape(w) for w in re.split(r'\W+', query) if len(w) > 2]
    if not words:
        return escaped_text
        
    # Create a regex pattern to match any of the words (case-insensitive)
    pattern = re.compile(rf"\b({'|'.join(map(re.escape, words))})\b", re.IGNORECASE)
    return pattern.sub(r'<mark style="background: #fef08a; color: #1e293b; padding: 0 2px; border-radius: 2px;">\1</mark>', escaped_text)


with tabs[1]:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 01 · Interview ingestion")

    upload_file = st.file_uploader(
        "Upload a transcript (.txt, .md) or audio (.mp3, .wav, .m4a, .flac)",
        type=["txt", "md", "mp3", "wav", "m4a", "flac"],
        help="Upload text transcripts or raw audio files. Audio is transcribed via Whisper API.",
    )

    whisper_prompt = st.text_input(
        "Audio transcription prompt (optional)",
        placeholder="e.g. product names, acronyms, or industry jargon like: ChromaDB, LLM, Streamlit",
        help="Whisper uses this context to correctly spell specific terms during transcription.",
    )
    
    st.markdown("🎧 *Don't have an audio file? Right click and save this [jfk.flac](https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac) (OpenAI official test audio) to try Whisper.*")

    if st.button("🚀 Load full demo project (3 interviews)", type="secondary", help="Instantly populates the system with 3 realistic interviews to test cross-session insights."):
        with st.spinner("Building demo project..."):
            st.toast("✅ Demo project loaded!")
            _load_demo_project()

    if upload_file is not None:
        # Ensure we only read the file once when it's uploaded/changed
        if st.session_state.get("last_uploaded_file_id") != upload_file.file_id:
            try:
                file_ext = upload_file.name.split('.')[-1].lower()
                
                if file_ext in ['mp3', 'wav', 'm4a', 'flac']:
                    with st.spinner("🎧 Transcribing audio with Whisper API... This might take a minute."):
                        from src.ingestion import transcribe_audio
                        text = transcribe_audio(upload_file.getvalue(), upload_file.name, prompt=whisper_prompt)
                    
                    if llm_ready:
                        with st.spinner("🤖 Identifying speakers and formatting dialogue..."):
                            from src.insights import format_transcript_dialogue
                            text = format_transcript_dialogue(text, backend=llm_backend)
                else:
                    text = load_transcript(text=upload_file.getvalue().decode("utf-8"))
                
                st.session_state.pending_transcript = text
                st.session_state.last_uploaded_file_id = upload_file.file_id
            except Exception as exc:
                st.error(f"Could not read upload: {exc}")

    # Optimize rendering for very long transcripts
    current_transcript = st.session_state.pending_transcript
    if len(current_transcript) > 10000:
        st.info("Transcript is too long to display fully. Preview is truncated to maintain UI performance.")
        st.text_area(
            "Transcript preview (View Only)",
            value=current_transcript[:10000] + "\n\n... [Content truncated for preview. Full text will be processed.]",
            height=240,
            disabled=True,
        )
    else:
        updated_transcript = st.text_area(
            "Transcript preview",
            value=current_transcript,
            height=240,
        )
        if updated_transcript != current_transcript:
            st.session_state.pending_transcript = updated_transcript

    st.text_input("Interview name", key="pending_name")
    st.text_input("Participant name", key="pending_participant_name")
    st.text_input("Segment", key="pending_segment")
    st.text_input("Role", key="pending_role")
    st.text_area("Notes", key="pending_notes", height=120)

    add_clicked = st.button("Add interview to project", type="primary")
    st.button("Clear form", type="secondary", on_click=_clear_intake_form)

    if add_clicked:
        if not st.session_state.pending_transcript.strip():
            st.error("Please upload or paste a transcript before adding it to the project.")
        else:
            with st.spinner("Processing text, computing embeddings, and updating index..."):
                try:
                    chunks = chunk_transcript(st.session_state.pending_transcript)
                    inv_name = st.session_state.pending_name or f"Interview {interview_count + 1}"
                    part_name = st.session_state.pending_participant_name or "Unknown"
                    for c in chunks:
                        c["interview_name"] = inv_name
                        c["participant_name"] = part_name
                        
                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.project_entries.append(
                        {
                            "name": inv_name,
                            "participant_name": part_name,
                            "segment": st.session_state.pending_segment,
                            "role": st.session_state.pending_role,
                            "notes": st.session_state.pending_notes,
                            "text": st.session_state.pending_transcript,
                        }
                    )
                    st.session_state.search_results = None
                    st.session_state.research_chat = []
                    st.session_state.assistant_answer = None
                    st.session_state.evidence_insight = None
                    st.session_state.aggregate_insights = None
                    st.session_state.evaluation_data = None
                    if embedding_backend == "openai" and not openai_key:
                        st.warning("OpenAI key is missing — cannot build OpenAI embeddings. Switch to local backend or add OPENAI_API_KEY.")
                    else:
                        refresh_collection(embedding_backend)
                    st.success("Interview added to the project and indexed for search.")
                    _clear_intake_form()
                    save_app_state()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to add interview: {exc}")

    if st.session_state.project_entries:
        st.markdown("### Project interviews")
        for i, entry in enumerate(st.session_state.project_entries):
            with st.expander(f"📄 **{entry['name']}** ({entry.get('segment', 'Unspecified')} / {entry.get('role', 'Unspecified')})"):
                st.markdown(f"**Participant:** {entry.get('participant_name', 'N/A')} &nbsp; | &nbsp; **Notes:** {entry.get('notes', 'None')}")
                st.markdown("**Transcript:**")
                st.text_area("Transcript content", value=entry["text"], height=150, disabled=True, label_visibility="collapsed", key=f"proj_entry_{i}")
                
                if st.button("🗑️ Remove this interview", key=f"remove_entry_{i}"):
                    st.session_state.pending_remove_index = i

        # Handle the removal safely outside of the rendering loop
        if st.session_state.get("pending_remove_index") is not None:
            remove_idx = st.session_state.pop("pending_remove_index")
            if 0 <= remove_idx < len(st.session_state.project_entries):
                st.session_state.project_entries.pop(remove_idx)
                sync_project_chunks()
                st.session_state.search_results = None
                st.session_state.research_chat = []
                st.session_state.aggregate_insights = None
                st.session_state.evaluation_data = None
                with st.spinner("Rebuilding index..."):
                    refresh_collection(st.session_state.last_embedding_backend or "local")
                save_app_state()
                st.rerun()
    else:
        st.info("No interviews added yet. Use the uploader above to begin.")


with tabs[3]:
    # Research workflow: Semantic Search + Evidence-backed Insight + Research Assistant
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 02 · Semantic Search & Evidence-backed Insight")

    if st.session_state.collection is None:
        st.markdown('<div class="empty-state">⬆️ Upload a transcript first to enable search.</div>', unsafe_allow_html=True)
    else:
        if not llm_ready:
            st.info(
                "Local embedding search is available, but research assistant and evidence-backed insight generation "
                "require an OpenAI or Gemini API key. Set one in .env and restart the app to enable those features."
            )
            
        with st.expander("⚙️ Advanced search settings", expanded=False):
            use_expansion = st.checkbox("🧠 Enable AI Query Expansion (Higher Recall)", value=False, disabled=not llm_ready, help="Uses LLM to rewrite your query into multiple variants.")
            use_rerank = st.checkbox("🎯 Enable Cross-Encoder Reranking (Higher Precision)", value=False, help="Uses a local ms-marco model to accurately re-score search results.")

        query_col, btn_col = st.columns([4, 1])

        with query_col:
            query = st.text_input(
                "Ask a question about the interview",
                placeholder="e.g. pricing pain points, subscription frustration, what features do they want",
                label_visibility="collapsed",
            )

        with btn_col:
            search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

        # Quick suggestion chips
        st.markdown("**Quick searches:**")
        chip_cols = st.columns(4)
        suggestions = [
            "subscription problems",
            "notification fatigue",
            "sync and reliability",
            "feature requests",
        ]
        for i, suggestion in enumerate(suggestions):
            if chip_cols[i].button(f"💡 {suggestion}", key=f"chip_{i}"):
                query = suggestion
                search_clicked = True

        if search_clicked and query:
            with st.spinner(f"Searching for '{query}'…"):
                try:
                    search_queries = query
                    if use_expansion and llm_ready:
                        from src.insights import expand_research_query
                        with st.spinner("Expanding query using LLM..."):
                            search_queries = expand_research_query(query, backend=llm_backend)
                            st.caption(f"**Expanded search:** {', '.join(search_queries)}")

                    hits = semantic_search(
                        search_queries,
                        st.session_state.collection,
                        n_results=n_results,
                        rerank=use_rerank,
                    )
                    quotes = extract_quotes_from_hits(hits)
                    st.session_state.search_results = {
                        "query": query,
                        "hits": hits,
                        "quotes": quotes,
                    }
                    st.session_state.research_chat = []
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    logger.exception("Search error")

        if st.session_state.search_results:
            results = st.session_state.search_results
            st.markdown(f"**Results for:** _{results['query']}_")

            tab1, tab2 = st.tabs(["💬 Customer Quotes", "📦 Raw Chunks"])

            with tab1:
                quotes = results["quotes"]
                if quotes:
                    for q in quotes:
                        highlighted_q = highlight_keywords(q["text"], results["query"])
                        source_label = html.escape(q.get("source", "Unknown"))
                        st.markdown(
                            f'<div class="quote-card"><div class="quote-text">"{highlighted_q}"</div>'
                            f'<div style="text-align: right; font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">— {source_label}</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No customer quotes extracted from results. Try a different query.")

            with tab2:
                for hit in results["hits"]:
                    meta = hit.get("metadata", {})
                    source_label = f"{meta.get('interview_name', 'Unknown')} ({meta.get('participant_name', 'Unknown')})"
                    with st.expander(
                        f"Chunk {hit['id']} · {format_hit_score(hit)} · {source_label}"
                    ):
                        highlighted_chunk = highlight_keywords(hit["text"], results["query"])
                        st.markdown(
                            f'<div style="white-space: pre-wrap; font-size: 0.9rem;">{highlighted_chunk}</div>',
                            unsafe_allow_html=True
                        )

            if results["quotes"]:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 🤖 Chat with Research Assistant")

                if not llm_ready:
                    st.warning("Research assistant requires an LLM key. Add OPENAI_API_KEY or GEMINI_API_KEY to use this feature.")
                else:
                    # Render existing chat
                    for msg in st.session_state.research_chat:
                        with st.chat_message(msg["role"]):
                            render_chat_markdown(msg["content"])
                            if msg["role"] == "assistant" and msg.get("supporting_quotes"):
                                with st.expander("Supporting quotes"):
                                    for q in msg["supporting_quotes"]:
                                        render_chat_quote(q)

                    if not st.session_state.research_chat:
                        if st.button("Generate summary & start chat", type="primary"):
                            with st.spinner("Analyzing results..."):
                                try:
                                    formatted_quotes = [f'{q["text"]} (Source: {q["source"]})' for q in results["quotes"]]
                                    assistant = answer_research_query(
                                        results["query"],
                                        formatted_quotes,
                                        chat_history=[],
                                        backend=llm_backend,
                                    )
                                    ans_text = assistant.get('answer', '')
                                    if assistant.get('recommended_next_steps'):
                                        ans_text += (
                                            f"\n\n**Recommended next step:** "
                                            f"{_esc(assistant.get('recommended_next_steps'))}"
                                        )
                                    
                                    st.session_state.research_chat.append({
                                        "role": "user",
                                        "content": results["query"]
                                    })
                                    st.session_state.research_chat.append({
                                        "role": "assistant",
                                        "content": ans_text,
                                        "supporting_quotes": assistant.get('supporting_quotes', [])
                                    })
                                    save_app_state()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Research assistant failed: {e}")
                                    logger.exception("Research assistant error")

                    if st.session_state.research_chat:
                        with st.form("chat_input_form", clear_on_submit=True):
                            chat_cols = st.columns([5, 1])
                            follow_up = chat_cols[0].text_input(
                                "Follow-up",
                                label_visibility="collapsed",
                                placeholder="Ask a follow-up question..."
                            )
                            submit_chat = chat_cols[1].form_submit_button("Send")

                        if submit_chat and follow_up:
                            st.session_state.research_chat.append({"role": "user", "content": follow_up})
                            with st.chat_message("user"):
                                render_chat_markdown(follow_up)
                            
                            with st.chat_message("assistant"):
                                with st.spinner("Thinking..."):
                                    try:
                                        formatted_quotes = [f'{q["text"]} (Source: {q["source"]})' for q in results["quotes"]]
                                        assistant = answer_research_query(
                                            follow_up,
                                            formatted_quotes,
                                            chat_history=st.session_state.research_chat[:-1],
                                            backend=llm_backend,
                                        )
                                        ans_text = assistant.get('answer', '')
                                        if assistant.get('recommended_next_steps'):
                                            ans_text += (
                                                f"\n\n**Recommended next step:** "
                                                f"{_esc(assistant.get('recommended_next_steps'))}"
                                            )
                                            
                                        render_chat_markdown(ans_text)
                                        if assistant.get('supporting_quotes'):
                                            with st.expander("Supporting quotes"):
                                                for q in assistant['supporting_quotes']:
                                                    render_chat_quote(q)
                                                    
                                        st.session_state.research_chat.append({
                                            "role": "assistant",
                                            "content": ans_text,
                                            "supporting_quotes": assistant.get('supporting_quotes', [])
                                        })
                                        save_app_state()
                                    except Exception as e:
                                        st.error(f"Research assistant failed: {e}")
                                        st.session_state.research_chat.pop()

        # Evidence-backed insight generator
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## Evidence-backed Insight")
        st.caption("Generate a focused insight card from a specific topic and the supporting quotes retrieved above.")

        topic_col, btn_col2 = st.columns([4, 1])

        with topic_col:
            insight_topic = st.text_input(
                "Insight topic",
                placeholder="e.g. subscription tracking frustration",
                label_visibility="collapsed",
                key="insight_topic_input",
            )

        with btn_col2:
            evidence_clicked = st.button(
                "💡 Generate",
                use_container_width=True,
                type="primary",
                disabled=not llm_ready,
            )
            if not llm_ready:
                st.caption("LLM key required for evidence-backed insight generation.")

        if evidence_clicked and insight_topic:
            with st.spinner("Retrieving evidence and synthesizing insight…"):
                try:
                    hits = semantic_search(
                        insight_topic,
                        st.session_state.collection,
                        n_results=6,
                    )
                    quotes = extract_quotes_from_hits(hits, min_score=0.2)

                    if not quotes:
                        st.warning("No strong quotes found for this topic. Try rephrasing.")
                    else:
                        formatted_quotes = [f'{q["text"]} (Source: {q["source"]})' for q in quotes[:4]]
                        insight = generate_evidence_insight(
                            topic=insight_topic,
                            quotes=formatted_quotes,
                            backend=llm_backend,
                        )
                        st.session_state.evidence_insight = insight
                except Exception as e:
                    st.error(f"Evidence insight failed: {e}")
                    logger.exception("Evidence insight error")

        if st.session_state.evidence_insight:
            ev = st.session_state.evidence_insight
            confidence = _safe_css_token(ev.get("confidence", "medium"), "medium")

            st.markdown(
                f'<div class="insight-card">'
                f'<h4>INSIGHT <span style="opacity:0.5">·</span> '
                f'<span style="font-size:0.75rem; background:rgba(255,255,255,0.15); '
                f'padding:2px 10px; border-radius:20px;">{_esc(confidence)} confidence</span></h4>'
                f'<p style="font-size:1.15rem; font-family:\'DM Serif Display\', serif; '
                f'margin:0 0 1.25rem; line-height:1.5;">{_esc(ev.get("insight", ""))}</p>'
                f'<hr style="border-color:rgba(255,255,255,0.15); margin:1rem 0;">'
                f'<h4>SUPPORTING EVIDENCE</h4>',
                unsafe_allow_html=True,
            )

            evidence = ev.get("evidence", [])
            for quote in evidence:
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.08); border-radius:8px; '
                    f'padding:0.75rem 1rem; margin-bottom:0.5rem; font-style:italic; '
                    f'font-size:0.9rem;">&ldquo;{_esc(quote)}&rdquo;</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<hr style="border-color:rgba(255,255,255,0.15); margin:1rem 0;">'
                f'<h4>RECOMMENDATION</h4>'
                f'<p style="font-size:0.95rem; opacity:0.9; margin:0;">'
                f'{_esc(ev.get("recommendation", ""))}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

with tabs[2]:
    # Insights tab — generate and browse aggregate insights
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 03 · Aggregate Research Insights")

    if not st.session_state.project_entries:
        st.markdown('<div class="empty-state">⬆️ Upload at least one transcript to start project-level insights.</div>', unsafe_allow_html=True)
    else:
        gen_col, eval_col = st.columns([2, 2])
        with gen_col:
            generate_clicked = st.button(
                "✨ Generate cross-interview insights",
                use_container_width=True,
                type="primary",
                disabled=st.session_state.aggregate_insights is not None or not llm_ready,
            )
        with eval_col:
            evaluate_clicked = st.button(
                "🧪 Evaluate insights",
                use_container_width=True,
                type="secondary",
                disabled=st.session_state.aggregate_insights is None or not llm_ready,
            )

        if not llm_ready:
            st.warning("LLM key required to generate or evaluate aggregate insights. Add OPENAI_API_KEY or GEMINI_API_KEY.")

        if st.session_state.aggregate_insights is not None:
            st.caption("✅ Aggregate insights generated. Add or remove interviews to regenerate.")

        if generate_clicked:
            with st.spinner("Analyzing interview set with LLM…"):
                try:
                    data = extract_aggregate_insights(
                        st.session_state.project_entries,
                        backend=llm_backend,
                    )
                    st.session_state.aggregate_insights = data
                    save_app_state()
                    st.rerun()
                except Exception as e:
                    st.error(f"Insight generation failed: {e}")
                    logger.exception("Aggregate insight error")

        if evaluate_clicked and st.session_state.aggregate_insights:
            with st.spinner("Evaluating insight quality…"):
                try:
                    st.session_state.evaluation_data = evaluate_insights(
                        get_full_text(st.session_state.all_chunks),
                        st.session_state.aggregate_insights,
                        backend=llm_backend,
                    )
                    save_app_state()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    logger.exception("Evaluation error")

        # ── Display insights ─────────────────────────────────────────────────────
        if st.session_state.aggregate_insights:
            data = st.session_state.aggregate_insights

            # Overall sentiment banner
            sentiment = _safe_css_token(data.get("overall_sentiment", "unknown"), "unknown")
            sentiment_summary = data.get("sentiment_summary", "")
            badge_class = f"badge-{sentiment}"
            st.markdown(
                f'<div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; '
                f'padding:1rem 1.5rem; margin-bottom:1.5rem;">'
                f'<strong>Overall Sentiment</strong> &nbsp; '
                f'<span class="{badge_class}">{_esc(sentiment.upper())}</span><br>'
                f'<span style="color:#475569; font-size:0.9rem; margin-top:0.25rem; display:block;">'
                f'{_esc(sentiment_summary)}</span></div>',
                unsafe_allow_html=True,
            )

            tabs_local = st.tabs([
                "🎯 Themes",
                "😤 Pain Points",
                "💡 Feature Requests",
                "📈 Trends",
                "🔍 Cross-session Findings",
                "📋 Recommendations",
                "🔧 Raw JSON",
            ])

            with tabs_local[0]:
                themes = data.get("themes", [])
                if themes:
                    for t in themes:
                        freq = _safe_css_token(t.get("frequency", "medium"), "medium")
                        sent = _safe_css_token(t.get("sentiment", "neutral"), "neutral")
                        st.markdown(
                            f'<div class="theme-card">'
                            f'<strong>{_esc(t.get("title", ""))}</strong> &nbsp;'
                            f'<span class="badge-{freq}">{_esc(freq)}</span> &nbsp;'
                            f'<span class="badge-{sent}">{_esc(sent)}</span>'
                            f'<p style="margin:0.5rem 0 0; color:#475569; font-size:0.9rem;">{_esc(t.get("description", ""))}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No themes extracted.")

            with tabs_local[1]:
                pain_points = data.get("pain_points", [])
                if pain_points:
                    for p in pain_points:
                        severity = _safe_css_token(p.get("severity", "moderate"), "moderate")
                        quote = p.get("quote", "")
                        st.markdown(
                            f'<div class="theme-card">'
                            f'<strong>{_esc(p.get("title", ""))}</strong> &nbsp;'
                            f'<span class="badge-{severity}">{_esc(severity)}</span>'
                            f'<p style="margin:0.5rem 0 0; color:#475569; font-size:0.9rem;">{_esc(p.get("description", ""))}</p>'
                            + (f'<div class="quote-card" style="margin-top:0.75rem;"><div class="quote-text">&ldquo;{_esc(quote)}&rdquo;</div></div>' if quote else "")
                            + f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No pain points extracted.")

            with tabs_local[2]:
                features = data.get("feature_requests", [])
                if features:
                    for f in features:
                        quote = f.get("quote", "")
                        st.markdown(
                            f'<div class="theme-card">'
                            f'<strong>🔧 {_esc(f.get("title", ""))}</strong>'
                            f'<p style="margin:0.5rem 0 0; color:#475569; font-size:0.9rem;">{_esc(f.get("description", ""))}</p>'
                            + (f'<div class="quote-card" style="margin-top:0.75rem;"><div class="quote-text">&ldquo;{_esc(quote)}&rdquo;</div></div>' if quote else "")
                            + f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No feature requests extracted.")

            with tabs_local[3]:
                trends = data.get("trend_signals", [])
                if trends:
                    for t in trends:
                        st.markdown(
                            f'<div class="theme-card">'
                            f'<strong>{_esc(t.get("signal", "Trend"))}</strong>'
                            f'<p style="margin:0.5rem 0 0; color:#475569; font-size:0.9rem;">{_esc(t.get("description", ""))}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No trend signals extracted.")

            with tabs_local[4]:
                findings = data.get("cross_session_findings", [])
                if findings:
                    for f in findings:
                        sessions = ", ".join(_esc(s) for s in f.get("supporting_sessions", []))
                        st.markdown(
                            f'<div class="theme-card">'
                            f'<strong>{_esc(f.get("title", ""))}</strong>'
                            f'<p style="margin:0.5rem 0 0; color:#475569; font-size:0.9rem;">{_esc(f.get("description", ""))}</p>'
                            f'<p style="margin:0.5rem 0 0; font-size:0.8rem; color:#64748b;">Supporting sessions: {sessions}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No cross-session findings extracted.")

            with tabs_local[5]:
                recs = data.get("recommendations", [])
                if recs:
                    for r in recs:
                        priority = _safe_css_token(r.get("priority", "medium"), "medium")
                        st.markdown(
                            f'<div class="theme-card">'
                            f'<strong>{_esc(r.get("title", ""))}</strong> &nbsp;'
                            f'<span class="badge-{priority}">{_esc(priority)} priority</span>'
                            f'<p style="margin:0.5rem 0 0; color:#475569; font-size:0.9rem;">{_esc(r.get("description", ""))}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No recommendations generated.")

            with tabs_local[6]:
                st.json(data)
                st.download_button(
                    "⬇️ Download insights JSON",
                    data=json.dumps(data, indent=2),
                    file_name="research_insights.json",
                    mime="application/json",
                )

            if st.session_state.evaluation_data:
                with st.expander("🧠 Insight evaluation details", expanded=True):
                    eval_data = st.session_state.evaluation_data
                    if "error" in eval_data:
                        st.error(f"Evaluation error: {eval_data['error']}")
                        st.json(eval_data)
                    else:
                        st.markdown(f"#### Overall Score: {eval_data.get('overall_score', 0)} / 5.0")
                        st.markdown(f"*{eval_data.get('summary', '')}*")
                        st.markdown("<br>", unsafe_allow_html=True)

                        cols = st.columns(4)
                        metrics = ["groundedness", "specificity", "actionability", "coverage"]

                        for i, m in enumerate(metrics):
                            if m in eval_data and isinstance(eval_data[m], dict):
                                score = eval_data[m].get("score", 0)
                                justification = eval_data[m].get("justification", "")
                                with cols[i]:
                                    st.metric(label=m.capitalize(), value=f"{score}/5")
                                    # Ensure score stays within 0.0 to 1.0 range for the progress bar
                                    safe_score = min(max(float(score) / 5.0, 0.0), 1.0)
                                    st.progress(safe_score)
                                    st.caption(justification)


with tabs[4]:
    # Export / share tab
    st.markdown('## Export / share')
    if not st.session_state.project_entries:
        st.markdown('<div class="empty-state">No project data available to export.</div>', unsafe_allow_html=True)
    else:
        if not st.session_state.aggregate_insights:
            st.info("💡 You have transcripts loaded, but haven't generated insights yet. Head over to the **Insights** tab to generate them before exporting your final report!")

        export_package = {
            "project_entries": [
                {
                    "name": entry["name"],
                    "participant_name": entry["participant_name"],
                    "segment": entry["segment"],
                    "role": entry["role"],
                    "notes": entry["notes"],
                    "text": entry["text"],
                }
                for entry in st.session_state.project_entries
            ],
            "aggregate_insights": st.session_state.aggregate_insights,
            "evaluation": st.session_state.evaluation_data,
        }

        st.markdown("### Project export")
        st.download_button(
            "⬇️ Download project package",
            data=json.dumps(export_package, indent=2),
            file_name="research_project_package.json",
            mime="application/json",
        )




# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#94a3b8; font-size:0.82rem;">'
    'AI Interview Insight Agent · Built with Streamlit + ChromaDB + OpenAI'
    '</p>',
    unsafe_allow_html=True,
)
