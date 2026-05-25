# AI Customer Interview Insight Agent

A lightweight, production-style tool for extracting themes, pain points, quotes, and actionable recommendations from customer interview transcripts — using semantic search, embeddings, and LLM reasoning.

Built as a portfolio project inspired by user research tooling (Great Question, Dovetail).

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-green?style=flat-square)

---

## Motivation

Qualitative user research produces rich signal — but buried in hours of interviews. Analysts spend days manually tagging themes, pulling quotes, and writing synthesis documents.

This project explores how AI systems can accelerate that workflow: surface recurring themes, retrieve evidence-backed quotes, and generate actionable product insights — from a raw transcript in seconds.

---

## Features

| Feature | Description |
|---|---|
| **Multimodal Ingestion** | Supports text transcripts and raw audio (`.mp3`, `.wav`) transcribed via Whisper API |
| **Speaker Diarization** | Auto-infer and format speaker turns from unstructured text using LLM logic |
| **Advanced RAG Search** | High-recall AI query expansion + high-precision Cross-Encoder reranking |
| **Persistent Storage** | SQLite-backed ChromaDB + JSON project state under `chroma_db/` |
| **Multi-interview analysis** | Aggregate themes and trends across multiple sessions |
| **Research dashboard** | Track project health, interview metadata coverage, and workflow progress |
| **Evidence-backed insights** | Synthesize a focused insight card from a topic + supporting quotes |
| **Evaluation** | LLM-as-judge scoring for groundedness, specificity, and actionability |

---

## Architecture

```
transcript (.txt / audio)
     │
     ▼
┌──────────────┐     ┌────────────────────┐
│  ingestion   │────▶│  speaker-turn       │
│  (load +     │     │  chunking           │
│   clean)     │     └────────┬───────────┘
└──────────────┘              │
                              ▼
                    ┌────────────────────┐
                    │  embeddings        │
                    │  (OpenAI or local  │
                    │   sentence-trans.) │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │  ChromaDB          │
                    │  (persistent       │
                    │   SQLite store)    │
                    └────────┬───────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   ┌──────────────────┐           ┌──────────────────┐
   │  semantic search │           │  LLM insight     │
   │  (+ rerank opt.) │           │  extraction      │
   └──────────────────┘           │  OpenAI / Gemini │
             │                    │  / Ollama        │
             ▼                    └──────────────────┘
   retrieved quotes                      │
                              themes, pain points,
                              recommendations, sentiment
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/ai-interview-insight-agent
cd ai-interview-insight-agent
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or GEMINI_API_KEY if you have one.
```

> **No API key?** Choose `Embedding backend = local` in the sidebar to search transcripts without an OpenAI key. For LLM features you can also use **Ollama** locally (no cloud key).

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Trial guide

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

3. In the sidebar, set `Embedding backend` to `local` if you do not have an OpenAI key.

4. Go to the **Interviews** tab and click **Load full demo project (3 interviews)** or upload a `.txt` / `.md` transcript.

5. Use the **Research** tab to ask questions, scan for customer quotes, and generate evidence-backed answers.

6. For cloud LLMs, add `OPENAI_API_KEY` or `GEMINI_API_KEY` in `.env`. For a fully local LLM, run [Ollama](https://ollama.com/) and select `ollama` as the LLM backend.

---

## No-key / local mode details

- `local` embedding mode uses `sentence-transformers` and works without any API key.
- Local embedding search is enough to explore transcripts, pull quotes, and test the interface.
- Heavy models are cached via Streamlit (`retrieval_streamlit.py`) so they do not reload on every interaction.
- LLM-powered features need an API key **or** a local Ollama server (`ollama run llama3`).
- Audio transcription (Whisper) requires `OPENAI_API_KEY` even when embeddings are local.
- Project data is saved to `chroma_db/app_state.json` (interviews + insights). Chunks are rebuilt from transcripts on load — not duplicated on disk.

---

## Usage

1. **Upload** one or more transcripts (or click **Load full demo project**) and fill in participant metadata
2. **Review the dashboard** — check interview counts, segment distribution, and workflow status
3. **Search** across all interviews — type a query like `"pricing frustration"` to retrieve relevant quotes
4. **Generate Insights** — extract cross-session themes, trends, and recommendations
5. **Ask research assistant** — evidence-backed Q&A with safe Markdown rendering
6. **Export** — download the structured insights and project package as JSON

---

## Project Structure

```
ai-interview-insight-agent/
├── app.py                      # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
│
├── src/
│   ├── ingestion.py            # Transcript loading and chunking
│   ├── retrieval.py            # Embeddings + ChromaDB vector search
│   ├── retrieval_streamlit.py  # Streamlit model cache (spinners)
│   ├── insights.py             # LLM theme extraction and insight generation
│   ├── prompts.py              # All prompt templates (centralized)
│   ├── evaluation.py           # LLM-as-judge quality scoring
│   └── constants.py            # Shared regex patterns
│
├── tests/
│   └── test_pipeline.py        # Unit tests (pytest)
│
├── sample_data/
│   └── sample_interview.txt
│
└── chroma_db/                  # Created at runtime (gitignored)
    ├── app_state.json          # Persisted project + insights
    └── …                       # Chroma SQLite files
```

---

## Configuration

All settings are adjustable in the sidebar at runtime:

| Setting | Options | Notes |
|---|---|---|
| LLM Backend | `openai`, `gemini`, `ollama` | Ollama uses `http://localhost:11434/v1` (no cloud key) |
| Embedding Backend | `openai`, `local` | `local` = sentence-transformers, no key needed |
| Search results | 2–10 | Number of chunks to retrieve per query |

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests cover ingestion, chunking, prompts, and retrieval helpers — no API key required.

---

## Design Decisions

**Why Streamlit?** Fastest path to a working, shareable demo. A React frontend would add days of work with no user-facing benefit for a portfolio project.

**Why ChromaDB (persistent)?** SQLite-backed storage under `chroma_db/` means restarts reuse embeddings when transcript content is unchanged. No Docker or separate vector DB server.

**Why separate `prompts.py`?** Prompt iteration is the highest-leverage activity in LLM systems. Separating prompts from logic makes A/B testing one change at a time easy.

**Why both OpenAI and local embeddings?** Run semantic search for free with sentence-transformers, while still offering OpenAI embeddings when a key is available.

**Why rebuild chunks on load?** Interview transcripts are the source of truth; chunk metadata is derived at startup so persisted state cannot drift.

---

## Extending This

Some ideas if you want to go further:

- **Retrieval evaluation UI** — wire up `score_retrieval_hits()` in the Research tab
- **Single-interview insights** — expose `extract_themes()` for one transcript in the UI
- **Evaluation dashboard** — visualize insight quality scores over time
- **Slack / Notion export** — push insights directly to your team's tools

---

## License

MIT
