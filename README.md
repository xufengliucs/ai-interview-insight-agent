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
| **Upload transcript** | Supports `.txt` and `.md` files with or without speaker labels |
| **Semantic search** | Retrieve relevant quotes using embedding-based vector search |
| **Multi-interview analysis** | Aggregate themes and trends across multiple sessions |
| **Research dashboard** | Track project health, interview metadata coverage, and workflow progress |
| **Participant metadata** | Track interview segment, role, and screener notes across your research project |
| **Evidence-backed insights** | Synthesize a focused insight card from a topic + supporting quotes |
| **Evaluation** | LLM-as-judge scoring for groundedness, specificity, and actionability |

---

## Architecture

```
transcript (.txt)
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
                    │  (in-memory        │
                    │   vector store)    │
                    └────────┬───────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   ┌──────────────────┐           ┌──────────────────┐
   │  semantic search │           │  LLM insight     │
   │  (cosine sim.)   │           │  extraction      │
   └──────────────────┘           │  (GPT / Gemini)  │
             │                    └──────────────────┘
             ▼                               │
   retrieved quotes               themes, pain points,
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
# Edit .env and add your OPENAI_API_KEY or GEMINI_API_KEY if you have one.
```

> **No API key?** This app supports a free local embedding mode. Choose `Embedding backend = local` in the sidebar and you can still search transcripts and explore quotes without an OpenAI key.

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

3. In the sidebar, set `Embedding backend` to `local` if you do not have an API key.

4. Go to the **Interviews** tab and click **Load sample transcript** or upload a `.txt` / `.md` transcript.

5. Use the **Research** tab to ask questions, scan for customer quotes, and generate evidence-backed answers.

6. If you add `OPENAI_API_KEY` or `GEMINI_API_KEY`, switch the LLM backend to unlock full theme extraction, aggregate insights, and evaluation.

---

## No-key / local mode details

- `local` embedding mode uses `sentence-transformers` and works without any API key.
- Local embedding search is enough to explore transcripts, pull quotes, and test the interface.
- The app caches the local embedding model during a Streamlit session so it does not reload on every interaction.
- LLM-powered features like theme extraction, evidence insights, and evaluation still require an OpenAI or Gemini API key.
- If you do have a key, put it in `.env`, restart Streamlit, and choose the desired LLM backend.
- If you experience memory pressure, keep transcripts moderate in size and use local embedding mode for search-only exploration.

---

## Usage

1. **Upload** one or more `.txt` transcripts (or click "Load sample transcript") and optionally fill in participant metadata
2. **Review the dashboard** — check interview counts, segment distribution, metadata coverage, and workflow status
3. **Search** across all interviews — type a query like `"pricing frustration"` to retrieve relevant quotes
4. **Generate Insights** — click the button to extract cross-session themes, trends, and recommendations
5. **Ask research assistant** — use evidence-backed quotes to answer a focused research question
6. **Export** — download the structured insights and project package as JSON

---

## Project Structure

```
ai-interview-insight-agent/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
│
├── src/
│   ├── ingestion.py        # Transcript loading and chunking
│   ├── retrieval.py        # Embeddings + ChromaDB vector search
│   ├── insights.py         # LLM theme extraction and insight generation
│   ├── prompts.py          # All prompt templates (centralized)
│   └── evaluation.py       # LLM-as-judge quality scoring
│
├── tests/
│   └── test_pipeline.py    # Unit tests (pytest)
│
└── sample_data/
    └── sample_interview.txt
```

---

## Configuration

All settings are adjustable in the sidebar at runtime:

| Setting | Options | Notes |
|---|---|---|
| LLM Backend | `openai`, `gemini` | Requires API key in `.env` |
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

**Why ChromaDB?** Runs fully in-memory — no Docker, no server, no setup. Perfect for a demo that needs to just work.

**Why separate `prompts.py`?** Prompt iteration is the highest-leverage activity in LLM systems. Separating prompts from logic makes A/B testing one change at a time easy.

**Why both OpenAI and local embeddings?** Lets the user run the full semantic search pipeline for free (sentence-transformers), while still offering the higher quality OpenAI embeddings when available.

---

## Extending This

Some ideas if you want to go further:

- **Multi-transcript analysis** — upload 5–10 interviews and cluster themes across them
- **Persistent storage** — swap in-memory Chroma for a persisted collection
- **Speaker diarization** — auto-detect speaker labels using Whisper
- **Evaluation dashboard** — visualize insight quality scores over time
- **Slack / Notion export** — push insights directly to your team's tools

---

## License

MIT
