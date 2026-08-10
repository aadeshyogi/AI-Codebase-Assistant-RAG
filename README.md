# 🧠 RepoMind — AI Codebase Assistant (RAG)

Ask natural-language questions about any GitHub repository and get grounded
answers with exact file + line-number citations — powered by `gpt-4o-mini`.

> Think of it as a mini "explain this codebase" tool: paste a repo URL,
> ask "how is authentication handled?", and get a cited answer instead of
> reading 40 files yourself.

## Why this is different from a basic PDF-RAG chatbot

Most RAG tutorials split text every N characters. That's fine for prose, but
disastrous for code — it slices functions in half and destroys the very
structure that makes code understandable.

This project instead uses **`tree-sitter`** to parse the real Abstract Syntax
Tree of each file and extracts **whole functions and classes** as chunks,
each tagged with its exact file path and line range. Retrieval then combines
**semantic search** (embeddings) with **BM25 keyword search** (hybrid
retrieval) so that exact identifier matches (e.g. `handleLogin`) aren't
missed by pure vector similarity.

## Architecture

```
GitHub repo URL
      │
      ▼
AST-aware chunking (tree-sitter: function/class-level, not naive char-split)
      │
      ▼
Embeddings (text-embedding-3-small) → ChromaDB (persisted per-repo)
      │
      ▼
Hybrid retrieval (semantic + BM25 keyword rerank)
      │
      ▼
gpt-4o-mini (grounded, cited answer) → Streamlit UI
```

## Features

- **AST-aware chunking** — functions and classes stay intact, never split mid-body
- **Hybrid retrieval** — semantic + keyword search combined for more robust results
- **Grounded answers** — the LLM is instructed to cite file paths and never invent behavior
- **Source citations** — every answer shows exactly which `file.py:start-end` it came from
- **Per-repo caching** — re-opening a previously indexed repo loads instantly (no re-embedding cost)
- **Multi-language** — Python and JavaScript/TypeScript out of the box (easy to extend)
- **Chat with memory** — ask follow-ups like *"which file/code did you use for that?"* and the
  model answers from the conversation history, no need to re-ask the original question
- **Live token usage in the sidebar** — input tokens (your questions) and output tokens
  (AI answers) are tracked from the actual OpenAI response, running totals shown at all times
- **"Explain this code" button** — click it on any source chip to get a step-by-step
  explanation of that exact function/class, without writing a new question

## Setup

```bash
git clone <this-repo>
cd repomind
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
streamlit run app.py
```

## Usage

1. Paste any public GitHub repo URL (e.g. `https://github.com/psf/requests.git`) in the sidebar
2. Click **Index repo** — first-time indexing takes a bit (embeds every function/class); it's cached after that
3. Ask questions like:
   - "How does this repo handle retries on failed requests?"
   - "Where is the main entry point?"
   - "What does the `Session` class do?"
4. Read the grounded answer, then check the **Sources** chips to jump to the exact file/line

## Project structure

Kept intentionally small — just 2 Python files:

```
repomind/
├── app.py            
├── requirements.txt
└── .env.example
```

## Possible extensions (good "future work" talking points in an interview)

- Add an eval set (10-15 questions with expected file/function) and report retrieval@k accuracy
- Show a running **cost estimate** ($) alongside token counts, based on per-model pricing
- Support more languages (Go, Rust, Java tree-sitter grammars are drop-in)
- Swap ChromaDB for a hosted vector DB for multi-user deployment

## Tech stack

Python · Streamlit · LangChain · ChromaDB · tree-sitter · OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) · rank-bm25 · GitPython
