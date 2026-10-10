# SCSE '26 GEX3 — RAG part

A retrieval-augmented University IT support agent built with LangChain,
Chroma and a local Ollama-served Qwen model. The agent answers **only** from
the supplied knowledge base (`university_it_policies.txt`, 1196 policy
blocks loaded through the unmodified `policy_loader.py`).

## Goal

For the question

> "What exact phone number should I call for the University IT Service Desk?"

the knowledge base only says that users may *contact* the IT Service Desk —
it contains **no phone number**. The agent must therefore answer that it does
not know the number. It must not invent one (hallucination) and must not give
an irrelevant answer.

## How it works

- `rag_tool.py` — loads the policy documents **only** via
  `load_policy_documents()`, embeds them with the Ollama model
  `nomic-embed-text`, stores them in a persistent local Chroma vector store
  (`.chroma/`, created on first run) and exposes the
  `search_university_knowledge` LangChain tool (top-4 similarity matches).
- `agent.py` — creates a LangChain agent (Qwen served by Ollama) whose only
  tool is the RAG search tool, then runs a **reflection loop**
  draft → strict fact-checking review → revision. The reviewer forces a
  revision whenever the answer contains a phone number / contact detail that
  does not appear verbatim in the retrieved text, or whenever it fails to
  admit that a missing detail is not known.
- `rag.py` — entry point. Shows the retrieved evidence, the draft, the
  reflection verdicts and the final answer.
- `policy_loader.py` — provided loader, **not modified**.

## Model

The graded exercise names `qwen3-0.6b`. This implementation runs against the
locally installed Ollama model `qwen3:8b`; the model is a single constant,
`MODEL_NAME`, at the top of `agent.py` — change it to `qwen3:0.6b` (after
`ollama pull qwen3:0.6b`) to use the exercise's smaller model.

## Setup

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
ollama pull nomic-embed-text   # embedding model
ollama pull qwen3:8b           # or qwen3:0.6b
```

## Run

```powershell
python rag.py
python rag.py "How long are unreleased print jobs kept?"
```
