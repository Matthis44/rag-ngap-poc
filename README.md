# RAG on French nursing billing regulations (NGAP)

[![tests](https://github.com/Matthis44/RAG_KEYNOVE_v1/actions/workflows/tests.yml/badge.svg)](https://github.com/Matthis44/RAG_KEYNOVE_v1/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **Retrieval-Augmented Generation** system that answers billing questions for French
home-care nurses from the official texts: the NGAP (*Nomenclature Générale des Actes
Professionnels*, the national fee schedule) and the circulars that amend it.

Every stage is written from scratch in plain Python (no LangChain, no vector database):
PDF extraction and cleaning, three chunking strategies, embeddings with an on-disk cache,
multi-query retrieval with re-ranking and a composite score, and generation with mandatory
source citations.

The project ends with an honest assessment of **where pure RAG falls short** on this
problem, which motivated a different architecture for the next version
([see below](#results-and-limitations)).

## Pipeline

```
data/RAW/*.pdf ─▶ ingestion.py ─▶ cleaning_text.py ─▶ chunking.py ─▶ embeddings.py ─▶ data/embedded_chunks.json
                  (pdfplumber,     (headers, page     (naive /        (Mistral or        (743 chunks,
                   NGAP tables,     numbers, hyphen-   overlap /       OpenAI, SHA-256    1024 dims)
                   registry)        ation, unicode)    semantic)       cache)
                                                                                              │
 question ─▶ retrieval.py: rewrite + decompose into sub-queries ─▶ cosine search ─────────────┘
                           ─▶ cross-encoder re-rank ─▶ score = cosine + legal priority + recency + rerank
                           ─▶ near-duplicate removal ─▶ rag.py: answer with cited sources
```

## Highlights

- **Domain-specific cleaning** of official PDFs: *Journal officiel* headers and page numbers,
  end-of-line hyphenation (keeping real compound words such as *anti-inflammatoire*),
  obsolete articles, unicode normalisation.
- **Table-aware ingestion**: NGAP fee tables (act, key letter, coefficient) are detected and
  turned into one self-contained text per act instead of broken table rows.
- **Three chunking strategies**: fixed size, sliding window, and semantic splitting on legal
  structure (titles, chapters, articles), with min / max size guards.
- **Embeddings** through the Mistral or OpenAI HTTP APIs, batched, rate-limited and cached by
  content hash so an index is only paid for once.
- **Retrieval**: query rewriting and multi-query decomposition with an LLM, cosine search,
  `mistral-rerank` cross-encoder, a composite score that favours higher-ranking legal texts
  and recent ones, and vector-based deduplication.
- **Document registry** (`data/documents_registry.xlsx`): title, type, year, legal source and
  priority of each text, attached to every chunk as metadata.
- **Tests** for cleaning, chunking, scoring and the consistency of the shipped index, run in CI
  without any API key.

## Quick start

```bash
git clone https://github.com/Matthis44/RAG_KEYNOVE_v1.git
cd RAG_KEYNOVE_v1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                # no API key needed
```

The pre-computed index `data/embedded_chunks.json` is included, so you can query right away
with a Mistral key in `.env` (see `.env.example`):

```python
from embeddings import load_embedded_chunks
from rag import RAGAgent

agent = RAGAgent(load_embedded_chunks("data/embedded_chunks.json"))
response = agent.ask(
    "Camille, 4, comes home after surgery for a forearm fracture with an external fixator. "
    "The prescription asks for 3 visits a week to change the dressing and monitor the pins. "
    "How should the visits be billed?",
    top_k=100,
)
print(response.answer)
response.print_sources()
```

To rebuild the index (calls the embedding API):

```bash
python pipeline.py                                    # Mistral, semantic chunking, size 800
python pipeline.py --provider openai --model text-embedding-3-large --strategy overlap
```

## Main parameters

| Stage | Parameter | Default |
|---|---|---|
| Chunking | strategy / size / overlap | semantic / 800 / 150 characters |
| Embedding | provider | Mistral `mistral-embed` (1024 dims) |
| Retrieval | candidates per sub-query | 150 in `RAGAgent.ask`, 30 in `AdvancedRetriever` |
| Retrieval | min cosine / dedup threshold | 0.68 / 0.93 |
| Generation | model / temperature | `mistral-large-latest` / 0.2 |

## Corpus

Four official documents (in `data/RAW/`): the NGAP reference text (key letters, coefficients,
nursing acts), circular CIR-34/2019 (amendment no. 6), circular CIR-30/2024 (medical
prescriptions) and a complementary regulatory text. New PDFs can be added to `data/RAW/` and
the registry, then indexed with `pipeline.py`.

## Results and limitations

Retrieval works well: the right passages of the NGAP are found and cited. But on realistic
billing scenarios the generated answers are **not reliable enough**. Correct billing depends on
combining many conditional rules (first-day surcharges, travel allowances, child patients, acts
performed during the same visit, non-cumulative acts) that are scattered across documents.
Given retrieved text chunks, the LLM does not apply these combinations consistently.

**Conclusion**: for rule-based domains, retrieval of unstructured text is not enough. The next
version therefore structures the rules explicitly (acts, conditions, surcharges, cumulation
rules) and applies them with a deterministic engine, using the LLM only to read the
prescription.

## Project layout

| File | Role |
|---|---|
| `ingestion.py` | PDF loading, NGAP table extraction, registry metadata |
| `cleaning_text.py` | text normalisation for official documents |
| `chunking.py` | naive, overlap and semantic strategies |
| `embeddings.py` | embedding engine, cache, cosine search, index I/O |
| `retrieval.py` | query rewriting, decomposition, re-ranking, scoring, dedup |
| `rag.py` | end-to-end agent and answer formatting |
| `pipeline.py` | CLI to build the index |
| `tests/` | unit tests and index consistency checks |

## License

MIT for the code. The documents in `data/RAW/` are official French public texts.
