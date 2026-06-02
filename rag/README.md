# Legal RAG Skeleton

This directory contains a minimal project skeleton for building a legal
article-level RAG system on top of the existing corpus in this repository.

Current design choices:

- Embedding provider: SiliconFlow
- Embedding model: `Qwen/Qwen3-Embedding-4B`
- RAG framework: LlamaIndex
- Vector store: Qdrant
- Retrieval unit: one legal article per node
- Enrichment fields: article annotation, outgoing citations, incoming citations

## Directory layout

- `config.py`: environment-backed runtime settings
- `schema.py`: shared dataclasses for article documents
- `loader.py`: parses this repository's legal corpus into article documents
- `siliconflow.py`: SiliconFlow embedding wrapper for LlamaIndex
- `index.py`: builds and persists the vector index
- `retrieval.py`: graph-aware retrieval helpers
- `query.py`: sample retrieval entrypoint
- `requirements.txt`: Python dependencies for the RAG project
- `.env.example`: required environment variables

## Quick start

1. Create a virtual environment.
2. Install dependencies from `rag/requirements.txt`.
3. Copy `rag/.env.example` to `.env` and fill in your keys.
4. Build the index:

```bash
python -m rag.index --limit 100 --batch-size 64
```

5. Run a test query:

```bash
python -m rag.query "民法典中自然人民事权利能力从什么时候开始？"
```

6. Build a subset index for core laws:

```bash
python -m rag.index --law-ids 3346,4833,31,4910,13 --batch-size 128
```

7. If full-corpus indexing is unstable on your network, reduce embedding sub-batch size:

```bash
export SILICONFLOW_EMBED_BATCH_SIZE=64
python -m rag.index --batch-size 1024
```

8. If you want better throughput, enable bounded request parallelism:

```bash
export SILICONFLOW_EMBED_BATCH_SIZE=64
export SILICONFLOW_REQUEST_BATCH_SIZE=16
export SILICONFLOW_MAX_PARALLEL_REQUESTS=4
python -m rag.index --batch-size 512
```

## Data sources used

- `data/laws_annotation/*.json`: legal full text
- `data/annotations/*.json`: article annotations
- `all_laws_map.json`: law id to law name mapping
- `js/backlinks.js`: reverse citation graph

## Notes

- This skeleton is intentionally conservative. It focuses on article-level
  retrieval first, then adds one-hop graph expansion from explicit citation
  metadata.
- `rag.query` now returns results from both vector retrieval and graph
  expansion, and prints the reason labels for each hit.
- `rag.index` inserts nodes in batches and persists after each batch, which is
  safer for long-running full-corpus builds.
- Local Qdrant mode is convenient for development, but payload indexes do not
  take effect there. Use a Qdrant server if you need payload indexing/filter
  performance.
