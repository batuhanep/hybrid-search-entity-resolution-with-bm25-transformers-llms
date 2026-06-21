# Modular Entity Resolution Pipeline with Hybrid Search and LLM Judge

This repository contains a production-grade, two-phase software pipeline engineered for high-precision Entity Resolution (Record Linkage) processes. The architecture is designed to eliminate false positives in noisy datasets by coupling traditional lexical techniques with dense vector space modeling and deterministic LLM validation.

## Architecture & Logic Flow

The pipeline operates across four structured layers:

1. **Preprocessing & Normalization:** Implements deterministic tokenization and lower-level text cleaning. Numeric values, capacity indicators, and decimals are rigorously isolated to protect structural meaning.
2. **Dual-Indexing Retrieval:** Creates a sparse representation using the `BM25Okapi` algorithm alongside a dense vector representation using `BAAI/bge-m3` structural embeddings. Dense representations are indexed via a GPU-accelerated `FAISS` (Inner Product) index for sub-millisecond similarity queries.
3. **Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking:** Merges lexical and semantic candidate ranks mathematically using RRF. The fused top candidates are fed into a sequential Cross-Encoder model (`BAAI/bge-reranker-v2-m3`) to process true sequence-to-sequence logit evaluations.
4. **LLM-as-a-Judge Validation:** The Top 3 candidates are evaluated by a Large Language Model (`Qwen-2.5-72B-Instruct` via cloud API or `Qwen2.5-3B` locally via Ollama). The model enforces zero-false-positive limits and outputs a deterministic JSON payload.

---

## Repository Structure

```text
hybrid-search-entity-resolution-with-bm25-transformers-llms/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── phase1-hybrid-matcher-bm25-transformers-rrf.py
    ├── data/
    │   └── source_data.xlsx
    └── phase2_llm_judge/
        ├── phase2-llm-as-a-judge-qwen-api.py
        └── phase2-llm-as-a-judge-qwen-local.py
