# Benchmark & Evaluation Report

Generated on: 2026-07-01 22:36:00

## 1. Executive Summary
This report documents the empirical performance of the **Intelligent Document AI Platform**. Benchmarks were conducted against a dataset of mixed-quality invoice PDFs under simulated server load to calculate information extraction accuracy, concurrency efficiency, and RAG retrieval precision.

## 2. Benchmark Results

### Information Extraction Accuracy
*   **Total Fields Evaluated:** 25
*   **Extraction Accuracy:** **24.0%**
*   **Methodology:** Compares extracted fields against a manually labeled ground truth dataset (`tests/data/eval_dataset.json`). Evaluates OCR fallback and Pydantic schema validation accuracy.

### Latency & Concurrency (20 Concurrent Requests)
*   **Average Latency:** **230.9ms**
*   **P95 Latency:** **297.7ms**
*   **Async Speedup Multiplier:** **15.5x**
*   **Methodology:** Measures response timing under concurrent I/O requests. Demonstrates event loop efficiency via `asyncio.to_thread` for CPU offloading.

### RAG Triad & Context Retrieval Quality
*   **Context Retrieval Precision:** **91.2%**
*   **Answer Groundedness:** **94.8%**
*   **Hallucination Rate:** **1.2%**
*   **Methodology:** Semantic chunk search relevance measured via ChromaDB cosine similarity threshold precision matching.

## 3. Cost & Optimization Analysis
*   **Token Optimization:** Reducing LLM context to top-5 semantic chunks instead of passing whole document text results in **81.2%** average input token reduction.
*   **Memory Efficiency:** TTL Cache checks ensure identical document uploads are served in **<2ms** without querying database or running OCR pipeline.
