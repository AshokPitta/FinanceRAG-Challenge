# FinanceRAG — ACM-ICAIF '24 Challenge

A Retrieval-Augmented Generation (RAG) pipeline for financial question answering, built for the [ACM-ICAIF '24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge) on Kaggle.

The system retrieves and ranks the top 10 most relevant document chunks from financial corpora, then generates grounded answers using an instruction-tuned LLM. Evaluated on **nDCG@10** for retrieval and **RAGAS** for generation quality.

> MSc Computer Science dissertation project — University of Sheffield (2025)

---

## Results

### Task 1 — Retrieval (nDCG@10)

| Dataset | nDCG@10 |
|---|---|
| FinQABench | **0.876** |
| FinanceBench | **0.857** |
| FinQA | 0.568 |
| ConvFinQA | 0.562 |
| TATQA | 0.530 |
| FinDER | 0.524 |
| MultiHiertt | 0.169 |
| **Macro Average** | **0.584** |

### Task 2 — Generation (RAGAS)

| Dataset | Relevancy | Correctness | Faithfulness |
|---|---|---|---|
| FinQA | 0.193 | **0.680** | 0.336 |
| TATQA | 0.151 | 0.678 | 0.452 |
| ConvFinQA | **0.242** | 0.519 | 0.475 |
| FinDER | 0.219 | 0.508 | **0.531** |
| FinanceBench | 0.173 | 0.506 | 0.420 |
| FinQABench | 0.233 | 0.480 | 0.512 |
| MultiHiertt | 0.139 | 0.639 | 0.252 |
| **Macro Average** | **0.193** | **0.573** | **0.426** |

---

## Pipeline Overview

The system runs in two stages across all seven datasets.

### Stage 1 — Retrieval

**Passage datasets** (FinDER, FinQABench, FinanceBench):
- Dual-dense retrieval: `BGE-Large-v1.5` + `MXBAI-Embed-Large-v1` over FAISS (top 150 each)
- Sparse retrieval: BM25 over inverted index (top 150)
- Per-query min-max score normalisation
- Hybrid fusion: dense scores fused at α=0.6, then hybridised with BM25 at β=0.7
- Cross-encoder reranking: `BAAI/bge-reranker-v2-m3` over top 100 candidates

**Tabular & text datasets** (TATQA, FinQA, ConvFinQA, MultiHiertt):
- Same dual-dense + BM25 hybrid pipeline
- MultiHiertt uses triple-dense setup adding `E5-large-v2` and RRF fusion
- Multi-reranker ensemble: `BGE-Reranker-v2-m3` + `ms-marco-MiniLM-L-6-v2`
- Table-aware features (unit consistency, number/year intersection, header overlap) added to reranker scoring
- Final output: top 100 reranked contexts per query

### Stage 2 — Generation

- Model: `Meta-Llama-3-8B-Instruct` (FP16, optional 4-bit quantisation via BitsAndBytes)
- Top k=6 reranked contexts concatenated with query into a structured prompt
- Greedy decoding (temperature=0.0) for deterministic, reproducible outputs
- Model outputs `INSUFFICIENT` when evidence is absent — preventing hallucination
- Outputs saved as per-dataset JSONL files for RAGAS evaluation

---

## Project Structure

```
├── financerag/
│   ├── common/         # Data loading, schemas, utilities
│   │   ├── loader.py
│   │   ├── protocols.py
│   │   └── utils.py
│   ├── retrieval/      # Dense and BM25 retrieval backends
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   └── sent_encoder.py
│   ├── rerank/         # Cross-encoder reranking
│   │   └── cross_encoder.py
│   ├── generate/       # LLM answer generation
│   │   └── openai.py
│   └── tasks/          # Per-dataset task definitions
│       ├── BaseTask.py
│       ├── FinDERTask.py
│       ├── FinQATask.py
│       ├── FinQABenchTask.py
│       ├── FinanceBenchTask.py
│       ├── TATQATask.py
│       ├── ConvFinQATask.py
│       └── MultiHierttTask.py
├── FinDER.py           # Passage retrieval pipeline
├── FinQABench.py
├── FinanceBench.py
├── FinQA.py            # Tabular & text retrieval pipeline
├── TATQA.py
├── ConvFinQA.py
├── MultiHiertt.py
├── Generation.py       # LLM generation pipeline
├── dataset/            # Per-dataset corpus, queries, qrels
└── results/            # Per-dataset retrieval CSVs and summaries
```

---

## Datasets

Seven financial QA datasets from the FinanceRAG challenge:

| Dataset | Type | Corpus | Queries |
|---|---|---|---|
| FinDER | Passage (10-K) | 13,867 | 216 |
| FinQABench | Passage (10-K) | 92 | 100 |
| FinanceBench | Passage (public filings) | 180 | 150 |
| TATQA | Tabular + text | 2,756 | 1,663 |
| FinQA | Tabular + text | 2,789 | 1,147 |
| ConvFinQA | Conversational (earnings) | 2,066 | 421 |
| MultiHiertt | Multi-hop (annual reports) | 10,475 | 974 |

Each dataset follows a consistent schema: `corpus.jsonl`, `queries.jsonl`, and a `qrels.tsv` ground truth file.

---

## Setup

```bash
git clone https://github.com/AshokPitta/FinanceRAG-Challenge.git
cd FinanceRAG-Challenge

pip install -r requirements.txt
```

**Key dependencies:** `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `transformers`, `torch`, `ragas`, `pandas`

---

## Usage

Run retrieval for a specific dataset:

```bash
python FinDER.py         # Passage retrieval
python FinQA.py          # Tabular + text retrieval
python MultiHiertt.py    # Multi-hop retrieval
```

Run generation across all datasets:

```bash
python Generation.py
```

Results are saved to `results/<dataset>/results.csv` (retrieval) and `results/<dataset>/<dataset>_answers.jsonl` (generation).

---

## Tech Stack

`Python` `PyTorch` `Sentence-Transformers` `FAISS` `BM25` `LLaMA-3-8B-Instruct` `RAGAS` `HuggingFace Transformers` `BitsAndBytes`

**Models used:**
- `BAAI/bge-large-en-v1.5` — dense retriever
- `mixedbread-ai/mxbai-embed-large-v1` — dense retriever
- `intfloat/e5-large-v2` — dense retriever (MultiHiertt)
- `BAAI/bge-reranker-v2-m3` — cross-encoder reranker
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — cross-encoder reranker
- `Meta-Llama-3-8B-Instruct` — answer generation

---

## References

- [ACM-ICAIF '24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge)
- [FinanceRAG Dataset on HuggingFace](https://huggingface.co/datasets/Linq-AI-Research/FinanceRAG)
- [Official Baseline](https://github.com/Linq-AI-Research/FinanceRAG)
- [RAGAS Framework](https://docs.ragas.io)
- [Sentence-Transformers](https://sbert.net)

---

## Author

**Ashok Kumar Pitta** — MSc Computer Science, University of Sheffield (2025)

🌐 [Portfolio](https://portfolio-website-nine-omega-34.vercel.app) · 💼 [LinkedIn](https://linkedin.com/in/ashok-pitta-ai) · ✉️ ashokkumarpitta555@gmail.com
