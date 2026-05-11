# 🏦 FinanceRAG — ACM-ICAIF '24 Challenge

A Retrieval-Augmented Generation (RAG) system built for the
[ACM-ICAIF '24 FinanceRAG Challenge](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge)
on Kaggle. The goal is to accurately retrieve and reason over
textual and tabular data from real-world financial documents.

---

## 📌 Challenge Overview

The competition consists of two core tasks:

- **Task 1 — Retrieval:** Given a financial query, retrieve the top 10
  most relevant document chunks from a large corpus.
- **Task 2 — Generation:** Using the retrieved chunks, generate accurate
  and grounded answers — including reasoning over numerical/tabular data.

The dataset covers multiple financial document types including 10-K reports,
earnings calls, and more.

---

## 🚀 Approach

> _(Fill in what you actually did — here are some prompts to guide you)_

- **Embedding model used:** e.g. `sentence-transformers/...`
- **Retrieval strategy:** e.g. dense retrieval, BM25, hybrid
- **Reranking:** e.g. cross-encoder reranker
- **Generation:** e.g. GPT-4, open-source LLM
- **Any preprocessing:** query expansion, corpus filtering, chunking strategy

---

## 🗂️ Project Structure
├── retrieval/        # Document retrieval pipeline
├── rerank/           # Reranking logic
├── generate/         # Answer generation
├── data/             # Dataset scripts
├── notebooks/        # Exploration & experiments
├── results/          # Output CSV files
├── requirements.txt  # Dependencies
└── run.sh            # Full pipeline runner
---

## ⚙️ Setup & Usage

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# Run the full pipeline
bash run.sh
```

---

## 📊 Results

| Task | Score | Leaderboard Rank |
|------|-------|-----------------|
| Task 1 — Retrieval | _XX_ | _XX_ |
| Task 2 — Generation | _XX_ | _XX_ |

---

## 🛠️ Tech Stack

![Python](https://skillicons.dev/icons?i=python)

- `sentence-transformers`
- `langchain` / `llama-index` _(if used)_
- `openai` / `huggingface`
- `pandas`, `numpy`

---

## 📚 References

- [FinanceRAG Dataset on HuggingFace](https://huggingface.co/datasets/Linq-AI-Research/FinanceRAG)
- [Official Baseline on GitHub](https://github.com/Linq-AI-Research/FinanceRAG)
- [ACM-ICAIF '24 Competition Page](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge)

---

## 🙋 Author

**[Your Name]** — [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
