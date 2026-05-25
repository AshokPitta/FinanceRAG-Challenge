import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer, CrossEncoder
from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever
from financerag.tasks import FinanceBench
from financerag.common import Lexical
from rank_bm25 import BM25Okapi
from collections import defaultdict
import pandas as pd

# === Load Task ===
finqa_task = FinanceBench()

# === Print dataset info ===
dataset_name = "FinanceBench"
num_queries = len(finqa_task.queries)
num_corpus = len(finqa_task.corpus)
print(f"?? Dataset: {dataset_name}")
print(f"   Queries: {num_queries}")
print(f"   Corpus documents: {num_corpus}")

# === Step 1: Dense Retriever 1 (BGE) ===
encoder1 = SentenceTransformerEncoder(
    model_name_or_path="BAAI/bge-large-en-v1.5",
    query_prompt="Represent this sentence for searching relevant passages: ",
    doc_prompt="Represent this passage for retrieval: ",
)
retriever1 = DenseRetrieval(model=encoder1)
dense1_results = finqa_task.retrieve(retriever=retriever1, top_k=150)

# === Step 2: Dense Retriever 2 (MXBAI) ===
encoder2 = SentenceTransformerEncoder(
    model_name_or_path="mixedbread-ai/mxbai-embed-large-v1",
    query_prompt="Represent this sentence for searching relevant passages: ",
    doc_prompt="Represent this passage for retrieval: ",
)
retriever2 = DenseRetrieval(model=encoder2)
dense2_results = finqa_task.retrieve(retriever=retriever2, top_k=150)

# === Step 3: BM25 Retrieval ===
class SimpleBM25(Lexical):
    def __init__(self, corpus):
        self.corpus_ids = list(corpus.keys())
        tokenized = [
            (doc.get("title", "") + " " + doc.get("text", "")).lower().split()
            for doc in corpus.values()
        ]
        self.bm25 = BM25Okapi(tokenized)
    def get_scores(self, query_tokens):
        return self.bm25.get_scores(query_tokens)

bm25_model = SimpleBM25(finqa_task.corpus)
bm25_retriever = BM25Retriever(model=bm25_model)
bm25_results = finqa_task.retrieve(retriever=bm25_retriever, top_k=150)

# === Step 4: Score Fusion (Dual Dense + BM25) ===
def minmax(score_dict):
    if not score_dict:
        return {}
    values = list(score_dict.values())
    mn, mx = min(values), max(values)
    if mn == mx:
        return {k: 0.0 for k in score_dict}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

alpha = 0.6  
beta = 0.7   
fused_results = {}
for qid in finqa_task.queries:
    bge_norm = minmax(dense1_results.get(qid, {}))
    mxbai_norm = minmax(dense2_results.get(qid, {}))
    bm25_norm = minmax(bm25_results.get(qid, {}))

    # First fuse dense retrievers
    all_dense_docs = set(bge_norm) | set(mxbai_norm)
    dense_fusion = {
        doc_id: alpha * bge_norm.get(doc_id, 0.0) + (1 - alpha) * mxbai_norm.get(doc_id, 0.0)
        for doc_id in all_dense_docs
    }

    # Then fuse with BM25
    all_docs = set(dense_fusion) | set(bm25_norm)
    fused_results[qid] = {
        doc_id: beta * dense_fusion.get(doc_id, 0.0) + (1 - beta) * bm25_norm.get(doc_id, 0.0)
        for doc_id in all_docs
    }

# === Step 5: Reranking (BGE CrossEncoder) ===
reranker = CrossEncoderReranker(
    model=CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
)
reranked = finqa_task.rerank(
    reranker=reranker,
    results=fused_results,
    top_k=100,
    batch_size=32
)

# === Step 6: Save Results ===
finqa_task.rerank_results = reranked
out_dir = os.path.join(BASE_DIR, "results/FinanceBench/")
finqa_task.save_results(output_dir=out_dir)
print(f"Saved results to {out_dir}/FinanceBench/results.csv")

# === Step 7: Evaluate (nDCG@10) ===
qrels_path = os.path.join(BASE_DIR, "dataset/FinanceBench/FinanceBench_qrels.tsv")
df = pd.read_csv(qrels_path, sep="\t", names=["query_id", "doc_id", "relevance"], skiprows=1)
qrels = defaultdict(dict)
for _, row in df.iterrows():
    qrels[row["query_id"]][row["doc_id"]] = int(row["relevance"])

from financerag.tasks.BaseTask import BaseTask
ndcg, *_ = BaseTask.evaluate(qrels=qrels, results=reranked, k_values=[10])
print(f"FinanceBench Dual-Dense + BM25 + BGE rerank nDCG@10: {ndcg['NDCG@10']:.4f}")
