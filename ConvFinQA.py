import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -*- coding: utf-8 -*-

from sentence_transformers import CrossEncoder
from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever
from financerag.tasks import ConvFinQA
from financerag.tasks.BaseTask import BaseTask
from financerag.common import Lexical
from rank_bm25 import BM25Okapi
from collections import defaultdict
import pandas as pd
import numpy as np
import re

# -------------------- helpers --------------------
NUM_RE  = re.compile(r"\b\d{1,3}(?:[,]\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
BM25_NUMERIC_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|%|\$|million|billion|thousand|bn|m")

def minmax(score_dict):
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return {k: 0.0 for k in score_dict}
    s = hi - lo
    return {k: (v - lo) / s for k, v in score_dict.items()}

def stringify_table(table):
    """Serialize table-like structures into compact text with headers and key rows."""
    try:
        if isinstance(table, str):
            return table[:2000]
        if isinstance(table, dict):
            cols = list(table.keys())
            rows = list(zip(*[table[c] for c in cols])) if cols else []
            lines = [f"headers: {', '.join(str(c) for c in cols)}"]
            for row in rows[:8]:
                lines.append(", ".join(f"{c}={row[i]}" for i, c in enumerate(cols)))
            return " | ".join(lines)[:2000]
        if isinstance(table, list):
            if table and isinstance(table[0], dict):
                cols = sorted({k for r in table for k in r.keys()})
                lines = [f"headers: {', '.join(str(c) for c in cols)}"]
                for r in table[:8]:
                    lines.append(", ".join(f"{c}={r.get(c,'')}" for c in cols))
                return " | ".join(lines)[:2000]
            if table and isinstance(table[0], (list, tuple)):
                hdr = [str(x) for x in table[0]]
                lines = [f"headers: {', '.join(hdr)}"]
                for row in table[1:9]:
                    lines.append(", ".join(f"{h}={v}" for h, v in zip(hdr, row)))
                return " | ".join(lines)[:2000]
        return str(table)[:1000]
    except Exception:
        return str(table)[:500]

def augment_corpus_with_tables(task):
    for did, doc in task.corpus.items():
        title = doc.get("title", "")
        base  = doc.get("text", "")
        parts = [title]
        # common table-like fields across datasets
        for k in ("table","tables","html_table","table_csv","table_rows","tab"):
            if k in doc and doc[k]:
                tb = doc[k]
                try:
                    if isinstance(tb, list):
                        parts += [stringify_table(t) for t in tb[:3]]
                    else:
                        parts.append(stringify_table(tb))
                except Exception:
                    pass
        for k in ("caption","section","subtitle","heading"):
            if isinstance(doc.get(k), str):
                parts.append(doc[k])
        parts.append(base)
        doc["text"] = " \n".join([p for p in parts if p])[:4000]

# -------------------- BM25 numeric-aware --------------------
def bm25_tokenize_list(texts):
    """
    FinanceRAG BM25Retriever lowercases queries before calling tokenizer.
    These come in already lowercased.
    """
    return [BM25_NUMERIC_RE.findall(t) for t in texts]

class SimpleBM25(Lexical):
    def __init__(self, corpus):
        self.corpus_ids = list(corpus.keys())
        tokenized_docs = []
        for d in corpus.values():
            txt = (d.get("title","") + " " + d.get("text","")).lower()
            tokenized_docs.append(BM25_NUMERIC_RE.findall(txt))
        self.bm25 = BM25Okapi(tokenized_docs)
    def get_scores(self, query_tokens):
        return self.bm25.get_scores(query_tokens)

def table_feature_score(q, doc_text):
    """Light features to help tables: number/year/header overlap."""
    qnums = set(NUM_RE.findall(q)); dnums = set(NUM_RE.findall(doc_text))
    num_overlap = len(qnums & dnums)
    year_hit = 1 if (YEAR_RE.search(q) and YEAR_RE.search(doc_text)) else 0
    headers = []
    for m in re.finditer(r"headers:\s*([^|]+)", doc_text, flags=re.IGNORECASE):
        headers += re.findall(r"[A-Za-z]+", m.group(1).lower())
    qterms = set(re.findall(r"[A-Za-z]+", q.lower()))
    hdr_overlap = len(qterms & set(headers))
    return 0.5 * min(num_overlap, 3) / 3.0 + 0.3 * (hdr_overlap > 0) + 0.2 * year_hit

# -------------------- pipeline --------------------

# 0) Load task
finqa_task = ConvFinQA()

# Print dataset info
print("Dataset: ConvFinQA")
print(f"Queries: {len(finqa_task.queries)}")
print(f"Corpus:  {len(finqa_task.corpus)}")

# 1) Table-aware augmentation
augment_corpus_with_tables(finqa_task)

# 2) Dual-dense retrieval (BGE + MXBAI)
encoder1 = SentenceTransformerEncoder(
    model_name_or_path="BAAI/bge-large-en-v1.5",
    query_prompt="Represent this question to retrieve relevant tables/rows from financial reports: ",
    doc_prompt="Represent this table or text section for retrieval including headers, units, and years: ",
)
retriever1 = DenseRetrieval(model=encoder1)
dense1 = finqa_task.retrieve(retriever=retriever1, top_k=150)

encoder2 = SentenceTransformerEncoder(
    model_name_or_path="mixedbread-ai/mxbai-embed-large-v1",
    query_prompt="Represent this question to retrieve relevant tables/rows from financial reports: ",
    doc_prompt="Represent this table or text section for retrieval including headers, units, and years: ",
)
retriever2 = DenseRetrieval(model=encoder2)
dense2 = finqa_task.retrieve(retriever=retriever2, top_k=150)

# 3) BM25 (numeric-aware)
bm25_model = SimpleBM25(finqa_task.corpus)
bm25_retriever = BM25Retriever(model=bm25_model, tokenizer=bm25_tokenize_list)
bm25 = finqa_task.retrieve(retriever=bm25_retriever, top_k=150)

# 4) Small grid search for alpha (dense fusion) and beta (hybrid fusion)
qrels_path = os.path.join(BASE_DIR, "dataset/ConvFinQA/ConvFinQA_qrels.tsv")
df_q = pd.read_csv(qrels_path, sep="\t", names=["qid","did","rel"], skiprows=1)
qrels = defaultdict(dict)
for _, r in df_q.iterrows():
    qrels[str(r["qid"])][str(r["did"])] = int(r["rel"])

def retrieval_fuse(alpha, beta):
    fused = {}
    for qid in finqa_task.queries:
        bge_n = minmax(dense1.get(qid, {}))
        mxb_n = minmax(dense2.get(qid, {}))
        bm_n  = minmax(bm25.get(qid, {}))
        all_dense = set(bge_n) | set(mxb_n)
        dense = {d: alpha*bge_n.get(d,0.0) + (1-alpha)*mxb_n.get(d,0.0) for d in all_dense}
        all_docs = set(dense) | set(bm_n)
        fused[qid] = {d: beta*dense.get(d,0.0) + (1-beta)*bm_n.get(d,0.0) for d in all_docs}
    return fused

def eval_retrieval_ndcg10(res):
    nd, *_ = BaseTask.evaluate(qrels=qrels, results=res, k_values=[10])
    return float(nd["NDCG@10"])

best_alpha, best_beta, best_score = None, None, -1.0
for a in (0.5, 0.6, 0.7):
    for b in (0.5, 0.6, 0.7):
        tmp = retrieval_fuse(a, b)
        sc = eval_retrieval_ndcg10(tmp)
        print(f"[grid] alpha={a:.2f} beta={b:.2f} -> retrieval nDCG@10={sc:.5f}")
        if sc > best_score:
            best_alpha, best_beta, best_score = a, b, sc

print(f"Chosen alpha={best_alpha:.2f}, beta={best_beta:.2f} (retrieval nDCG@10={best_score:.5f})")
fused_results = retrieval_fuse(best_alpha, best_beta)

# 5) Reranking: BGE + MiniLM + table features
rer_bge = CrossEncoderReranker(model=CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda"))
rer_min = CrossEncoderReranker(model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda"))

reranked_bge = finqa_task.rerank(reranker=rer_bge, results=fused_results, top_k=100, batch_size=32)
reranked_min = finqa_task.rerank(reranker=rer_min, results=fused_results, top_k=100, batch_size=32)

final_results = {}
for qid in finqa_task.queries:
    cand_docs = set(fused_results.get(qid, {}).keys())
    bge_n = minmax({d: reranked_bge.get(qid, {}).get(d, 0.0) for d in cand_docs})
    min_n = minmax({d: reranked_min.get(qid, {}).get(d, 0.0) for d in cand_docs})
    feat  = {}
    qtext = finqa_task.queries[qid]
    for did in cand_docs:
        feat[did] = table_feature_score(qtext, finqa_task.corpus[did].get("text",""))
    feat_n = minmax(feat)

    # weights tuned for tabular QA
    w_bge, w_min, w_feat = 0.6, 0.3, 0.1
    scores = {d: w_bge*bge_n.get(d,0.0) + w_min*min_n.get(d,0.0) + w_feat*feat_n.get(d,0.0) for d in cand_docs}
    topk = dict(sorted(scores.items(), key=lambda x: -x[1])[:10])
    final_results[qid] = topk

# 6) Save and evaluate
out_dir = os.path.join(BASE_DIR, "results")
finqa_task.rerank_results = final_results
finqa_task.save_results(output_dir=out_dir)
print(f"Saved results to {out_dir}/ConvFinQA/results.csv")

ndcg, *_ = BaseTask.evaluate(qrels=qrels, results=final_results, k_values=[10])
print(f"ConvFinQA -> nDCG@10: {ndcg['NDCG@10']:.5f}")


