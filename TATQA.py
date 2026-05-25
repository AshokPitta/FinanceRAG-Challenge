# -*- coding: utf-8 -*-

import os, re, shutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from financerag.tasks import TATQA
from financerag.tasks.BaseTask import BaseTask
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever
from financerag.rerank import CrossEncoderReranker
from financerag.common import Lexical

# ---------- helpers ----------
NUM_RE  = re.compile(r"\b\d{1,3}(?:[,]\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
BM25_NUMERIC_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|%|\$|million|billion|thousand|bn|m")

INTENT_WORDS = {
    "diff":  ("difference","change","delta"),
    "ratio": ("ratio","proportion","per","/"),
    "pct":   ("percent","%"),
    "sum":   ("total","sum","aggregate"),
    "cmp":   ("greater","less","higher","lower","vs","compare"),
}

def ensure_inputs():
    root = "./dataset/TATQA"
    os.makedirs(root, exist_ok=True)
    q_src = "/mnt/data/queries.jsonl"
    c_src = "/mnt/data/corpus.jsonl"
    if os.path.exists(q_src):
        shutil.copy(q_src, os.path.join(root, "queries.jsonl"))
        shutil.copy(q_src, os.path.join(root, "queries_prep.jsonl"))
        print("[prep] copied /mnt/data/queries.jsonl ? ./dataset/TATQA/")
    if os.path.exists(c_src):
        shutil.copy(c_src, os.path.join(root, "corpus.jsonl"))
        shutil.copy(c_src, os.path.join(root, "corpus_prep.jsonl"))
        print("[prep] copied /mnt/data/corpus.jsonl ? ./dataset/TATQA/")

    for a,b in (("queries.jsonl","queries_prep.jsonl"),("corpus.jsonl","corpus_prep.jsonl")):
        src, dst = os.path.join(root,a), os.path.join(root,b)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"[prep] created {b} from {a}")

def minmax(d):
    if not d: return {}
    vals = list(d.values()); lo, hi = min(vals), max(vals)
    if lo == hi: return {k:0.0 for k in d}
    s = hi-lo;  return {k:(v-lo)/s for k,v in d.items()}

def extract_markdown_tables(text, max_tables=2):
    lines = text.splitlines()
    tables, start = [], None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|"):
            start = i if start is None else start
        elif start is not None:
            block = "\n".join(lines[start:i])
            if block.count("|") >= 4: tables.append(block)
            start = None
        if len(tables) >= max_tables: break
    if start is not None and len(tables) < max_tables:
        tables.append("\n".join(lines[start:]))
    return tables

def augment_corpus_with_tables(task):
    for _, doc in task.corpus.items():
        title = doc.get("title","")
        base  = doc.get("text","")
        tbls  = extract_markdown_tables(base, max_tables=2)
        parts = [title]
        for tb in tbls:
            rows = [r.strip() for r in tb.splitlines() if r.strip().startswith("|")]
            if rows:
                head = rows[0]
                body = rows[1:5]
                parts.append(f"headers: {head.replace('|',' ').strip()}")
                for r in body:
                    parts.append(r.replace("|"," ").strip())
        parts.append(base)
        doc["text"] = " \n".join([p for p in parts if p])[:4000]

def bm25_tokenize_list(texts):
    return [BM25_NUMERIC_RE.findall(t) for t in texts]

class SimpleBM25(Lexical):
    def __init__(self, corpus):
        toks = []
        for d in corpus.values():
            txt = (d.get("title","")+" "+d.get("text","")).lower()
            toks.append(BM25_NUMERIC_RE.findall(txt))
        self.bm25 = BM25Okapi(toks)
    def get_scores(self, query_tokens):
        return self.bm25.get_scores(query_tokens)

def parse_intents(q):
    ql = q.lower()
    return {
        "diff": any(w in ql for w in INTENT_WORDS["diff"]),
        "ratio": any(w in ql for w in INTENT_WORDS["ratio"]),
        "pct": any(w in ql for w in INTENT_WORDS["pct"]),
        "sum": any(w in ql for w in INTENT_WORDS["sum"]),
        "cmp": any(w in ql for w in INTENT_WORDS["cmp"]),
    }

def feature_score(q, doc_text):
    intents = parse_intents(q)
    qnums = set(NUM_RE.findall(q)); dnums = set(NUM_RE.findall(doc_text))
    num_overlap = len(qnums & dnums)
    year_hit = 1 if (YEAR_RE.search(q) and YEAR_RE.search(doc_text)) else 0
    dl = doc_text.lower()
    unit_pct = ("%" in q.lower() and "%" in dl) or ("percent" in q.lower() and "percent" in dl)
    unit_cur = (("$" in q) and ("$" in dl)) or ("usd" in q.lower() and "usd" in dl)
    ratio_cues = any(k in dl for k in (" ratio"," per ","/",":"))
    diff_cues  = any(k in dl for k in ("increase","decrease","change","delta"))
    sum_cues   = any(k in dl for k in ("total","sum","aggregate"))
    cmp_cues   = any(k in dl for k in ("greater","less","higher","lower","vs","compare"))
    score = 0.0
    score += 0.5 * min(num_overlap,3) / 3.0
    score += 0.15 * year_hit
    score += 0.10 * unit_pct
    score += 0.10 * unit_cur
    score += 0.05 * (intents["ratio"] and ratio_cues)
    score += 0.05 * (intents["diff"] and diff_cues)
    score += 0.03 * (intents["sum"] and sum_cues)
    score += 0.02 * (intents["cmp"] and cmp_cues)
    return max(0.0, min(1.0, score))

# ---------- main ----------
if __name__ == "__main__":
    ensure_inputs()

    # load
    task = TATQA()
    print("Dataset:", task.metadata.name)
    print("Queries:", len(task.queries))
    print("Corpus: ", len(task.corpus))

    # quick dataset-aware tweaks (no heavy EDA)
    tbl_ratio = sum(1 for d in task.corpus.values() if "|" in d.get("text","")) / max(1,len(task.corpus))
    q_len_avg = sum(len(task.queries[qid].split()) for qid in task.queries) / max(1,len(task.queries))
    q_numeric = sum(1 for q in task.queries.values() if any(ch.isdigit() for ch in q)) / max(1,len(task.queries))
    print(f"[stats] table_ratio={tbl_ratio:.2f} avg_q_len={q_len_avg:.1f} numeric_q_frac={q_numeric:.2f}")

    augment_corpus_with_tables(task)

    # prompts adapted slightly to dataset stats
    q_prompt = "Represent this financial question to retrieve relevant tables/passages with numeric facts: "
    d_prompt = "Represent this table/paragraph for retrieval; emphasize headers, units (%, $, bn, m) and fiscal periods: "
    if q_len_avg < 8:
        q_prompt = "Represent this short financial query for retrieving exact rows/figures from reports: "
    if tbl_ratio > 0.3:
        d_prompt = "Represent this TABLE-HEAVY financial content; include headers, units, periods and key row names: "

    # dual-dense
    enc1 = SentenceTransformerEncoder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        query_prompt=q_prompt, doc_prompt=d_prompt,
    )
    ret1 = DenseRetrieval(model=enc1)
    dense1 = task.retrieve(retriever=ret1, top_k=180)

    enc2 = SentenceTransformerEncoder(
        model_name_or_path="mixedbread-ai/mxbai-embed-large-v1",
        query_prompt=q_prompt, doc_prompt=d_prompt,
    )
    ret2 = DenseRetrieval(model=enc2)
    dense2 = task.retrieve(retriever=ret2, top_k=180)

    # BM25 numeric-aware
    bm25_model = SimpleBM25(task.corpus)
    bm25_ret = BM25Retriever(model=bm25_model, tokenizer=bm25_tokenize_list)
    bm25 = task.retrieve(retriever=bm25_ret, top_k=180)

    # fusion grid using qrels
    qrels_path = "./dataset/TATQA/TATQA_qrels.tsv"
    if not os.path.exists(qrels_path):
        raise FileNotFoundError("Missing ./dataset/TATQA/TATQA_qrels.tsv")
    df = pd.read_csv(qrels_path, sep="\t", names=["qid","did","rel"], skiprows=1)
    qrels = defaultdict(dict)
    for _, r in df.iterrows():
        qrels[str(r["qid"])][str(r["did"])] = int(r["rel"])

    def fuse(alpha, beta):
        fused = {}
        for qid in task.queries:
            bge_n = minmax(dense1.get(qid, {}))
            mxb_n = minmax(dense2.get(qid, {}))
            bm_n  = minmax(bm25.get(qid, {}))
            all_dense = set(bge_n) | set(mxb_n)
            dense = {d: alpha*bge_n.get(d,0.0) + (1-alpha)*mxb_n.get(d,0.0) for d in all_dense}
            all_docs = set(dense) | set(bm_n)
            fused[qid] = {d: beta*dense.get(d,0.0) + (1-beta)*bm_n.get(d,0.0) for d in all_docs}
        return fused

    def ndcg10(res):
        nd, *_ = BaseTask.evaluate(qrels=qrels, results=res, k_values=[10])
        return float(nd["NDCG@10"])


    beta_grid = (0.5, 0.6, 0.7) if q_numeric < 0.6 else (0.45, 0.55, 0.65)
    best_a, best_b, best_sc = None, None, -1.0
    for a in (0.5, 0.6, 0.7):
        for b in beta_grid:
            sc = ndcg10(fuse(a,b))
            print(f"[fuse] alpha={a:.2f} beta={b:.2f} -> nDCG@10={sc:.5f}")
            if sc > best_sc:
                best_a, best_b, best_sc = a, b, sc
    print(f"Chosen alpha={best_a:.2f}, beta={best_b:.2f} (retrieval nDCG@10={best_sc:.5f})")
    fused = fuse(best_a, best_b)

    # rerankers
    rr_bge = CrossEncoderReranker(CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda"))
    rr_min = CrossEncoderReranker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda"))
    rer_bge = task.rerank(reranker=rr_bge, results=fused, top_k=120, batch_size=32)
    rer_min = task.rerank(reranker=rr_min, results=fused, top_k=120, batch_size=32)

    # optional third CE (safe; skip if OOM or not available)
    have_gte, rer_gte = False, {}
    try:
        rr_gte = CrossEncoderReranker(CrossEncoder("Alibaba-NLP/gte-multilingual-reranker-base", device="cuda"))
        rer_gte = task.rerank(reranker=rr_gte, results=fused, top_k=120, batch_size=32)
        have_gte = True
    except Exception as e:
        print(f"[warn] GTE reranker skipped: {e}")

    def eval_mix(wb, wm, wf, wg=0.0):
        agg = {}
        for qid in task.queries:
            cand = set(fused.get(qid, {}).keys())
            bge_n = minmax({d: rer_bge.get(qid, {}).get(d, 0.0) for d in cand})
            min_n = minmax({d: rer_min.get(qid, {}).get(d, 0.0) for d in cand})
            feat  = {d: feature_score(task.queries[qid], task.corpus[d].get("text","")) for d in cand}
            feat_n = minmax(feat)
            if have_gte:
                gte_n = minmax({d: rer_gte.get(qid, {}).get(d, 0.0) for d in cand})
            scores = {}
            for d in cand:
                s = wb*bge_n.get(d,0.0) + wm*min_n.get(d,0.0) + wf*feat_n.get(d,0.0)
                if have_gte: s += wg*gte_n.get(d,0.0)
                scores[d] = s
            agg[qid] = dict(sorted(scores.items(), key=lambda x: -x[1])[:10])
        nd, *_ = BaseTask.evaluate(qrels=qrels, results=agg, k_values=[10])
        return float(nd["NDCG@10"]), agg

    best_r, best_nd = None, -1.0
    if have_gte:
        grids = [(0.55,0.25,0.10,0.10),(0.60,0.20,0.10,0.10)]
    else:
        grids = [(0.60,0.25,0.15,0.0),(0.55,0.30,0.15,0.0)]
    for wb, wm, wf, wg in grids:
        sc, agg = eval_mix(wb, wm, wf, wg)
        print(f"[mix] bge={wb:.2f} min={wm:.2f} feat={wf:.2f} gte={wg:.2f} -> nDCG@10={sc:.5f}")
        if sc > best_nd:
            best_nd, best_r = sc, agg
    print(f"Chosen weights -> nDCG@10={best_nd:.5f}")

    # save & report
    out_dir = os.path.join(BASE_DIR, "results/")
    task.rerank_results = best_r
    task.save_results(output_dir=out_dir)
    print(f"Saved results to {out_dir}/TATQA/results.csv")

    ndcg, *_ = BaseTask.evaluate(qrels=qrels, results=best_r, k_values=[10])
    print(f"TATQA -> nDCG@10: {ndcg['NDCG@10']:.5f}")
