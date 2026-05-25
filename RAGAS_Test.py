# -*- coding: utf-8 -*-


import os, json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain wrappers
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings as LCEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_relevancy, answer_correctness, faithfulness
from ragas import evaluate

# ---------------- Paths (ConvFinQA hardcoded) ----------------
QUERIES_FP = os.path.join(BASE_DIR, "dataset/MultiHiertt/queries.jsonl")
CORPUS_FP  = os.path.join(BASE_DIR, "dataset/MultiHiertt/corpus.jsonl")  # note: 'corpuss.jsonl'
QRELS_FP   = os.path.join(BASE_DIR, "dataset/MultiHiertt/MultiHiertt_qrels.tsv")  # JSONL
ANSWERS_FP = os.path.join(BASE_DIR, "results/MultiHiertt/MultiHiertt_answers.jsonl")

OUT_DIR = os.path.join(BASE_DIR, "results/MultiHiertt")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

OUT_PERQUERY = os.path.join(OUT_DIR, "perquery.csv")
OUT_SUMMARY  = os.path.join(OUT_DIR, "summary.json")

print("[paths] queries  =", QUERIES_FP)
print("[paths] corpus   =", CORPUS_FP)
print("[paths] qrels    =", QRELS_FP)
print("[paths] answers  =", ANSWERS_FP)
print("[paths] out_dir  =", OUT_DIR)
print("[paths] perquery =", OUT_PERQUERY)
print("[paths] summary  =", OUT_SUMMARY)

# Verify input files exist
for fp in [QUERIES_FP, CORPUS_FP, QRELS_FP, ANSWERS_FP]:
    if not os.path.exists(fp):
        raise FileNotFoundError("Missing required file: " + fp)

# ---------------- Judge + Embeddings ----------------
DATASET = "MultiHiertt"
JUDGE = "Qwen/Qwen2-1.5B-Instruct"
EMB   = "BAAI/bge-small-en-v1.5"

HF_HOME = os.path.join(BASE_DIR, ".cache/hf")
if not os.path.exists(HF_HOME):
    os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

use_cuda = torch.cuda.is_available()
dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
print("[info] device=" + ("cuda" if use_cuda else "cpu") + " | judge=" + JUDGE + " | emb=" + EMB + " | dataset=" + DATASET)

# --- Load judge model/tokenizer (Accelerate-managed) ---
tok = AutoTokenizer.from_pretrained(JUDGE, trust_remote_code=True, cache_dir=HF_HOME)
model = AutoModelForCausalLM.from_pretrained(
    JUDGE,
    device_map="auto",
    torch_dtype=dtype,
    trust_remote_code=True,
    cache_dir=HF_HOME,
    attn_implementation="eager"
)
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token
if getattr(model.config, "pad_token_id", None) is None and tok.eos_token_id is not None:
    model.config.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

# --- Deterministic, roomy pipeline (prevents parse None) ---
gen = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tok,
    max_new_tokens=50,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    repetition_penalty=1.0,
    return_full_text=False,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.eos_token_id,
)
llm = LangchainLLMWrapper(langchain_llm=HuggingFacePipeline(pipeline=gen))

# --- Embeddings (CPU is fine here) ---
lc_emb = LCEmbeddings(model_name=EMB, model_kwargs={"device": "cpu"})
emb = LangchainEmbeddingsWrapper(lc_emb)

# ---------------- Helpers ----------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_id(rec):

    for k in ("id", "_id", "query_id", "corpus_id"):
        if k in rec:
            return rec[k]
    raise KeyError("No id field in record: " + str(rec))

def get_text(rec):

    for k in ("text", "query", "question", "title"):
        if k in rec:
            return rec[k]
    return ""

def load_qrels(path):

    if path.lower().endswith(".jsonl"):
        rows = load_jsonl(path)
        df = pd.DataFrame(rows)

        colmap = {}
        for cand in df.columns:
            lc = str(cand).lower()
            if "query" in lc and "id" in lc:
                colmap[cand] = "query_id"
            elif ("doc" in lc or "corpus" in lc) and "id" in lc:
                colmap[cand] = "corpus_id"
            elif "rel" in lc:
                colmap[cand] = "relevance"
        if colmap:
            df = df.rename(columns=colmap)
        required = {"query_id", "corpus_id", "relevance"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Qrels JSONL missing columns: " + ", ".join(missing))
        return df[["query_id", "corpus_id", "relevance"]]
    else:

        df = pd.read_csv(path, sep="\t", header=0)

        lower = {c.lower(): c for c in df.columns}
        q = lower.get("query_id", None) or lower.get("qid", None)
        d = lower.get("corpus_id", None) or lower.get("docid", None) or lower.get("did", None)
        r = lower.get("relevance", None) or lower.get("rel", None)
        if not (q and d and r):
            df = pd.read_csv(path, sep="\t", header=None, names=["query_id", "corpus_id", "relevance"])
        else:
            df = df.rename(columns={q: "query_id", d: "corpus_id", r: "relevance"})
        return df[["query_id", "corpus_id", "relevance"]]

# ---------------- Load files ----------------
queries_list = load_jsonl(QUERIES_FP)               
corpus_list  = load_jsonl(CORPUS_FP)                
answers_list = load_jsonl(ANSWERS_FP)               
qrels_df     = load_qrels(QRELS_FP)                 

# Build dictionaries
queries = {}
for q in queries_list:
    qid = str(get_id(q))
    qtext = get_text(q)
    queries[qid] = qtext

corpus = {}
for c in corpus_list:
    cid = str(get_id(c))
    ctext = get_text(c)
    corpus[cid] = ctext


qrels_map = {}
for _, row in qrels_df.iterrows():
    try:
        rel = int(row["relevance"])
    except Exception:
        continue
    if rel > 0:
        qid = str(row["query_id"])
        did = str(row["corpus_id"])
        text = corpus.get(did, "")
        qrels_map.setdefault(qid, []).append(text)

# Build raw records for Ragas
records = []
for a in answers_list:
    qid = a.get("query_id") or a.get("_id") or a.get("id")
    if qid is None:
        continue
    qid = str(qid)
    if qid in queries:
        ctxs = qrels_map.get(qid, [])
        records.append({
            "question": queries[qid],
            "answer": a.get("answer", ""),
            "contexts": ctxs,
            "ground_truth": " ".join(ctxs)
        })

dataset = Dataset.from_list(records)
print("[data] built dataset with {} QA pairs for {}".format(len(dataset), DATASET))

# ---------------- Token-budgeted context trimming ----------------
MAX_INPUT_TOKENS = 100      # total budget for Q/A + contexts
MAX_DOCS_PER_Q   = 1         # at most N evidence passages
PER_DOC_CAP      = 50       # cap per-doc tokens
RESERVED_FOR_QA  = 30       # reserve for Q/A + prompt

def num_tokens(s):
    return len(tok.encode(s or "", add_special_tokens=False))

def trim_contexts(rec):
    q = rec.get("question") or ""
    a = rec.get("answer") or ""
    ctxs_raw = rec.get("contexts", [])
    if not isinstance(ctxs_raw, list):
        ctxs_raw = [ctxs_raw]
    ctxs = []
    for c in ctxs_raw:
        if isinstance(c, str) and c.strip():
            ctxs.append(c)
        if len(ctxs) >= MAX_DOCS_PER_Q:
            break

    # per-doc clipping
    clipped = []
    for c in ctxs:
        ids = tok.encode(c, add_special_tokens=False)[:PER_DOC_CAP]
        clipped.append(tok.decode(ids))

    # enforce overall budget
    used_by_qa = num_tokens(q) + num_tokens(a) + RESERVED_FOR_QA
    budget = max(256, MAX_INPUT_TOKENS - used_by_qa)
    packed = []
    used_now = 0
    for c in clipped:
        t = num_tokens(c)
        if used_now + t <= budget:
            packed.append(c)
            used_now += t
        else:
            left = budget - used_now
            if left > 64:
                ids = tok.encode(c, add_special_tokens=False)[:left]
                packed.append(tok.decode(ids))
            break

    rec["contexts"] = packed
    if not rec.get("ground_truth"):
        rec["ground_truth"] = " ".join(packed)
    return rec

dataset = dataset.map(trim_contexts)
avg_ctxs = sum(len(x["contexts"]) for x in dataset) / max(1, len(dataset))
avg_ctx_tokens = sum(sum(num_tokens(c) for c in x["contexts"]) for x in dataset) / max(1, len(dataset))
print("[data] after trimming: " + str({"avg_ctxs": avg_ctxs, "avg_ctx_tokens": avg_ctx_tokens}))

# ---------------- Evaluate ----------------
result = evaluate(
    dataset,
    metrics=[answer_relevancy, answer_correctness, faithfulness],
    llm=llm,
    embeddings=emb,
)

# ---------------- Save results (version-agnostic) ----------------
import json as _json

def per_query_df_from_ragas(res):
    if hasattr(res, "to_pandas"):
        return res.to_pandas()
    if isinstance(res, dict):
        for k in ("per_query", "per_question", "results"):
            if k in res:
                return pd.DataFrame(res[k])
    return pd.DataFrame(res)

df = per_query_df_from_ragas(result)
df.to_csv(OUT_PERQUERY, index=False)

metric_cols = [c for c in df.columns if c in {"answer_relevancy", "answer_correctness", "faithfulness"}]
summary = {"dataset": DATASET, "n": int(len(df))}
for c in metric_cols:
    summary[c + "_mean"] = float(pd.to_numeric(df[c], errors="coerce").mean())

with open(OUT_SUMMARY, "w") as f:
    _json.dump(summary, f, indent=2)

# Show a quick look + count parse failures
bad = df[df[metric_cols].isna().any(axis=1)] if len(metric_cols) > 0 else pd.DataFrame()
print("[out] per-query -> " + OUT_PERQUERY)
print("[out] summary  -> " + OUT_SUMMARY)
print(df.head(3))
print("[warn] parse failures: {} / {}".format(len(bad), len(df)))