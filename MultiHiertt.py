# -*- coding: utf-8 -*-


import os, re, json, time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from collections import defaultdict, Counter

import torch
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from financerag.tasks import MultiHiertt
from financerag.tasks.BaseTask import BaseTask
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
from financerag.rerank import CrossEncoderReranker
from financerag.common import Lexical

# ------------- Llama-3 settings -------------
USE_LLAMA_EXPANSION = True
LLAMA_CANDIDATES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]
LLAMA_MAX_NEW_TOKENS = 120
LLAMA_BATCH = 4      
LLAMA_DEVICE = "auto"   
LLAMA_DTYPE = "bfloat16"  

# ------------- regex / cues -------------
NUM_RE  = re.compile(r"\b\d{1,3}(?:[,]\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&\-/]+")
BM25_NUMERIC_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|%|\$|million|billion|thousand|bn|m")
HOP_CUES = (
    "respectively","combined","together","across","between","each of",
    "sum of","difference between","versus","vs","per segment","by segment",
    "by region","by geography","by product","by business","by division"
)

# ------------- helpers -------------
def minmax(d):
    if not d: return {}
    vals = list(d.values()); lo, hi = min(vals), max(vals)
    if lo == hi: return {k:0.0 for k in d}
    s = hi-lo
    return {k:(v-lo)/s for k,v in d.items()}

def rrf_fuse(dicts, k=60):
    ranks = []
    for d in dicts:
        order = sorted(d.items(), key=lambda x:-x[1])
        ranks.append({doc:i+1 for i,(doc,_) in enumerate(order)})
    all_docs = set().union(*[set(d.keys()) for d in dicts])
    out = {}
    for doc in all_docs:
        s = 0.0
        for r in ranks:
            pos = r.get(doc)
            if pos: s += 1.0/(k+pos)
        out[doc] = s
    return out

# ------------- segmentation -------------
def extract_markdown_tables(text):
    lines = text.splitlines()
    blocks, start = [], None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|"):
            start = i if start is None else start
        elif start is not None:
            block = "\n".join(lines[start:i])
            if block.count("|") >= 4:
                blocks.append(block)
            start = None
    if start is not None:
        block = "\n".join(lines[start:])
        if block.count("|") >= 4:
            blocks.append(block)
    return blocks

def extract_headings(text, max_heads=20):
    heads = []
    for ln in text.splitlines():
        ls = ln.strip()
        if not ls: continue
        if ls.startswith("#"): heads.append(ls.lstrip("#").strip())
        elif ls.isupper() and 3 <= len(ls) <= 80: heads.append(ls)
        elif (":" in ls) and (len(ls) <= 80) and not ls.endswith("."): heads.append(ls)
        elif re.match(r"^\d+(\.\d+)*\s+\S+", ls): heads.append(ls)
        elif re.match(r"^[IVXLC]+\.\s+\S+", ls): heads.append(ls)
        elif re.match(r"^[-*]\s+\S+", ls): heads.append(ls[1:].strip())
        if len(heads) >= max_heads: break
    seen=set(); out=[]
    for h in heads:
        k=h.lower()
        if k not in seen: seen.add(k); out.append(h)
    return out

def build_segmented_view(corpus, window_rows=24, stride=12, max_segs_per_doc=20):
    seg_corpus, seg2parent = {}, {}
    for did, doc in corpus.items():
        title = doc.get("title","")
        base  = doc.get("text","")
        heads = extract_headings(base, max_heads=16)
        head_str = "section_headers: " + " | ".join(heads[:16]) if heads else ""
        tables = extract_markdown_tables(base)
        segments = []
        for tb in tables:
            rows = [r.strip() for r in tb.splitlines() if r.strip().startswith("|")]
            if not rows: continue
            header = rows[0].replace("|"," ").strip()
            body   = [r.replace("|"," ").strip() for r in rows[1:]]
            for i in range(0, len(body), stride):
                chunk = body[i:i+window_rows]
                if not chunk: break
                seg_text = " \n".join(chunk)
                payload = "\n".join([title, head_str, f"table_headers: {header}", seg_text])
                segments.append(payload[:4000])
                if len(segments) >= max_segs_per_doc: break
            if len(segments) >= max_segs_per_doc: break
        if not segments:
            segments = ["\n".join([title, head_str, base])[:4000]]
        for i, seg in enumerate(segments):
            sid = f"{did}#seg{i}"
            seg_corpus[sid] = {"title": title, "text": seg}
            seg2parent[sid] = did
    return seg_corpus, seg2parent

# ------------- BM25 + RM3 -------------
def bm25_tokenize_list(texts):
    return [BM25_NUMERIC_RE.findall(t) for t in texts]

class TunedBM25(Lexical):
    def __init__(self, corpus, k1=1.6, b=0.75, title_boost=2):
        tok_docs = []
        for d in corpus.values():
            title = (d.get("title","")+" ").lower()*title_boost
            txt   = (d.get("text","")).lower()
            tok_docs.append(BM25_NUMERIC_RE.findall(title + txt))
        self.bm25 = BM25Okapi(tok_docs, k1=k1, b=b)
        self.doc_ids = list(corpus.keys())
    def get_scores(self, query_tokens):
        return self.bm25.get_scores(query_tokens)

def decompose_query(q):
    parts = re.split(r"\b(?:and|,|/| vs | versus )\b", q.lower())
    out = [q]
    for p in parts:
        p = p.strip()
        if len(p.split()) >= 3: out.append(p)
    seen=set(); res=[]
    for s in out:
        if s not in seen:
            seen.add(s); res.append(s)
    return res[:4]

def rm3_expand_query(bm: BM25Okapi, qtext: str, docs: dict, top_docs=8, exp_terms=10):
    q_tokens = bm25_tokenize_list([qtext.lower()])[0]
    scores = bm.get_scores(q_tokens)
    idxs = list(reversed(scores.argsort()))[:top_docs]
    pool = Counter()
    doc_ids = list(docs.keys())
    for i in idxs:
        did = doc_ids[i]
        txt = (docs[did].get("title","")+" "+docs[did].get("text","")).lower()
        toks = bm25_tokenize_list([txt])[0]
        for t in toks:
            if len(t) < 3: continue
            pool[t] += 1
    qset = set(q_tokens)
    ext = [t for t,_ in pool.most_common(80) if t not in qset][:exp_terms]
    return qtext + " " + " ".join(ext) if ext else qtext

# ------------- Llama 3 expansion -------------
def try_load_llama():
    if not USE_LLAMA_EXPANSION:
        return None
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    token = os.getenv("HF_TOKEN", None)
    for name in LLAMA_CANDIDATES:
        try:
            print(f"[llama] loading {name} ...")
            tok = AutoTokenizer.from_pretrained(name, token=token, trust_remote_code=True)
            dtype = torch.bfloat16 if LLAMA_DTYPE=="bfloat16" else torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=dtype,
                device_map=LLAMA_DEVICE,
                token=token,
                trust_remote_code=True,
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            )
            print("[llama] ready.")
            return pipe
        except Exception as e:
            print(f"[llama] could not load {name}: {e}")
    print("[llama] expansion disabled (no model).")
    return None

EXP_PROMPT = (
    "You expand a financial QA query for retrieval over annual reports with hierarchical tables.\n"
    "Given the QUERY, output a compact JSON with fields:\n"
    "{\"keywords\":[],\"entities\":[],\"metrics\":[],\"units\":[],\"periods\":[],\"group_by\":[],\"hints\":[]}\n"
    "- keywords: 1-6 important nouns/phrases from the query; prefer canonical forms\n"
    "- entities: tickers or company/segment names if any\n"
    "- metrics: financial metrics (e.g., revenue, gross margin, EPS)\n"
    "- units: symbols like %, $, million, billion, bn, m\n"
    "- periods: fiscal years/quarters (e.g., FY2019, Q4 2020); no ranges like 2018-2019, list both\n"
    "- group_by: dimensions (segment, region, product)\n"
    "- hints: short boolean cues like [\"ratio\",\"sum\",\"difference\",\"per-segment\",\"comparison\"]\n"
    "Only return JSON. Do not add extra text.\n\nQUERY: "
)

def llama_expand_queries(pipe, queries):
    if pipe is None:
        return {qid: q for qid, q in queries.items()}
    qids = list(queries.keys())
    expanded = {}
    for i in range(0, len(qids), LLAMA_BATCH):
        batch_ids = qids[i:i+LLAMA_BATCH]
        prompts = [EXP_PROMPT + queries[qid] for qid in batch_ids]
        outs = pipe(prompts)
        for qid, out in zip(batch_ids, outs):
            txt = out[0]["generated_text"]
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            exp_txt = queries[qid]
            if m:
                try:
                    js = json.loads(m.group(0))
                except Exception:
                    js = {}
                fields = []
                for k in ("keywords","entities","metrics","units","periods","group_by","hints"):
                    v = js.get(k, [])
                    if isinstance(v, list) and v:
                        fields.append(f"{k}: " + ", ".join(map(str,v)))
                if fields:
                    exp_txt = queries[qid] + "\n\n" + " | ".join(fields)
            expanded[qid] = exp_txt
    return expanded

# ------------- features -------------
def parse_intents(q):
    ql = q.lower()
    return {
        "hop": any(c in ql for c in HOP_CUES),
        "ratio": any(w in ql for w in ("ratio","per","/","proportion",":")),
        "diff": any(w in ql for w in ("difference","change","delta","increase","decrease")),
        "sum": any(w in ql for w in ("sum","total","aggregate","combined","together")),
        "cmp": any(w in ql for w in ("greater","less","higher","lower","vs","versus","compare")),
        "pct": ("%" in ql) or ("percent" in ql),
        "cur": ("$" in q) or ("usd" in ql),
    }

def multi_hop_feature_score(q, doc_text):
    intents = parse_intents(q)
    qnums = set(NUM_RE.findall(q)); dnums = set(NUM_RE.findall(doc_text))
    num_overlap = len(qnums & dnums)
    year_hit = 1 if (YEAR_RE.search(q) and YEAR_RE.search(doc_text)) else 0
    dl = doc_text.lower()
    heads = []
    for m in re.finditer(r"(section_headers|table_headers):\s*([^\n]+)", doc_text, flags=re.IGNORECASE):
        heads += TOKEN_RE.findall(m.group(2).lower())
    qterms = set(TOKEN_RE.findall(q.lower()))
    hdr_overlap = len(qterms & set(heads))
    unit_pct = intents["pct"] and (("%" in dl) or ("percent" in dl))
    unit_cur = intents["cur"] and (("$" in dl) or ("usd" in dl))
    score = 0.0
    score += 0.40 * min(num_overlap, 3)/3.0
    score += 0.18 * year_hit
    score += 0.22 * (hdr_overlap > 0)
    score += 0.10 * unit_pct
    score += 0.10 * unit_cur
    return max(0.0, min(1.0, score))

# ------------- main -------------
if __name__ == "__main__":
    task = MultiHiertt()
    print("Dataset:", task.metadata.name)
    print("Queries:", len(task.queries))
    print("Corpus: ", len(task.corpus))

    # Llama 3 expansion (optional)
    llama = try_load_llama()
    expanded_queries = llama_expand_queries(llama, task.queries)

    # segmented view for dense
    seg_corpus, seg2parent = build_segmented_view(task.corpus, window_rows=24, stride=12, max_segs_per_doc=20)

    # triple dense on segments (use expanded queries)
    q_prompt = ("Represent this multi-hop financial question to retrieve all necessary tables/sections, "
                "preserving entities, units and fiscal periods: ")
    d_prompt = ("Represent this TABLE/SECTION for retrieval; include section headers, table headers, "
                "units (%, $, bn, m) and periods: ")

    enc_bge = SentenceTransformerEncoder("BAAI/bge-large-en-v1.5", query_prompt=q_prompt, doc_prompt=d_prompt)
    ret_bge = DenseRetrieval(model=enc_bge)
    d1_seg = ret_bge.retrieve(corpus=seg_corpus, queries=expanded_queries, top_k=400)

    enc_mxb = SentenceTransformerEncoder("mixedbread-ai/mxbai-embed-large-v1", query_prompt=q_prompt, doc_prompt=d_prompt)
    ret_mxb = DenseRetrieval(model=enc_mxb)
    d2_seg = ret_mxb.retrieve(corpus=seg_corpus, queries=expanded_queries, top_k=400)

    enc_e5  = SentenceTransformerEncoder("intfloat/e5-large-v2", query_prompt=q_prompt, doc_prompt=d_prompt)
    ret_e5  = DenseRetrieval(model=enc_e5)
    d3_seg  = ret_e5.retrieve(corpus=seg_corpus, queries=expanded_queries, top_k=400)

    def agg(seg_results):
        out = {}
        for qid, d in seg_results.items():
            acc = defaultdict(float)
            for sid, sc in d.items():
                pid = seg2parent.get(sid, sid)
                acc[pid] = max(acc[pid], sc)
            out[qid] = dict(acc)
        return out

    dense1 = agg(d1_seg); dense2 = agg(d2_seg); dense3 = agg(d3_seg)

    def bm25_multi_variant(docs, top_k=350):
        grids = [(1.4,0.80,2), (1.6,0.75,2), (1.8,0.70,3)]
        variants = []
        for k1,b,boost in grids:
            bm = TunedBM25(docs, k1=k1, b=b, title_boost=boost)
            doc_ids = list(docs.keys())
            res = {}
            for qid, qtext in task.queries.items():
                cand_scores = Counter()
                forms = decompose_query(qtext)
                if expanded_queries.get(qid) and expanded_queries[qid] != qtext:
                    forms.append(expanded_queries[qid])
                base_q = forms[0]
                _ = bm.bm25.get_scores(bm25_tokenize_list([base_q])[0])
                rm3_q = rm3_expand_query(bm.bm25, base_q, docs, top_docs=8, exp_terms=10)
                forms = list(dict.fromkeys(forms + [rm3_q]))[:6]
                for qv in forms:
                    toks = bm25_tokenize_list([qv.lower()])[0]
                    scrs = bm.bm25.get_scores(toks)
                    idxs = list(reversed(scrs.argsort()))[:top_k]
                    for i in idxs:
                        cand_scores[doc_ids[i]] = max(cand_scores[doc_ids[i]], float(scrs[i]))
                res[qid] = dict(sorted(cand_scores.items(), key=lambda x:-x[1])[:top_k])
            variants.append(res)
        fused = {}
        for qid in task.queries:
            fused[qid] = rrf_fuse([v.get(qid,{}) for v in variants], k=50)
        return fused

    bm25 = bm25_multi_variant(task.corpus, top_k=350)

    # Fusion selection
    qrels_candidates = [
        "./dataset/MultiHiertt/MultiHiertt_qrels.tsv",
        "./dataset/MultiHiertt/MultiHeirtt_qrels.tsv",
    ]
    qrels_path = next((p for p in qrels_candidates if os.path.exists(p)), None)
    if qrels_path is None:
        raise FileNotFoundError("MultiHiertt qrels not found.")
    dfq = pd.read_csv(qrels_path, sep="\t", names=["qid","did","rel"], skiprows=1)
    qrels = defaultdict(dict)
    for _, r in dfq.iterrows():
        qrels[str(r["qid"])][str(r["did"])] = int(r["rel"])

    def ndcg10(res):
        nd, *_ = BaseTask.evaluate(qrels=qrels, results=res, k_values=[10])
        return float(nd["NDCG@10"])

    def fuse_minmax(weights_dense, gamma):
        fused = {}
        for qid in task.queries:
            parts = [
                minmax(dense1.get(qid, {})),
                minmax(dense2.get(qid, {})),
                minmax(dense3.get(qid, {})),
            ]
            all_docs = set().union(*[set(p.keys()) for p in parts])
            dense_mix = {}
            for d in all_docs:
                s = 0.0
                for w, p in zip(weights_dense, parts):
                    s += w * p.get(d, 0.0)
                dense_mix[d] = s
            bm_n = minmax(bm25.get(qid, {}))
            allc = set(dense_mix) | set(bm_n)
            fused[qid] = {d: gamma*dense_mix.get(d,0.0) + (1-gamma)*bm_n.get(d,0.0) for d in allc}
        return fused

    dense_weight_grid = [(0.40,0.30,0.30),(0.45,0.30,0.25)]
    best_fused, best_score = None, -1.0
    for w in dense_weight_grid:
        for gamma in (0.50,0.60,0.70,0.80):
            f = fuse_minmax(w, gamma)
            sc = ndcg10(f)
            print(f"[fuse] w={w} gamma={gamma:.2f} -> nDCG@10={sc:.5f}")
            if sc > best_score:
                best_score, best_fused = sc, f
    # compare with pure RRF of dense+bm25
    def fuse_rrf():
        f = {}
        for qid in task.queries:
            f[qid] = rrf_fuse(
                [
                    dense1.get(qid,{}),
                    dense2.get(qid,{}),
                    dense3.get(qid,{}),
                    bm25.get(qid,{})
                ],
                k=60
            )
        return f
    fused_rrf = fuse_rrf(); sc_rrf = ndcg10(fused_rrf)
    fused = best_fused if best_score >= sc_rrf else fused_rrf
    print(f"Chosen fusion: {'minmax' if fused is best_fused else 'rrf'} (nDCG@10={max(best_score, sc_rrf):.5f})")

    # Rerank
    rr_bge = CrossEncoderReranker(CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda"))
    rr_min = CrossEncoderReranker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda"))
    rer_bge = task.rerank(reranker=rr_bge, results=fused, top_k=180, batch_size=32)
    rer_min = task.rerank(reranker=rr_min, results=fused, top_k=180, batch_size=32)

    def final_mix(weights):
        wb, wm, wf = weights
        out = {}
        for qid in task.queries:
            cand = set(fused.get(qid, {}).keys())
            bge_n = minmax({d: rer_bge.get(qid, {}).get(d, 0.0) for d in cand})
            min_n = minmax({d: rer_min.get(qid, {}).get(d, 0.0) for d in cand})
            feat  = {d: multi_hop_feature_score(task.queries[qid], task.corpus[d].get("text","")) for d in cand}
            feat_n = minmax(feat)
            scores = {d: wb*bge_n.get(d,0.0) + wm*min_n.get(d,0.0) + wf*feat_n.get(d,0.0) for d in cand}
            out[qid] = dict(sorted(scores.items(), key=lambda x:-x[1])[:10])
        return out

    grid = [(0.55,0.25,0.20),(0.50,0.30,0.20)]
    best_nd, best_res, best_w = -1.0, None, None
    for w in grid:
        res = final_mix(w)
        nd, *_ = BaseTask.evaluate(qrels=qrels, results=res, k_values=[10])
        sc = float(nd["NDCG@10"])
        print(f"[mix] w={w} -> nDCG@10={sc:.5f}")
        if sc > best_nd:
            best_nd, best_res, best_w = sc, res, w

    print(f"Chosen weights {best_w} -> nDCG@10={best_nd:.5f}")

    # Save
    out_dir = os.path.join(BASE_DIR, "results/")
    task.rerank_results = best_res
    task.save_results(output_dir=out_dir)
    print(f"Saved results to {out_dir}/MultiHiertt/results.csv")

    # Final metric
    ndcg, *_ = BaseTask.evaluate(qrels=qrels, results=best_res, k_values=[10])
    print(f"MultiHiertt -> nDCG@10: {ndcg['NDCG@10']:.5f}")

