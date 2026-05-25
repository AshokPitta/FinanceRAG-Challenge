# -*- coding: utf-8 -*-
"""
This generation code is same for all datasets. just replace the dataset name and path.
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import json
import torch
import pandas as pd
from pathlib import Path
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from financerag.tasks import FinanceBench

# === Authenticate with HF token (needed for Llama 3) ===
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("[auth] Hugging Face login successful")

# === Load FinanceBench task (queries + corpus) ===
finqa_task = FinanceBench()

# === Load reranked results from CSV (query_id, corpus_id) ===
results_csv = os.path.join(BASE_DIR, "results/FinanceBench/results.csv")

df = pd.read_csv(results_csv)

reranked = {}
for _, row in df.iterrows():
    qid = str(row["query_id"])
    docid = str(row["corpus_id"])
    if qid not in reranked:
        reranked[qid] = {}
    # assign increasing scores so order is preserved
    reranked[qid][docid] = len(reranked[qid])

# === Setup output directory ===
out_dir = os.path.join(BASE_DIR, "results/FinanceBench")
Path(out_dir).mkdir(parents=True, exist_ok=True)
answers_out = Path(out_dir) / "FinanceBench_answers.jsonl"

# === Load Llama-3 model ===
gen_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
use_4bit = True
max_ctx_per_query, max_new_tokens = 6, 128

print("[gen] loading LLM:", gen_model_name)
tok = AutoTokenizer.from_pretrained(gen_model_name, use_fast=True)
load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
if use_4bit:
    try:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    except Exception:
        load_kwargs["load_in_4bit"] = True

model = AutoModelForCausalLM.from_pretrained(gen_model_name, **load_kwargs)
generate = pipeline("text-generation", model=model, tokenizer=tok,
                    device_map="auto", torch_dtype=torch.float16)

# === Helper functions ===
def build_prompt(query_text, contexts):
    sep = "\n\n---\n\n"
    return (
        "You are a financial QA assistant. "
        "Answer concisely using ONLY the provided contexts. "
        "If the answer cannot be found, reply exactly 'INSUFFICIENT'.\n\n"
        "Question:\n" + query_text + "\n\n"
        "Contexts:" + sep.join(contexts) + "\n\n"
        "Answer:"
    )

def doc_text(doc):
    return (doc.get("title", "") + "\n" + doc.get("text", "")).strip()

# === Generate answers ===
n_total = len(finqa_task.queries)
print("[gen] generating answers for {} queries (top {} contexts each)...".format(n_total, max_ctx_per_query))

with open(answers_out, "w", encoding="utf-8") as fw:
    for qi, (qid, qtext) in enumerate(finqa_task.queries.items(), 1):
        # take top-k contexts
        scored = list(reranked.get(qid, {}).keys())[:max_ctx_per_query]
        contexts = [doc_text(finqa_task.corpus[docid]) for docid in scored if docid in finqa_task.corpus]
        if not contexts:
            contexts = ["(no context)"]

        prompt = build_prompt(qtext, contexts)
        outputs = generate(prompt, max_new_tokens=max_new_tokens, do_sample=False,
                           temperature=0.0, top_p=1.0, pad_token_id=tok.eos_token_id)
        text = outputs[0]["generated_text"]
        ans = text.split("Answer:")[-1].strip()

        fw.write(json.dumps({"query_id": qid, "answer": ans}, ensure_ascii=False) + "\n")

        if qi % 50 == 0 or qi == n_total:
            print("[gen] {} / {} done".format(qi, n_total))