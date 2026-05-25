import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------- I/O helpers ----------

def clean_text(text: str) -> str:
  
    return re.sub(r"(\\u[0-9A-Fa-f]{4})+", " ", text)

def load_jsonl(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    items: List[Dict] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(clean_text(line)))
    return items

def save_jsonl(file_path: Path, data: List[Dict], ensure_ascii: bool = False) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")

def load_prompt(subset: str, key: str = "queries") -> str:

    prompt_path = Path("./prompt.json")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)["pre_retrieval"][key]
    if subset not in prompts:
        raise ValueError(f"Prompt not found for subset '{subset}'")
    return prompts[subset]

# ---------- Corpus processing ----------

def _extract_table_from_corpus(c_text: str, subset: str) -> str:

    lines = c_text.split("\n")
    results = []
    table_start = None

    for i, line in enumerate(lines):
        if line.startswith("| "):
            if table_start is None:
                table_start = i
            # end of table block?
            nxt_is_table = (i + 1 < len(lines) and lines[i + 1].startswith("| "))
            if not nxt_is_table:
                table_content = "\n".join(lines[table_start : i + 1])
                results.append(table_content.strip())
                table_start = None

    if results:
        return "\n\n".join(results)

    # No table found -> subset-specific fallback
    parts = c_text.split("\n\n")
    if subset in {"TATQA", "FinQA", "ConvFinQA"}:
        # TATQA often wants the last chunk (tables often rendered near end);
        # for FinQA / ConvFinQA the second chunk often carries the salient text.
        if subset == "TATQA":
            return parts[-1] if parts else c_text
        else:
            return parts[1] if len(parts) > 1 else parts[0] if parts else c_text
    elif subset == "MultiHiertt":
        return c_text
    else:
        return c_text  # safe default

def compress_corpus(subset: str, dataset_dir: str) -> None:
    corpus_path = Path(dataset_dir) / subset / "corpus.jsonl"
    out_path = Path(dataset_dir) / subset / "corpus_prep.jsonl"
    corpus = load_jsonl(corpus_path)
    for item in corpus:
        item["text"] = _extract_table_from_corpus(item.get("text", ""), subset)
    save_jsonl(out_path, corpus)

def copy_corpus(subset: str, dataset_dir: str) -> None:
    src = Path(dataset_dir) / subset / "corpus.jsonl"
    dst = Path(dataset_dir) / subset / "corpus_prep.jsonl"
    shutil.copy(src, dst)

# ---------- Local HF "chat" wrapper (mimics .invoke().content) ----------

class HFLocalChat:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 32,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
    ):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=trust_remote_code
        )

        # Model
        kwargs = dict(
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=trust_remote_code,
        )
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        if load_in_4bit:
            kwargs["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            **kwargs,
        )

        # IMPORTANT: don't pass device= when using device_map/Accelerate
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.max_new_tokens = max_new_tokens

    class _Resp:
        def __init__(self, text: str):
            self.content = text

    def invoke(self, prompt: str):
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,  # only the continuation
        )[0]["generated_text"]
        return HFLocalChat._Resp(out.strip())

# ---------- Query expansion ----------

def expand_queries(subset: str, dataset_dir: str, llm: HFLocalChat, overwrite: bool = True) -> None:

    data = load_jsonl(Path(dataset_dir) / subset / "queries.jsonl")
    prompt_template = load_prompt(subset, "queries")

    expanded_queries: List[Dict] = []
    for item in data:
        q_text = item["text"]
        prompt = f"{prompt_template}\n\n# Query\n{q_text}"
        new_text = llm.invoke(prompt).content
        merged = {
            "_id": item["_id"],
            "title": item.get("title", ""),
            "text": f"{q_text}\n\n{new_text}".strip(),
        }
        expanded_queries.append(merged)

    out_path = Path(dataset_dir) / subset / "queries_prep.jsonl"
    if overwrite or (not out_path.exists()):
        save_jsonl(out_path, expanded_queries, ensure_ascii=False)

# ---------- Orchestration ----------

def pre_retrieval(dataset_dir: str) -> None:
    model_name = os.environ.get("FINRAG_QE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    print(f"[pre_retrieval] Using HF model for query expansion: {model_name}")

    llm = HFLocalChat(
        model_name=model_name,
        max_new_tokens=int(os.environ.get("FINRAG_QE_MAX_NEW_TOKENS", "32")),
        load_in_8bit=os.environ.get("FINRAG_QE_8BIT", "0") == "1",
        load_in_4bit=os.environ.get("FINRAG_QE_4BIT", "0") == "1",
        trust_remote_code=True,
    )

    subsets = [
        "FinanceBench"
    ]

    for subset in subsets:
        print(f"\n=== Pre-retrieval for '{subset}' ===")
        q_path = Path(dataset_dir) / subset / "queries.jsonl"
        c_path = Path(dataset_dir) / subset / "corpus.jsonl"
        # counts before
        try:
            q_before = len(load_jsonl(q_path))
            c_before = len(load_jsonl(c_path))
        except FileNotFoundError as e:
            print(f"[{subset}] ERROR: {e}")
            continue

        print(f"[{subset}] BEFORE  -> queries: {q_before:,} | corpus: {c_before:,}")

        # Expand queries
        try:
            expand_queries(subset, dataset_dir, llm, overwrite=True)
        except Exception as e:
            print(f"[{subset}] Query expansion error: {e}")
            continue

        # Prepare corpus
        try:
            if subset == "MultiHiertt":
                compress_corpus(subset, dataset_dir)
            else:
                copy_corpus(subset, dataset_dir)
        except Exception as e:
            print(f"[{subset}] Corpus preparation error: {e}")
            continue

        # counts after
        q_after_path = Path(dataset_dir) / subset / "queries_prep.jsonl"
        c_after_path = Path(dataset_dir) / subset / "corpus_prep.jsonl"
        try:
            q_after = len(load_jsonl(q_after_path))
            c_after = len(load_jsonl(c_after_path))
        except FileNotFoundError as e:
            print(f"[{subset}] ERROR reading prepared files: {e}")
            continue

        print(f"[{subset}] AFTER   -> queries_prep: {q_after:,} | corpus_prep: {c_after:,}")
        print(f"[{subset}] Completed.")

# ---------- CLI ----------

if __name__ == "__main__":
    # Optional: honor HF cache/env if you set them outside
    os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", ""))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ.get("HF_HOME", ""))

    pre_retrieval(dataset_dir="./dataset")

