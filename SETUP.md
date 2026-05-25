# Setup Guide

## Requirements

- Python 3.11
- CUDA-capable GPU (recommended: 24GB+ VRAM for generation with LLaMA-3-8B)
- 64GB+ RAM for larger datasets (MultiHiertt corpus is ~32GB uncompressed)
- A [Hugging Face account](https://huggingface.co) with access to `meta-llama/Meta-Llama-3-8B-Instruct`

---

## 1. Clone and install

```bash
git clone https://github.com/AshokPitta/FinanceRAG-Challenge.git
cd FinanceRAG-Challenge

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Download the datasets

Download the seven datasets from the [Kaggle competition page](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge/data) and place them under `dataset/`:

```
dataset/
├── FinDER/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── FinDER_qrels.tsv
├── FinQABench/
├── FinanceBench/
├── TATQA/
├── FinQA/
├── ConvFinQA/
└── MultiHiertt/
```

Each dataset folder must contain `corpus.jsonl`, `queries.jsonl`, and a `*_qrels.tsv` ground truth file.

---

## 3. Set your Hugging Face token

LLaMA-3 requires accepting the model licence on Hugging Face and authenticating:

```bash
export HF_TOKEN="your_token_here"
```

Or log in via the CLI:

```bash
huggingface-cli login
```

---

## 4. Update paths in scripts

Each pipeline script has hardcoded paths pointing to the Sheffield HPC scratch directory (`/mnt/parscratch/...`). Before running locally, update these two variables at the top of each script:

```python
qrels_path = "dataset/<DATASET>/<DATASET>_qrels.tsv"
out_dir    = "results/<DATASET>"
```

Do the same in `Generation.py`:

```python
results_csv = "results/<DATASET>/results.csv"
out_dir     = "results/<DATASET>"
```

---

## 5. Run retrieval

Run each dataset's retrieval pipeline independently:

```bash
# Passage retrieval
python FinDER.py
python FinQABench.py
python FinanceBench.py

# Tabular + text retrieval
python TATQA.py
python FinQA.py
python ConvFinQA.py
python MultiHiertt.py
```

Results are saved to `results/<DATASET>/results.csv`.

---

## 6. Run generation

After retrieval is complete for a dataset, run generation:

```bash
python Generation.py
```

Update the `results_csv` path and dataset class at the top of `Generation.py` for each dataset you want to generate answers for. Outputs are saved as `results/<DATASET>/<DATASET>_answers.jsonl`.

---

## 7. Running on HPC (SLURM)

A SLURM job script is provided. Submit a job for a specific dataset:

```bash
mkdir -p logs
sbatch run_hpc.sh FinDER
sbatch run_hpc.sh FinQA
sbatch run_hpc.sh Generation
```

Adjust `--time`, `--mem`, and `--partition` in `run_hpc.sh` to match your cluster's configuration.

Set your HF token inside `run_hpc.sh`:

```bash
export HF_TOKEN="your_token_here"
```

---

## Notes

- **MultiHiertt** uses a triple-dense setup (`BGE + MXBAI + E5`) and takes significantly longer than other datasets.
- **Generation** requires LLaMA-3-8B-Instruct. With 4-bit quantisation (`use_4bit=True`) it fits in ~10GB VRAM.
- Model weights are downloaded automatically from Hugging Face on first run and cached in `$HF_HOME`.
- All scripts use greedy decoding (`temperature=0.0`) for reproducibility.
