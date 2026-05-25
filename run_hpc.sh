#!/bin/bash
#SBATCH --job-name=financerag
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ── Environment ──────────────────────────────────────────────────────────────
module load Anaconda3/2024.02-1
source activate financerag

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_TOKEN="your_huggingface_token_here"

# ── Run a specific dataset (change DATASET to run others) ────────────────────
# Options: FinDER | FinQABench | FinanceBench | TATQA | FinQA | ConvFinQA | MultiHiertt | Generation

DATASET=${1:-FinDER}

echo "[$(date)] Starting $DATASET"
python ${DATASET}.py
echo "[$(date)] Finished $DATASET"
