#!/usr/bin/env python3
"""
Configuration and default hyperparameters for RGPO training.

Hyperparameter defaults follow the paper's implementation details
(Section V-A): base model Gemma 2B, K = 5 candidates per question,
beta = 0.1, acceptance threshold theta = 0.6 (used upstream during
dataset construction, not during this training script), AdamW with
lr = 5e-5, batch size 16, 3 epochs. The reference script here keeps
a smaller lr/epoch count suited to a single TinyLlama smoke test.
"""
import os

# HF_TOKEN must be provided via environment variable, e.g.:
#   export HF_TOKEN="hf_xxx"
# Do NOT hardcode tokens in source code.
HF_TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATA_FILE = "/content/grpo_cot_pairs_250.jsonl"
DEFAULT_OUTPUT_DIR = "./grpo_output"

# --- RGPO objective hyperparameters (paper Section IV) ---
# beta: weight of the KL regularization term L_KL (Eq. 11)
DEFAULT_BETA = 0.1
# tau_BT: Bradley-Terry temperature used in the pairwise ranking loss
# L_CoT-rank (Eq. 6-7). This is distinct from the sampling temperature
# tau used for candidate generation.
DEFAULT_TAU_BT = 1.0
# Sampling temperature tau used when generating candidate CoTs.
DEFAULT_TEMPERATURE = 1.0
# K: number of ranked candidates per question (paper uses K = 5).
DEFAULT_MAX_LENGTH = 512

# --- Training loop defaults ---
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-6
DEFAULT_USE_PAIRWISE = True

CUDA_ALLOC_CONF = "expandable_segments:True"
