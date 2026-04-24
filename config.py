# file: config.py
"""
Runtime configuration for RGPO data generation pipeline.
"""
from __future__ import annotations

import os

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = os.getenv("INPUT_PATH", "/content/ori_pqal_2.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/content/rgpo_cot_pairs.jsonl")

# -----------------------------
# Refinement
# -----------------------------
ACCEPTANCE_THRESHOLD = float(os.getenv("ACCEPTANCE_THRESHOLD", 0.6))

# -----------------------------
# Model backend selection
# -----------------------------
USE_OPENAI = os.getenv("OPENAI_API_KEY") is not None

OPENAI_MODEL_GENERATE = os.getenv("OPENAI_MODEL_GENERATE", "gpt-4o-mini")
OPENAI_MODEL_SCORE    = os.getenv("OPENAI_MODEL_SCORE",    "gpt-4o-mini")
OPENAI_MODEL_REVISE   = os.getenv("OPENAI_MODEL_REVISE",   "gpt-4o-mini")

HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

# -----------------------------
# RGPO generation settings
# (K=5 generate, top M=4 selected — per paper Section III.C)
# -----------------------------
NUM_COT_VARIANTS  = 5   # K: number of candidates generated per question
NUM_TOP_CANDIDATES = 4  # M: top candidates kept after quality ranking
SAMPLE_SIZE = 900
