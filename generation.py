# file: generation.py
"""
LLM backend initialisation, CoT variant generation, scoring calls, and
probability-driven revision for the RGPO data generation pipeline.
"""
from __future__ import annotations

import os
from typing import Dict, List

from config import (
    USE_OPENAI,
    OPENAI_MODEL_GENERATE,
    OPENAI_MODEL_SCORE,
    OPENAI_MODEL_REVISE,
    HF_MODEL,
    ACCEPTANCE_THRESHOLD,
)
from prompts import get_reasoning_prompt, score_prompt, self_reflection_prompt
from scoring import parse_score_response, needs_revision, determine_start_step


# ──────────────────────────────────────────────
# Backend initialisation
# ──────────────────────────────────────────────

if USE_OPENAI:
    from openai import OpenAI
    _client = OpenAI()
    hf_pipeline = None
else:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    hf_pipeline = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
        max_new_tokens=512,
    )
    hf_pipeline("Hello")  # warmup
    _client = None


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _call_llm_generate(prompt: str, n: int = 1, temperature: float = 0.7) -> List[str]:
    """Return n completions for `prompt` using the active backend."""
    if USE_OPENAI:
        res = _client.chat.completions.create(
            model=OPENAI_MODEL_GENERATE,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
            max_tokens=512,
            n=n,
        )
        return [choice.message.content.strip() for choice in res.choices]
    else:
        outputs = hf_pipeline(
            prompt,
            num_return_sequences=n,
            do_sample=(temperature > 0),
            temperature=temperature,
        )
        return [out["generated_text"].split("Reasoning:")[-1].strip() for out in outputs]


def _call_llm_score(prompt: str) -> str:
    """Return a single scoring completion."""
    if USE_OPENAI:
        res = _client.chat.completions.create(
            model=OPENAI_MODEL_SCORE,
            messages=[
                {"role": "system", "content": "You are a rational alignment scorer."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        return res.choices[0].message.content.strip()
    else:
        outputs = hf_pipeline(prompt, num_return_sequences=1, do_sample=False)
        return outputs[0]["generated_text"]


def _call_llm_revise(prompt: str) -> str:
    """Return a single revision completion."""
    if USE_OPENAI:
        res = _client.chat.completions.create(
            model=OPENAI_MODEL_REVISE,
            messages=[
                {"role": "system", "content": "You are a factual medical reviser."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5,
            max_tokens=512,
        )
        return res.choices[0].message.content.strip()
    else:
        outputs = hf_pipeline(prompt, num_return_sequences=1, do_sample=False)
        return outputs[0]["generated_text"].split("Improved Reasoning:")[-1].strip()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def generate_cot_variants(
    question: str,
    context: str,
    answer: str,
    task: str = "qa",
    n: int = 5,
) -> List[str]:
    """Generate K=5 candidate CoT chains for one medical question (Section III.C).

    Args:
        task: "qa" → tau_QA template; "diag" → tau_Diag template.
        n:    number of candidates (K=5 per paper).
    """
    prompt = get_reasoning_prompt(task, question, context, answer)
    try:
        return _call_llm_generate(prompt, n=n, temperature=0.7)
    except Exception as e:
        print(f"Generation failed: {e}")
        return []


def score_cot(question: str, context: str, answer: str, reasoning: str) -> Dict[str, float]:
    """Score one CoT on Coverage, Factual Accuracy, Redundancy (Section III.E.1)."""
    prompt = score_prompt(question, context, answer, reasoning)
    try:
        text = _call_llm_score(prompt)
        return parse_score_response(text)
    except Exception as e:
        print(f"Scoring failed: {e}")
        return {"coverage": 0.0, "factual_accuracy": 0.0, "redundancy": 0.0}


def revise_cot_if_needed(
    question: str,
    context: str,
    answer: str,
    reasoning: str,
    breakdown: Dict[str, float],
    task: str = "qa",
    acceptance_threshold: float = ACCEPTANCE_THRESHOLD,
) -> str:
    """Probabilistic revision gate (Section III.E.2 / Eq. 3–4).

    If P_accept(c) < threshold, trigger targeted revision starting from the
    step with the highest conditional fix probability.
    Returns the (possibly revised) reasoning string.
    """
    should_revise, probs = needs_revision(breakdown, acceptance_threshold)
    if not should_revise:
        return reasoning

    start_step = determine_start_step(probs, task=task)
    prompt = self_reflection_prompt(
        question, context, answer, reasoning, breakdown, start_step, task=task
    )
    try:
        return _call_llm_revise(prompt)
    except Exception as e:
        print(f"Revision failed: {e}")
        return reasoning
