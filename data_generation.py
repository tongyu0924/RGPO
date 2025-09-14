# file: cot_rpro_pairs_5_to_4.py
"""
RPRO data generation: Generate 5 CoT variants, rank them, select top 4 for RPRO training.
Uses probabilistic acceptance model for refinement.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import List, Dict, Tuple
from textwrap import dedent

# -----------------------------
# Runtime config
# -----------------------------
INPUT_PATH = os.getenv("INPUT_PATH", "/content/ori_pqal_2.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/content/grpo_cot_pairs.jsonl")
ACCEPTANCE_THRESHOLD = float(os.getenv("ACCEPTANCE_THRESHOLD", 0.6))  # tuned threshold

USE_OPENAI = os.getenv("OPENAI_API_KEY") is not None

if USE_OPENAI:
    # OpenAI path
    from openai import OpenAI
    client = OpenAI()
else:
    # HF fallback path
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch

    model_id = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    _ = hf_pipeline("Hello")  # warmup


# -----------------------------
# Prompt builders
# -----------------------------

def reasoning_prompt(q: str, c: str, a: str) -> str:
    return (
        "You are a biomedical QA explainer. Given the question, supporting context, and the answer, "
        "generate a 4-step reasoning chain to justify the answer. Structure the reasoning in these 4 labeled steps:\n"
        "1. Question Decomposition: Identify what the question is asking.\n"
        "2. Background Knowledge: Recall any relevant biomedical facts.\n"
        "3. Logical Connection: Link knowledge to the question logically.\n"
        "4. Final Justification: Conclude the answer with reasoning support.\n"
        "Ensure factual consistency, no hallucination.\n"
        f"\nQuestion: {q}\nContext: {c}\nAnswer: {a}\n\nReasoning:"
    )


def score_prompt(q: str, c: str, a: str, reasoning: str) -> str:
    return (
        "You are an alignment evaluator. Given a question, context, answer, and stepwise reasoning, "
        "score each step from 0-5 on these dimensions:\n"
        "- Coverage: How complete is the explanation?\n"
        "- Factual Accuracy: Does it align with the context and medical knowledge?\n"
        "- Redundancy: Is the step repetitive or verbose?\n"
        f"\nQuestion: {q}\nContext: {c}\nAnswer: {a}\nReasoning: {reasoning}\n"
        "Provide the result in the format: Coverage: x\nFactual Accuracy: y\nRedundancy: z"
    )


def self_reflection_prompt(q: str, c: str, a: str, reasoning: str, breakdown: Dict[str, float], start_step: int) -> str:
    return dedent(f"""
    You are a biomedical QA assistant. Given a flawed reasoning and feedback, rewrite it to fix the issues.
    Avoid hallucinated medical terms.

    Question: {q}
    Context: {c}
    Answer: {a}

    Flawed Reasoning: {reasoning}

    Issues: Coverage={breakdown['coverage']}, Factual Accuracy={breakdown['factual_accuracy']}, Redundancy={breakdown['redundancy']}

    Revise starting from Step {start_step}. Preserve earlier steps; only minimally adjust them if required for consistency.

    Output exactly 4 labeled steps (1..4):
    1. Question Decomposition
    2. Background Knowledge
    3. Logical Connection
    4. Final Justification

    Ensure factual accuracy (Does it align with the context and medical knowledge?) and avoid hallucinations.

    Improved Reasoning:
    """).strip()


# -----------------------------
# Generation & scoring helpers
# -----------------------------

def generate_cot_variants(question: str, context: str, answer: str, n: int = 5) -> List[str]:
    prompt = reasoning_prompt(question, context, answer)
    try:
        if USE_OPENAI:
            res = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_GENERATE", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=512,
                n=n,
            )
            return [choice.message.content.strip() for choice in res.choices]
        else:
            outputs = hf_pipeline(prompt, num_return_sequences=n, do_sample=True, temperature=0.7)
            return [out["generated_text"].split("Reasoning:")[-1].strip() for out in outputs]
    except Exception as e:
        print(f"Generation failed: {e}")
        return []


def gpt_judge_score(coverage: float, factual_accuracy: float, redundancy: float) -> float:
    # Legacy aggregate retained for metadata/debugging
    return coverage + factual_accuracy - redundancy


def parse_score_response(text: str) -> Dict[str, float]:
    """Parse model scoring output. Accepts 'Factual Accuracy' or legacy 'Faithfulness'."""
    try:
        matches = re.findall(
            r"(Coverage|Faithfulness|Factual\s*Accuracy|Redundancy)\s*:\s*([0-5](?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        out = {"coverage": 0.0, "factual_accuracy": 0.0, "redundancy": 0.0}
        for key, val in matches:
            label = key.strip().lower()
            score = float(val)
            if label == "coverage":
                out["coverage"] = score
            elif label == "redundancy":
                out["redundancy"] = score
            elif label == "faithfulness" or label.startswith("factual"):
                out["factual_accuracy"] = score
        return out
    except Exception as e:
        print(f"Score parse failed: {e}\nRaw output: {text[:200]}")
        return {"coverage": 0.0, "factual_accuracy": 0.0, "redundancy": 0.0}


def gpt_score_components(q: str, c: str, a: str, reasoning: str) -> Dict[str, float]:
    prompt = score_prompt(q, c, a, reasoning)
    try:
        if USE_OPENAI:
            res = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_SCORE", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a rational alignment scorer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            text = res.choices[0].message.content.strip()
        else:
            outputs = hf_pipeline(prompt, num_return_sequences=1, do_sample=False)
            text = outputs[0]["generated_text"]
        return parse_score_response(text)
    except Exception as e:
        print(f"GPT-Judge score breakdown failed: {e}")
        return {"coverage": 0.0, "factual_accuracy": 0.0, "redundancy": 0.0}


# -----------------------------
# Probabilistic decision model
# -----------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def calculate_joint_probability(coverage: float, factual_accuracy: float, redundancy: float) -> Dict[str, float]:
    """Probabilistic acceptance model based on normalized scores.

    Math:
    1) normalize: s/5 → [0,1]
    2) P(C∩A) = P(C) × P(A)
    3) P(¬R) = 1 - P(R)
    4) P(accept) = P(C∩A) × P(¬R)
    5) Conditional fix probs: 1-P(metric) for positive metrics; P(R) for redundancy.
    """
    coverage_norm = _clip01(coverage / 5.0)
    factual_accuracy_norm = _clip01(factual_accuracy / 5.0)
    redundancy_norm = _clip01(redundancy / 5.0)

    joint_positive = coverage_norm * factual_accuracy_norm
    anti_redundancy = 1.0 - redundancy_norm
    acceptance_probability = joint_positive * anti_redundancy

    p_fix_given_coverage = 1.0 - coverage_norm
    p_fix_given_accuracy = 1.0 - factual_accuracy_norm
    p_fix_given_redundancy = redundancy_norm

    return {
        "acceptance_probability": acceptance_probability,
        "joint_positive": joint_positive,
        "anti_redundancy": anti_redundancy,
        "p_fix_given_coverage": p_fix_given_coverage,
        "p_fix_given_accuracy": p_fix_given_accuracy,
        "p_fix_given_redundancy": p_fix_given_redundancy,
        "coverage_norm": coverage_norm,
        "factual_accuracy_norm": factual_accuracy_norm,
        "redundancy_norm": redundancy_norm,
    }


def determine_start_step_probabilistic(probabilities: Dict[str, float]) -> int:
    """Choose the primary fix by the largest conditional fix probability.
    mapping: coverage→1, accuracy→2, redundancy→3
    """
    fix_probs = {
        "coverage": probabilities["p_fix_given_coverage"],
        "accuracy": probabilities["p_fix_given_accuracy"],
        "redundancy": probabilities["p_fix_given_redundancy"],
    }
    primary_issue = max(fix_probs.items(), key=lambda kv: kv[1])[0]
    return {"coverage": 1, "accuracy": 2, "redundancy": 3}[primary_issue]


# -----------------------------
# Revision (probability-driven)
# -----------------------------

def revise_reasoning_if_needed(
    q: str,
    c: str,
    a: str,
    reasoning: str,
    breakdown: Dict[str, float],
    acceptance_threshold: float = ACCEPTANCE_THRESHOLD,
) -> str:
    """Probability-driven revision decision.

    If P(accept) < threshold → revise, starting from argmax conditional-fix probability.
    """
    probabilities = calculate_joint_probability(
        breakdown.get("coverage", 0.0),
        breakdown.get("factual_accuracy", 0.0),
        breakdown.get("redundancy", 0.0),
    )

    needs_revision = probabilities["acceptance_probability"] < acceptance_threshold
    if not needs_revision:
        return reasoning

    start_step = determine_start_step_probabilistic(probabilities)

    prompt = self_reflection_prompt(q, c, a, reasoning, breakdown, start_step)
    try:
        if USE_OPENAI:
            res = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_REVISE", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a factual medical reviser."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=512,
            )
            return res.choices[0].message.content.strip()
        else:
            outputs = hf_pipeline(prompt, num_return_sequences=1, do_sample=False)
            return outputs[0]["generated_text"].split("Improved Reasoning:")[-1].strip()
    except Exception as e:
        print(f"Revision failed: {e}")
        return reasoning


# -----------------------------
# Dataset generation (RPRO 5→4)
# -----------------------------

def generate_rpro_dataset(input_path: str, output_path: str, sample_size: int = 900) -> None:
    from tqdm import tqdm

    with open(input_path) as f:
        data = json.load(f)

    items = list(data.items())
    random.shuffle(items)
    items = items[:sample_size]

    results = []
    for pmid, item in tqdm(items, desc="Generating RPRO CoT pairs (5→4)"):
        q = item["QUESTION"]
        a = item["final_decision"]
        context_full = " ".join(item["CONTEXTS"]) if isinstance(item.get("CONTEXTS"), list) else str(item.get("CONTEXTS", ""))

        # Generate 5 CoT variants
        cot_variants = generate_cot_variants(q, context_full, a, n=5)
        if len(cot_variants) < 5:
            continue

        # Score all 5 variants
        scored_variants = []
        for i, cot in enumerate(cot_variants):
            breakdown = gpt_score_components(q, context_full, a, cot)
            score = gpt_judge_score(**breakdown)
            
            # Skip if breakdown parsing failed completely
            if (breakdown["coverage"] == 0.0 and 
                breakdown["factual_accuracy"] == 0.0 and 
                breakdown["redundancy"] == 0.0):
                continue
                
            scored_variants.append((cot, breakdown, score))

        # Need at least 4 valid variants
        if len(scored_variants) < 4:
            continue

        # Sort by score (highest first) and take top 4
        scored_variants.sort(key=lambda x: x[2], reverse=True)
        top_4 = scored_variants[:4]

        # Apply probabilistic refinement to all top 4
        refined_variants = []
        refined_breakdowns = []
        refined_scores = []
        
        for cot, breakdown, score in top_4:
            refined_cot = revise_reasoning_if_needed(
                q, context_full, a, cot, breakdown, acceptance_threshold=ACCEPTANCE_THRESHOLD
            )
            refined_variants.append(refined_cot)
            refined_breakdowns.append(breakdown)
            refined_scores.append(score)

        results.append({
            "id": pmid,
            "question": q,
            "context": context_full,
            "answer": a,
            "ranked": refined_variants,  # Top 4 refined reasoning chains, ranked by quality
            "scores": refined_scores,    # Corresponding aggregate scores
            "breakdowns": refined_breakdowns  # Corresponding detailed breakdowns
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(results)} RPRO ranking pairs (5→4) written to {output_path}")


# -----------------------------
# Quick self-test (optional)
# -----------------------------

def _demo_prob_model() -> None:
    cases = {
        "high_quality": {"coverage": 4.5, "factual_accuracy": 4.8, "redundancy": 1.5},
        "low_coverage": {"coverage": 2.5, "factual_accuracy": 4.2, "redundancy": 2.0},
        "low_accuracy": {"coverage": 4.0, "factual_accuracy": 2.2, "redundancy": 2.3},
        "high_redundancy": {"coverage": 4.2, "factual_accuracy": 4.1, "redundancy": 4.8},
    }
    for name, br in cases.items():
        probs = calculate_joint_probability(br["coverage"], br["factual_accuracy"], br["redundancy"])
        needs = probs["acceptance_probability"] < ACCEPTANCE_THRESHOLD
        step = determine_start_step_probabilistic(probs) if needs else 0
        print(
            f"{name:>14} | C={br['coverage']:.1f} A={br['factual_accuracy']:.1f} R={br['redundancy']:.1f} | "
            f"P(acc)={probs['acceptance_probability']:.3f} | decision={'REV' if needs else 'OK'}"
            + (f" | start_step={step}" if needs else "")
        )


if __name__ == "__main__":
    # Run RPRO dataset generation by default; toggle to _demo_prob_model() for a quick check.
    # _demo_prob_model()
    generate_rpro_dataset(INPUT_PATH, OUTPUT_PATH)
