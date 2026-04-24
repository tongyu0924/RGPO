# file: cot_rgpo_pairs_5_to_4.py
"""
RGPO data generation: Generate K=5 CoT variants per question, rank them by
multi-dimensional quality score, select the top M=4 candidates, apply
probabilistic refinement, and write RGPO-format JSONL for training.

Pipeline (Section III of the paper):
  1. Task identification  → tau_QA or tau_Diag
  2. CoT generation       → K=5 candidates per question
  3. Quality scoring      → Coverage / Factual Accuracy / Redundancy
  4. Ranking & filtering  → retain top M=4, discard lowest-ranked
  5. Probabilistic refinement → revise if P_accept(c) < theta
  6. Dataset construction → ranked JSONL for RGPO training

Usage:
    python cot_rgpo_pairs_5_to_4.py
"""
from __future__ import annotations

import json
import random

from tqdm import tqdm

from config import (
    INPUT_PATH,
    OUTPUT_PATH,
    ACCEPTANCE_THRESHOLD,
    NUM_COT_VARIANTS,
    NUM_TOP_CANDIDATES,
    SAMPLE_SIZE,
)
from generation import generate_cot_variants, score_cot, revise_cot_if_needed
from scoring import aggregate_score


# ──────────────────────────────────────────────
# Task classifier
# ──────────────────────────────────────────────

def classify_task(item: dict) -> str:
    """Simple heuristic task classifier → "diag" (tau_Diag) or "qa" (tau_QA).

    Checks for clinical-vignette signals (age, patient, symptoms) to distinguish
    MedQA-USMLE-style diagnostic questions from PubMedQA-style research QA.
    """
    q = item.get("QUESTION", "").lower()
    clinical_keywords = ("year-old", "patient", "presents", "symptoms", "diagnosis",
                         "treatment", "physical exam", "history of")
    if any(kw in q for kw in clinical_keywords):
        return "diag"
    return "qa"


# ──────────────────────────────────────────────
# Main dataset generation
# ──────────────────────────────────────────────

def generate_rgpo_dataset(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    sample_size: int = SAMPLE_SIZE,
) -> None:
    """Generate ranked RGPO preference data (5→4 selection).

    Output JSONL fields per entry:
      id         — unique identifier
      question   — medical question
      context    — supporting abstract / passage
      answer     — gold label
      ranked     — list of M=4 refined reasoning chains (best → worst)
      scores     — corresponding aggregate quality scores
      breakdowns — per-chain {coverage, factual_accuracy, redundancy}
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    random.shuffle(items)
    items = items[:sample_size]

    results = []

    for pmid, item in tqdm(items, desc="Generating RGPO CoT pairs (5→4)"):
        question = item["QUESTION"]
        answer   = item["final_decision"]
        context  = (
            " ".join(item["CONTEXTS"])
            if isinstance(item.get("CONTEXTS"), list)
            else str(item.get("CONTEXTS", ""))
        )
        task = classify_task(item)

        # ── Step 1: Generate K=5 candidate CoT chains ──────────────────────
        cot_variants = generate_cot_variants(question, context, answer, task=task, n=NUM_COT_VARIANTS)
        if len(cot_variants) < NUM_COT_VARIANTS:
            continue

        # ── Step 2: Score all K=5 candidates ───────────────────────────────
        scored: list[tuple[str, dict, float]] = []
        for cot in cot_variants:
            breakdown = score_cot(question, context, answer, cot)
            # Skip candidates where scoring failed entirely
            if all(v == 0.0 for v in breakdown.values()):
                continue
            scored.append((cot, breakdown, aggregate_score(breakdown)))

        if len(scored) < NUM_TOP_CANDIDATES:
            continue

        # ── Step 3: Rank and keep top M=4 ──────────────────────────────────
        scored.sort(key=lambda x: x[2], reverse=True)
        top_m = scored[:NUM_TOP_CANDIDATES]

        # ── Step 4: Probabilistic refinement ───────────────────────────────
        refined_reasonings: list[str]  = []
        refined_breakdowns: list[dict] = []
        refined_scores:     list[float]= []

        for cot, breakdown, score in top_m:
            revised_cot = revise_cot_if_needed(
                question, context, answer, cot, breakdown,
                task=task,
                acceptance_threshold=ACCEPTANCE_THRESHOLD,
            )
            refined_reasonings.append(revised_cot)
            refined_breakdowns.append(breakdown)
            refined_scores.append(score)

        # ── Step 5: Write RGPO training record ─────────────────────────────
        results.append({
            "id":         pmid,
            "question":   question,
            "context":    context,
            "answer":     answer,
            "ranked":     refined_reasonings,   # M=4, best → worst
            "scores":     refined_scores,
            "breakdowns": refined_breakdowns,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✓ {len(results)} RGPO ranking pairs (5→4) written to {output_path}")


# ──────────────────────────────────────────────
# Quick self-test for the probabilistic model
# ──────────────────────────────────────────────

def _demo_prob_model() -> None:
    from scoring import calculate_acceptance_probability, determine_start_step

    cases = {
        "high_quality":   {"coverage": 4.5, "factual_accuracy": 4.8, "redundancy": 0.15},
        "low_coverage":   {"coverage": 2.5, "factual_accuracy": 4.2, "redundancy": 0.40},
        "low_accuracy":   {"coverage": 4.0, "factual_accuracy": 2.2, "redundancy": 0.46},
        "high_redundancy":{"coverage": 4.2, "factual_accuracy": 4.1, "redundancy": 0.96},
    }
    for name, br in cases.items():
        probs    = calculate_acceptance_probability(**br)
        revise   = probs["acceptance_probability"] < ACCEPTANCE_THRESHOLD
        step     = determine_start_step(probs) if revise else 0
        decision = f"REVISE (step {step})" if revise else "OK"
        print(
            f"{name:>16} | C={br['coverage']:.1f} A={br['factual_accuracy']:.1f} "
            f"R={br['redundancy']:.2f} | P_accept={probs['acceptance_probability']:.3f} | {decision}"
        )


if __name__ == "__main__":
    # Uncomment the line below for a quick probabilistic-model check:
    # _demo_prob_model()
    generate_rgpo_dataset()
