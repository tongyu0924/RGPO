# file: scoring.py
"""
Multi-dimensional quality scoring and probabilistic refinement decision.

Implements the acceptance function P_accept(c) from Section III.E of the paper:

    P_accept(c) = p_cov(c) · p_fact(c) · (1 − p_red(c))          [Eq. 3]

Scoring dimensions (Supplementary Section S3):
  - Coverage      : 0–5 scale, higher = better
  - Factual Accuracy : 0–5 scale, higher = better
  - Redundancy    : 0–1 scale, lower = better
"""
from __future__ import annotations

import re
from typing import Dict, Tuple


# ──────────────────────────────────────────────
# Score parsing
# ──────────────────────────────────────────────

def parse_score_response(text: str) -> Dict[str, float]:
    """Parse LLM-judge output into {coverage, factual_accuracy, redundancy}.

    Accepts 'Factual Accuracy' or legacy 'Faithfulness' labels.
    """
    try:
        matches = re.findall(
            r"(Coverage|Faithfulness|Factual\s*Accuracy|Redundancy)\s*:\s*([0-5](?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        out: Dict[str, float] = {"coverage": 0.0, "factual_accuracy": 0.0, "redundancy": 0.0}
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


def aggregate_score(breakdown: Dict[str, float]) -> float:
    """Legacy aggregate score used for ranking candidates before probabilistic filtering.

    Higher score = better candidate.
    """
    return breakdown["coverage"] + breakdown["factual_accuracy"] - breakdown["redundancy"]


# ──────────────────────────────────────────────
# Probabilistic acceptance model  (Eq. 3–4)
# ──────────────────────────────────────────────

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def calculate_acceptance_probability(
    coverage: float,
    factual_accuracy: float,
    redundancy: float,
) -> Dict[str, float]:
    """Compute P_accept(c) and per-dimension fix probabilities.

    Normalization:
      p_cov  = coverage / 5          ∈ [0, 1]
      p_fact = factual_accuracy / 5  ∈ [0, 1]
      p_red  = redundancy / 1        ∈ [0, 1]   (already 0–1 scale per paper)

    Acceptance (Eq. 3):
      P_accept = p_cov · p_fact · (1 − p_red)

    Conditional fix probabilities guide which step to revise first:
      p_fix_coverage   = 1 − p_cov
      p_fix_accuracy   = 1 − p_fact
      p_fix_redundancy = p_red
    """
    p_cov  = _clip01(coverage / 5.0)
    p_fact = _clip01(factual_accuracy / 5.0)
    p_red  = _clip01(redundancy)          # already on 0–1 scale (Section S3)

    p_accept = p_cov * p_fact * (1.0 - p_red)

    return {
        "acceptance_probability": p_accept,
        "p_cov":  p_cov,
        "p_fact": p_fact,
        "p_red":  p_red,
        "p_fix_coverage":   1.0 - p_cov,
        "p_fix_accuracy":   1.0 - p_fact,
        "p_fix_redundancy": p_red,
    }


def determine_start_step(probs: Dict[str, float], task: str = "qa") -> int:
    """Return the step index (1–4) where revision should begin.

    Picks the dimension with the highest conditional fix probability.
    Step mapping per task type:

    tau_QA (qa):
      coverage   → Step 1 (Question Decomposition)
      accuracy   → Step 2 (Background Knowledge)
      redundancy → Step 3 (Logical Connection)

    tau_Diag (diag):
      coverage   → Step 1 (Case Summary)
      accuracy   → Step 2 (Clinical Significance)
      redundancy → Step 3 (Differential Diagnosis)
    """
    fix_probs = {
        "coverage":   probs["p_fix_coverage"],
        "accuracy":   probs["p_fix_accuracy"],
        "redundancy": probs["p_fix_redundancy"],
    }
    primary_issue = max(fix_probs, key=fix_probs.__getitem__)
    return {"coverage": 1, "accuracy": 2, "redundancy": 3}[primary_issue]


def needs_revision(breakdown: Dict[str, float], threshold: float) -> Tuple[bool, Dict[str, float]]:
    """Return (should_revise, full probability dict) for a candidate reasoning chain."""
    probs = calculate_acceptance_probability(
        breakdown.get("coverage", 0.0),
        breakdown.get("factual_accuracy", 0.0),
        breakdown.get("redundancy", 0.0),
    )
    return probs["acceptance_probability"] < threshold, probs
