# file: prompts.py
"""
Task-adaptive prompt templates for RGPO data generation.

Two task types are supported (Section III.B–III.D of the paper):
  - tau_QA   : General medical question answering (e.g. PubMedQA)
  - tau_Diag : Clinical diagnostic reasoning      (e.g. MedQA-USMLE)

Prompt wording is aligned with the Supplementary Material (Section S2).
"""
from __future__ import annotations

from textwrap import dedent


# ──────────────────────────────────────────────
# tau_QA  — General Medical Reasoning (PubMedQA)
# ──────────────────────────────────────────────

def reasoning_prompt_qa(q: str, c: str, a: str) -> str:
    """4-step CoT template for general medical QA (tau_QA).

    Steps: Question Decomposition → Background Knowledge →
           Logical Connection → Final Justification.
    Aligned with Supplementary Material Section S2, General Medical Reasoning template.
    """
    return (
        "You are a medical reasoning expert. Based on the question, context, and final answer, "
        "produce a 4-step reasoning process:\n"
        "1. Question Decomposition: Identify what the question is asking.\n"
        "2. Background Knowledge: Recall relevant biomedical facts.\n"
        "3. Logical Connection: Link knowledge to the question logically.\n"
        "4. Final Justification: Conclude the answer with reasoning support.\n"
        "Ensure factual consistency, no hallucination.\n"
        f"\nQuestion: {q}\nContext: {c}\nAnswer: {a}\n\nReasoning:"
    )


# ──────────────────────────────────────────────
# tau_Diag — Clinical Diagnostic Reasoning (MedQA-USMLE)
# ──────────────────────────────────────────────

def reasoning_prompt_diag(q: str, c: str, a: str) -> str:
    """4-step CoT template for clinical diagnostic reasoning (tau_Diag).

    Steps: Case Summary → Clinical Significance →
           Differential Diagnosis → Most Likely Diagnosis.
    Aligned with Supplementary Material Section S2, Clinical Diagnostic Reasoning template.
    """
    return (
        "You are a clinical reasoning expert. Given the patient case and final diagnosis, "
        "produce a 4-step diagnostic reasoning process:\n"
        "1. Case Summary: Summarize the key patient information.\n"
        "2. Clinical Significance: Explain the important findings.\n"
        "3. Differential Diagnosis: Consider possible alternatives.\n"
        "4. Most Likely Diagnosis: Justify the final diagnosis.\n"
        "Ensure medical accuracy, no hallucination.\n"
        f"\nQuestion: {q}\nContext: {c}\nAnswer: {a}\n\nReasoning:"
    )


def get_reasoning_prompt(task: str, q: str, c: str, a: str) -> str:
    """Dispatch to task-appropriate reasoning template.

    Args:
        task: "qa" for tau_QA, "diag" for tau_Diag.
    """
    if task == "diag":
        return reasoning_prompt_diag(q, c, a)
    return reasoning_prompt_qa(q, c, a)


# ──────────────────────────────────────────────
# Scoring prompt  (Section III.E / Supplementary S3)
# ──────────────────────────────────────────────

def score_prompt(q: str, c: str, a: str, reasoning: str) -> str:
    """LLM-judge prompt for multi-dimensional quality scoring.

    Dimensions (Supplementary Section S3):
      - Coverage      (0–5): completeness of key clinical/biomedical aspects
      - Factual Accuracy (0–5): correctness vs. medical knowledge and context
      - Redundancy    (0–1): unnecessary repetition (lower = better)
    """
    return (
        "You are an alignment evaluator. Given a question, context, answer, and stepwise reasoning, "
        "score the reasoning on these three dimensions:\n"
        "- Coverage (0–5): How comprehensively does the reasoning address the key clinical or biomedical aspects?\n"
        "- Factual Accuracy (0–5): Are the statements and inferences medically or scientifically correct?\n"
        "- Redundancy (0–1): Does the reasoning contain unnecessary repetition or irrelevant content? "
        "(0 = highly concise; 1 = severe redundancy)\n"
        f"\nQuestion: {q}\nContext: {c}\nAnswer: {a}\nReasoning: {reasoning}\n"
        "Provide the result in the format:\n"
        "Coverage: x\nFactual Accuracy: y\nRedundancy: z"
    )


# ──────────────────────────────────────────────
# Self-reflection / revision prompt
# ──────────────────────────────────────────────

def self_reflection_prompt(
    q: str,
    c: str,
    a: str,
    reasoning: str,
    breakdown: dict,
    start_step: int,
    task: str = "qa",
) -> str:
    """Targeted revision prompt triggered when P_accept < threshold.

    start_step is determined by the largest conditional fix probability
    (Section III.E.2 of the paper).
    Step labels adapt to task type so the reviser knows the correct structure.
    """
    if task == "diag":
        step_labels = (
            "1. Case Summary\n"
            "2. Clinical Significance\n"
            "3. Differential Diagnosis\n"
            "4. Most Likely Diagnosis"
        )
    else:
        step_labels = (
            "1. Question Decomposition\n"
            "2. Background Knowledge\n"
            "3. Logical Connection\n"
            "4. Final Justification"
        )

    return dedent(f"""
    You are a biomedical QA assistant. Given a flawed reasoning and feedback, rewrite it to fix the issues.
    Avoid hallucinated medical terms.

    Question: {q}
    Context: {c}
    Answer: {a}

    Flawed Reasoning: {reasoning}

    Issues: Coverage={breakdown['coverage']}, Factual Accuracy={breakdown['factual_accuracy']}, Redundancy={breakdown['redundancy']}

    Revise starting from Step {start_step}. Preserve earlier steps; only minimally adjust them if required for consistency.

    Output exactly 4 labeled steps:
    {step_labels}

    Ensure factual accuracy and avoid hallucinations.

    Improved Reasoning:
    """).strip()
