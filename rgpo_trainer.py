#!/usr/bin/env python3
"""
RGPO training code — Ranking-Guided Preference Optimization for Reliable Clinical Reasoning.

Implements the full RGPO training objective (Section IV of the paper):

    L_total = L_RGPO-CoT + L_Linear                              [Eq. 13]

where:
    L_RGPO-CoT = L_CoT-rank + L_KL                               [Eq. 7 + 11]
    L_Linear   = -(1/K) Σ a_j · s_j                              [Eq. 9]
    L_CoT-rank = (1/C(K,2)) Σ_{i<j} log(1 + exp(-(s_i-s_j)/τ_BT))  [Eq. 7]
    L_KL       = β · (1/K) Σ_k DKL(c_k | z, τ)                  [Eq. 11]
"""
import csv
import json
import logging
import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class JsonlRGPODataset(Dataset):
    """Loads ranked CoT pairs from the JSONL produced by cot_rgpo_pairs_5_to_4.py."""

    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load(data_file)
        logger.info(f"Loaded {len(self.data)} RGPO training instances.")

    def _load(self, data_file: str):
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if all(k in item for k in ["question", "ranked"]):
                        data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed: {e}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build prompt (question + optional context + answer header)
        parts = [f"Question: {item['question']}"]
        if "context" in item and str(item["context"]).strip():
            parts.append(f"Context: {item['context']}")
        if "answer" in item and str(item["answer"]).strip():
            parts.append(f"Answer: {item['answer']}")
        parts.append("Detailed Analysis:")
        prompt = "\n\n".join(parts)

        enc_prompt = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = int(enc_prompt["attention_mask"][0].sum().item())

        inputs = []
        for reasoning in item["ranked"]:
            full_text = prompt + "\n" + reasoning
            enc_full = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc_full["input_ids"].squeeze(0)
            attention_mask = enc_full["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            seq_len       = int(attention_mask.sum().item())
            eff_prompt_len = min(prompt_len, seq_len)
            labels[:eff_prompt_len] = -100  # only compute log π on the generated segment

            inputs.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
                "full_text":      full_text,
            })

        return inputs


def collate_fn(batch):
    """Dynamic padding — pad each batch only to its own maximum length."""
    all_inputs = []
    max_len = 0
    for sample in batch:
        for inp in sample:
            max_len = max(max_len, len(inp["input_ids"]))
            all_inputs.append(inp)

    for inp in all_inputs:
        pad_len = max_len - len(inp["input_ids"])
        if pad_len > 0:
            pad_id = inp["input_ids"][-1].item() if len(inp["input_ids"]) > 0 else 0
            inp["input_ids"]      = torch.cat([inp["input_ids"],      torch.full((pad_len,), pad_id)])
            inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.zeros(pad_len)])
            inp["labels"]         = torch.cat([inp["labels"],         torch.full((pad_len,), -100)])

    return batch


# ──────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────

class JsonlRGPOTrainer:
    """
    RGPO trainer implementing the full objective from Section IV of the paper.

    Args:
        model_name: HuggingFace model identifier. Paper uses Gemma 2B (google/gemma-2b).
        beta:       KL-regularization weight β (Eq. 11). Paper optimal: 0.1.
        temperature: Sampling temperature for policy.
        tau_bt:     Bradley-Terry temperature τ_BT (Eq. 6-7) controlling ranking sharpness.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        device: str = "auto",
        beta: float = 0.1,
        temperature: float = 1.0,
        tau_bt: float = 1.0,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        logger.info(f"Device: {self.device}")

        self.beta        = beta
        self.temperature = temperature
        self.tau_bt      = tau_bt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model    = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        self._last_rewards: torch.Tensor | None = None

        # Loss history for plotting / CSV export
        self._global_step  = 0
        self._hist_steps:  List[int]   = []
        self._hist_total:  List[float] = []
        self._hist_linear: List[float] = []   # L_Linear (PG term)
        self._hist_kl:     List[float] = []   # L_KL
        self._hist_rank:   List[float] = []   # L_CoT-rank (pairwise / BT)

    # ── Core computations ──────────────────────────────────────────────────

    def compute_log_probs_and_kl(self, input_ids, attention_mask, labels):
        """
        Compute sequence-level log π_θ(c|z,τ) and token-averaged forward KL divergence.

        Implements Eq. 5 (preference score s_j) and Eq. 10-12 (KL term).
        """
        with torch.set_grad_enabled(self.policy_model.training):
            policy_logits = self.policy_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        with torch.no_grad():
            ref_logits = self.reference_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs    = F.log_softmax(ref_logits,    dim=-1)
        policy_probs     = F.softmax(policy_logits,     dim=-1)

        # Shift for next-token prediction
        shift_plp  = policy_log_probs[:, :-1, :]
        shift_rlp  = ref_log_probs[:, :-1, :]
        shift_pp   = policy_probs[:, :-1, :]
        shift_lbl  = labels[:, 1:]
        shift_mask = (shift_lbl != -100) & (attention_mask[:, 1:] == 1)

        valid_lbl = shift_lbl.clamp(min=0, max=shift_plp.size(-1) - 1)
        valid_cnt = shift_mask.sum(dim=-1).float() + 1e-8

        # Sequence-level log probability (Eq. 5) — averaged over generated tokens
        policy_gathered = shift_plp.gather(-1, valid_lbl.unsqueeze(-1)).squeeze(-1)
        policy_masked   = torch.where(shift_mask, policy_gathered, torch.zeros_like(policy_gathered))
        seq_log_prob    = policy_masked.sum(dim=-1) / valid_cnt

        # Token-level forward KL: KL(π_θ || π_ref) = Σ π_θ log(π_θ / π_ref)  (Eq. 12)
        kl_per_token = (shift_pp * (shift_plp - shift_rlp)).sum(dim=-1)
        kl_masked    = torch.where(shift_mask, kl_per_token, torch.zeros_like(kl_per_token))
        seq_kl       = kl_masked.sum(dim=-1) / valid_cnt

        # Numeric guards
        if torch.isnan(seq_log_prob).any() or torch.isinf(seq_log_prob).any():
            logger.warning("seq_log_prob contains NaN/Inf — using fallback.")
            seq_log_prob = torch.full_like(seq_log_prob, -5.0)

        if torch.isnan(seq_kl).any() or torch.isinf(seq_kl).any():
            logger.warning("seq_kl contains NaN/Inf — using fallback.")
            seq_kl = torch.full_like(seq_kl, 0.1)

        return seq_log_prob, seq_kl

    def _linear_rewards_from_rank(self, k: int) -> torch.Tensor:
        """
        Linear advantage values a_j = (K - r_j) - (K-1)/2  (Eq. 8).

        r_j=1 is best (index 0); advantages sum to zero.
        """
        if k <= 1:
            return torch.zeros(1, device=self.device)
        advantages = torch.arange(k - 1, -1, -1, dtype=torch.float32, device=self.device) - (k - 1) / 2.0
        return advantages

    def compute_preference_scores(self, batch_inputs):
        """Forward pass for all K candidates; returns (policy_logp [K], kl_div [K])."""
        policy_scores, kl_divs = [], []

        for inp in batch_inputs:
            input_ids      = inp["input_ids"].unsqueeze(0).to(self.device)
            attention_mask = inp["attention_mask"].unsqueeze(0).to(self.device)
            labels         = inp["labels"].unsqueeze(0).to(self.device)

            if hasattr(self.policy_model, "gradient_checkpointing_enable"):
                self.policy_model.gradient_checkpointing_enable()

            log_prob, kl = self.compute_log_probs_and_kl(input_ids, attention_mask, labels)
            policy_scores.append(log_prob)
            kl_divs.append(kl)

            del input_ids, attention_mask, labels
            torch.cuda.empty_cache()

        policy_logp = torch.stack(policy_scores).squeeze(-1)   # [K]
        kl_div      = torch.stack(kl_divs).squeeze(-1)          # [K]

        self._last_rewards = self._linear_rewards_from_rank(int(policy_logp.shape[0]))
        return policy_logp, kl_div

    def compute_rgpo_loss(self, policy_logp: torch.Tensor, kl_div: torch.Tensor):
        """
        L_RGPO-CoT = L_Linear + L_KL   (combined in this method; L_CoT-rank added in train()).

        L_Linear = -(1/K) Σ a_j · s_j  (Eq. 9)
        L_KL     = β · (1/K) Σ DKL     (Eq. 11)
        """
        advantages = (
            self._last_rewards.to(policy_logp.device)
            if self._last_rewards is not None
            else torch.zeros_like(policy_logp)
        )
        linear_loss = -(advantages * policy_logp).mean()   # L_Linear (Eq. 9)
        kl_penalty  = self.beta * kl_div.mean()            # L_KL     (Eq. 11)
        total       = linear_loss + kl_penalty
        return total, linear_loss, kl_penalty

    def compute_pairwise_loss(self, policy_logp: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry ranking loss L_CoT-rank (Eq. 7).

            L_CoT-rank = (1/C(K,2)) Σ_{i<j} log(1 + exp(-(s_i - s_j) / τ_BT))

        τ_BT controls the sharpness of the preference distribution (Eq. 6).
        """
        losses = []
        n = int(policy_logp.shape[0])
        for i in range(n):
            for j in range(i + 1, n):
                # candidate i is ranked higher than j (better), so s_i > s_j expected
                diff = (policy_logp[i] - policy_logp[j]) / self.tau_bt
                losses.append(torch.log(1.0 + torch.exp(-diff)))
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.device)

    # ── Training ───────────────────────────────────────────────────────────

    def train(
        self,
        data_file: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 5e-5,
        use_pairwise: bool = True,
        pairwise_weight: float = 0.1,
    ):
        """
        Full RGPO training loop.

        Paper settings (Section V.A):
            base model : Gemma 2B
            optimizer  : AdamW, lr=5e-5, batch_size=16, epochs=3
            β=0.1, K=5 candidates (top M=4 used), θ=0.6, τ_BT=1.0
        """
        dataset    = JsonlRGPODataset(data_file, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer  = AdamW(self.policy_model.parameters(), lr=lr, weight_decay=0.01)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.policy_model.train()
            epoch_loss, n_batches = 0.0, 0

            for step, batch in enumerate(dataloader):
                ranked_list = batch[0]

                policy_logp, kl_div = self.compute_preference_scores(ranked_list)

                if torch.isnan(policy_logp).any() or torch.isinf(policy_logp).any():
                    logger.warning("NaN/Inf in policy_logp — skipping batch.")
                    continue
                if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
                    logger.warning("NaN/Inf in kl_div — skipping batch.")
                    continue

                # L_RGPO-CoT = L_Linear + L_KL
                rgpo_loss, linear_loss, kl_penalty = self.compute_rgpo_loss(policy_logp, kl_div)
                total_loss = rgpo_loss
                rank_loss_val = 0.0

                # L_total = L_RGPO-CoT + L_CoT-rank  (Eq. 13)
                if use_pairwise:
                    rank_loss     = self.compute_pairwise_loss(policy_logp)
                    total_loss    = total_loss + pairwise_weight * rank_loss
                    rank_loss_val = float(rank_loss.item())

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning("NaN/Inf in total_loss — skipping batch.")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += float(total_loss.item())
                n_batches  += 1
                self._append_history(
                    float(total_loss.item()),
                    float(linear_loss.item()),
                    float(kl_penalty.item()),
                    rank_loss_val,
                )

                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}  Step {step}")
                    logger.info(f"  Total Loss   : {total_loss.item():.4f}")
                    logger.info(f"  L_Linear (PG): {linear_loss.item():.4f}")
                    logger.info(f"  L_KL         : {kl_penalty.item():.4f}")
                    logger.info(f"  Grad Norm    : {grad_norm:.4f}")
                    if use_pairwise:
                        logger.info(f"  L_CoT-rank   : {rank_loss_val:.4f}")
                    logger.info(f"  logp range   : [{policy_logp.min().item():.2f}, {policy_logp.max().item():.2f}]")
                    logger.info(f"  KL range     : [{kl_div.min().item():.4f}, {kl_div.max().item():.4f}]")

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

            ckpt_dir = output_path / f"checkpoint-epoch-{epoch+1}"
            ckpt_dir.mkdir(exist_ok=True)
            self.policy_model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)

        # Export loss history
        self._save_history_csv(output_path / "loss_history.csv")
        self._plot_history_png(output_path / "loss_curve.png")

        final_dir = output_path / "final_model"
        final_dir.mkdir(exist_ok=True)
        self.policy_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"RGPO training complete. Model saved to {final_dir}")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate_preferences(self, test_data_file: str) -> float:
        """
        Ranking accuracy: fraction of samples where policy log-probs follow
        the expected ranking order (best > 2nd > ... > worst).
        """
        dataset = JsonlRGPODataset(test_data_file, self.tokenizer)
        self.policy_model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for i, inputs in enumerate(dataset):
                policy_logp, _ = self.compute_preference_scores(inputs)
                is_correct = all(
                    policy_logp[j] >= policy_logp[j + 1]
                    for j in range(len(policy_logp) - 1)
                )
                correct += int(is_correct)
                total   += 1
                if i < 5:
                    logger.info(
                        f"Sample {i+1}: logp={policy_logp.detach().cpu().numpy()}  correct={is_correct}"
                    )

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Ranking Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy

    # ── History helpers ─────────────────────────────────────────────────────

    def _append_history(self, total: float, linear: float, kl: float, rank: float):
        self._global_step += 1
        self._hist_steps.append(self._global_step)
        self._hist_total.append(total)
        self._hist_linear.append(linear)
        self._hist_kl.append(kl)
        self._hist_rank.append(rank)

    def _save_history_csv(self, out_csv: Path):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "total", "linear", "kl", "cot_rank"])
            for row in zip(self._hist_steps, self._hist_total, self._hist_linear, self._hist_kl, self._hist_rank):
                w.writerow(row)
        logger.info(f"Loss CSV saved: {out_csv}")

    def _plot_history_png(self, out_png: Path, ma_window: int = 50):
        def moving_avg(x, w):
            w = max(1, int(w))
            c = np.convolve(x, np.ones(w) / w, mode="valid")
            return np.concatenate([np.full(len(x) - len(c), np.nan), c])

        steps   = np.array(self._hist_steps,  dtype=float)
        total   = np.array(self._hist_total,  dtype=float)
        linear  = np.array(self._hist_linear, dtype=float)
        kl      = np.array(self._hist_kl,     dtype=float)
        rank    = np.array(self._hist_rank,   dtype=float)

        plt.figure(figsize=(12, 7))
        for arr, label in [(total, "Total Loss"), (linear, "L_Linear"), (kl, "L_KL"), (rank, "L_CoT-rank")]:
            plt.plot(steps, arr, alpha=0.3, label=label)
            plt.plot(steps, moving_avg(arr, ma_window), linewidth=2, label=f"{label} (MA{ma_window})")

        plt.title("RGPO Training Loss Curves")
        plt.xlabel("Global Training Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info(f"Loss curve saved: {out_png}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Paper settings (Section V.A):
    #   model: Gemma 2B | optimizer: AdamW | lr=5e-5 | batch=16 | epochs=3
    #   β=0.1 | K=5 rollouts (top M=4) | acceptance threshold θ=0.6
    trainer = JsonlRGPOTrainer(
        model_name="google/gemma-2b",
        beta=0.1,          # KL regularization weight β (Fig. 2 optimal)
        temperature=1.0,
        tau_bt=1.0,        # Bradley-Terry temperature τ_BT (Eq. 6)
    )

    trainer.train(
        data_file="/content/rgpo_cot_pairs.jsonl",
        output_dir="./rgpo_output",
        epochs=3,
        batch_size=16,
        lr=5e-5,
        use_pairwise=True,
        pairwise_weight=0.1,
    )

    trainer.evaluate_preferences("/content/rgpo_cot_pairs.jsonl")
