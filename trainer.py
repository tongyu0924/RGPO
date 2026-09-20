#!/usr/bin/env python3
"""
RGPO trainer: ties together the dataset, the policy/reference models,
and the loss terms defined in losses.py into a training and
evaluation loop.
"""
import csv
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DEFAULT_TAU_BT
from dataset import JsonlGRPODataset, collate_fn
from losses import compute_log_probs_and_kl, compute_rgpo_losses

logger = logging.getLogger(__name__)


class JsonlGRPOTrainer:
    def __init__(self, model_name: str = "gpt2", device: str = "auto",
                 beta: float = 0.1, temperature: float = 1.0,
                 tau_bt: float = DEFAULT_TAU_BT,
                 hf_token: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        logger.info(f"Using device: {self.device}")

        self.beta = beta
        self.temperature = temperature
        self.tau_bt = tau_bt

        tok_kwargs = {}
        if hf_token:
            tok_kwargs["token"] = hf_token
            tok_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token
            model_kwargs["trust_remote_code"] = True
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(self.device)

        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        # History for loss curves / CSV export.
        self._global_step = 0
        self._hist_steps: List[int] = []
        self._hist_total: List[float] = []
        self._hist_linear: List[float] = []
        self._hist_kl: List[float] = []
        self._hist_rank: List[float] = []

    def compute_preference_scores(self, batch_inputs):
        """Run each of the K ranked candidates through the policy and
        reference models, returning per-candidate log-probabilities
        (Eq. 5) and KL divergences (Eq. 10/12)."""
        policy_scores = []
        kl_divs = []

        for inp in batch_inputs:
            input_ids = inp['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = inp['attention_mask'].unsqueeze(0).to(self.device)
            labels = inp['labels'].unsqueeze(0).to(self.device)

            if hasattr(self.policy_model, 'gradient_checkpointing_enable'):
                self.policy_model.gradient_checkpointing_enable()

            policy_log_prob, kl_div = compute_log_probs_and_kl(
                self.policy_model, self.reference_model, input_ids, attention_mask, labels
            )

            policy_scores.append(policy_log_prob)
            kl_divs.append(kl_div)

            del input_ids, attention_mask, labels
            torch.cuda.empty_cache()

        policy_scores = torch.stack(policy_scores).squeeze(-1).to(self.device)  # [K]
        kl_divs = torch.stack(kl_divs).squeeze(-1).to(self.device)              # [K]
        return policy_scores, kl_divs

    def _append_history(self, total_loss: float, linear_loss: float, kl_loss: float, rank_loss: float):
        self._global_step += 1
        self._hist_steps.append(self._global_step)
        self._hist_total.append(float(total_loss))
        self._hist_linear.append(float(linear_loss))
        self._hist_kl.append(float(kl_loss))
        self._hist_rank.append(float(rank_loss))

    def _save_history_csv(self, out_csv: Path):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["step", "total", "linear", "kl", "rank"])
            for s, t, lin, k, r in zip(self._hist_steps, self._hist_total, self._hist_linear,
                                        self._hist_kl, self._hist_rank):
                w.writerow([s, t, lin, k, r])
        logger.info(f"Saved loss history CSV: {out_csv}")

    def _plot_history_png(self, out_png: Path, ma_window: int = 50):
        steps = np.array(self._hist_steps, dtype=float)
        total = np.array(self._hist_total, dtype=float)
        linear = np.array(self._hist_linear, dtype=float)
        kl = np.array(self._hist_kl, dtype=float)
        rank = np.array(self._hist_rank, dtype=float)

        def moving_avg(x, w):
            if len(x) < 1:
                return x
            w = max(1, int(w))
            c = np.convolve(x, np.ones(w) / w, mode='valid')
            pad = np.full((len(x) - len(c),), np.nan)
            return np.concatenate([pad, c])

        total_ma = moving_avg(total, ma_window)
        linear_ma = moving_avg(linear, ma_window)
        kl_ma = moving_avg(kl, ma_window)
        rank_ma = moving_avg(rank, ma_window)

        plt.figure(figsize=(12, 7))
        plt.plot(steps, total, alpha=0.3, label='Total Loss (L_total)')
        plt.plot(steps, linear, alpha=0.3, label='Linear Reward Loss (L_Linear)')
        plt.plot(steps, kl, alpha=0.3, label='KL Penalty (L_KL)')
        plt.plot(steps, rank, alpha=0.3, label='Ranking Loss (L_CoT-rank)')

        plt.plot(steps, total_ma, linestyle='-', linewidth=2, label='Total (MA50)')
        plt.plot(steps, linear_ma, linestyle='-', linewidth=2, label='Linear (MA50)')
        plt.plot(steps, kl_ma, linestyle='-', linewidth=2, label='KL (MA50)')
        plt.plot(steps, rank_ma, linestyle='-', linewidth=2, label='Ranking (MA50)')

        plt.title('RGPO Training Loss Curves')
        plt.xlabel('Global Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info(f"Saved loss curve plot: {out_png}")

    def train(self, data_file: str, output_dir: str, epochs: int = 3, batch_size: int = 1,
              lr: float = 1e-5, use_pairwise: bool = True):
        dataset = JsonlGRPODataset(data_file, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        optimizer = AdamW(self.policy_model.parameters(), lr=lr, weight_decay=0.01)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.policy_model.train()
            epoch_total_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(dataloader):
                ranked_list = batch[0]
                policy_logp, kl_div = self.compute_preference_scores(ranked_list)

                if torch.isnan(policy_logp).any() or torch.isinf(policy_logp).any():
                    logger.warning("NaN/Inf found in policy_logp, skipping this batch")
                    continue
                if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
                    logger.warning("NaN/Inf found in kl_div, skipping this batch")
                    continue

                # Eq. (13): L_total = L_CoT-rank + L_KL + L_Linear
                total_loss_step, rank_loss, kl_loss, linear_loss = compute_rgpo_losses(
                    policy_logp, kl_div, beta=self.beta, tau_bt=self.tau_bt, use_pairwise=use_pairwise
                )

                if torch.isnan(total_loss_step) or torch.isinf(total_loss_step):
                    logger.warning("Abnormal total loss detected, skipping this batch")
                    continue

                optimizer.zero_grad()
                total_loss_step.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_total_loss += float(total_loss_step.item())
                num_batches += 1

                self._append_history(
                    float(total_loss_step.item()), float(linear_loss.item()),
                    float(kl_loss.item()), float(rank_loss.item())
                )

                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Step {step}")
                    logger.info(f"  Total Loss: {total_loss_step.item():.4f}")
                    logger.info(f"  Linear Reward Loss: {linear_loss.item():.4f}")
                    logger.info(f"  KL Penalty: {kl_loss.item():.4f}")
                    logger.info(f"  Ranking Loss: {rank_loss.item():.4f}")
                    logger.info(f"  Grad Norm: {grad_norm:.4f}")
                    logger.info(f"  Policy logp range: [{policy_logp.min().item():.2f}, {policy_logp.max().item():.2f}]")
                    logger.info(f"  KL div range: [{kl_div.min().item():.4f}, {kl_div.max().item():.4f}]")

            avg_loss = epoch_total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} done, average loss: {avg_loss:.4f}")

            checkpoint_dir = output_path / f"checkpoint-epoch-{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            self.policy_model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

        csv_path = Path(output_dir) / "loss_history.csv"
        png_path = Path(output_dir) / "loss_curve.png"
        self._save_history_csv(csv_path)
        self._plot_history_png(png_path, ma_window=50)

        final_dir = Path(output_dir) / "final_model"
        final_dir.mkdir(exist_ok=True)
        self.policy_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"RGPO training complete, model saved to {final_dir}")

    def evaluate_preferences(self, test_data_file: str):
        dataset = JsonlGRPODataset(test_data_file, self.tokenizer)
        self.policy_model.eval()

        correct_rankings = 0
        total_rankings = 0

        with torch.no_grad():
            for i, inputs in enumerate(dataset):
                policy_logp, _ = self.compute_preference_scores(inputs)
                is_correctly_ranked = all(
                    policy_logp[j] >= policy_logp[j + 1] for j in range(len(policy_logp) - 1)
                )
                if is_correctly_ranked:
                    correct_rankings += 1
                total_rankings += 1
                if i < 5:
                    logger.info(
                        f"Sample {i+1}: logp = {policy_logp.detach().cpu().numpy()}, Correct = {is_correctly_ranked}"
                    )

        accuracy = correct_rankings / total_rankings if total_rankings > 0 else 0.0
        logger.info(f"Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
        return accuracy
