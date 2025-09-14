#!/usr/bin/env python3
"""
PRPO training code
"""
import torch
import torch.nn.functional as F
import json
import os
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JsonlRPRODataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int = 512): 
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_jsonl_data(data_file)
        logger.info(f"Loaded {len(self.data)} RPRO data")

    def load_jsonl_data(self, data_file: str):
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if all(k in item for k in ['question', 'ranked']):
                        data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed: {e}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_parts = [f"Question: {item['question']}"]
        if 'context' in item and str(item['context']).strip():
            prompt_parts.append(f"Context: {item['context']}")
        if 'answer' in item and str(item['answer']).strip():
            prompt_parts.append(f"Answer: {item['answer']}")
        prompt_parts.append("Detailed Analysis:")
        prompt = "\n\n".join(prompt_parts)

        enc_prompt = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,  
            truncation=True,
            return_tensors='pt'
        )
        prompt_len = int(enc_prompt['attention_mask'][0].sum().item())

        inputs = []
        for reasoning in item['ranked']:
            full_text = prompt + "\n" + reasoning
            enc_full = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding=False,  # 動態padding
                truncation=True,
                return_tensors='pt'
            )
            input_ids = enc_full['input_ids'].squeeze(0)
            attention_mask = enc_full['attention_mask'].squeeze(0)

            labels = input_ids.clone()
            seq_len = int(attention_mask.sum().item())
            eff_prompt_len = min(prompt_len, seq_len)
            labels[:eff_prompt_len] = -100  # Do logπ on the generated segment

            inputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'full_text': full_text,
            })

        return inputs


def collate_fn(batch):
    # Dynamic padding, only padding to the maximum length of the current batch
    max_len = 0
    all_inputs = []
    
    for sample in batch:
        for inp in sample:
            max_len = max(max_len, len(inp['input_ids']))
            all_inputs.append(inp)
    
    # Maximum length of Pad to batch
    for inp in all_inputs:
        pad_len = max_len - len(inp['input_ids'])
        if pad_len > 0:
            pad_token_id = inp['input_ids'][-1] if len(inp['input_ids']) > 0 else 0
            inp['input_ids'] = torch.cat([inp['input_ids'], torch.full((pad_len,), pad_token_id)])
            inp['attention_mask'] = torch.cat([inp['attention_mask'], torch.zeros(pad_len)])
            inp['labels'] = torch.cat([inp['labels'], torch.full((pad_len,), -100)])
    
    return batch


class JsonlRPROTrainer:
    def __init__(self, model_name: str = "gpt2", device: str = "auto",
                 beta: float = 0.1, temperature: float = 1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        logger.info(f"Equipment used: {self.device}")

        self.beta = beta
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad = False

        self._last_rewards: torch.Tensor | None = None

        # Record loss for plotting
        self._global_step = 0
        self._hist_steps: List[int] = []
        self._hist_total: List[float] = []
        self._hist_pg: List[float] = []     # Used as Ranking Loss line
        self._hist_kl: List[float] = []
        self._hist_pw: List[float] = []

    def compute_log_probs_and_kl(self, policy_model, reference_model, input_ids, attention_mask, labels):
        """
        Calculate log probability and token-level forward KL divergence
        """
        with torch.set_grad_enabled(policy_model.training):
            policy_outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
            policy_logits = policy_outputs.logits
            
        with torch.no_grad():
            reference_outputs = reference_model(input_ids=input_ids, attention_mask=attention_mask)
            reference_logits = reference_outputs.logits
        
        # Calculate log_softmax and softmax, numerical stability
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Shift for next token prediction
        shift_policy_log_probs = policy_log_probs[:, :-1, :]
        shift_reference_log_probs = reference_log_probs[:, :-1, :]
        shift_policy_probs = policy_probs[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = (shift_labels != -100) & (attention_mask[:, 1:] == 1)

        # Calculate the average log probability of the generated segment of the sequence (to avoid length bias)
        valid_labels = shift_labels.clamp(min=0, max=shift_policy_log_probs.size(-1)-1)
        
        policy_gathered = shift_policy_log_probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
        policy_masked = torch.where(shift_mask, policy_gathered, torch.zeros_like(policy_gathered))
        
        reference_gathered = shift_reference_log_probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
        reference_masked = torch.where(shift_mask, reference_gathered, torch.zeros_like(reference_gathered))
        
        # Calculate the number of valid tokens for averaging
        valid_token_count = shift_mask.sum(dim=-1).float() + 1e-8
        
        # Average log probability to avoid length bias
        seq_policy_log_probs = policy_masked.sum(dim=-1) / valid_token_count
        seq_reference_log_probs = reference_masked.sum(dim=-1) / valid_token_count
        
        # Calculate forward KL divergence: KL(π_θ || π_ref) = Σ π_θ(a|s) * (log π_θ(a|s) - log π_ref(a|s))
        # Calculate KL divergence on the generated token
        kl_per_token = torch.sum(
            shift_policy_probs * (shift_policy_log_probs - shift_reference_log_probs), 
            dim=-1
        )  # [batch, seq_len-1]
        
        # Calculate KL on the generated tokens and take the average
        kl_masked = torch.where(shift_mask, kl_per_token, torch.zeros_like(kl_per_token))
        seq_kl_div = kl_masked.sum(dim=-1) / valid_token_count  # [batch] - 平均KL散度
        
        # Check calculation results
        if torch.isnan(seq_policy_log_probs).any() or torch.isinf(seq_policy_log_probs).any():
            logger.warning("seq_policy_log_probs contains NaN/Inf, using fallback value")
            seq_policy_log_probs = torch.full_like(seq_policy_log_probs, -5.0)  
            
        if torch.isnan(seq_kl_div).any() or torch.isinf(seq_kl_div).any():
            logger.warning("seq_kl_div contains NaN/Inf, using fallback value")
            seq_kl_div = torch.full_like(seq_kl_div, 0.1)  
                
        return seq_policy_log_probs, seq_kl_div

    def _linear_rewards_from_rank(self, k: int) -> torch.Tensor:
        if k <= 1:
            return torch.zeros(1, device=self.device)  # 若只有一個候選樣本，沒有排名可比，優勢值設為0
        # 將排名轉換為優勢函數，排名越高（越好）的樣本，優勢值越大
        # 公式：advantage = (k-1-i) - (k-1)/2
        # - (k-1-i)：讓最佳樣本分數最高，最差樣本分數最低
        # - 減去 (k-1)/2：確保所有樣本的平均值為0，避免偏置
        advantages = torch.arange(k-1, -1, -1, dtype=torch.float32, device=self.device) - (k-1)/2
        return advantages

    def compute_preference_scores(self, batch_inputs):
        policy_scores = []
        kl_divs = []

        # 使用gradient accumulation處理更大的batch
        for inp in batch_inputs:
            input_ids = inp['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = inp['attention_mask'].unsqueeze(0).to(self.device)
            labels = inp['labels'].unsqueeze(0).to(self.device)

            # 使用gradient checkpointing節省內存
            if hasattr(self.policy_model, 'gradient_checkpointing_enable'):
                self.policy_model.gradient_checkpointing_enable()

            policy_log_prob, kl_div = self.compute_log_probs_and_kl(
                self.policy_model, self.reference_model, input_ids, attention_mask, labels
            )

            policy_scores.append(policy_log_prob)  
            kl_divs.append(kl_div)
            
            # 清理顯存
            del input_ids, attention_mask, labels
            torch.cuda.empty_cache()

        policy_scores = torch.stack(policy_scores).squeeze(-1).to(self.device)   # [K]
        kl_divs = torch.stack(kl_divs).squeeze(-1).to(self.device)  # [K]

        K = int(policy_scores.shape[0])
        self._last_rewards = self._linear_rewards_from_rank(K) 
        return policy_scores, kl_divs

    def compute_rpro_loss(self, policy_logp: torch.Tensor, kl_div: torch.Tensor):
        if self._last_rewards is None:
            advantages = torch.zeros_like(policy_logp)
        else:
            advantages = self._last_rewards.to(policy_logp.device)

        # RPRO損失：直接使用advantages加權的policy gradient
        pg_loss = -(advantages * policy_logp).mean()
        
        # 真正的KL散度懲罰項
        kl_penalty = self.beta * kl_div.mean()
        
        # 總損失
        total_loss = pg_loss + kl_penalty

        return total_loss, pg_loss, kl_penalty

    def compute_pairwise_loss(self, policy_logp: torch.Tensor):
        pairwise_losses = []
        n = int(policy_logp.shape[0])
        for i in range(n):
            for j in range(i + 1, n):
                # i應該比j更好（更高的logp），所以diff應該為正
                diff = policy_logp[i] - policy_logp[j]
                # 使用sigmoid loss，當diff為正時loss接近0
                pairwise_loss = torch.log(1 + torch.exp(-diff))
                pairwise_losses.append(pairwise_loss)
        if pairwise_losses:
            return torch.stack(pairwise_losses).mean()
        return torch.tensor(0.0, device=self.device)

    def _append_history(self, total_loss: float, pg_only: float, kl_penalty: float, pw_loss: float):
        self._global_step += 1
        self._hist_steps.append(self._global_step)
        self._hist_total.append(float(total_loss))
        self._hist_pg.append(float(pg_only))
        self._hist_kl.append(float(kl_penalty))
        self._hist_pw.append(float(pw_loss))

    def _save_history_csv(self, out_csv: Path):
        import csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["step", "total", "ranking", "kl", "pairwise"])  # ranking=PG項
            for s, t, r, k, p in zip(self._hist_steps, self._hist_total, self._hist_pg, self._hist_kl, self._hist_pw):
                w.writerow([s, t, r, k, p])
        logger.info(f"已保存損失CSV: {out_csv}")

    def _plot_history_png(self, out_png: Path, ma_window: int = 50):
        steps = np.array(self._hist_steps, dtype=float)
        total = np.array(self._hist_total, dtype=float)
        ranking = np.array(self._hist_pg, dtype=float)
        kl = np.array(self._hist_kl, dtype=float)
        pair = np.array(self._hist_pw, dtype=float)

        def moving_avg(x, w):
            if len(x) < 1:
                return x
            w = max(1, int(w))
            c = np.convolve(x, np.ones(w)/w, mode='valid')
            pad = np.full((len(x) - len(c),), np.nan)
            return np.concatenate([pad, c])

        total_ma = moving_avg(total, ma_window)
        ranking_ma = moving_avg(ranking, ma_window)
        kl_ma = moving_avg(kl, ma_window)
        pair_ma = moving_avg(pair, ma_window)

        plt.figure(figsize=(12, 7))
        plt.plot(steps, total, alpha=0.3, label='Total Loss')
        plt.plot(steps, ranking, alpha=0.3, label='Ranking Loss')
        plt.plot(steps, kl, alpha=0.3, label='KL Penalty')
        plt.plot(steps, pair, alpha=0.3, label='Pairwise Loss')

        plt.plot(steps, total_ma, linestyle='-', linewidth=2, label='Total (MA50)')
        plt.plot(steps, ranking_ma, linestyle='-', linewidth=2, label='Ranking (MA50)')
        plt.plot(steps, kl_ma, linestyle='-', linewidth=2, label='KL (MA50)')
        plt.plot(steps, pair_ma, linestyle='-', linewidth=2, label='Pairwise (MA50)')

        plt.title('RPRO Training Loss Curves (Fixed KL Divergence)')
        plt.xlabel('Global Training Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        plt.close()
        logger.info(f"已保存損失曲線: {out_png}")

    def train(self, data_file: str, output_dir: str, epochs: int = 3, batch_size: int = 1,
              lr: float = 1e-5, use_pairwise: bool = True, pairwise_weight: float = 0.1):
        dataset = JsonlRPRODataset(data_file, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        optimizer = AdamW(self.policy_model.parameters(), lr=lr, weight_decay=0.01)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.policy_model.train()
            epoch_total_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(dataloader):
                ranked_list = batch[0]  # 保持原行為
                policy_logp, kl_div = self.compute_preference_scores(ranked_list)
                
                # 檢查是否有異常值
                if torch.isnan(policy_logp).any() or torch.isinf(policy_logp).any():
                    logger.warning(f"發現NaN/Inf在policy_logp，跳過這個batch")
                    continue
                    
                if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
                    logger.warning(f"發現NaN/Inf在kl_div，跳過這個batch")
                    continue
                    
                rpro_loss, pg_only, kl_penalty = self.compute_rpro_loss(policy_logp, kl_div)
                total_loss_step = rpro_loss
                pw_loss_val = 0.0

                if use_pairwise:
                    pw_loss = self.compute_pairwise_loss(policy_logp)
                    total_loss_step = total_loss_step + pairwise_weight * pw_loss
                    pw_loss_val = float(pw_loss.item())

                # 檢查總損失是否異常
                if torch.isnan(total_loss_step) or torch.isinf(total_loss_step):
                    logger.warning(f"發現異常，跳過這個batch")
                    continue

                optimizer.zero_grad()
                total_loss_step.backward()
                
                # 添加梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                
                optimizer.step()

                epoch_total_loss += float(total_loss_step.item())
                num_batches += 1

                self._append_history(float(total_loss_step.item()), float(pg_only.item()), float(kl_penalty.item()), pw_loss_val)

                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Step {step}")
                    logger.info(f"  Total Loss: {total_loss_step.item():.4f}")
                    logger.info(f"  Ranking Loss(PG): {pg_only.item():.4f}")
                    logger.info(f"  KL Divergence: {kl_penalty.item():.4f}")
                    logger.info(f"  Grad Norm: {grad_norm:.4f}")
                    if use_pairwise:
                        logger.info(f"  Pairwise Loss: {pw_loss_val:.4f}")
                    logger.info(f"  Policy logp range: [{policy_logp.min().item():.2f}, {policy_logp.max().item():.2f}]")
                    logger.info(f"  KL div range: [{kl_div.min().item():.4f}, {kl_div.max().item():.4f}]")

            avg_loss = epoch_total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} 完成，平均損失: {avg_loss:.4f}")

            # 每個 epoch 結束保存 checkpoint
            checkpoint_dir = output_path / f"checkpoint-epoch-{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            self.policy_model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)

        # 導出 CSV 紀錄與圖
        csv_path = Path(output_dir) / "loss_history.csv"
        png_path = Path(output_dir) / "loss_curve.png"
        self._save_history_csv(csv_path)
        self._plot_history_png(png_path, ma_window=50)

        final_dir = Path(output_dir) / "final_model"
        final_dir.mkdir(exist_ok=True)
        self.policy_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        logger.info(f"RPRO訓練完成，模型保存到 {final_dir}")

    def evaluate_preferences(self, test_data_file: str):
        dataset = JsonlRPRODataset(test_data_file, self.tokenizer)
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


if __name__ == '__main__':
    # 設置環境變量優化顯存使用
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    data_file = "/content/rpro_cot_pairs.jsonl"
    output_dir = "./rpro_output"

    # 調整參數讓訓練更穩定
    trainer = JsonlRPROTrainer(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        beta=0.01,  # KL係數
        temperature=1.0
    )

    # 優化的訓練設置
    trainer.train(
        data_file=data_file,
        output_dir=output_dir,
        epochs=30,      
        batch_size=1,  
        lr=1e-6,      # 保持較低學習率
        use_pairwise=True,   # 可以嘗試啟用 pairwise loss
        pairwise_weight=0.1  # 適中的權重
    )

    # 評估模型性能
    trainer.evaluate_preferences(data_file)
