#!/usr/bin/env python3
"""
Dataset and collate function for RGPO training.

Each JSONL record must contain a 'question' and a 'ranked' list of K
candidate reasoning chains, ordered from best (index 0) to worst
(index K-1) as produced by the probabilistic refinement / ranking
step described in the paper (Section III-E). This module only
consumes the already-ranked data; it does not perform the ranking
itself.
"""
import json
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class JsonlGRPODataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_jsonl_data(data_file)
        logger.info(f"Loaded {len(self.data)} RGPO training examples")

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
                    logger.warning(f"Failed to parse JSON line: {e}")
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

        # item['ranked'] is assumed sorted best -> worst (rank 1 .. K),
        # matching the advantage assignment in losses.rank_advantages.
        inputs = []
        for reasoning in item['ranked']:
            full_text = prompt + "\n" + reasoning
            enc_full = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = enc_full['input_ids'].squeeze(0)
            attention_mask = enc_full['attention_mask'].squeeze(0)

            labels = input_ids.clone()
            seq_len = int(attention_mask.sum().item())
            eff_prompt_len = min(prompt_len, seq_len)
            # Mask the prompt tokens so log-probabilities are computed
            # only over the generated CoT tokens (Eq. 5).
            labels[:eff_prompt_len] = -100

            inputs.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'full_text': full_text,
            })

        return inputs


def collate_fn(batch):
    """Dynamically pad all candidates in the batch to the same length."""
    max_len = 0
    all_inputs = []

    for sample in batch:
        for inp in sample:
            max_len = max(max_len, len(inp['input_ids']))
            all_inputs.append(inp)

    for inp in all_inputs:
        pad_len = max_len - len(inp['input_ids'])
        if pad_len > 0:
            pad_token_id = inp['input_ids'][-1] if len(inp['input_ids']) > 0 else 0
            inp['input_ids'] = torch.cat([inp['input_ids'], torch.full((pad_len,), pad_token_id)])
            inp['attention_mask'] = torch.cat([inp['attention_mask'], torch.zeros(pad_len)])
            inp['labels'] = torch.cat([inp['labels'], torch.full((pad_len,), -100)])

    return batch
