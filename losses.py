#!/usr/bin/env python3
"""
Core RGPO loss terms, implementing the equations from Section IV of
the paper ("Training Framework"):

  Eq. (5)  s_j            : mean log-probability of candidate c_j
  Eq. (6)  P(c_i > c_j)   : Bradley-Terry pairwise preference
  Eq. (7)  L_CoT-rank     : listwise ranking loss over all K(K-1)/2 pairs
  Eq. (8)  a_j            : rank -> linear advantage mapping
  Eq. (9)  L_Linear       : linear-reward-weighted policy gradient term
  Eq. (10-12) D_KL        : token-level forward KL divergence and L_KL
  Eq. (13) L_total        : L_CoT-rank + L_KL + L_Linear (equal weights)

These are implemented as standalone functions (rather than methods
tied to a trainer class) so they can be unit-tested and checked
against the paper independently of the training loop.
"""
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_log_probs_and_kl(policy_model, reference_model, input_ids, attention_mask, labels):
    """
    Compute the sequence-level mean log-probability (Eq. 5) and the
    token-level forward KL divergence averaged over generated tokens
    (Eq. 10, 12): KL(pi_theta || pi_ref) = sum_v pi_theta(v) * (log pi_theta(v) - log pi_ref(v)).
    """
    with torch.set_grad_enabled(policy_model.training):
        policy_logits = policy_model(input_ids=input_ids, attention_mask=attention_mask).logits

    with torch.no_grad():
        reference_logits = reference_model(input_ids=input_ids, attention_mask=attention_mask).logits

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    policy_probs = F.softmax(policy_logits, dim=-1)

    # Shift for next-token prediction.
    shift_policy_log_probs = policy_log_probs[:, :-1, :]
    shift_reference_log_probs = reference_log_probs[:, :-1, :]
    shift_policy_probs = policy_probs[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = (shift_labels != -100) & (attention_mask[:, 1:] == 1)

    valid_labels = shift_labels.clamp(min=0, max=shift_policy_log_probs.size(-1) - 1)

    policy_gathered = shift_policy_log_probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
    policy_masked = torch.where(shift_mask, policy_gathered, torch.zeros_like(policy_gathered))

    reference_gathered = shift_reference_log_probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
    reference_masked = torch.where(shift_mask, reference_gathered, torch.zeros_like(reference_gathered))

    valid_token_count = shift_mask.sum(dim=-1).float() + 1e-8

    # Eq. (5): mean log-probability over the generated (unmasked) tokens.
    seq_policy_log_probs = policy_masked.sum(dim=-1) / valid_token_count
    seq_reference_log_probs = reference_masked.sum(dim=-1) / valid_token_count

    # Eq. (12): token-level forward KL, then averaged over generated tokens.
    kl_per_token = torch.sum(
        shift_policy_probs * (shift_policy_log_probs - shift_reference_log_probs),
        dim=-1
    )
    kl_masked = torch.where(shift_mask, kl_per_token, torch.zeros_like(kl_per_token))
    seq_kl_div = kl_masked.sum(dim=-1) / valid_token_count

    if torch.isnan(seq_policy_log_probs).any() or torch.isinf(seq_policy_log_probs).any():
        logger.warning("seq_policy_log_probs contains NaN/Inf, using fallback value")
        seq_policy_log_probs = torch.full_like(seq_policy_log_probs, -5.0)

    if torch.isnan(seq_kl_div).any() or torch.isinf(seq_kl_div).any():
        logger.warning("seq_kl_div contains NaN/Inf, using fallback value")
        seq_kl_div = torch.full_like(seq_kl_div, 0.1)

    return seq_policy_log_probs, seq_kl_div


def rank_advantages(k: int, device) -> torch.Tensor:
    """
    Eq. (8): a_j = (K - r_j) - (K-1)/2, for rank r_j = 1 (best) .. K (worst),
    assigned in order to candidates 0..K-1 of the (pre-sorted) ranked list.
    Sums to zero by construction.
    """
    if k <= 1:
        return torch.zeros(1, device=device)
    ranks = torch.arange(1, k + 1, dtype=torch.float32, device=device)  # r_j = 1..K
    return (k - ranks) - (k - 1) / 2


def linear_reward_loss(advantages: torch.Tensor, policy_logp: torch.Tensor) -> torch.Tensor:
    """Eq. (9): L_Linear = -(1/K) * sum_j a_j * s_j."""
    return -(advantages.to(policy_logp.device) * policy_logp).mean()


def pairwise_ranking_loss(policy_logp: torch.Tensor, tau_bt: float = 1.0) -> torch.Tensor:
    """
    Eq. (6-7): Bradley-Terry listwise ranking loss over all K(K-1)/2 pairs,
        L_CoT-rank = 1/C(K,2) * sum_{i<j} log(1 + exp(-(s_i - s_j) / tau_BT)),
    where candidates are pre-sorted so index i < j implies c_i is preferred
    over c_j (i.e. s_i should be larger than s_j).
    """
    device = policy_logp.device
    n = int(policy_logp.shape[0])
    if n < 2:
        return torch.tensor(0.0, device=device)

    pairwise_losses = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (policy_logp[i] - policy_logp[j]) / tau_bt
            pairwise_losses.append(torch.log1p(torch.exp(-diff)))
    return torch.stack(pairwise_losses).mean()


def kl_regularization(kl_div: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq. (11): L_KL = beta * (1/K) * sum_k D_KL(c_k)."""
    return beta * kl_div.mean()


def compute_rgpo_losses(policy_logp: torch.Tensor, kl_div: torch.Tensor, beta: float,
                         tau_bt: float = 1.0, use_pairwise: bool = True):
    """
    Assemble the full RGPO objective, Eq. (13):
        L_total = L_CoT-rank + L_KL + L_Linear
    combined with equal weighting, as specified in the paper (no extra
    hyperparameter is introduced to balance L_CoT-rank against L_Linear).

    Returns (total_loss, rank_loss, kl_loss, linear_loss).
    """
    k = int(policy_logp.shape[0])
    advantages = rank_advantages(k, policy_logp.device)

    linear_loss = linear_reward_loss(advantages, policy_logp)
    kl_loss = kl_regularization(kl_div, beta)
    rank_loss = pairwise_ranking_loss(policy_logp, tau_bt) if use_pairwise else torch.tensor(
        0.0, device=policy_logp.device)

    total_loss = rank_loss + kl_loss + linear_loss
    return total_loss, rank_loss, kl_loss, linear_loss
