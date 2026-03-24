"""GRPO (Group Relative Policy Optimization) loss for RL training.

Uses PPO-style clipped surrogate objective with behavioral TIS weight
correction.  The PPO ratio is computed against pre-computed proximal
logprobs (from a forward pass before training), and the TIS weight
corrects for the train-inference gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import tinker
import torch


def make_ppo_loss_fn(
    advantages: List[float],
    inf_logprob_list: List[List[float]],
    eps_clip: float = 0.2,
) -> ...:
    def loss_fn(
        data: List[tinker.Datum],
        logprobs_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=logprobs_list[0].device, requires_grad=True)

        for datum, target_logprobs, inf_logprobs in zip(data, logprobs_list, inf_logprob_list):
            prob_ratio = torch.exp(target_logprobs - inf_logprobs)
            clipped_ratio = torch.clamp(prob_ratio, 1 - eps_clip, 1 + eps_clip)
            unclipped_objective = prob_ratio * advantages
            clipped_objective = clipped_ratio * advantages
            ppo_objective = torch.min(unclipped_objective, clipped_objective)
            loss = -ppo_objective.sum()
            total_loss = total_loss + loss

        return total_loss, {
        }

    return loss_fn


def ppo_loss_fn(
    data: List[tinker.Datum],
    logprobs_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # In a real implementation, advantages and inf_logprob_list would be computed based on the data
    # target_tokens_td = [datum.loss_fn_inputs["target_tokens"] for datum in data]
    eps_clip = 0.2
    total_loss = torch.tensor(0.0, device=logprobs_list[0].device, requires_grad=True)

    for datum, target_logprobs in zip(data, logprobs_list):
        advantages = datum.loss_fn_inputs["advantages"]
        inf_logprobs = datum.loss_fn_inputs["logprobs"]
        prob_ratio = torch.exp(target_logprobs - inf_logprobs)
        clipped_ratio = torch.clamp(prob_ratio, 1 - eps_clip, 1 + eps_clip)
        unclipped_objective = prob_ratio * advantages
        clipped_objective = clipped_ratio * advantages
        ppo_objective = torch.min(unclipped_objective, clipped_objective)
        loss = -ppo_objective.sum()
        total_loss = total_loss + loss

    return total_loss, {
    }