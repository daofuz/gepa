#!/usr/bin/env python3
"""Run outcome-only routing with ranking restricted to mini-only examples."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import run_sembench_outcome_pairwise_router as experiment
from run_sembench_accuracy_targeted_router import REGRET_WEIGHTS


def critical_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    buckets: list[str],
    pairwise_weight: float,
    margin: float,
) -> tuple[torch.Tensor, float, float]:
    per_example = F.cross_entropy(logits, labels, reduction="none")
    weights = torch.tensor(
        [REGRET_WEIGHTS[bucket] for bucket in buckets], dtype=logits.dtype
    )
    regret_ce = (weights * per_example).sum() / weights.sum()

    # Only mini-only affects the accuracy advantage of mini.  Both-wrong keeps
    # its conservative mini target in CE but contributes no ranking pairs.
    critical_indices = torch.tensor(
        [index for index, bucket in enumerate(buckets) if bucket == "mini_only"],
        dtype=torch.long,
    )
    safe_indices = torch.tensor(
        [
            index for index, bucket in enumerate(buckets)
            if bucket in {"both_correct", "nano_only"}
        ],
        dtype=torch.long,
    )
    if pairwise_weight <= 0.0 or not len(critical_indices) or not len(safe_indices):
        return regret_ce, float(regret_ce.detach()), 0.0

    critical_scores = logits[critical_indices, 1] - logits[critical_indices, 0]
    safe_scores = logits[safe_indices, 1] - logits[safe_indices, 0]
    score_gaps = critical_scores[:, None] - safe_scores[None, :]
    pair_weights = torch.sqrt(
        weights[critical_indices, None] * weights[safe_indices][None, :]
    )
    ranking = (
        pair_weights * F.softplus(margin - score_gaps)
    ).sum() / pair_weights.sum()
    total = regret_ce + pairwise_weight * ranking
    return total, float(regret_ce.detach()), float(ranking.detach())


if __name__ == "__main__":
    experiment.outcome_pairwise_loss = critical_pairwise_loss
    experiment.main()
