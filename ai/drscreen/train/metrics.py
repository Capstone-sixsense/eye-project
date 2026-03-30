from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class BinaryClassificationMetrics:
    accuracy: float
    auroc: float | None
    f1: float
    sensitivity: float | None
    specificity: float | None
    precision: float
    threshold: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    num_examples: int
    positive_examples: int
    negative_examples: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _binary_auroc(probabilities: np.ndarray, targets: np.ndarray) -> float | None:
    positive_mask = targets == 1
    negative_mask = targets == 0
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    if positive_count == 0 or negative_count == 0:
        return None

    ranks = _average_ranks(probabilities)
    positive_rank_sum = float(ranks[positive_mask].sum())
    auc = (
        positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)
    return float(auc)


def compute_binary_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    flattened_logits = logits.detach().float().view(-1).cpu()
    flattened_targets = targets.detach().long().view(-1).cpu()

    if flattened_logits.numel() != flattened_targets.numel():
        raise ValueError("Logits and targets must contain the same number of examples.")
    if flattened_logits.numel() == 0:
        raise ValueError("At least one example is required to compute classification metrics.")

    probabilities = torch.sigmoid(flattened_logits)
    predictions = (probabilities >= threshold).long()

    true_positive = int(((predictions == 1) & (flattened_targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (flattened_targets == 0)).sum().item())
    false_positive = int(((predictions == 1) & (flattened_targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (flattened_targets == 1)).sum().item())

    num_examples = int(flattened_targets.numel())
    positive_examples = int((flattened_targets == 1).sum().item())
    negative_examples = int((flattened_targets == 0).sum().item())

    accuracy = (true_positive + true_negative) / num_examples
    precision = _safe_ratio(true_positive, true_positive + false_positive) or 0.0
    sensitivity = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)

    f1_denominator = (2 * true_positive) + false_positive + false_negative
    f1 = 0.0 if f1_denominator == 0 else (2 * true_positive) / f1_denominator
    auroc = _binary_auroc(probabilities.numpy(), flattened_targets.numpy())

    return BinaryClassificationMetrics(
        accuracy=float(accuracy),
        auroc=auroc,
        f1=float(f1),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=float(precision),
        threshold=float(threshold),
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        num_examples=num_examples,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
    )
