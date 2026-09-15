from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from .schemas import DAMAGE_LABELS


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(value)


def classification_metrics(
    y_true: Iterable[int], probabilities: np.ndarray, ece_bins: int = 10
) -> dict:
    truth = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (len(truth), len(DAMAGE_LABELS)):
        raise ValueError(f"Expected probabilities shape {(len(truth), len(DAMAGE_LABELS))}, got {probs.shape}")
    return classification_metrics_nclass(truth, probs, tuple(DAMAGE_LABELS), ece_bins)


def classification_metrics_nclass(
    y_true: Iterable[int], probabilities: np.ndarray,
    labels: tuple[str, ...], ece_bins: int = 10,
) -> dict:
    """Classification metrics for arbitrary candidate-label tasks."""
    truth = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    n_classes = len(labels)
    if probs.shape != (len(truth), n_classes):
        raise ValueError(f"Expected probabilities shape {(len(truth), n_classes)}, got {probs.shape}")
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=range(n_classes), zero_division=0
    )
    one_hot = np.eye(n_classes)[truth]
    result = {
        "count": int(len(truth)),
        "macro_f1": float(f1_score(truth, pred, average="macro", labels=range(n_classes), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "ordinal_mae": float(np.abs(truth - pred).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(truth, pred, weights="quadratic")),
        "nll": float(log_loss(truth, probs, labels=range(n_classes))),
        "brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(truth, probs, bins=ece_bins),
        "per_class": {
            labels[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(truth, pred, labels=range(n_classes)).tolist(),
    }
    return result


def metrics_by_event(rows: Iterable[dict], ece_bins: int = 10) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["event_id"])].append(row)
    events = {
        event: classification_metrics(
            [row["label_id"] for row in group],
            np.asarray([row["probabilities"] for row in group]),
            ece_bins,
        )
        for event, group in groups.items()
    }
    all_rows = [row for group in groups.values() for row in group]
    overall = classification_metrics(
        [row["label_id"] for row in all_rows],
        np.asarray([row["probabilities"] for row in all_rows]),
        ece_bins,
    )
    overall["per_event"] = events
    overall["worst_event_macro_f1"] = min(value["macro_f1"] for value in events.values())
    overall["mean_event_macro_f1"] = float(np.mean([value["macro_f1"] for value in events.values()]))
    return overall


def select_checkpoint(rows: Iterable[dict]) -> int:
    """Maximize support-CV macro-F1, then minimize NLL, ECE, and steps."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["steps"])].append(row)
    if not grouped:
        raise ValueError("No checkpoint-validation records")
    summary = {}
    for steps, values in grouped.items():
        summary[steps] = {
            "macro_f1": float(np.mean([value["macro_f1"] for value in values])),
            "nll": float(np.mean([value["nll"] for value in values])),
            "ece": float(np.mean([value["ece"] for value in values])),
        }
    return max(
        summary,
        key=lambda steps: (
            summary[steps]["macro_f1"],
            -summary[steps]["nll"],
            -summary[steps]["ece"],
            -steps,
        ),
    )
