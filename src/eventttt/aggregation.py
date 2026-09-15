from __future__ import annotations

import numpy as np


def product_of_experts(view_log_scores) -> np.ndarray:
    """D4 product-of-experts: average log-scores, then normalize."""
    scores = np.asarray(view_log_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Expected [views, classes] log scores")
    mean = scores.mean(axis=0)
    probabilities = np.exp(mean - mean.max())
    return probabilities / probabilities.sum()
