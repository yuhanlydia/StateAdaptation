import numpy as np

from eventttt.metrics import classification_metrics, select_checkpoint


def test_perfect_metrics():
    probabilities = np.asarray([[0.99, 0.005, 0.005], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])
    result = classification_metrics([0, 1, 2], probabilities)
    assert result["macro_f1"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["ordinal_mae"] == 0.0
    assert result["nll"] < 0.03


def test_checkpoint_selection_tie_breaks_on_nll_then_steps():
    records = [
        {"steps": 4, "macro_f1": 0.8, "nll": 0.5, "ece": 0.2},
        {"steps": 8, "macro_f1": 0.8, "nll": 0.4, "ece": 0.3},
        {"steps": 16, "macro_f1": 0.7, "nll": 0.2, "ece": 0.1},
    ]
    assert select_checkpoint(records) == 8
