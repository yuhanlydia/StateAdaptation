import numpy as np

from eventttt.aggregation import product_of_experts


def test_product_of_experts_uses_all_views():
    probabilities = product_of_experts([[0.0, -2.0, -4.0], [-1.0, 0.0, -5.0]])
    assert probabilities.shape == (3,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_product_of_experts_rejects_vector():
    try:
        product_of_experts([0.0, 1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("one-dimensional scores should fail")
