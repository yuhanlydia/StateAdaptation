from visual_lens.audit import gradient_agrees


def test_gradient_agreement_uses_documented_relative_tolerance():
    comparison = {'shape_match': True, 'relative_l2': 0.00997}

    assert gradient_agrees(comparison, 0.05)
    assert not gradient_agrees(comparison, 0.005)


def test_gradient_agreement_rejects_shape_mismatch():
    assert not gradient_agrees({'shape_match': False, 'relative_l2': 0.0}, 0.05)
