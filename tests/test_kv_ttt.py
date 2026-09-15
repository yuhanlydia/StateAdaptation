import torch
import torch.nn as nn

from eventttt.kv_ttt import (
    KVGradientCollector,
    ResidualKVController,
    _disable_input_require_grads,
    build_post_image_mask_fn,
    consistency_js_loss,
    default_layers,
    discover_language_decoder_kv,
)


def test_consistency_js_loss_zero_for_identical_views_and_positive_otherwise():
    identical = torch.tensor([[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]])
    disagree = torch.tensor([[5.0, 0.0, -1.0], [-1.0, 0.0, 5.0]])
    assert torch.allclose(consistency_js_loss(identical), torch.tensor(0.0), atol=1e-7)
    assert float(consistency_js_loss(disagree)) > 0.5


def test_consistency_js_loss_is_differentiable_and_label_free():
    scores = torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], requires_grad=True)
    consistency_js_loss(scores).backward()
    assert scores.grad is not None
    assert torch.any(scores.grad != 0)


def test_post_image_mask_groups_second_image_batch():
    image_id = 128
    build = build_post_image_mask_fn(image_id)
    row = torch.tensor(
        [1, image_id, image_id, 2, image_id, image_id, image_id, 3], dtype=torch.long
    )
    mask = build(row.unsqueeze(0))
    assert mask.shape == (1, 8)
    assert mask[0].nonzero(as_tuple=False).flatten().tolist() == [4, 5, 6]
    assert not bool(mask[0, 1])
    assert not bool(mask[0, 2])


def test_build_post_mask_fails_without_exactly_two_groups():
    build = build_post_image_mask_fn(128)
    for bad in (torch.tensor([[128, 128, 1]]), torch.tensor([[1, 2, 3]])):
        try:
            build(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_build_post_mask_multibatch():
    image_id = 9
    build = build_post_image_mask_fn(image_id)
    batch = torch.tensor(
        [
            [0, image_id, image_id, image_id, 1, image_id, image_id, 2],
            [0, image_id, 1, image_id, image_id, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    mask = build(batch)
    assert mask[0].nonzero(as_tuple=False).flatten().tolist() == [5, 6]
    assert mask[1].nonzero(as_tuple=False).flatten().tolist() == [3, 4]


def test_covariance_eigvecs_match_svd_right_singulars():
    torch.manual_seed(0)
    num_post, dim, rank = 40, 24, 8
    gradient = torch.randn(num_post, dim, dtype=torch.float32)
    covariance = gradient.t() @ gradient
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    values, order = torch.sort(eigenvalues, descending=True)
    basis = eigenvectors[:, order[:rank]]
    _, _, vh = torch.linalg.svd(gradient)
    svd_basis = vh[:rank].t()
    agreement = torch.abs(basis.t() @ svd_basis).diag()
    assert torch.all(agreement > 0.99)


def test_random_basis_is_orthonormal_and_seeded():
    torch.manual_seed(0)
    generator = torch.Generator(device="cpu").manual_seed(7)
    first, _ = torch.linalg.qr(torch.randn(24, 5, generator=generator))
    generator = torch.Generator(device="cpu").manual_seed(7)
    second, _ = torch.linalg.qr(torch.randn(24, 5, generator=generator))
    assert torch.allclose(first, second)
    assert torch.allclose(first.T @ first, torch.eye(5), atol=1e-5)


def test_controller_zero_coefficients_is_identity():
    torch.manual_seed(1)
    dim, rank = 16, 4
    proj = nn.Linear(dim, dim, bias=False)
    basis = torch.linalg.qr(torch.randn(dim, dim))[0][:, :rank]
    controller = ResidualKVController([(0, "K", proj)], {(0, "K"): basis}, rank=rank)
    x = torch.randn(2, 8, dim)
    controller.set_mask(torch.ones(2, 8, dtype=torch.bool))
    modified = proj(x)
    controller.clear_mask()
    assert torch.equal(modified, proj(x))


def test_controller_mask_only_modifies_post_span():
    torch.manual_seed(2)
    dim, rank = 16, 4
    proj = nn.Linear(dim, dim, bias=False)
    basis = torch.linalg.qr(torch.randn(dim, dim))[0][:, :rank]
    controller = ResidualKVController([(0, "K", proj)], {(0, "K"): basis}, rank=rank)
    x = torch.randn(1, 8, dim)
    original = proj(x)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, 4:] = True
    with torch.no_grad():
        for parameter in controller.ttt_parameters():
            parameter.fill_(1.0)
    controller.set_mask(mask)
    modified = proj(x)
    controller.clear_mask()
    assert torch.allclose(modified[:, :4], original[:, :4])
    assert not torch.allclose(modified[:, 4:], original[:, 4:])
    controller.reset_coefficients()
    assert torch.equal(proj(x), original)


def test_controller_gradients_flow_to_coefficients():
    dim, rank = 16, 4
    proj = nn.Linear(dim, dim, bias=False)
    basis = torch.linalg.qr(torch.randn(dim, dim))[0][:, :rank]
    controller = ResidualKVController([(0, "K", proj)], {(0, "K"): basis}, rank=rank)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[:, 2:] = True
    use = torch.randn(1, 4, dim, requires_grad=True)
    with torch.enable_grad():
        controller.set_mask(mask)
        out = proj(use)
        controller.clear_mask()
        out.mean().backward()
    grads = [p.grad for p in controller.ttt_parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_full_controller_mixes_subspace_directions_and_is_bounded():
    torch.manual_seed(3)
    dim, rank, alpha = 12, 3, 0.4
    proj = nn.Linear(dim, dim, bias=False)
    basis = torch.linalg.qr(torch.randn(dim, dim))[0][:, :rank]
    controller = ResidualKVController(
        [(0, "K", proj)], {(0, "K"): basis}, rank=rank,
        alpha_max=alpha, coefficient_mode="full",
    )
    assert controller.num_scalars() == rank * rank
    raw = controller.coefficients["0:K"]
    with torch.no_grad():
        raw[0, 1] = 2.0
    mixing = alpha * raw / (1.0 + torch.linalg.vector_norm(raw))
    assert float(torch.linalg.matrix_norm(mixing.detach(), ord=2)) < alpha
    x = torch.randn(1, 4, dim)
    mask = torch.ones(1, 4, dtype=torch.bool)
    original = proj(x)
    controller.set_mask(mask)
    modified = proj(x)
    controller.clear_mask()
    expected = original + (original @ basis @ mixing @ basis.t())
    assert torch.allclose(modified, expected, atol=1e-6)


def test_full_controller_zero_is_identity_and_has_gradients():
    dim, rank = 8, 2
    proj = nn.Linear(dim, dim, bias=False)
    basis = torch.linalg.qr(torch.randn(dim, dim))[0][:, :rank]
    controller = ResidualKVController(
        [(0, "V", proj)], {(0, "V"): basis}, rank=rank,
        coefficient_mode="full",
    )
    x = torch.randn(1, 3, dim, requires_grad=True)
    baseline = proj(x)
    controller.set_mask(torch.ones(1, 3, dtype=torch.bool))
    identity = proj(x)
    assert torch.equal(identity, baseline)
    identity.sum().backward()
    assert torch.any(controller.coefficients["0:V"].grad != 0)
    controller.clear_mask()


def test_default_layers_middle_and_last():
    assert default_layers(28) == [14, 27]
    assert default_layers(4) == [2, 3]


def test_discover_regex_through_nested_wrapper():
    lm = nn.Module()
    layers = nn.ModuleList()
    for _ in range(4):
        layer = nn.Module()
        attention = nn.Module()
        attention.k_proj = nn.Linear(8, 8)
        attention.v_proj = nn.Linear(8, 8)
        layer.self_attn = attention
        layers.append(layer)
    lm.layers = layers

    outer = nn.Module()
    outer.base_model = nn.Module()
    outer.base_model.language_model = lm
    found, count = discover_language_decoder_kv(outer)
    assert count == 4
    assert sorted({layer_id for layer_id, _, _ in found}) == [0, 1, 2, 3]
    assert len(found) == 8


def test_kv_collector_close_unregisters_hooks():
    lm = nn.Module()
    layers = nn.ModuleList()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.k_proj = nn.Linear(8, 8)
    layer.self_attn.v_proj = nn.Linear(8, 8)
    layers.append(layer)
    lm.layers = layers

    outer = nn.Module()
    outer.base_model = nn.Module()
    outer.base_model.language_model = lm
    modules, _ = discover_language_decoder_kv(outer)
    assert len(modules) == 2
    collector = KVGradientCollector(modules)
    assert len(collector.handles) == 2
    assert all(len(m._forward_hooks) == 1 for _, _, m in modules)
    collector.close()
    assert collector.handles == []
    assert all(len(m._forward_hooks) == 0 for _, _, m in modules)


def test_kv_collector_keeps_multiple_forward_gradients():
    projection = nn.Linear(4, 4, bias=False)
    collector = KVGradientCollector([(0, "K", projection)])
    first = projection(torch.randn(1, 3, 4, requires_grad=True))
    second = projection(torch.randn(1, 3, 4, requires_grad=True))
    (first.sum() + 2 * second.sum()).backward()
    masks = [torch.tensor([[False, True, True]]), torch.tensor([[True, False, True]])]
    gradients = collector.history_gradients(masks)[(0, "K")]
    assert gradients.shape == (4, 4)
    assert torch.all(gradients[:2] == 1)
    assert torch.all(gradients[2:] == 2)
    collector.close()


def test_controller_close_unregisters_hooks():
    dim, rank = 8, 2
    proj = nn.Linear(dim, dim, bias=False)
    basis = torch.linalg.qr(torch.randn(dim, dim))[0][:, :rank]
    controller = ResidualKVController([(0, "K", proj)], {(0, "K"): basis}, rank=rank)
    assert len(proj._forward_hooks) == 1
    controller.close()
    assert controller._hooks == []
    assert len(proj._forward_hooks) == 0


def test_disable_input_require_grads_is_guarded():
    plain = nn.Linear(2, 2)
    _disable_input_require_grads(plain)

    class Boom:
        def disable_input_require_grads(self):
            raise AttributeError("no hook registered")

    _disable_input_require_grads(Boom())

    class Removes:
        def __init__(self):
            self.called = False

        def disable_input_require_grads(self):
            self.called = True

    target = Removes()
    _disable_input_require_grads(target)
    assert target.called
