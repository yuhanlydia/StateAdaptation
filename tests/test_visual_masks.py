import pytest
import torch

from eventttt.visual_masks import build_visual_mask_fn, contiguous_token_groups


def test_contiguous_groups_and_visual_modes():
    ids = torch.tensor([[9, 7, 7, 3, 7, 7, 7, 4]])
    assert [x.tolist() for x in contiguous_token_groups(ids[0], 7)] == [[1, 2], [4, 5, 6]]
    assert build_visual_mask_fn(7, "first_visual", expected_groups=2)(ids)[0].nonzero().flatten().tolist() == [1, 2]
    assert build_visual_mask_fn(7, "second_visual", expected_groups=2)(ids)[0].nonzero().flatten().tolist() == [4, 5, 6]
    assert build_visual_mask_fn(7, "all_visual", expected_groups=2)(ids)[0].nonzero().flatten().tolist() == [1, 2, 4, 5, 6]


def test_visual_masks_fail_closed():
    ids = torch.tensor([[1, 7, 7, 2]])
    with pytest.raises(ValueError, match="expected 2"):
        build_visual_mask_fn(7, "all_visual", expected_groups=2)(ids)
    with pytest.raises(ValueError, match="second_visual"):
        build_visual_mask_fn(7, "second_visual")(ids)


def test_text_mask_excludes_visual_and_special_tokens():
    ids = torch.tensor([[1, 7, 7, 4, 0]])
    mask = build_visual_mask_fn(7, "text", text_exclude_ids=(0, 1))(ids)
    assert mask[0].nonzero().flatten().tolist() == [3]
