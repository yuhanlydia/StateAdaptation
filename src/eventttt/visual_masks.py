"""Fail-closed token masks for task-relevant visual-state adaptation."""

from __future__ import annotations

from collections.abc import Callable

import torch


MASK_MODES = ("first_visual", "second_visual", "all_visual", "text")


def contiguous_token_groups(row: torch.Tensor, token_id: int) -> list[torch.Tensor]:
    """Return contiguous groups containing ``token_id`` in one token row."""
    if row.ndim != 1:
        raise ValueError(f"row must be one-dimensional, got {list(row.shape)}")
    positions = (row == token_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    boundaries = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
    return list(torch.tensor_split(positions, boundaries.tolist()))


def build_visual_mask_fn(
    image_token_id: int,
    mode: str = "all_visual",
    *,
    expected_groups: int | None = None,
    text_exclude_ids: tuple[int, ...] = (),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a mask over visual groups or non-special text positions.

    ``first_visual`` and ``second_visual`` are intended for paired-image BRIGHT;
    ``all_visual`` is the cross-application default for single-image medical and
    robotics samples. Group-count mismatches raise instead of silently masking
    the wrong tokens.
    """
    if mode not in MASK_MODES:
        raise ValueError(f"unknown mask mode {mode!r}; choose from {MASK_MODES}")

    def build(input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B,T], got {list(input_ids.shape)}")
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row_index, row in enumerate(input_ids):
            groups = contiguous_token_groups(row, image_token_id)
            if expected_groups is not None and len(groups) != expected_groups:
                raise ValueError(
                    f"row {row_index}: expected {expected_groups} visual groups, found {len(groups)}"
                )
            if not groups:
                raise ValueError(f"row {row_index}: no visual tokens found")
            if mode == "first_visual":
                mask[row_index, groups[0]] = True
            elif mode == "second_visual":
                if len(groups) < 2:
                    raise ValueError(f"row {row_index}: second_visual requires two groups")
                mask[row_index, groups[1]] = True
            elif mode == "all_visual":
                for group in groups:
                    mask[row_index, group] = True
            else:
                row_mask = row != image_token_id
                for token_id in text_exclude_ids:
                    row_mask &= row != token_id
                mask[row_index] = row_mask
        return mask

    return build
