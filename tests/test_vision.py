import numpy as np
from PIL import Image

from eventttt.vision import crop_pair, d4_pair, expanded_square_bbox


def test_crop_pair_is_aligned_and_resized():
    array = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    pre = Image.fromarray(array).convert("RGB")
    post = Image.fromarray(array).convert("RGB")
    pre_crop, post_crop = crop_pair(pre, post, (20, 10, 30, 40), margin=1.5, size=32)
    assert pre_crop.size == (32, 32)
    assert np.array_equal(np.asarray(pre_crop), np.asarray(post_crop))


def test_d4_identity_and_count():
    image = Image.new("RGB", (32, 32), "red")
    views = list(d4_pair(image, image, 8))
    assert len(views) == 8
    assert np.array_equal(np.asarray(views[0][0]), np.asarray(image))


def test_bbox_clamps_at_image_border():
    left, top, right, bottom = expanded_square_bbox((-4, -2, 5, 5), 20, 20, margin=2)
    assert left == 0 and top == 0
    assert right <= 20 and bottom <= 20
