from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance


def _to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = array[..., None]
    array = array[..., :3]
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros((*array.shape[:2], 3), dtype=np.uint8)
    channels = []
    for index in range(array.shape[-1]):
        band = array[..., index].astype(np.float32)
        good = band[np.isfinite(band)]
        lo, hi = np.percentile(good, [1, 99]) if good.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        band = np.nan_to_num((band - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0)
        channels.append(np.clip(band * 255.0, 0, 255).astype(np.uint8))
    result = np.stack(channels, axis=-1)
    if result.shape[-1] == 1:
        result = np.repeat(result, 3, axis=-1)
    elif result.shape[-1] == 2:
        result = np.concatenate([result, result[..., :1]], axis=-1)
    return result


def load_image(path: str | Path) -> Image.Image:
    """Load common images and GeoTIFFs, normalizing high-bit-depth rasters."""
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            array = np.asarray(image)
            if array.dtype == np.uint8 and image.mode in ("RGB", "RGBA", "L"):
                return image.convert("RGB")
            return Image.fromarray(_to_uint8(array), mode="RGB")
    except Exception as pil_error:
        try:
            import tifffile

            return Image.fromarray(_to_uint8(tifffile.imread(image_path)), mode="RGB")
        except Exception as tif_error:
            raise RuntimeError(f"Cannot read image {image_path}: {pil_error}; {tif_error}") from tif_error


def expanded_square_bbox(
    bbox_xyxy: Iterable[float], width: int, height: int, margin: float = 1.5
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1, 2.0) * margin
    left = max(0, int(np.floor(cx - side / 2)))
    top = max(0, int(np.floor(cy - side / 2)))
    right = min(width, int(np.ceil(cx + side / 2)))
    bottom = min(height, int(np.ceil(cy + side / 2)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def crop_pair(
    pre: Image.Image,
    post: Image.Image,
    bbox_xyxy: Iterable[float] | None,
    margin: float = 1.5,
    size: int = 448,
) -> tuple[Image.Image, Image.Image]:
    if pre.size != post.size:
        post = post.resize(pre.size, Image.Resampling.BILINEAR)
    if bbox_xyxy is not None:
        box = expanded_square_bbox(bbox_xyxy, *pre.size, margin=margin)
        pre, post = pre.crop(box), post.crop(box)
    return (
        pre.resize((size, size), Image.Resampling.BICUBIC),
        post.resize((size, size), Image.Resampling.BICUBIC),
    )


def materialize_pair(
    pre_path: str | Path,
    post_path: str | Path,
    bbox_xyxy: Iterable[float] | None,
    output_dir: str | Path,
    sample_id: str,
    margin: float = 1.5,
    size: int = 448,
) -> tuple[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pre, post = crop_pair(
        load_image(pre_path), load_image(post_path), bbox_xyxy, margin=margin, size=size
    )
    pre_out = output / f"{sample_id}_pre.png"
    post_out = output / f"{sample_id}_post.png"
    pre.save(pre_out)
    post.save(post_out)
    return str(pre_out.resolve()), str(post_out.resolve())


def d4_transform(image: Image.Image, view: int) -> Image.Image:
    """Eight dihedral views; view 0 is identity."""
    if view not in range(8):
        raise ValueError("D4 view must be in [0, 7]")
    if view >= 4:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        view -= 4
    if view:
        image = image.rotate(90 * view, expand=False)
    return image


def d4_pair(pre: Image.Image, post: Image.Image, views: int = 8):
    if views not in (1, 2, 4, 8):
        raise ValueError("views must be one of 1, 2, 4, 8")
    for view in range(views):
        yield d4_transform(pre, view), d4_transform(post, view), view


def enhance_sar_for_display(image: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(image.convert("L")).enhance(1.5).convert("RGB")
