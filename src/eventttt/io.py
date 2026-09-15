from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .schemas import Sample, TaskSample


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: str | Path, names: Sequence[str]) -> str:
    """Combined SHA-256 over the content of ``names`` inside ``path``.

    Missing files raise; the fold is over sorted names so the ordering is
    stable regardless of the input order."""
    base = Path(path)
    digest = hashlib.sha256()
    for name in sorted(set(names)):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sha256_file(base / name).encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def adapter_fingerprint(path: str | Path) -> str:
    """Adapter-identity hash over the LoRA weights and config."""
    return directory_sha256(path, ["adapter_config.json", "adapter_model.safetensors"])


def manifest_fingerprint(path: str | Path) -> str:
    return sha256_file(path)


def build_eval_config(
    *,
    model_id: str,
    adapter: str | None,
    kv_state: str | None,
    manifest: str,
    d4_views: int,
    crop_size: int,
    no_lora: bool,
) -> dict:
    """Full fingerprint of one evaluation invocation.

    Model, adapter and kv-state identity are content hashes so a resumed run
    can refuse to mix predictions produced against different weights."""
    return {
        "model_id": model_id,
        "model_sha256": model_fingerprint(model_id),
        "adapter": adapter or None,
        "adapter_sha256": adapter_fingerprint(adapter) if adapter else None,
        "kv_state": str(Path(kv_state).resolve()) if kv_state else None,
        "kv_state_sha256": sha256_file(kv_state) if kv_state else None,
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": manifest_fingerprint(manifest),
        "d4_views": int(d4_views),
        "crop_size": int(crop_size),
        "no_lora": bool(no_lora),
    }


_MODEL_FINGERPRINT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
)


def _candidate_model_root(model_id: str) -> Path | None:
    direct = Path(model_id)
    if direct.is_dir():
        return direct
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    hits = sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime)
    return hits[-1] if hits else None


def model_fingerprint(model_id: str) -> str:
    """Hash of the locally used base model.

    Uses the resolved snapshot: config/tokenizer byte content plus the names
    and sizes of the weight shards, so a different checkpoint is caught without
    re-reading the multi-GB weights. Falls back to the model id itself when no
    local snapshot is found."""
    root = _candidate_model_root(model_id)
    digest = hashlib.sha256()
    if root is None:
        digest.update(model_id.encode("utf-8"))
        return digest.hexdigest()
    for name in sorted(_MODEL_FINGERPRINT_FILES):
        path = root / name
        if not path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\x00")
    for shard in sorted(root.glob("model-*.safetensors")):
        digest.update(shard.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(shard.stat().st_size).encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def read_samples(path: str | Path, resolve: bool = True) -> list[Sample]:
    manifest = Path(path).resolve()
    samples = [Sample.from_dict(row) for row in iter_jsonl(manifest)]
    if resolve:
        samples = [sample.resolve_paths(manifest.parent) for sample in samples]
    return samples


def read_task_samples(path: str | Path, resolve: bool = True) -> list[TaskSample]:
    manifest = Path(path).resolve()
    samples = [TaskSample.from_dict(row) for row in iter_jsonl(manifest)]
    if resolve:
        samples = [sample.resolve_paths(manifest.parent) for sample in samples]
    return samples


def write_samples(path: str | Path, samples: Iterable[Sample]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = sample.to_dict()
            for key in ("pre_image", "post_image", "mask_path"):
                value = row.get(key)
                if value is not None and Path(value).is_absolute():
                    row[key] = os.path.relpath(value, start=output.parent.resolve())
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_manifest(path: str | Path, check_paths: bool = True) -> dict:
    manifest = Path(path).resolve()
    sample_ids: set[str] = set()
    event_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    tiles: set[tuple[str, str]] = set()
    image_paths: set[Path] = set()
    count = 0
    for row in iter_jsonl(manifest):
        sample = Sample.from_dict(row).resolve_paths(manifest.parent)
        if sample.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        sample_ids.add(sample.sample_id)
        event_counts[sample.event_id] += 1
        label_counts[sample.label] += 1
        tiles.add((sample.event_id, sample.tile_id))
        image_paths.update((Path(sample.pre_image), Path(sample.post_image)))
        count += 1
    if not count:
        raise ValueError(f"Manifest is empty: {manifest}")
    missing = sorted(str(item) for item in image_paths if check_paths and not item.is_file())
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} image files; first entries:\n{preview}")
    return {
        "manifest": str(manifest),
        "samples": count,
        "events": dict(sorted(event_counts.items())),
        "labels": dict(sorted(label_counts.items())),
        "event_tiles": len(tiles),
        "unique_images": len(image_paths),
        "paths_checked": check_paths,
    }
