from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PUBLISHABLE_DIRECTORIES = (
    "source_adapter",
    "source_eval",
    "source_gate",
    "source_diagnostic_eval",
    "event_adapter",
    "event_eval",
)
PUBLISHABLE_FILES = ("adaptation_gain.json", "launcher.json")
REQUIRED_COMPLETE_FILES = (
    "source_adapter/adapter_model.safetensors",
    "source_adapter/train_summary.json",
    "source_eval/metrics.json",
    "event_adapter/adapter_model.safetensors",
    "event_adapter/selection.json",
    "event_eval/metrics.json",
    "adaptation_gain.json",
)
PORTABLE_JSON_FILES = {"launcher.json", "selection.json", "train_summary.json"}


def portable_value(value: Any) -> Any:
    """Remove machine-specific prefixes from JSON run metadata."""
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    for directory in ("data", "runs", "scripts", "configs", "artifacts"):
        marker = f"/{directory}/"
        if marker in value:
            suffix = value.split(marker, 1)[1]
            return f"${{EVENTTUNE_ROOT}}/{directory}/{suffix}"
    return f"<local-path>/{Path(value).name}"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _portable_payload(path: Path) -> bytes:
    if path.name not in PORTABLE_JSON_FILES:
        return path.read_bytes()
    value = json.loads(path.read_text(encoding="utf-8"))
    return (
        json.dumps(portable_value(value), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _publish(destination: Path, payload: bytes) -> None:
    if destination.exists():
        if destination.read_bytes() == payload:
            return
        raise FileExistsError(
            f"refusing to replace a different exported artifact: {destination}"
        )
    _atomic_write(destination, payload)


def export_run(run_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    """Export a completed run without raw imagery, caches, or local paths."""
    source = Path(run_dir).resolve()
    target = Path(destination).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {source}")
    missing = [name for name in REQUIRED_COMPLETE_FILES if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"run is incomplete; missing: {', '.join(missing)}")

    sources: list[tuple[Path, Path]] = []
    for directory in PUBLISHABLE_DIRECTORIES:
        base = source / directory
        for path in sorted(base.rglob("*")):
            if path.is_file() and ".cache" not in path.parts and not path.name.endswith(".tmp"):
                sources.append((path, Path(directory) / path.relative_to(base)))
    for name in PUBLISHABLE_FILES:
        path = source / name
        if path.is_file():
            sources.append((path, Path(name)))

    records = []
    for path, relative in sources:
        payload = _portable_payload(path)
        output = target / relative
        _publish(output, payload)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_name": source.name,
        "complete": True,
        "files": records,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish(target / "export_manifest.json", manifest_payload)
    return manifest
