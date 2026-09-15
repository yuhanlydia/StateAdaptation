import json

import pytest

from eventttt.artifacts import REQUIRED_COMPLETE_FILES, export_run, portable_value


def make_complete_run(path):
    for name in REQUIRED_COMPLETE_FILES:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".json":
            target.write_text(json.dumps({"value": 1}) + "\n", encoding="utf-8")
        else:
            target.write_bytes(b"weights")
    (path / "launcher.json").write_text(
        json.dumps(
            {
                "command": ["bash", "/usr/Remote/scripts/run_event.sh"],
                "log": "/usr/Remote/runs/event/run/launcher.log",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "source_eval" / "predictions.jsonl").write_text(
        '{"sample_id": "one"}\n', encoding="utf-8"
    )
    (path / "source_diagnostic_eval").mkdir()
    (path / "source_diagnostic_eval" / "metrics.json").write_text(
        '{"macro_f1": 0.5}\n', encoding="utf-8"
    )
    (path / "source_gate").mkdir()
    (path / "source_gate" / "gate.json").write_text(
        '{"passed": true}\n', encoding="utf-8"
    )


def test_portable_value_rewrites_known_and_unknown_absolute_paths():
    assert portable_value("/usr/Remote/data/splits/a.jsonl") == (
        "${EVENTTUNE_ROOT}/data/splits/a.jsonl"
    )
    assert portable_value("/tmp/private.txt") == "<local-path>/private.txt"
    assert portable_value("Qwen/Qwen2.5-VL-7B-Instruct") == (
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )


def test_export_run_is_complete_portable_and_idempotent(tmp_path):
    run = tmp_path / "run-v1"
    destination = tmp_path / "export"
    make_complete_run(run)

    first = export_run(run, destination)
    second = export_run(run, destination)

    assert first == second
    assert first["complete"] is True
    launcher = json.loads((destination / "launcher.json").read_text(encoding="utf-8"))
    assert launcher["command"][1] == "${EVENTTUNE_ROOT}/scripts/run_event.sh"
    assert launcher["log"] == "${EVENTTUNE_ROOT}/runs/event/run/launcher.log"
    assert (destination / "source_diagnostic_eval" / "metrics.json").is_file()
    assert (destination / "source_gate" / "gate.json").is_file()
    manifest = json.loads(
        (destination / "export_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["files"]) == len(first["files"])
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])


def test_export_run_refuses_incomplete_or_conflicting_exports(tmp_path):
    (tmp_path / "missing").mkdir()
    with pytest.raises(RuntimeError, match="run is incomplete"):
        export_run(tmp_path / "missing", tmp_path / "export")

    run = tmp_path / "run-v1"
    destination = tmp_path / "export"
    make_complete_run(run)
    export_run(run, destination)
    (run / "source_eval" / "metrics.json").write_text(
        '{"value": 2}\n', encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        export_run(run, destination)
