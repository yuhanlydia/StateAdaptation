import os

from eventttt.io import (
    adapter_fingerprint,
    build_eval_config,
    directory_sha256,
    manifest_fingerprint,
    model_fingerprint,
    sha256_file,
)


def test_sha256_file_stable(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"hello" * 10)
    assert sha256_file(path) == sha256_file(path)


def test_directory_sha256_missing_file_raises(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    try:
        directory_sha256(tmp_path, ["adapter_config.json", "adapter_model.safetensors"])
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_adapter_fingerprint_missing_file_raises(tmp_path):
    try:
        adapter_fingerprint(tmp_path)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_adapter_fingerprint_stable_and_flips_on_change(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"r": 4}')
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights-v1")
    first = adapter_fingerprint(tmp_path)
    assert first == adapter_fingerprint(tmp_path)

    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights-v2")
    second = adapter_fingerprint(tmp_path)
    assert second != first


def test_directory_fingerprint_order_independent(tmp_path):
    (tmp_path / "a").write_bytes(b"1")
    (tmp_path / "b").write_bytes(b"2")
    assert directory_sha256(tmp_path, ["a", "b"]) == directory_sha256(tmp_path, ["b", "a"])


def test_model_fingerprint_falls_back_to_id():
    identifier = "Qwen/NoSuchModel-OnEarth-20B"
    assert model_fingerprint(identifier) == model_fingerprint(identifier)
    assert model_fingerprint("labs/model-b") != model_fingerprint(identifier)


def test_manifest_fingerprint_stable(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"sample_id": "a"}\n')
    assert manifest_fingerprint(path) == manifest_fingerprint(path)


def test_build_eval_config_full_configuration(tmp_path):
    detector_dir = tmp_path / "adapter"
    detector_dir.mkdir()
    (detector_dir / "adapter_config.json").write_text("{}")
    (detector_dir / "adapter_model.safetensors").write_bytes(b"w")
    manifest = tmp_path / "query.jsonl"
    manifest.write_text('{"sample_id": "q1"}\n')
    kv_state = tmp_path / "kv_state.pt"
    kv_state.write_bytes(b"kv")

    config = build_eval_config(
        model_id="Model/A",
        adapter=str(detector_dir),
        kv_state=str(kv_state),
        manifest=str(manifest),
        d4_views=1,
        crop_size=336,
        no_lora=False,
    )
    assert config["d4_views"] == 1
    assert config["crop_size"] == 336
    assert not config["no_lora"]
    assert config["adapter"] == str(detector_dir)
    assert config["kv_state_sha256"]
    assert config["manifest_sha256"]
    assert config["adapter_sha256"]

    same = build_eval_config(
        model_id="Model/A",
        adapter=str(detector_dir),
        kv_state=str(kv_state),
        manifest=str(manifest),
        d4_views=1,
        crop_size=336,
        no_lora=False,
    )
    assert same == config

    changed = build_eval_config(
        model_id="Model/A",
        adapter=str(detector_dir),
        kv_state=str(kv_state),
        manifest=str(manifest),
        d4_views=8,
        crop_size=336,
        no_lora=False,
    )
    assert changed != config

    other_model = build_eval_config(
        model_id="Model/B",
        adapter=str(detector_dir),
        kv_state=str(kv_state),
        manifest=str(manifest),
        d4_views=1,
        crop_size=336,
        no_lora=False,
    )
    assert other_model != config
    assert other_model["model_sha256"] != config["model_sha256"]