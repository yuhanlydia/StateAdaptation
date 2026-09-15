import json

from eventttt.io import read_samples, validate_manifest, write_samples
from eventttt.schemas import Sample


def test_manifest_paths_are_portable_and_round_trip(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    pre = images / "pre.png"
    post = images / "post.png"
    pre.touch()
    post.touch()
    sample = Sample(
        sample_id="one",
        event_id="event",
        tile_id="tile",
        pre_image=str(pre.resolve()),
        post_image=str(post.resolve()),
        label="intact",
        label_id=0,
        dataset="test",
    )
    manifest = tmp_path / "manifests" / "samples.jsonl"

    write_samples(manifest, [sample])

    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert not row["pre_image"].startswith("/")
    restored = read_samples(manifest)[0]
    assert restored.pre_image == str(pre.resolve())
    assert restored.post_image == str(post.resolve())

    summary = validate_manifest(manifest)
    assert summary["samples"] == 1
    assert summary["events"] == {"event": 1}
    assert summary["labels"] == {"intact": 1}
    assert summary["unique_images"] == 2
