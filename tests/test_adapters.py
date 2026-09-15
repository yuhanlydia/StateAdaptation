import json

import numpy as np
from PIL import Image

from eventttt.datasets.bright import adapt_bright_coco, adapt_bright_rasters
from eventttt.datasets.xbd import adapt_xbd


def test_bright_raster_adapter(tmp_path):
    for directory in ("pre-event", "post-event", "target"):
        (tmp_path / directory).mkdir()
    tile = "quake-test_00000000"
    Image.new("RGB", (32, 32), "gray").save(tmp_path / "pre-event" / f"{tile}_pre_disaster.tif")
    Image.new("L", (32, 32), 100).save(tmp_path / "post-event" / f"{tile}_post_disaster.tif")
    target = np.zeros((32, 32), dtype=np.uint8)
    target[2:6, 2:6] = 1
    target[12:16, 12:16] = 2
    target[22:27, 22:27] = 3
    Image.fromarray(target).save(tmp_path / "target" / f"{tile}_building_damage.tif")
    rows = adapt_bright_rasters(tmp_path, min_pixels=4)
    assert len(rows) == 3
    assert {row.label for row in rows} == {"intact", "damaged", "destroyed"}
    assert {row.event_id for row in rows} == {"quake-test"}


def test_xbd_adapter(tmp_path):
    labels = tmp_path / "train" / "labels"
    images = tmp_path / "train" / "images"
    labels.mkdir(parents=True)
    images.mkdir(parents=True)
    tile = "mexico-earthquake_00000000"
    pre_label = labels / f"{tile}_pre_disaster.json"
    post_label = labels / f"{tile}_post_disaster.json"
    pre_label.write_text("{}", encoding="utf-8")
    post_label.write_text(
        json.dumps(
            {
                "features": {
                    "xy": [
                        {
                            "wkt": "POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))",
                            "properties": {"uid": "one", "subtype": "major-damage"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (16, 16), "gray").save(images / f"{tile}_pre_disaster.png")
    Image.new("RGB", (16, 16), "gray").save(images / f"{tile}_post_disaster.png")
    rows = adapt_xbd(tmp_path)
    assert len(rows) == 1
    assert rows[0].label == "damaged"
    assert rows[0].bbox_xyxy == (2.0, 2.0, 8.0, 8.0)


def test_official_bright_tilewise_coco_directory(tmp_path):
    (tmp_path / "pre-event").mkdir()
    (tmp_path / "post-event").mkdir()
    labels = tmp_path / "target_instance_level"
    labels.mkdir()
    tile = "turkey-earthquake_00000455"
    Image.new("RGB", (32, 32), "gray").save(
        tmp_path / "pre-event" / f"{tile}_pre_disaster.tif"
    )
    Image.new("RGB", (32, 32), "gray").save(
        tmp_path / "post-event" / f"{tile}_post_disaster.tif"
    )
    payload = {
        "images": [{"id": 1, "sample_id": tile, "width": 32, "height": 32}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [3, 4, 8, 10]}
        ],
        "categories": [
            {"id": 1, "name": "intact"},
            {"id": 2, "name": "damaged"},
            {"id": 3, "name": "destroyed"},
        ],
    }
    (labels / f"{tile}_instance_damage.json").write_text(
        "\ufeff" + json.dumps(payload), encoding="utf-8"
    )
    rows = adapt_bright_coco(tmp_path, labels)
    assert len(rows) == 1
    assert rows[0].sample_id == f"bright-instance-{tile}-1"
    assert rows[0].event_id == "turkey-earthquake"
    assert rows[0].label == "destroyed"
