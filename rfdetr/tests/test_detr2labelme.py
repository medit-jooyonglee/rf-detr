import json

import cv2
import numpy as np

from rfdetr.util.detr2labelme import (
    build_labelme_annotation,
    fdi_to_universal_label,
    instance_result_to_labelme_shapes,
    save_labelme_annotation,
)


def _sample_result():
    masks = np.zeros((2, 20, 30), dtype=np.uint8)
    cv2.rectangle(masks[0], (2, 3), (8, 10), 1, -1)
    cv2.rectangle(masks[0], (12, 3), (16, 8), 1, -1)
    cv2.rectangle(masks[1], (20, 5), (27, 15), 1, -1)
    return {
        "fdi": np.array([11, 12]),
        "scores": np.array([0.9, 0.8]),
        "masks": masks.astype(bool),
    }


def test_build_labelme_annotation_uses_polygon_shapes_and_instance_groups():
    annotation = build_labelme_annotation("nested/pano.jpg", _sample_result())

    assert annotation["imagePath"] == "pano.jpg"
    assert annotation["imageHeight"] == 20
    assert annotation["imageWidth"] == 30
    assert annotation["imageData"] is None
    assert len(annotation["shapes"]) == 3
    assert {shape["shape_type"] for shape in annotation["shapes"]} == {"polygon"}
    assert [shape["label"] for shape in annotation["shapes"]] == ["7", "8", "8"]
    assert [shape["group_id"] for shape in annotation["shapes"]] == [1, 0, 0]
    assert all(len(shape["points"]) >= 3 for shape in annotation["shapes"])


def test_save_labelme_annotation_writes_loadable_json(tmp_path):
    output_path = tmp_path / "pano.json"

    saved_path = save_labelme_annotation(
        output_path, tmp_path / "pano.jpg", _sample_result()
    )

    assert saved_path == output_path
    with saved_path.open(encoding="utf-8") as file:
        annotation = json.load(file)
    assert annotation["imagePath"] == "pano.jpg"
    assert annotation["shapes"][0]["label"] == "7"
    assert annotation["shapes"][0]["description"] == "confidence=0.800000"


def test_build_labelme_annotation_rejects_dimension_mismatch():
    try:
        build_labelme_annotation(
            "pano.jpg", _sample_result(), image_height=10, image_width=30
        )
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("Expected mismatched dimensions to raise ValueError")


def test_rdp_simplification_reduces_polygon_vertices():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 35, 1, -1)
    result = {
        "fdi": np.array([31]),
        "scores": np.array([0.9]),
        "masks": mask[None].astype(bool),
    }

    unsimplified = instance_result_to_labelme_shapes(result, polygon_epsilon=0)
    simplified = instance_result_to_labelme_shapes(result, polygon_epsilon=1.0)

    assert len(simplified) == len(unsimplified) == 1
    assert len(simplified[0]["points"]) < len(unsimplified[0]["points"])


def test_fdi_to_universal_label_covers_all_quadrants():
    assert [fdi_to_universal_label(fdi) for fdi in (18, 11, 21, 28)] == [1, 8, 9, 16]
    assert [fdi_to_universal_label(fdi) for fdi in (38, 31, 41, 48)] == [17, 24, 25, 32]


def test_shapes_are_saved_in_universal_number_order():
    result = _sample_result()
    result["fdi"] = np.array([41, 18])

    shapes = instance_result_to_labelme_shapes(result)

    assert [int(shape["label"]) for shape in shapes] == [1, 25, 25]
