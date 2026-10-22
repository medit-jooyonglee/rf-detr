"""Convert RF-DETR instance predictions to Labelme polygon annotations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import cv2
import numpy as np


PathLike = Union[str, Path]

UNIVERSAL_TO_FDI = np.array(
    [
        0,
        18, 17, 16, 15, 14, 13, 12, 11,
        21, 22, 23, 24, 25, 26, 27, 28,
        38, 37, 36, 35, 34, 33, 32, 31,
        41, 42, 43, 44, 45, 46, 47, 48,
    ],
    dtype=np.int64,
)
FDI_TO_UNIVERSAL = {
    int(fdi): universal
    for universal, fdi in enumerate(UNIVERSAL_TO_FDI)
    if universal > 0
}


def fdi_to_universal_label(fdi: int) -> int:
    """Convert an FDI permanent-tooth number to universal numbering (1-32)."""
    try:
        return FDI_TO_UNIVERSAL[int(fdi)]
    except KeyError as error:
        raise ValueError(f"Unsupported FDI tooth number: {fdi!r}") from error


def instance_result_to_labelme_shapes(
    instance_result: Mapping[str, Any],
    *,
    min_contour_area: float = 0.0,
    polygon_epsilon: float = 2.0,
) -> List[Dict[str, Any]]:
    """Build Labelme polygon shapes from one RF-DETR instance result.

    Disconnected contours belonging to the same prediction share ``group_id``.
    This matches Labelme's instance convention and the grouping used by
    ``examples/instance_segmentation/labelme2coco.py``.
    """
    masks = np.asarray(instance_result["masks"], dtype=np.uint8)
    labels = np.asarray(instance_result["fdi"]).reshape(-1)
    scores = np.asarray(
        instance_result.get("scores", np.ones(len(labels), dtype=np.float32))
    ).reshape(-1)

    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}")
    if not (len(masks) == len(labels) == len(scores)):
        raise ValueError(
            "masks, fdi, and scores must contain the same number of instances"
        )
    if min_contour_area < 0:
        raise ValueError("min_contour_area must be non-negative")
    if polygon_epsilon < 0:
        raise ValueError("polygon_epsilon must be non-negative")

    shapes: List[Dict[str, Any]] = []
    for instance_index, (mask, label, score) in enumerate(
        zip(masks, labels, scores)
    ):
        universal_label = fdi_to_universal_label(int(label))
        contours, _ = cv2.findContours(
            np.ascontiguousarray(mask),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(
            contours,
            key=lambda contour: (*cv2.boundingRect(contour)[:2], -cv2.contourArea(contour)),
        )
        for contour in contours:
            if polygon_epsilon > 0:
                contour = cv2.approxPolyDP(
                    contour, epsilon=polygon_epsilon, closed=True
                )
            points = contour.reshape(-1, 2)
            if len(points) < 3 or cv2.contourArea(contour) <= min_contour_area:
                continue
            shapes.append(
                {
                    "label": str(universal_label),
                    "points": points.astype(float).tolist(),
                    "group_id": instance_index,
                    "description": f"confidence={float(score):.6f}",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                }
            )
    return shapes


def build_labelme_annotation(
    image_path: PathLike,
    instance_result: Mapping[str, Any],
    *,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    min_contour_area: float = 0.0,
    polygon_epsilon: float = 2.0,
) -> Dict[str, Any]:
    """Return a Labelme JSON object for an image and its predictions."""
    masks = np.asarray(instance_result["masks"])
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}")
    mask_height, mask_width = masks.shape[-2:]
    image_height = mask_height if image_height is None else int(image_height)
    image_width = mask_width if image_width is None else int(image_width)
    if (image_height, image_width) != (mask_height, mask_width):
        raise ValueError(
            "image dimensions must match the restored prediction masks: "
            f"image={(image_height, image_width)}, masks={(mask_height, mask_width)}"
        )

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": instance_result_to_labelme_shapes(
            instance_result,
            min_contour_area=min_contour_area,
            polygon_epsilon=polygon_epsilon,
        ),
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }


def save_labelme_annotation(
    output_path: PathLike,
    image_path: PathLike,
    instance_result: Mapping[str, Any],
    *,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    min_contour_area: float = 0.0,
    polygon_epsilon: float = 2.0,
) -> Path:
    """Write predictions as a Labelme JSON file and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation = build_labelme_annotation(
        image_path,
        instance_result,
        image_height=image_height,
        image_width=image_width,
        min_contour_area=min_contour_area,
        polygon_epsilon=polygon_epsilon,
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(annotation, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path
