from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# MediaPipe Face Mesh landmark indices.  These are deliberately kept local so
# this module does not depend on the reference repository at runtime.
UPPER_LIP_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291)
LOWER_LIP_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291)
MOUTH_LANDMARKS = tuple(dict.fromkeys(UPPER_LIP_OUTER + LOWER_LIP_OUTER))
UPPER_LIP_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308)
LOWER_LIP_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
INNER_MOUTH_POLYGON = UPPER_LIP_INNER + tuple(reversed(LOWER_LIP_INNER))

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_landmarker_model(
    path: Path,
    *,
    download: bool = True,
    url: str = DEFAULT_MODEL_URL,
) -> Path:
    """Return a local Face Landmarker bundle, downloading it atomically if needed."""
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.stat().st_size > 0:
        return path
    if not download:
        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model not found: {path}. "
            "Run without --no-download-model or supply --landmarker-model."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".download", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"Downloading MediaPipe Face Landmarker model to {path}")
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty model from {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class MouthCrop:
    image_bgr: np.ndarray
    metadata: dict[str, Any]


class MouthCropper:
    """Detect one face and create a fixed-aspect mouth crop from lip landmarks."""

    def __init__(
        self,
        model_path: Path,
        *,
        output_size: tuple[int, int] = (640, 320),
        horizontal_margin: float = 0.45,
        vertical_margin: float = 1.0,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ) -> None:
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output_size dimensions must be positive")
        if horizontal_margin < 0 or vertical_margin < 0:
            raise ValueError("crop margins must be non-negative")

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for mouth cropping. Install it with `pip install mediapipe`."
            ) from exc

        self._mp = mp
        self.output_size = output_size
        self.horizontal_margin = horizontal_margin
        self.vertical_margin = vertical_margin
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(Path(model_path).resolve())),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "MouthCropper":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def crop_box(
        points_xy: np.ndarray,
        *,
        output_size: tuple[int, int],
        horizontal_margin: float,
        vertical_margin: float,
    ) -> tuple[float, float, float, float]:
        mouth = points_xy[np.asarray(MOUTH_LANDMARKS)]
        x_min, y_min = mouth.min(axis=0)
        x_max, y_max = mouth.max(axis=0)
        raw_width = max(float(x_max - x_min), 1.0)
        raw_height = max(float(y_max - y_min), 1.0)
        width = raw_width * (1.0 + 2.0 * horizontal_margin)
        height = raw_height * (1.0 + 2.0 * vertical_margin)

        target_aspect = output_size[0] / output_size[1]
        if width / height < target_aspect:
            width = height * target_aspect
        else:
            height = width / target_aspect

        center_x = float(x_min + x_max) / 2.0
        center_y = float(y_min + y_max) / 2.0
        return (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )

    def detect_landmarks(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Return the first face's landmarks in source pixels (N, 2) and the face count."""
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")

        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.face_landmarks:
            raise ValueError("MediaPipe did not detect a face")

        source_height, source_width = image_bgr.shape[:2]
        points = np.asarray(
            [(point.x * source_width, point.y * source_height) for point in result.face_landmarks[0]],
            dtype=np.float32,
        )
        return points, len(result.face_landmarks)

    @staticmethod
    def teeth_roi_box(
        points_xy: np.ndarray,
        image_hw: tuple[int, int],
        *,
        margin: float = 0.08,
        min_height_ratio: float = 0.25,
    ) -> tuple[int, int, int, int]:
        """Tight teeth ROI from the inner-lip opening, matching RF-DETR ``teeth_roi`` crops.

        Training crops are the union of tooth boxes plus ``eval_margin`` (0.08) per side,
        clipped to the image. Visible teeth lie inside the inner-lip polygon, so its bbox
        stands in for the tooth union. ``min_height_ratio`` keeps a nearly closed mouth
        from collapsing to a sliver (height >= ratio * width, grown around the center).
        """
        inner = points_xy[np.asarray(INNER_MOUTH_POLYGON)]
        x0, y0 = inner.min(axis=0)
        x1, y1 = inner.max(axis=0)
        width = max(float(x1 - x0), 1.0)
        height = max(float(y1 - y0), 1.0)
        if height < width * min_height_ratio:
            center_y = float(y0 + y1) / 2.0
            height = width * min_height_ratio
            y0, y1 = center_y - height / 2.0, center_y + height / 2.0
        x0, x1 = x0 - width * margin, x1 + width * margin
        y0, y1 = y0 - height * margin, y1 + height * margin
        image_h, image_w = image_hw
        return (
            int(max(0, np.floor(x0))),
            int(max(0, np.floor(y0))),
            int(min(image_w, np.ceil(x1))),
            int(min(image_h, np.ceil(y1))),
        )

    def crop(self, image_bgr: np.ndarray) -> MouthCrop:
        points, face_count = self.detect_landmarks(image_bgr)
        source_height, source_width = image_bgr.shape[:2]
        x0, y0, x1, y1 = self.crop_box(
            points,
            output_size=self.output_size,
            horizontal_margin=self.horizontal_margin,
            vertical_margin=self.vertical_margin,
        )
        box_width = x1 - x0
        box_height = y1 - y0
        output_width, output_height = self.output_size

        # Map source coordinates to the output crop. BORDER_REFLECT_101 avoids
        # black wedges when a generated face is close to an image edge.
        affine = np.asarray(
            [
                [output_width / box_width, 0.0, -x0 * output_width / box_width],
                [0.0, output_height / box_height, -y0 * output_height / box_height],
            ],
            dtype=np.float32,
        )
        crop = cv2.warpAffine(
            image_bgr,
            affine,
            (output_width, output_height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mouth = points[np.asarray(MOUTH_LANDMARKS)]
        inner_mouth = points[np.asarray(INNER_MOUTH_POLYGON)]
        inner_mouth_h = np.column_stack((inner_mouth, np.ones(len(inner_mouth), dtype=np.float32)))
        inner_mouth_crop = inner_mouth_h @ affine.T
        lip_width = max(float(mouth[:, 0].max() - mouth[:, 0].min()), 1.0)
        opening = float(np.linalg.norm(points[13] - points[14]) / lip_width)
        metadata = {
            "source_size": [source_width, source_height],
            "crop_size": [output_width, output_height],
            "crop_box_xyxy": [round(value, 4) for value in (x0, y0, x1, y1)],
            "source_to_crop_affine": affine.round(8).tolist(),
            # The crop is centered on the outer-lip bounding box. Its geometric
            # center is stable for left/right ordering under profile perspective.
            "mouth_center_crop_xy": [output_width / 2.0, output_height / 2.0],
            "inner_mouth_polygon_crop_xy": inner_mouth_crop.round(3).tolist(),
            "mouth_opening_ratio": round(opening, 6),
            "face_count": face_count,
        }
        return MouthCrop(image_bgr=crop, metadata=metadata)
