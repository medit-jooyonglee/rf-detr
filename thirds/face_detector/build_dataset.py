from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from .mouth_crop import MouthCropper, ensure_landmarker_model, sha256_file
except ImportError:  # Support direct execution: python thirds/face_detector/build_dataset.py
    from mouth_crop import MouthCropper, ensure_landmarker_model, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(r"D:\workspace\repositories\flux2\output\dental_data")
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "mouth_seg_dataset"
DEFAULT_LANDMARKER = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
DEFAULT_TEETH_REPO = REPO_ROOT / "thirds" / "teeth_seg_natural"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jfif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mouth crops and FDI-valued instance masks from FLUX2 dental images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "crop", "segment", "export"), default="all")
    parser.add_argument("--limit", type=int, help="Process only this many selected inputs (smoke test).")
    parser.add_argument("--start", type=int, default=0, help="Skip this many sorted source images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    crop = parser.add_argument_group("mouth crop")
    crop.add_argument("--landmarker-model", type=Path, default=DEFAULT_LANDMARKER)
    crop.add_argument(
        "--download-model", action=argparse.BooleanOptionalAction, default=True,
        help="Download the official MediaPipe model when it is missing.",
    )
    crop.add_argument("--crop-width", type=int, default=640)
    crop.add_argument("--crop-height", type=int, default=320)
    crop.add_argument("--horizontal-margin", type=float, default=0.45)
    crop.add_argument("--vertical-margin", type=float, default=1.0)
    crop.add_argument(
        "--min-mouth-opening", type=float, default=0.0,
        help="Reject smaller inner-lip opening/outer-lip-width ratios; 0 disables filtering.",
    )

    segment = parser.add_argument_group("tooth segmentation")
    segment.add_argument("--teeth-repo", type=Path, default=DEFAULT_TEETH_REPO)
    segment.add_argument("--checkpoint", type=Path, help="Defaults to TEETH_REPO/checkpoints/CP_teeth_seg.pth")
    segment.add_argument("--device", default="cuda:0")
    segment.add_argument("--resize", action=argparse.BooleanOptionalAction, default=False)
    segment.add_argument("--min-instance-pixels", type=int, default=80)
    segment.add_argument(
        "--min-instance-area-ratio", type=float, default=0.10,
        help="Drop regions smaller than this fraction of the median instance area.",
    )
    segment.add_argument(
        "--min-roi-overlap", type=float, default=0.5,
        help="Reject a model instance if less than this fraction lies inside the inner-mouth ROI.",
    )
    segment.add_argument("--overlay-alpha", type=float, default=0.45)
    segment.add_argument(
        "--prune-work", action="store_true",
        help="Remove per-sample model intermediates after a successful export.",
    )
    args = parser.parse_args()
    if args.start < 0 or args.limit is not None and args.limit < 1:
        parser.error("--start must be >= 0 and --limit must be >= 1")
    if args.crop_width < 1 or args.crop_height < 1:
        parser.error("crop dimensions must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be between 0 and 1")
    if not 0.0 <= args.min_roi_overlap <= 1.0:
        parser.error("--min-roi-overlap must be between 0 and 1")
    return args


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def numeric_id(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else fallback


def selected_sources(input_dir: Path, start: int, limit: int | None) -> list[Path]:
    sources = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    selected = sources[start:]
    return selected[:limit] if limit is not None else selected


def read_source_metadata(image_path: Path) -> dict[str, Any] | None:
    sidecar = image_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"value": value}
    except (OSError, json.JSONDecodeError) as exc:
        return {"read_error": str(exc)}


@dataclass(frozen=True)
class Instance:
    label: int
    area: int
    centroid_x: float
    centroid_y: float
    bbox_xywh: tuple[int, int, int, int]


def instance_statistics(labels: np.ndarray) -> list[Instance]:
    instances: list[Instance] = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        ys, xs = np.nonzero(labels == label)
        if not len(xs):
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        instances.append(
            Instance(
                label=label,
                area=int(len(xs)),
                centroid_x=float(xs.mean()),
                centroid_y=float(ys.mean()),
                bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
            )
        )
    return instances


def _split_rows(
    instances: list[Instance],
    image_height: int,
    mouth_center_y: float,
    single_row_hint: str,
) -> tuple[dict[int, str], dict[str, Any]]:
    if not instances:
        return {}, {"mode": "none", "gap": None}
    ordered = sorted(instances, key=lambda item: item.centroid_y)
    heights = np.asarray([item.bbox_xywh[3] for item in ordered], dtype=float)
    gaps = np.diff([item.centroid_y for item in ordered])
    min_gap = max(image_height * 0.055, float(np.median(heights)) * 0.28)
    if len(ordered) >= 4 and len(gaps):
        split = int(np.argmax(gaps)) + 1
        gap = float(gaps[split - 1])
        if gap >= min_gap and split >= 2 and len(ordered) - split >= 2:
            rows = {item.label: "upper" for item in ordered[:split]}
            rows.update({item.label: "lower" for item in ordered[split:]})
            return rows, {"mode": "two_rows", "gap": round(gap, 4), "min_gap": round(min_gap, 4)}
    mean_y = float(np.mean([item.centroid_y for item in ordered]))
    row = single_row_hint
    return {item.label: row for item in ordered}, {
        "mode": "single_row",
        "row": row,
        "mean_y": round(mean_y, 4),
        "mouth_center_y": round(mouth_center_y, 4),
    }


def assign_fdi(
    labels: np.ndarray,
    *,
    mouth_center_xy: tuple[float, float] | list[float] | None = None,
    min_instance_pixels: int = 80,
    min_instance_area_ratio: float = 0.10,
    single_row_hint: str = "upper",
    single_row_hint_source: str = "dataset_default",
    roi_polygon_xy: list[list[float]] | None = None,
    min_roi_overlap: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert positive model instance labels into an uint8 mask containing FDI codes."""
    height, width = labels.shape
    center_x, center_y = mouth_center_xy or (width / 2.0, height / 2.0)
    raw_instances = instance_statistics(labels)
    roi_rejected: list[dict[str, Any]] = []
    clipped_labels = labels.copy()
    if roi_polygon_xy:
        roi = np.zeros(labels.shape, dtype=np.uint8)
        polygon = np.rint(np.asarray(roi_polygon_xy, dtype=float)).astype(np.int32)
        cv2.fillPoly(roi, [polygon.reshape(-1, 1, 2)], 1)
        # A small dilation tolerates landmark/segmentation boundary mismatch.
        roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        clipped_labels[roi == 0] = 0
        for item in raw_instances:
            kept_pixels = int(np.sum(clipped_labels == item.label))
            overlap = kept_pixels / item.area
            if overlap < min_roi_overlap:
                clipped_labels[clipped_labels == item.label] = 0
                roi_rejected.append(
                    {"model_label": item.label, "roi_overlap": round(overlap, 6), "area": item.area}
                )
    all_instances = instance_statistics(clipped_labels)
    median_area = float(np.median([item.area for item in all_instances])) if all_instances else 0.0
    area_threshold = max(int(min_instance_pixels), int(math.ceil(median_area * min_instance_area_ratio)))
    instances = [item for item in all_instances if item.area >= area_threshold]
    dropped = [item.label for item in all_instances if item.area < area_threshold]
    if single_row_hint not in ("upper", "lower"):
        raise ValueError("single_row_hint must be 'upper' or 'lower'")
    rows, row_details = _split_rows(instances, height, float(center_y), single_row_hint)
    row_details["hint_source"] = single_row_hint_source

    # Image-left is the patient's right.  Codes are ordered from the central
    # incisor toward the molars as distance from the mouth midline increases.
    code_sequences = {
        ("upper", "image_left"): [11, 12, 13, 14, 15, 16, 17, 18],
        ("upper", "image_right"): [21, 22, 23, 24, 25, 26, 27, 28],
        ("lower", "image_left"): [41, 42, 43, 44, 45, 46, 47, 48],
        ("lower", "image_right"): [31, 32, 33, 34, 35, 36, 37, 38],
    }
    mapping: dict[int, int] = {}
    overflow: list[int] = []
    for row in ("upper", "lower"):
        for side in ("image_left", "image_right"):
            side_instances = [
                item
                for item in instances
                if rows.get(item.label) == row
                and (item.centroid_x < center_x if side == "image_left" else item.centroid_x >= center_x)
            ]
            side_instances.sort(key=lambda item: abs(item.centroid_x - center_x))
            codes = code_sequences[(row, side)]
            for index, item in enumerate(side_instances):
                if index < len(codes):
                    mapping[item.label] = codes[index]
                else:
                    overflow.append(item.label)

    fdi_mask = np.zeros(labels.shape, dtype=np.uint8)
    for label, fdi in mapping.items():
        fdi_mask[clipped_labels == label] = fdi
    for label in overflow:
        fdi_mask[clipped_labels == label] = 255  # explicit ignore label

    per_instance = []
    by_label = {item.label: item for item in all_instances}
    for label in sorted(mapping):
        item = by_label[label]
        per_instance.append(
            {
                "model_label": label,
                "fdi": mapping[label],
                "row": rows[label],
                "area": item.area,
                "centroid_xy": [round(item.centroid_x, 3), round(item.centroid_y, 3)],
                "bbox_xywh": list(item.bbox_xywh),
            }
        )

    side_counts = {
        key: sum(
            1
            for item in instances
            if rows.get(item.label) == key.split("_")[0]
            and (item.centroid_x < center_x if key.endswith("left") else item.centroid_x >= center_x)
        )
        for key in ("upper_left", "upper_right", "lower_left", "lower_right")
    }
    review_reasons = []
    if dropped:
        review_reasons.append("small_instances_dropped")
    if roi_rejected:
        review_reasons.append("instances_rejected_outside_mouth_roi")
    if overflow:
        review_reasons.append("more_than_eight_teeth_on_one_side")
    if not mapping:
        review_reasons.append("no_tooth_instances")
    if row_details.get("mode") == "single_row" and single_row_hint_source == "dataset_default":
        review_reasons.append("single_row_assumed_upper")
    for row in ("upper", "lower"):
        if abs(side_counts[f"{row}_left"] - side_counts[f"{row}_right"]) > 2:
            review_reasons.append(f"asymmetric_{row}_row")
    return fdi_mask, {
        "assignment": "heuristic_from_row_and_distance_to_mouth_midline",
        "mouth_center_xy": [round(float(center_x), 4), round(float(center_y), 4)],
        "area_threshold": area_threshold,
        "raw_instance_count": len(raw_instances),
        "roi_instance_count": len(all_instances),
        "roi_rejected_instances": roi_rejected,
        "min_roi_overlap": min_roi_overlap,
        "kept_instance_count": len(mapping),
        "dropped_model_labels": dropped,
        "overflow_model_labels": overflow,
        "row_split": row_details,
        "side_counts": side_counts,
        "instances": per_instance,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def color_for_fdi(code: int) -> tuple[int, int, int]:
    hue = int((code * 37) % 180)
    hsv = np.uint8([[[hue, 190, 245]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def render_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = image_bgr.copy()
    colored = image_bgr.copy()
    for code in sorted(int(value) for value in np.unique(mask) if value not in (0, 255)):
        region = mask == code
        colored[region] = color_for_fdi(code)
    foreground = (mask > 0) & (mask != 255)
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, colored, alpha, 0)
    overlay[foreground] = blended[foreground]
    for code in sorted(int(value) for value in np.unique(mask) if value not in (0, 255)):
        region_u8 = np.where(mask == code, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_for_fdi(code), 2, cv2.LINE_AA)
        moments = cv2.moments(region_u8)
        if moments["m00"]:
            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])
            cv2.putText(overlay, str(code), (x - 11, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, str(code), (x - 11, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def load_teeth_runtime(teeth_repo: Path, checkpoint: Path | None, device: str, resize: bool) -> dict[str, Any]:
    teeth_repo = teeth_repo.resolve()
    checkpoint = (checkpoint or teeth_repo / "checkpoints" / "CP_teeth_seg.pth").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Tooth segmentation checkpoint not found: {checkpoint}")
    if str(teeth_repo) not in sys.path:
        sys.path.insert(0, str(teeth_repo))
    import src.myTools as mts
    from src.makeup import TeethSeg
    from src.teethSeg import PseudoER

    config = {
        "DEFAULT": {"ROOT": str(teeth_repo), "DEVICE": device},
        "DATA": {"DIR": ""},
        "MODEL": {"WEIGHTS": str(checkpoint)},
        "EVAL": {"DIR": ""},
        "RESIZE": bool(resize),
    }
    return {
        "repo": teeth_repo,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": config,
        "mts": mts,
        "TeethSeg": TeethSeg,
        "PseudoER": PseudoER,
        "net": None,
    }


def segment_one(runtime: dict[str, Any], crop_path: Path, work_dir: Path, model_id: int) -> np.ndarray:
    mts = runtime["mts"]
    input_dir = work_dir.parent / "model_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    model_input = input_dir / f"{model_id}.png"
    input_changed = not model_input.exists() or model_input.read_bytes() != crop_path.read_bytes()
    if input_changed:
        shutil.copy2(crop_path, model_input)
        # A crop change invalidates every cached downstream stage.
        (work_dir / f"{model_id:06d}.pth").unlink(missing_ok=True)
    runtime["config"]["DATA"]["DIR"] = str(input_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    ts = runtime["TeethSeg"](str(work_dir), model_id, mts.SaveTools(str(work_dir)), runtime["config"])
    if "img" not in ts._dt:
        pseudo = runtime["PseudoER"](runtime["config"], model_id)
        if runtime["net"] is None:
            runtime["net"] = pseudo.setModel()
            runtime["net"].eval()
        import torch

        with torch.no_grad():
            input_tensor, output_tensor = pseudo.inference(runtime["net"])
        image = input_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
        output = output_tensor.cpu().squeeze().numpy()
        ts._dt.update({"img": image, "output": output, "per0": np.where(output > 0.5, 1.0, 0.0)})
        mts.saveFile(ts._dt, ts.path_dict)
        ts.sts.imwrite(image * 255, "input.png")
        ts.sts.imwrite(output * 255, "output.png")
    if "phi0" not in ts._dt:
        ts.initContour()
    if "phi_res" not in ts._dt:
        ts.snake()
    if "res" not in ts._dt:
        # The legacy TEM routine renders an internal Matplotlib debug figure.
        # Some large/degenerate contour sets can crash Windows inside the
        # native image resampler before Python can raise an exception.  The
        # rendered file is not consumed by this dataset pipeline, so suppress
        # only that save while preserving TEM's region computation.
        savecfg = ts.sts.savecfg
        ts.sts.savecfg = lambda *_args, **_kwargs: None
        try:
            ts.tem()
        finally:
            ts.sts.savecfg = savecfg
    return np.asarray(ts._dt["res"])


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def single_row_hint(source_metadata: dict[str, Any] | None) -> tuple[str, str]:
    text = json.dumps(source_metadata or {}, ensure_ascii=False).lower()
    has_upper = "upper teeth" in text
    has_lower = "lower teeth" in text
    if has_lower and not has_upper:
        return "lower", "source_prompt"
    if has_upper and not has_lower:
        return "upper", "source_prompt"
    # This dataset was generated for natural-smile/anterior segmentation and
    # the legacy model generally returns the upper arch when it finds one row.
    return "upper", "dataset_default"


def source_review_reasons(source_metadata: dict[str, Any] | None) -> list[str]:
    text = json.dumps(source_metadata or {}, ensure_ascii=False).lower()
    reasons = []
    if any(term in text for term in ("profile", "three-quarter", "head turned", "turned slightly")):
        reasons.append("non_frontal_pose_can_shift_fdi_side")
    if any(term in text for term in ("missing tooth", "missing a tooth", "missing a front tooth")):
        reasons.append("missing_tooth_can_shift_positional_fdi")
    return reasons


def write_manifests(output: Path) -> dict[str, int]:
    records = []
    for path in sorted((output / "metadata").glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    atomic_json(output / "manifest.json", records)
    jsonl = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (output / "manifest.jsonl").write_text(jsonl, encoding="utf-8")
    counts: dict[str, int] = {"total": len(records)}
    for record in records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    atomic_json(output / "summary.json", counts)
    return counts


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output = args.output.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for name in ("images", "masks", "overlays", "metadata", "work"):
        (output / name).mkdir(parents=True, exist_ok=True)
    sources = selected_sources(input_dir, args.start, args.limit)
    if not sources:
        raise RuntimeError("No input images matched the selection")

    cropper = None
    model_hash = None
    if args.stage in ("all", "crop"):
        model_path = ensure_landmarker_model(args.landmarker_model, download=args.download_model)
        model_hash = sha256_file(model_path)
        cropper = MouthCropper(
            model_path,
            output_size=(args.crop_width, args.crop_height),
            horizontal_margin=args.horizontal_margin,
            vertical_margin=args.vertical_margin,
        )
    teeth_runtime = None
    if args.stage in ("all", "segment"):
        teeth_runtime = load_teeth_runtime(args.teeth_repo, args.checkpoint, args.device, args.resize)

    print(f"Selected {len(sources)} image(s); output={output}; stage={args.stage}")
    started = time.time()
    try:
        for offset, source in enumerate(sources, start=args.start):
            model_id = numeric_id(source, offset)
            sample_id = f"{model_id:06d}"
            crop_path = output / "images" / f"{sample_id}.png"
            mask_path = output / "masks" / f"{sample_id}.png"
            overlay_path = output / "overlays" / f"{sample_id}.png"
            metadata_path = output / "metadata" / f"{sample_id}.json"
            work_dir = output / "work" / sample_id
            record = {
                "schema_version": 1,
                "id": sample_id,
                "source_filename": source.name,
                "source_image": str(source),
                "source_json_filename": source.with_suffix(".json").name,
                "source_metadata": read_source_metadata(source),
                "image": relative(crop_path, output),
                "mask": relative(mask_path, output),
                "overlay": relative(overlay_path, output),
                "mask_encoding": {"dtype": "uint8", "background": 0, "ignore": 255, "foreground_values": "FDI tooth codes"},
                "status": "processing",
                "timings_sec": {},
            }
            if metadata_path.is_file():
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                record.update(previous)
            try:
                if args.stage in ("all", "crop") and (args.overwrite or not crop_path.is_file()):
                    t0 = time.time()
                    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("OpenCV could not read the source image")
                    mouth_crop = cropper.crop(image)
                    if mouth_crop.metadata["mouth_opening_ratio"] < args.min_mouth_opening:
                        raise ValueError(
                            f"mouth opening ratio {mouth_crop.metadata['mouth_opening_ratio']:.4f} "
                            f"is below {args.min_mouth_opening:.4f}"
                        )
                    if not cv2.imwrite(str(crop_path), mouth_crop.image_bgr):
                        raise OSError(f"Could not write crop: {crop_path}")
                    record["crop"] = mouth_crop.metadata | {
                        "landmarker_model_sha256": model_hash,
                        "horizontal_margin": args.horizontal_margin,
                        "vertical_margin": args.vertical_margin,
                    }
                    record["timings_sec"]["crop"] = round(time.time() - t0, 4)

                if args.stage in ("all", "segment"):
                    if not crop_path.is_file():
                        raise FileNotFoundError(f"Crop missing; run --stage crop first: {crop_path}")
                    if args.overwrite or not mask_path.is_file():
                        t0 = time.time()
                        labels = segment_one(teeth_runtime, crop_path, work_dir, model_id)
                        center = record.get("crop", {}).get("mouth_center_crop_xy")
                        row_hint, row_hint_source = single_row_hint(record.get("source_metadata"))
                        roi_polygon = record.get("crop", {}).get("inner_mouth_polygon_crop_xy")
                        mask, assignment = assign_fdi(
                            labels,
                            mouth_center_xy=center,
                            min_instance_pixels=args.min_instance_pixels,
                            min_instance_area_ratio=args.min_instance_area_ratio,
                            single_row_hint=row_hint,
                            single_row_hint_source=row_hint_source,
                            roi_polygon_xy=roi_polygon,
                            min_roi_overlap=args.min_roi_overlap,
                        )
                        for reason in source_review_reasons(record.get("source_metadata")):
                            if reason not in assignment["review_reasons"]:
                                assignment["review_reasons"].append(reason)
                        assignment["review_required"] = bool(assignment["review_reasons"])
                        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                        overlay = render_overlay(image, mask, args.overlay_alpha)
                        if not cv2.imwrite(str(mask_path), mask):
                            raise OSError(f"Could not write mask: {mask_path}")
                        if not cv2.imwrite(str(overlay_path), overlay):
                            raise OSError(f"Could not write overlay: {overlay_path}")
                        record["segmentation"] = assignment | {
                            "checkpoint": str(teeth_runtime["checkpoint"]),
                            "checkpoint_sha256": teeth_runtime["checkpoint_sha256"],
                            "device": args.device,
                        }
                        record["timings_sec"]["segment"] = round(time.time() - t0, 4)
                        if args.prune_work:
                            shutil.rmtree(work_dir)

                if args.stage == "export":
                    if not crop_path.is_file() or not mask_path.is_file():
                        raise FileNotFoundError("Export requires both an image crop and FDI mask")
                    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
                        raise ValueError("Image/mask pair is unreadable or has mismatched dimensions")
                    cv2.imwrite(str(overlay_path), render_overlay(image, mask, args.overlay_alpha))

                has_crop = crop_path.is_file()
                has_mask = mask_path.is_file()
                if has_crop and has_mask:
                    record["status"] = (
                        "review" if record.get("segmentation", {}).get("review_required") else "complete"
                    )
                else:
                    record["status"] = "cropped" if has_crop else "failed"
                record.pop("error", None)
                print(f"[{sample_id}] {source.name}: {record['status']}")
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                print(f"[{sample_id}] {source.name}: FAILED: {exc}", file=sys.stderr)
                if args.fail_fast:
                    atomic_json(metadata_path, record)
                    raise
            atomic_json(metadata_path, record)
    finally:
        if cropper is not None:
            cropper.close()

    counts = write_manifests(output)
    print(f"Finished in {time.time() - started:.1f}s: {counts}")
    return 0 if counts.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
