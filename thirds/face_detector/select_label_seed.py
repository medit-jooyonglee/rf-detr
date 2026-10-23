from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "artifacts" / "teeth_crop_dataset"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "teeth_label_seed_50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a diverse annotation seed set with clearly exposed lower teeth."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--candidate-multiplier", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    return args


def lower_teeth_metrics(image: np.ndarray, record: dict[str, Any]) -> dict[str, float]:
    height, width = image.shape[:2]
    polygon = np.asarray(record["crop"]["inner_mouth_polygon_crop_xy"], dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    inner_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(inner_mask, [np.rint(polygon).astype(np.int32)], 255)
    y_center = float(record["crop"]["mouth_center_crop_xy"][1])
    inner_top = float(polygon[:, 1].min())
    inner_bottom = float(polygon[:, 1].max())
    lower_start = max(y_center + 5.0, inner_top + 0.54 * (inner_bottom - inner_top))
    lower_mask = inner_mask.copy()
    lower_mask[: max(0, int(round(lower_start))), :] = 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_values = gray[lower_mask > 0]
    if lower_values.size == 0:
        return {"score": 0.0, "area_ratio": 0.0, "width_ratio": 0.0, "components": 0.0}

    # Teeth are relatively bright and neutral compared with the dark oral cavity
    # and saturated lips/gums. The adaptive threshold keeps yellowed teeth eligible.
    adaptive_l = max(92.0, float(np.percentile(lower_values, 58)))
    tooth_like = (
        (lower_mask > 0)
        & (lab[:, :, 0] >= adaptive_l)
        & (hsv[:, :, 1] <= 115)
        & (lab[:, :, 1] >= 104)
        & (lab[:, :, 1] <= 146)
    ).astype(np.uint8)
    tooth_like = cv2.morphologyEx(tooth_like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(tooth_like, 8)
    cleaned = np.zeros_like(tooth_like)
    kept_components = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 18:
            cleaned[labels == label] = 1
            kept_components += 1

    lower_area = max(1, int(np.count_nonzero(lower_mask)))
    tooth_area = int(np.count_nonzero(cleaned))
    area_ratio = tooth_area / lower_area
    ys, xs = np.nonzero(cleaned)
    poly_width = max(1.0, float(polygon[:, 0].max() - polygon[:, 0].min()))
    width_ratio = 0.0 if xs.size == 0 else float(xs.max() - xs.min() + 1) / poly_width
    vertical_ratio = 0.0 if ys.size == 0 else float(np.percentile(ys, 90) - lower_start) / max(1.0, inner_bottom - lower_start)

    metadata = record.get("source_metadata") or {}
    expression = str(metadata.get("expression", "")).lower()
    shot = str(metadata.get("shot", "")).lower()
    expression_bonus = 0.15 if "laughing" in expression else 0.09 if "broadly" in expression else -0.55

    area_term = min(area_ratio / 0.28, 1.0)
    width_term = min(width_ratio / 0.72, 1.0)
    depth_term = min(max(vertical_ratio, 0.0) / 0.23, 1.0)
    component_term = min(kept_components / 4.0, 1.0)
    score = (
        0.42 * area_term
        + 0.28 * width_term
        + 0.15 * depth_term
        + 0.15 * component_term
        + expression_bonus
    )
    return {
        "score": float(score),
        "area_ratio": float(area_ratio),
        "width_ratio": float(width_ratio),
        "vertical_ratio": float(vertical_ratio),
        "components": float(kept_components),
    }


def visual_feature(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (32, 16), interpolation=cv2.INTER_AREA)
    feature = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1)
    feature -= feature.mean()
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def category(record: dict[str, Any], key: str) -> str:
    return str((record.get("source_metadata") or {}).get(key, "unknown"))


def select_diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(candidates) <= count:
        return candidates

    scores = np.asarray([item["selection_metrics"]["score"] for item in candidates], np.float32)
    lo, hi = float(scores.min()), float(scores.max())
    normalized_scores = (scores - lo) / max(hi - lo, 1e-6)
    selected: list[int] = []
    category_counts: dict[str, Counter[str]] = {
        key: Counter() for key in ("shot", "age_group", "ethnicity", "gender", "dental_condition")
    }

    while len(selected) < count:
        best_index = -1
        best_utility = -math.inf
        for index, item in enumerate(candidates):
            if index in selected:
                continue
            shot = category(item, "shot")
            clinical_count = sum(
                value for name, value in category_counts["shot"].items() if "cheek retractors" in name.lower()
            )
            if "cheek retractors" in shot.lower() and clinical_count >= max(6, count // 5):
                continue
            caps = {
                "shot": max(5, math.ceil(count / 8)),
                "age_group": max(8, math.ceil(count / 4)),
                "ethnicity": max(6, math.ceil(count / 6)),
                "gender": math.ceil(count * 0.56),
                "dental_condition": max(4, math.ceil(count / 10)),
            }
            if any(category_counts[key][category(item, key)] >= cap for key, cap in caps.items()):
                continue
            if not selected:
                visual_novelty = 1.0
            else:
                visual_novelty = min(
                    float(np.linalg.norm(item["_feature"] - candidates[chosen]["_feature"]))
                    for chosen in selected
                ) / 2.0
            category_novelty = np.mean([
                1.0 / (1.0 + category_counts[key][category(item, key)])
                for key in category_counts
            ])
            utility = 0.44 * float(normalized_scores[index]) + 0.30 * visual_novelty + 0.26 * category_novelty
            if utility > best_utility:
                best_index, best_utility = index, utility

        selected.append(best_index)
        for key, counter in category_counts.items():
            counter[category(candidates[best_index], key)] += 1
    return [candidates[index] for index in selected]


def write_contact_sheets(selected: list[dict[str, Any]], image_dir: Path, output_dir: Path) -> None:
    tile_width, tile_height = 320, 190
    columns, rows = 5, 5
    for page_start in range(0, len(selected), columns * rows):
        page = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
        for offset, item in enumerate(selected[page_start: page_start + columns * rows]):
            image = cv2.imread(str(image_dir / item["image"]))
            if image is None:
                continue
            preview = cv2.resize(image, (tile_width, 160), interpolation=cv2.INTER_AREA)
            row, column = divmod(offset, columns)
            x, y = column * tile_width, row * tile_height
            page[y: y + 160, x: x + tile_width] = preview
            label = f'{page_start + offset + 1:02d}  {item["id"]}  score={item["selection_metrics"]["score"]:.2f}'
            cv2.putText(page, label, (x + 6, y + 181), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"contact_sheet_{page_start // (columns * rows) + 1:02d}.jpg"), page)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    image_output = output / "images"
    mask_output = output / "masks"
    metadata_output = output / "metadata"
    image_output.mkdir(parents=True)
    mask_output.mkdir(parents=True)
    metadata_output.mkdir(parents=True)

    candidates: list[dict[str, Any]] = []
    for metadata_path in sorted((dataset / "metadata").glob("*.json")):
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record.get("status") != "cropped":
            continue
        image_path = dataset / record["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        metrics = lower_teeth_metrics(image, record)
        record["selection_metrics"] = metrics
        record["_feature"] = visual_feature(image)
        candidates.append(record)

    pool_size = min(len(candidates), max(args.count, args.count * args.candidate_multiplier))
    candidates.sort(key=lambda item: item["selection_metrics"]["score"], reverse=True)
    pool = candidates[:pool_size]
    selected = select_diverse(pool, min(args.count, len(pool)))
    selected.sort(key=lambda item: item["selection_metrics"]["score"], reverse=True)

    manifest: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        clean_item = {key: value for key, value in item.items() if not key.startswith("_")}
        clean_item["selection_rank"] = rank
        source_image = dataset / item["image"]
        target_name = f'{rank:03d}_{item["id"]}.png'
        clean_item["dataset_image"] = item["image"]
        clean_item["image"] = f"images/{target_name}"
        clean_item["mask"] = f"masks/{target_name}"
        clean_item.pop("overlay", None)
        clean_item["mask_encoding"] = {
            "dtype": "uint8",
            "background": 0,
            "teeth": 255,
        }
        clean_item["annotation_status"] = "unlabeled"
        shutil.copy2(source_image, image_output / target_name)
        (metadata_output / f'{rank:03d}_{item["id"]}.json').write_text(
            json.dumps(clean_item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append(clean_item)

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_contact_sheets(selected, dataset, output)
    summary = {
        "dataset": str(dataset),
        "eligible_crops": len(candidates),
        "candidate_pool": pool_size,
        "selected": len(selected),
        "selection_goal": "clearly exposed lower teeth with visual and metadata diversity",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
