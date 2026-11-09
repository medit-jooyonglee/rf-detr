"""Deployment-readiness check: is outputs/peri_data_fusion_1106_ce fit to
replace the currently deployed baseline (panoramic_overlapfix)?

Reuses the exact same predict.py inference pipeline as evaluate.py (see that
file's docstring for the fdi_refinement caveats -- not repeated here), but
breaks results down by data type instead of one combined average, over the
FULL 1106 valid split (not a sample):

  cbct_pano   -- cbct_ios_dcm_latest_1106/valid, large-res (full panoramic)
  cbct_peri   -- cbct_ios_dcm_latest_1106/valid, <=1000x1000 (periapical-like)
  seoul_pano  -- seoul_univ_inspect_1106/valid, filename contains "pano"
  seoul_peri  -- seoul_univ_inspect_1106/valid, filename contains "peri"
  seoul_other -- seoul_univ_inspect_1106/valid, neither (implant closeups etc.)

fdi_refinement is left at its real deployed default (True) only -- evaluate.py
already established it's strictly neutral-to-positive and removes all
duplicate FDI numbers, so there is no need to re-run the off case here.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/evaluate_by_type.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import resize_image, MASK_PROBABILITY_THRESHOLD
from rfdetr.deploy.evaluate import (
    XRAY_DATA, PERI_DATA, MODELS, load_val_coco, count_duplicate_teeth, METRIC_KEYS,
)
from rfdetr.engine import draw_preditions_boxes
from rfdetr.datasets.xraypanoramic import label2fdi
from tools.compare_own1022 import metrics, aggregate
from trainer import image_utils, torch_utils

OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs/eval_fdi_refinement'


def categorize(source_name, file_name, width, height):
    if source_name == 'cbct_ios_dcm_latest_1106':
        return 'cbct_peri' if (width <= 1000 and height <= 1000) else 'cbct_pano'
    name = file_name.lower()
    if 'peri' in name:
        return 'seoul_peri'
    if 'pano' in name:
        return 'seoul_pano'
    return 'seoul_other'


def run_model(name, checkpoint_path, datasets, device='cuda'):
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=checkpoint_path, device=device)

    rows_by_cat = {}
    dup_by_cat = {}
    n_images = 0
    for source_name, coco in datasets:
        images = list(coco.imgs.values())
        for im in images:
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            cat = categorize(source_name, im['file_name'], im['width'], im['height'])

            rsz_img, content_box = resize_image(img, stride=64)
            img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
            img_tensor = torch_utils.data_convert(img2[None])
            with torch.no_grad():
                outputs = model(img_tensor)
                if not isinstance(outputs, dict):
                    outputs = dict(zip(('pred_boxes', 'pred_logits', 'pred_masks'), outputs))

            instance_results = []
            draw_preditions_boxes(
                img_tensor, outputs, save=False, origin_size=img.shape[:2],
                segmentation_mode='crop_and_resize',
                mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                segmentation_crop_box_scale=1.15,
                paste_masks_at_original_size=True,
                input_content_box=content_box,
                drawing_mode='polygon',
                instance_results=instance_results,
                fdi_refinement=True,
            )
            r = instance_results[0]
            universal_labels = label2fdi[r['fdi']]

            gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])
            row = metrics(gt, r['masks'], universal_labels, r['scores'], r['boxes'], anns, h, w)
            row['file_name'] = im['file_name']
            rows_by_cat.setdefault(cat, []).append(row)
            dup_by_cat[cat] = dup_by_cat.get(cat, 0) + count_duplicate_teeth(universal_labels)

            n_images += 1
            if n_images % 20 == 0:
                print(name, source_name, n_images, flush=True)

    overall_by_cat = {cat: aggregate(rows) for cat, rows in rows_by_cat.items()}
    all_rows = [row for rows in rows_by_cat.values() for row in rows]
    overall_by_cat['ALL'] = aggregate(all_rows)
    dup_by_cat['ALL'] = sum(dup_by_cat.values())
    return overall_by_cat, dup_by_cat, {cat: len(rows) for cat, rows in rows_by_cat.items()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_val_coco()

    summary = {}
    for name, ckpt in MODELS.items():
        print(f'\n=== {name} ({ckpt}) ===')
        overall_by_cat, dup_by_cat, n_by_cat = run_model(name, ckpt, datasets)
        summary[name] = {}
        for cat in sorted(overall_by_cat, key=lambda c: (c != 'ALL', c)):
            m = overall_by_cat[cat]
            n = n_by_cat.get(cat, sum(n_by_cat.values()) if cat == 'ALL' else 0)
            print(f"  {cat:<12s} n={n:<4d} dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
                  f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}  "
                  f"duplicate_teeth={dup_by_cat[cat]}")
            summary[name][cat] = dict({k: m[k] for k in METRIC_KEYS}, n_images=n, duplicate_teeth=dup_by_cat[cat])

    out_path = OUT_DIR / 'summary_by_type.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nsaved summary:', out_path)


if __name__ == '__main__':
    main()
