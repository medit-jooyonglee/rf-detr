"""Does rfdetr/deploy/predict.py's two_pass_predict() (one-pass full-image
inference vs. two-pass: coarse full-image pass -> teeth_roi crop -> high-res
second pass) actually improve segmentation accuracy on panoramic (pano)
images? Peri-type images are included as a control -- two_pass_predict's own
crop-shrink check (DEFAULT_TWO_PASS_SHRINK_RATIO) should make it fall back to
one-pass automatically there, so results should barely move.

Model: outputs/peri_data_fusion_1106_ce/checkpoint_best_ema.pth only (the
deployment candidate already established in outputs/eval_fdi_refinement/).

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/evaluate_two_pass.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import (
    MASK_PROBABILITY_THRESHOLD, _predict_single_pass, two_pass_predict,
    DEFAULT_TWO_PASS_ROI_MARGIN, DEFAULT_TWO_PASS_SHRINK_RATIO,
)
from rfdetr.deploy.evaluate import MODELS, load_val_coco, METRIC_KEYS
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import label2fdi
from tools.compare_own1022 import metrics, aggregate
from trainer import image_utils

OUT_DIR = Path(__file__).resolve().parents[2] / 'outputs/eval_fdi_refinement'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_val_coco()

    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    rows = {'one_pass': {}, 'two_pass': {}}
    two_pass_used = {}
    n_by_cat = {}
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
            gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])

            _, _, inst1 = _predict_single_pass(img, model, convert_options, drawing_mode='polygon')
            u1 = label2fdi[inst1['fdi']]
            row1 = metrics(gt, inst1['masks'], u1, inst1['scores'], inst1['boxes'], anns, h, w)
            rows['one_pass'].setdefault(cat, []).append(row1)

            _, _, inst2, used_two_pass, _ = two_pass_predict(
                img, model, drawing_mode='polygon', convert_options=convert_options,
                roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN, roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO)
            u2 = label2fdi[inst2['fdi']]
            row2 = metrics(gt, inst2['masks'], u2, inst2['scores'], inst2['boxes'], anns, h, w)
            rows['two_pass'].setdefault(cat, []).append(row2)
            two_pass_used[cat] = two_pass_used.get(cat, 0) + int(used_two_pass)
            n_by_cat[cat] = n_by_cat.get(cat, 0) + 1

            n_images += 1
            if n_images % 20 == 0:
                print(source_name, n_images, flush=True)

    summary = {}
    print(f'\n=== {MODEL_NAME}: one-pass vs two-pass ===')
    all_cats = sorted(rows['one_pass'])
    for cat in all_cats + ['ALL']:
        if cat == 'ALL':
            r1 = [r for rs in rows['one_pass'].values() for r in rs]
            r2 = [r for rs in rows['two_pass'].values() for r in rs]
            n = sum(n_by_cat.values())
            used = sum(two_pass_used.values())
        else:
            r1, r2 = rows['one_pass'][cat], rows['two_pass'][cat]
            n = n_by_cat[cat]
            used = two_pass_used[cat]
        m1, m2 = aggregate(r1), aggregate(r2)
        print(f"  {cat:<12s} n={n:<4d} two_pass_used={used}/{n}")
        print(f"    one-pass  dice={m1['instance_dice']:.4f}  iou={m1['instance_iou']:.4f}  "
              f"boundary_f1={m1['boundary_f1']:.4f}  contact_f1={m1['contact_f1']:.4f}")
        print(f"    two-pass  dice={m2['instance_dice']:.4f}  iou={m2['instance_iou']:.4f}  "
              f"boundary_f1={m2['boundary_f1']:.4f}  contact_f1={m2['contact_f1']:.4f}")
        summary[cat] = dict(
            n_images=n, two_pass_used=used,
            one_pass={k: m1[k] for k in METRIC_KEYS},
            two_pass={k: m2[k] for k in METRIC_KEYS},
        )

    out_path = OUT_DIR / 'summary_two_pass.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nsaved summary:', out_path)


if __name__ == '__main__':
    main()
