"""두 가지를 검증한다 (사용자 가설):

1. "ROI의 x-center가 1차 검출 상태(구치부 일부 누락 등)에 따라 흔들린다" -- 11/21(중절치) 경계를
   origin으로 고정하고, 좌/우 중 더 넓은 쪽 폭으로 대칭 crop을 만드는 center-aware ROI가
   기존(단순 bbox 합집합) ROI보다 나은지 cbct_pano/seoul_pano 전체에서 정량 비교.
2. "pass-2(crop 후 재추론)만 confidence threshold를 낮추면 recall이 늘지만 오분류도 늘지 않나"
   -- pass-2 confidence_threshold를 0.3/0.4/0.5로 바꿔가며 평가 (pass-1은 배포값 0.5 고정).

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/debug_two_pass_center.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import _predict_single_pass, DEFAULT_TWO_PASS_ROI_MARGIN, DEFAULT_TWO_PASS_SHRINK_RATIO
from rfdetr.deploy.evaluate import MODELS, load_val_coco, METRIC_KEYS
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import label2fdi
from rfdetr.datasets.xray_crop_aug import teeth_roi_box, _clip_box
from tools.compare_own1022 import metrics, aggregate
from trainer import image_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]


def center_aware_crop_box(boxes_xyxy, fdi_labels, img_hw, eval_margin=DEFAULT_TWO_PASS_ROI_MARGIN):
    """11/21(중절치) 경계를 x 원점으로 고정하고, 원점 기준 좌/우 중 더 넓은 쪽 폭으로 대칭
    crop을 만든다 (구치부 일부가 1차에서 누락돼도 crop 중심이 한쪽으로 쏠리지 않도록).
    11/21 둘 다 없으면 기존 합집합 중심으로 폴백."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(boxes_xyxy) == 0:
        return None
    x0u, y0u = boxes_xyxy[:, 0].min(), boxes_xyxy[:, 1].min()
    x1u, y1u = boxes_xyxy[:, 2].max(), boxes_xyxy[:, 3].max()
    fdi_labels = np.asarray(fdi_labels).reshape(-1)
    has11, has21 = (fdi_labels == 11), (fdi_labels == 21)
    if has11.any() and has21.any():
        cx = (boxes_xyxy[has11][0, [0, 2]].mean() + boxes_xyxy[has21][0, [0, 2]].mean()) / 2
    elif has11.any():
        cx = boxes_xyxy[has11][0, 0]
    elif has21.any():
        cx = boxes_xyxy[has21][0, 2]
    else:
        cx = (x0u + x1u) / 2
    half_w = max(cx - x0u, x1u - cx)
    bw, bh = x1u - x0u, y1u - y0u
    m = eval_margin
    box = [cx - half_w - bw * m, y0u - bh * m, cx + half_w + bw * m, y1u + bh * m]
    return _clip_box(box, img_hw)


def two_pass_with_crop_fn(img, model, convert_options, crop_fn, roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO,
                           pass2_confidence_threshold=0.5):
    h, w = img.shape[:2]
    _, _, inst1 = _predict_single_pass(img, model, convert_options)
    if inst1['boxes'].shape[0] == 0:
        return inst1, False
    crop_box = crop_fn(inst1['boxes'], inst1['fdi'], (h, w))
    if crop_box is None:
        return inst1, False
    x0, y0, x1, y1 = crop_box
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    crop_w, crop_h = x1i - x0i, y1i - y0i
    if crop_w <= 0 or crop_h <= 0 or crop_w * crop_h >= roi_shrink_ratio * w * h:
        return inst1, False
    crop_img = img[y0i:y1i, x0i:x1i]
    _, _, inst2 = _predict_single_pass(crop_img, model, convert_options,
                                        confidence_threshold=pass2_confidence_threshold)
    offset = np.array([x0i, y0i, x0i, y0i], dtype=inst2['boxes'].dtype)
    inst2['boxes'] = inst2['boxes'] + offset
    full_masks = np.zeros((inst2['masks'].shape[0], h, w), dtype=bool)
    full_masks[:, y0i:y1i, x0i:x1i] = inst2['masks']
    inst2['masks'] = full_masks
    return inst2, True


def naive_crop_fn(boxes_xyxy, fdi_labels, img_hw, eval_margin=DEFAULT_TWO_PASS_ROI_MARGIN):
    return teeth_roi_box(boxes_xyxy, img_hw, train=False, cfg={'eval_margin': eval_margin})


def run_eval(model, convert_options, datasets, variants):
    """variants: dict(name -> callable(img, model, convert_options) -> (inst, used))."""
    rows = {name: {} for name in variants}
    rows['one_pass'] = {}
    n_by_cat = {}
    n_images = 0
    for source_name, coco in datasets:
        for im in coco.imgs.values():
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            cat = categorize(source_name, im['file_name'], im['width'], im['height'])
            if cat not in ('cbct_pano', 'seoul_pano'):
                continue
            img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])

            _, _, inst0 = _predict_single_pass(img, model, convert_options)
            u0 = label2fdi[inst0['fdi']]
            rows['one_pass'].setdefault(cat, []).append(
                metrics(gt, inst0['masks'], u0, inst0['scores'], inst0['boxes'], anns, h, w))

            for name, fn in variants.items():
                inst, used = fn(img, model, convert_options)
                u = label2fdi[inst['fdi']]
                rows[name].setdefault(cat, []).append(
                    metrics(gt, inst['masks'], u, inst['scores'], inst['boxes'], anns, h, w))

            n_by_cat[cat] = n_by_cat.get(cat, 0) + 1
            n_images += 1
            if n_images % 20 == 0:
                print(source_name, n_images, flush=True)
    return rows, n_by_cat


def report(rows, n_by_cat, variant_names, tag):
    summary = {}
    print(f'\n=== {tag} ===')
    all_cats = sorted(rows['one_pass'])
    for cat in all_cats + ['ALL']:
        if cat == 'ALL':
            n = sum(n_by_cat.values())
            getrows = lambda name: [r for rs in rows[name].values() for r in rs]
        else:
            n = n_by_cat[cat]
            getrows = lambda name: rows[name][cat]
        print(f'  {cat:<12s} n={n}')
        agg = {}
        for name in ['one_pass'] + variant_names:
            m = aggregate(getrows(name))
            agg[name] = {k: m[k] for k in METRIC_KEYS}
            print(f"    {name:<16s} dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
                  f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}")
        summary[cat] = dict(n_images=n, **agg)
    return summary


def main():
    datasets = load_val_coco()
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    # part A: naive(기존) vs center-aware crop, pass-2 threshold는 배포값 0.5로 고정
    variants_a = {
        'naive_crop': lambda img, m, co: two_pass_with_crop_fn(img, m, co, naive_crop_fn),
        'center_aware_crop': lambda img, m, co: two_pass_with_crop_fn(img, m, co, center_aware_crop_box),
    }
    rows_a, n_by_cat = run_eval(model, convert_options, datasets, variants_a)
    summary_a = report(rows_a, n_by_cat, list(variants_a), 'naive vs center-aware crop (pass2_conf=0.5)')
    (OUT_DIR / 'summary_center_aware_crop.json').write_text(json.dumps(summary_a, indent=2))

    # part B: naive crop 유지, pass-2 confidence threshold만 sweep (pass-1은 0.5 고정)
    variants_b = {
        f'pass2_conf_{th}'.replace('.', ''): (
            lambda img, m, co, th=th: two_pass_with_crop_fn(img, m, co, naive_crop_fn, pass2_confidence_threshold=th)
        )
        for th in (0.3, 0.4, 0.5)
    }
    rows_b, n_by_cat_b = run_eval(model, convert_options, datasets, variants_b)
    summary_b = report(rows_b, n_by_cat_b, list(variants_b), 'pass-2 confidence threshold sweep (naive crop)')
    (OUT_DIR / 'summary_pass2_threshold_sweep.json').write_text(json.dumps(summary_b, indent=2))

    print('\nsaved:', OUT_DIR / 'summary_center_aware_crop.json')
    print('saved:', OUT_DIR / 'summary_pass2_threshold_sweep.json')


if __name__ == '__main__':
    main()
