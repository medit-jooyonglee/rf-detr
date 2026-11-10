"""two-pass에서 pano 성능이 떨어지는 원인을 더 깊게 판다.

1. GT-box 기반 crop(학습 때 oracle crop과 동일 규칙) vs 현재처럼 1차 예측 박스 기반 crop으로
   2차를 돌렸을 때 segmentation 정확도를 비교한다. GT crop도 나쁘면 crop-resize+재추론 자체의
   문제, GT crop만 좋으면 "1차 예측 불완전성"이 원인이라는 뜻.
2. seoul_pano_4 (000004_993944 pano 1.jpg)에서 사라진 하악 전치부(31,32,41,42)의 GT bbox가
   1차가 만든 crop_box 안에 기하학적으로 들어있었는지 확인.
3. 2차(크롭) 추론의 raw output(NMS/threshold 이전)에 그 치아 위치에 신호가 있었는지 확인 --
   threshold/NMS에서 걸러진 건지, 애초에 모델이 crop 스케일에서 그 위치를 전혀 못 찾은 건지.
4. 2차 입력(크롭+리사이즈 이미지)을 디버깅용으로 저장.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/debug_two_pass_roi.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import (
    resize_image, MASK_PROBABILITY_THRESHOLD, _predict_single_pass, two_pass_predict,
    DEFAULT_TWO_PASS_ROI_MARGIN, DEFAULT_TWO_PASS_SHRINK_RATIO,
)
from rfdetr.deploy.evaluate import MODELS, load_val_coco, METRIC_KEYS
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import label2fdi, universal_to_fdi
from rfdetr.datasets.xray_crop_aug import teeth_roi_box
from tools.compare_own1022 import metrics, aggregate
from trainer import image_utils, torch_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement'
DEBUG_DIR = OUT_DIR / 'two_pass_debug' / 'roi_inputs'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]
TARGET_FILE = '000004_993944 pano 1.jpg'
TARGET_UNIVERSAL = {23, 24, 25, 26}  # FDI 32, 31, 41, 42


def gt_crop_predict(img, model, convert_options, gt_boxes_xyxy, drawing_mode='polygon',
                     roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN):
    """GT bbox 합집합(학습 때 oracle teeth_roi crop과 동일 규칙, train=False 고정 margin)으로
    crop 후 1회 추론. two_pass_predict와 동일한 후처리(점수 threshold/NMS/fdi_refinement 포함)를
    거친 instance_result를 원본 좌표로 돌려준다."""
    h, w = img.shape[:2]
    crop_box = teeth_roi_box(gt_boxes_xyxy, (h, w), train=False, cfg={'eval_margin': roi_margin})
    if crop_box is None:
        return None, None
    x0, y0, x1, y1 = crop_box
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    crop_img = img[y0i:y1i, x0i:x1i]
    _, _, inst = _predict_single_pass(crop_img, model, convert_options, drawing_mode=drawing_mode)
    offset = np.array([x0i, y0i, x0i, y0i], dtype=inst['boxes'].dtype)
    inst['boxes'] = inst['boxes'] + offset
    full_masks = np.zeros((inst['masks'].shape[0], h, w), dtype=bool)
    full_masks[:, y0i:y1i, x0i:x1i] = inst['masks']
    inst['masks'] = full_masks
    return inst, (x0i, y0i, x1i, y1i)


def gt_boxes_xyxy_from_anns(anns):
    boxes = []
    for a in anns:
        x, y, w, h = a['bbox']
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def part1_quantitative_gt_vs_pred_crop(model, convert_options, datasets):
    """cbct_pano / seoul_pano에서 one-pass vs pred-crop two-pass vs GT-crop two-pass 비교."""
    rows = {'one_pass': {}, 'pred_crop': {}, 'gt_crop': {}}
    used = {'pred_crop': {}, 'gt_crop': {}}
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
            gt_boxes = gt_boxes_xyxy_from_anns(anns)

            _, _, inst1 = _predict_single_pass(img, model, convert_options)
            u1 = label2fdi[inst1['fdi']]
            rows['one_pass'].setdefault(cat, []).append(
                metrics(gt, inst1['masks'], u1, inst1['scores'], inst1['boxes'], anns, h, w))

            _, _, inst2, used2, _ = two_pass_predict(
                img, model, convert_options=convert_options,
                roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN, roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO)
            u2 = label2fdi[inst2['fdi']]
            rows['pred_crop'].setdefault(cat, []).append(
                metrics(gt, inst2['masks'], u2, inst2['scores'], inst2['boxes'], anns, h, w))
            used['pred_crop'][cat] = used['pred_crop'].get(cat, 0) + int(used2)

            inst3, crop_box3 = gt_crop_predict(img, model, convert_options, gt_boxes)
            if inst3 is not None:
                u3 = label2fdi[inst3['fdi']]
                rows['gt_crop'].setdefault(cat, []).append(
                    metrics(gt, inst3['masks'], u3, inst3['scores'], inst3['boxes'], anns, h, w))
                used['gt_crop'][cat] = used['gt_crop'].get(cat, 0) + 1

            n_by_cat[cat] = n_by_cat.get(cat, 0) + 1
            n_images += 1
            if n_images % 20 == 0:
                print(source_name, n_images, flush=True)

    summary = {}
    print(f'\n=== {MODEL_NAME}: one-pass vs pred-crop two-pass vs GT-crop two-pass ===')
    all_cats = sorted(rows['one_pass'])
    for cat in all_cats + ['ALL']:
        if cat == 'ALL':
            r1 = [r for rs in rows['one_pass'].values() for r in rs]
            r2 = [r for rs in rows['pred_crop'].values() for r in rs]
            r3 = [r for rs in rows['gt_crop'].values() for r in rs]
            n = sum(n_by_cat.values())
        else:
            r1, r2, r3 = rows['one_pass'][cat], rows['pred_crop'][cat], rows['gt_crop'][cat]
            n = n_by_cat[cat]
        m1, m2, m3 = aggregate(r1), aggregate(r2), aggregate(r3)
        print(f"  {cat:<12s} n={n}")
        print(f"    one-pass        dice={m1['instance_dice']:.4f}  iou={m1['instance_iou']:.4f}  "
              f"boundary_f1={m1['boundary_f1']:.4f}  contact_f1={m1['contact_f1']:.4f}")
        print(f"    pred-crop 2pass dice={m2['instance_dice']:.4f}  iou={m2['instance_iou']:.4f}  "
              f"boundary_f1={m2['boundary_f1']:.4f}  contact_f1={m2['contact_f1']:.4f}")
        print(f"    GT-crop   2pass dice={m3['instance_dice']:.4f}  iou={m3['instance_iou']:.4f}  "
              f"boundary_f1={m3['boundary_f1']:.4f}  contact_f1={m3['contact_f1']:.4f}")
        summary[cat] = dict(n_images=n, one_pass={k: m1[k] for k in METRIC_KEYS},
                             pred_crop_two_pass={k: m2[k] for k in METRIC_KEYS},
                             gt_crop_two_pass={k: m3[k] for k in METRIC_KEYS})
    out_path = OUT_DIR / 'summary_gt_crop_vs_pred_crop.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nsaved:', out_path)


def _raw_forward_probs(img, model, convert_options):
    """threshold/NMS 이전 raw (boxes_xyxy, probs[num_queries, num_classes+1], content_box, resized_hw)."""
    rsz_img, content_box = resize_image(img, stride=64)
    img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
    img_tensor = torch_utils.data_convert(img2[None], **convert_options)
    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, dict):
            pass
        else:
            bboxes, logits, masks = outputs
            outputs = {'pred_boxes': bboxes, 'pred_logits': logits, 'pred_masks': masks}
    boxes_n = torch_utils.to_numpy(outputs['pred_boxes'])[0]
    probs = torch_utils.to_numpy(outputs['pred_logits'].sigmoid().to(torch.float32))[0]
    h_in, w_in = rsz_img.shape[-2:]
    x_c, y_c, bw, bh = [boxes_n[:, i] for i in range(4)]
    boxes_xyxy = np.stack([
        (x_c - 0.5 * bw) * w_in, (y_c - 0.5 * bh) * h_in,
        (x_c + 0.5 * bw) * w_in, (y_c + 0.5 * bh) * h_in,
    ], axis=1)
    return boxes_xyxy, probs, content_box, (h_in, w_in)


def part2_and_3_image4_deep_dive(model, convert_options, datasets):
    coco = dict(datasets)['seoul_univ_inspect_1106']
    im = next(x for x in coco.imgs.values() if x['file_name'].endswith(TARGET_FILE))
    anns = coco.imgToAnns[im['id']]
    img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    gt_boxes = gt_boxes_xyxy_from_anns(anns)
    gt_cats = np.array([a['category_id'] for a in anns])

    missing_mask = np.isin(gt_cats, list(TARGET_UNIVERSAL))
    missing_boxes = gt_boxes[missing_mask]
    missing_fdi = universal_to_fdi[gt_cats[missing_mask]]
    print(f'\n=== part2/3: {TARGET_FILE} (h={h}, w={w}) ===')
    print('missing teeth GT boxes (FDI):')
    for fdi, box in zip(missing_fdi, missing_boxes):
        print(f'  FDI {fdi}: xyxy={box.tolist()}')

    # pass-1 (NMS/threshold 적용된, 실제 배포 경로와 동일) -> 1차가 실제로 만든 crop_box
    _, _, inst1 = _predict_single_pass(img, model, convert_options)
    u1 = label2fdi[inst1['fdi']]
    print('pass-1 detected universal labels:', sorted(u1.tolist()))
    crop_box = teeth_roi_box(inst1['boxes'], (h, w), train=False,
                              cfg={'eval_margin': DEFAULT_TWO_PASS_ROI_MARGIN})
    print('pass-1 predicted crop_box (xyxy):', crop_box)

    # part2: GT box가 crop_box 안에 100% 포함되는지
    x0, y0, x1, y1 = crop_box
    for fdi, box in zip(missing_fdi, missing_boxes):
        bx0, by0, bx1, by1 = box
        contained = (bx0 >= x0) and (by0 >= y0) and (bx1 <= x1) and (by1 <= y1)
        # crop 안에 들어오는 비율도 같이
        iw = max(0.0, min(bx1, x1) - max(bx0, x0))
        ih = max(0.0, min(by1, y1) - max(by0, y0))
        area = max((bx1 - bx0) * (by1 - by0), 1e-6)
        visible = iw * ih / area
        print(f'  FDI {fdi}: fully_inside_pred_crop={contained}, visible_fraction={visible:.3f}')

    # part4 준비: crop 이미지 저장 + raw(사전 threshold/NMS) 신호 확인
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    crop_img = img[y0i:y1i, x0i:x1i]
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_DIR / 'seoul_pano_4_pred_crop_raw.png'), crop_img)
    rsz_crop, content_box = resize_image(crop_img, stride=64)
    cv2.imwrite(str(DEBUG_DIR / 'seoul_pano_4_pred_crop_resized_model_input.png'),
                (rsz_crop * 255 if rsz_crop.max() <= 1.0 else rsz_crop).astype(np.uint8))
    print('saved 2nd-pass crop debug images to', DEBUG_DIR)
    print('pred crop_box clipped int:', (x0i, y0i, x1i, y1i), 'crop_img shape:', crop_img.shape)

    # part3: 2차(크롭) raw output에서 missing teeth 근처에 신호가 있었는지
    # universal label idx = argmax 위치 (주의: pred_logits 마지막 차원은 num_classes+1, index0=배경)
    boxes_xyxy_crop, probs_crop, content_box2, resized_hw = _raw_forward_probs(crop_img, model, convert_options)
    # crop-local 모델입력 좌표계 boxes_xyxy_crop을 crop 원본 좌표계로 복원 (letterbox 보정)
    cx0, cy0, cx1, cy1 = content_box2
    h_in, w_in = resized_hw
    ch, cw = crop_img.shape[:2]
    scale_x = cw / max(cx1 - cx0, 1e-6)
    scale_y = ch / max(cy1 - cy0, 1e-6)
    boxes_full = boxes_xyxy_crop.copy()
    boxes_full[:, [0, 2]] = (boxes_full[:, [0, 2]] - cx0) * scale_x
    boxes_full[:, [1, 3]] = (boxes_full[:, [1, 3]] - cy0) * scale_y
    # full-image(원본) 좌표로 한번 더 이동
    boxes_full[:, [0, 2]] += x0i
    boxes_full[:, [1, 3]] += y0i

    pred_scores_all = probs_crop[:, 1:].max(axis=-1)  # 배경(0) 제외 최대 class prob
    pred_label_all = probs_crop[:, 1:].argmax(axis=-1) + 1  # universal idx(1~32)

    for fdi, gt_box in zip(missing_fdi, missing_boxes):
        gcx, gcy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
        pred_cx = (boxes_full[:, 0] + boxes_full[:, 2]) / 2
        pred_cy = (boxes_full[:, 1] + boxes_full[:, 3]) / 2
        dist = np.hypot(pred_cx - gcx, pred_cy - gcy)
        near = np.argsort(dist)[:5]
        print(f'\n  FDI {fdi} (GT center=({gcx:.0f},{gcy:.0f})) -- 2차 raw output 근접 query top5:')
        for qi in near:
            print(f'    query#{qi}: dist={dist[qi]:.1f}px, pred_label_univ={pred_label_all[qi]} '
                  f'(FDI {universal_to_fdi[pred_label_all[qi]] if pred_label_all[qi] < 33 else -1}), '
                  f'score={pred_scores_all[qi]:.4f}, box={boxes_full[qi].round(1).tolist()}')


def main():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_val_coco()
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    part2_and_3_image4_deep_dive(model, convert_options, datasets)
    part1_quantitative_gt_vs_pred_crop(model, convert_options, datasets)


if __name__ == '__main__':
    main()
