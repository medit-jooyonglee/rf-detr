"""one-pass vs two-pass(2차, crop 후 재추론)의 confidence threshold 선택을 위한
precision-recall curve. box-level 매칭(IoU>=IOU_THR) + FDI 라벨 일치를 TP 조건으로 삼는다.

참고(사용자 질문에 대한 답): pred_logits는 softmax가 아니라 class별 독립 sigmoid라서
("32개 클래스라 확률이 작다"는 정규화 효과는 아님), 특정 치아의 0.3~0.6대 점수는 32-class
정규화 때문이 아니라 crop 스케일에서 모델이 실제로 덜 확신한다는 뜻 그대로다.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/pr_curve_two_pass.py
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import (
    _predict_single_pass, DEFAULT_TWO_PASS_ROI_MARGIN, DEFAULT_TWO_PASS_SHRINK_RATIO,
)
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import label2fdi
from rfdetr.datasets.xray_crop_aug import teeth_roi_box
from trainer import image_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]
LOW_THRESHOLD = 0.02  # NMS/후보 생성용 하한 -- 곡선은 이 위에서만 그릴 수 있음
IOU_THR = 0.3
CATS = ('cbct_pano', 'seoul_pano')


def box_iou(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M) iou."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0, ax1, ay1 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx0, by0, bx1, by1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix0, iy0 = np.maximum(ax0, bx0), np.maximum(ay0, by0)
    ix1, iy1 = np.minimum(ax1, bx1), np.minimum(ay1, by1)
    iw, ih = np.clip(ix1 - ix0, 0, None), np.clip(iy1 - iy0, 0, None)
    inter = iw * ih
    area_a = np.clip((ax1 - ax0) * (ay1 - ay0), 1e-6, None)
    area_b = np.clip((bx1 - bx0) * (by1 - by0), 1e-6, None)
    return inter / (area_a + area_b - inter)


def collect_two_pass_candidates(img, model, convert_options):
    """pass-1은 배포값(threshold=0.5)으로 ROI를 만들고, pass-2는 LOW_THRESHOLD로 후보를
    최대한 살려서 반환한다 (fdi_refinement=False: DP 재배열이 원점수-기반 랭킹을 왜곡하지
    않도록)."""
    h, w = img.shape[:2]
    _, _, inst1 = _predict_single_pass(img, model, convert_options)
    if inst1['boxes'].shape[0] == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))
    crop_box = teeth_roi_box(inst1['boxes'], (h, w), train=False,
                              cfg={'eval_margin': DEFAULT_TWO_PASS_ROI_MARGIN})
    if crop_box is None:
        u1 = label2fdi[inst1['fdi']]
        return inst1['boxes'], u1, inst1['scores']
    x0, y0, x1, y1 = crop_box
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1i - x0i <= 0 or y1i - y0i <= 0 or (x1i - x0i) * (y1i - y0i) >= DEFAULT_TWO_PASS_SHRINK_RATIO * w * h:
        u1 = label2fdi[inst1['fdi']]
        return inst1['boxes'], u1, inst1['scores']
    crop_img = img[y0i:y1i, x0i:x1i]
    _, _, inst2 = _predict_single_pass(crop_img, model, convert_options,
                                        confidence_threshold=LOW_THRESHOLD, fdi_refinement=False)
    boxes2 = inst2['boxes'] + np.array([x0i, y0i, x0i, y0i], dtype=inst2['boxes'].dtype)
    u2 = label2fdi[inst2['fdi']]
    return boxes2, u2, inst2['scores']


def collect(mode, model, convert_options, datasets):
    preds, gts = [], []
    n = 0
    for source_name, coco in datasets:
        for im in coco.imgs.values():
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            cat = categorize(source_name, im['file_name'], im['width'], im['height'])
            if cat not in CATS:
                continue
            img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
            gt_boxes = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                                   a['bbox'][1] + a['bbox'][3]] for a in anns], dtype=np.float64)
            gt_labels = np.array([a['category_id'] for a in anns])
            gts.append((im['id'], gt_boxes, gt_labels))

            if mode == 'one_pass':
                _, _, inst = _predict_single_pass(img, model, convert_options,
                                                   confidence_threshold=LOW_THRESHOLD, fdi_refinement=False)
                u = label2fdi[inst['fdi']]
                preds.append((im['id'], inst['boxes'], u, inst['scores']))
            else:
                boxes, labels, scores = collect_two_pass_candidates(img, model, convert_options)
                preds.append((im['id'], boxes, labels, scores))
            n += 1
            if n % 20 == 0:
                print(mode, n, flush=True)
    return preds, gts


def pr_curve(preds, gts, iou_thr=IOU_THR):
    gt_by_image = {iid: (boxes, labels) for iid, boxes, labels in gts}
    total_gt = sum(len(labels) for _, _, labels in gts)
    flat = []  # (score, image_id, box, label)
    for iid, boxes, labels, scores in preds:
        for b, l, s in zip(boxes, labels, scores):
            flat.append((s, iid, b, l))
    flat.sort(key=lambda x: -x[0])

    matched_gt = {iid: np.zeros(len(labels), dtype=bool) for iid, (_, labels) in gt_by_image.items()}
    tps, fps, scores_sorted = [], [], []
    tp_cum = fp_cum = 0
    for s, iid, box, label in flat:
        gt_boxes, gt_labels = gt_by_image.get(iid, (np.zeros((0, 4)), np.zeros((0,))))
        same_cls = np.where((gt_labels == label) & (~matched_gt[iid]))[0]
        is_tp = False
        if len(same_cls) > 0:
            ious = box_iou(box[None, :], gt_boxes[same_cls])[0]
            best = np.argmax(ious)
            if ious[best] >= iou_thr:
                matched_gt[iid][same_cls[best]] = True
                is_tp = True
        tp_cum += int(is_tp)
        fp_cum += int(not is_tp)
        tps.append(tp_cum)
        fps.append(fp_cum)
        scores_sorted.append(s)
    tps, fps = np.array(tps, dtype=np.float64), np.array(fps, dtype=np.float64)
    precision = tps / np.clip(tps + fps, 1e-9, None)
    recall = tps / max(total_gt, 1)
    # recall is already non-decreasing in this (score-descending) order, so integrate directly --
    # reversing both arrays here would flip dx's sign and make the integral negative.
    ap = float(np.trapz(precision, recall)) if len(recall) > 1 else 0.0
    return dict(precision=precision, recall=recall, scores=np.array(scores_sorted), ap=float(ap),
                total_gt=total_gt)


def main():
    datasets = load_val_coco()
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    curves = {}
    for mode in ('one_pass', 'two_pass'):
        preds, gts = collect(mode, model, convert_options, datasets)
        curves[mode] = pr_curve(preds, gts)
        print(f'{mode}: AP={curves[mode]["ap"]:.4f}  total_gt={curves[mode]["total_gt"]}')

    plt.figure(figsize=(6, 6))
    markers_at = (0.3, 0.4, 0.5)
    for mode, style in (('one_pass', 'C0'), ('two_pass', 'C1')):
        c = curves[mode]
        plt.plot(c['recall'], c['precision'], style, label=f"{mode} (AP={c['ap']:.3f})")
        for th in markers_at:
            idx = np.argmin(np.abs(c['scores'] - th))
            plt.scatter([c['recall'][idx]], [c['precision'][idx]], color=style, zorder=5)
            plt.annotate(f'{th}', (c['recall'][idx], c['precision'][idx]), fontsize=8)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(f'{MODEL_NAME}: box+FDI PR curve (IoU>={IOU_THR}), cbct_pano+seoul_pano')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.02)
    plt.xlim(0, 1.02)
    out_png = OUT_DIR / 'pr_curve_one_pass_vs_two_pass.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print('saved:', out_png)

    # F1-optimal operating point per curve
    summary = {}
    for mode in ('one_pass', 'two_pass'):
        c = curves[mode]
        f1 = 2 * c['precision'] * c['recall'] / np.clip(c['precision'] + c['recall'], 1e-9, None)
        best = int(np.argmax(f1))
        summary[mode] = dict(ap=c['ap'], best_score_threshold=float(c['scores'][best]),
                              best_precision=float(c['precision'][best]), best_recall=float(c['recall'][best]),
                              best_f1=float(f1[best]), total_gt=c['total_gt'])
        print(f"{mode}: best F1={f1[best]:.4f} at score>={c['scores'][best]:.3f} "
              f"(precision={c['precision'][best]:.4f}, recall={c['recall'][best]:.4f})")
    (OUT_DIR / 'pr_curve_summary.json').write_text(json.dumps(summary, indent=2))
    print('saved:', OUT_DIR / 'pr_curve_summary.json')


if __name__ == '__main__':
    main()
