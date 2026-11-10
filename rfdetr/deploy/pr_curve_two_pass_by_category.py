"""pr_curve_two_pass.py를 cbct_pano/seoul_pano로 나눠서 각각 PR curve/AP/F1-최적 threshold를
구한다. two_pass_predict()의 현재 기본값(center-aware ROI + pass2_confidence_threshold=0.35)을
그대로 사용 -- 반영 전(naive crop + 공용 threshold 0.5) 수치는 outputs/eval_fdi_refinement/
pr_curve_summary.json(ALL 합산)과 비교 가능.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/pr_curve_two_pass_by_category.py
"""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import _predict_single_pass, two_pass_predict
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import label2fdi
from rfdetr.deploy.pr_curve_two_pass import box_iou, IOU_THR, LOW_THRESHOLD
from trainer import image_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]
CATS = ('cbct_pano', 'seoul_pano')


def collect_two_pass_candidates_low_threshold(img, model, convert_options):
    """two_pass_predict()와 같은(center-aware) ROI 구성을 그대로 쓰되, 2차만 LOW_THRESHOLD로
    후보를 최대한 살려서 반환 (PR curve용 -- fdi_refinement=False로 DP 재배열이 원점수 랭킹을
    왜곡하지 않게 한다)."""
    # two_pass_predict 내부와 동일한 pass1->crop 로직을 타되, pass2만 낮은 threshold로 다시 돈다.
    _, _, inst1, used, crop_box = two_pass_predict(img, model, convert_options=convert_options)
    if not used:
        u1 = label2fdi[inst1['fdi']]
        return inst1['boxes'], u1, inst1['scores']
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop_box
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    crop_img = img[y0i:y1i, x0i:x1i]
    _, _, inst2 = _predict_single_pass(crop_img, model, convert_options,
                                        confidence_threshold=LOW_THRESHOLD, fdi_refinement=False)
    boxes2 = inst2['boxes'] + np.array([x0i, y0i, x0i, y0i], dtype=inst2['boxes'].dtype)
    u2 = label2fdi[inst2['fdi']]
    return boxes2, u2, inst2['scores']


def collect(mode, model, convert_options, datasets):
    """mode: 'one_pass' | 'two_pass'. return dict(cat -> dict(preds=[...], gts=[...]))."""
    out = {cat: {'preds': [], 'gts': []} for cat in CATS}
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
            out[cat]['gts'].append((im['id'], gt_boxes, gt_labels))

            if mode == 'one_pass':
                _, _, inst = _predict_single_pass(img, model, convert_options,
                                                   confidence_threshold=LOW_THRESHOLD, fdi_refinement=False)
                u = label2fdi[inst['fdi']]
                out[cat]['preds'].append((im['id'], inst['boxes'], u, inst['scores']))
            else:
                boxes, labels, scores = collect_two_pass_candidates_low_threshold(img, model, convert_options)
                out[cat]['preds'].append((im['id'], boxes, labels, scores))
            n += 1
            if n % 20 == 0:
                print(mode, n, flush=True)
    return out


def pr_curve(preds, gts, iou_thr=IOU_THR):
    gt_by_image = {iid: (boxes, labels) for iid, boxes, labels in gts}
    total_gt = sum(len(labels) for _, _, labels in gts)
    flat = []
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
    ap = float(np.trapz(precision, recall)) if len(recall) > 1 else 0.0
    return dict(precision=precision, recall=recall, scores=np.array(scores_sorted), ap=ap, total_gt=total_gt)


def main():
    datasets = load_val_coco()
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    collected = {}
    for mode in ('one_pass', 'two_pass'):
        collected[mode] = collect(mode, model, convert_options, datasets)

    fig, axes = plt.subplots(1, len(CATS), figsize=(6 * len(CATS), 6))
    summary = {}
    for ax, cat in zip(axes, CATS):
        summary[cat] = {}
        for mode, style in (('one_pass', 'C0'), ('two_pass', 'C1')):
            preds = collected[mode][cat]['preds']
            gts = collected[mode][cat]['gts']
            c = pr_curve(preds, gts)
            f1 = 2 * c['precision'] * c['recall'] / np.clip(c['precision'] + c['recall'], 1e-9, None)
            best = int(np.argmax(f1))
            ax.plot(c['recall'], c['precision'], style, label=f"{mode} (AP={c['ap']:.3f})")
            for th in (0.3, 0.4, 0.5):
                idx = np.argmin(np.abs(c['scores'] - th))
                ax.scatter([c['recall'][idx]], [c['precision'][idx]], color=style, zorder=5)
                ax.annotate(f'{th}', (c['recall'][idx], c['precision'][idx]), fontsize=7)
            summary[cat][mode] = dict(ap=c['ap'], best_score_threshold=float(c['scores'][best]),
                                       best_precision=float(c['precision'][best]),
                                       best_recall=float(c['recall'][best]), best_f1=float(f1[best]),
                                       total_gt=c['total_gt'])
            print(f"{cat}/{mode}: AP={c['ap']:.4f}  best F1={f1[best]:.4f} at score>={c['scores'][best]:.3f} "
                  f"(precision={c['precision'][best]:.4f}, recall={c['recall'][best]:.4f})  total_gt={c['total_gt']}")
        ax.set_title(f'{cat} (box+FDI PR, IoU>={IOU_THR})')
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    plt.tight_layout()
    out_png = OUT_DIR / 'pr_curve_by_category.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print('saved:', out_png)
    (OUT_DIR / 'pr_curve_by_category_summary.json').write_text(json.dumps(summary, indent=2))
    print('saved:', OUT_DIR / 'pr_curve_by_category_summary.json')


if __name__ == '__main__':
    main()
