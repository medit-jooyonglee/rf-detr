"""Q3 follow-up: formal COCO segm AP/AR + fixed-score-threshold precision/recall
for the real-peri subset (same methodology as docs/peri_data_fusion_eval.md
section 4's teeth_roi table, via rfdetr/scripts/eval_pr_utils.py), plus extra
GT/baseline/new render images spanning the full dice range (best/typical/worst).

Reuses the detections already cached by eval_peri_fusion_progress.py under
outputs/compare_peri_fusion/peri_subset_progress/ -- no GPU/model needed.

Run: PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_peri_precision_recall.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval

from rfdetr.scripts.compare_core import ROOT, load_coco
from rfdetr.scripts.eval_peri_fusion_progress import PERI_DATA, BASE_NAME, render

PERI_OUT = ROOT / 'outputs/compare_peri_fusion/peri_subset_progress'
IMG_DIR = ROOT / 'outputs/compare_peri_fusion/images'
NEW_NAME = 'peri_data_fusion_ep106_ema'

EXTRA_RENDER_IDS = [
    ('worst', [100, 108, 127, 151]),
    ('typical', [17, 58, 111, 141]),
    ('improved', [59, 134, 15]),
]


def mean_precision_recall(ev, iou_index=0, score_threshold=0.5):
    """Same approximation as eval_pr_utils.mean_precision_recall, but taking a
    COCOeval object (already evaluate()+accumulate()'d) directly instead of a
    saved eval.pth file."""
    P, S, R = ev.eval['precision'], ev.eval['scores'], ev.eval['recall']
    rec_thr = np.linspace(0, 1, 101)
    ps, rs = [], []
    for k in range(P.shape[2]):
        if R[iou_index, k, 0, 2] < 0:
            continue
        p, s = P[iou_index, :, k, 0, 2], S[iou_index, :, k, 0, 2]
        ok = np.where((s >= score_threshold) & (p > 0))[0]
        if len(ok) == 0:
            ps.append(0.0)
            rs.append(0.0)
            continue
        ps.append(float(p[ok[-1]]))
        rs.append(float(rec_thr[ok[-1]]))
    return dict(precision=float(np.mean(ps)), recall=float(np.mean(rs)), num_classes=len(ps))


def coco_ap_ar(ev):
    def _mean(arr):
        arr = arr[arr > -1]
        return float(np.mean(arr)) if arr.size else -1.0
    P, R = ev.eval['precision'], ev.eval['recall']
    return dict(AP=_mean(P[:, :, :, 0, 2]), AP50=_mean(P[0, :, :, 0, 2]),
                AP75=_mean(P[5, :, :, 0, 2]), AR100=_mean(R[:, :, 0, 2]))


def run_coco_eval(coco, name):
    dets = json.loads((PERI_OUT / f'{name}_predictions.json').read_text())
    dets = [{k: v for k, v in d.items() if k != 'bbox'} for d in dets]  # segm: no bbox key
    dt = coco.loadRes(dets)
    ev = COCOeval(coco, dt, 'segm')
    ev.evaluate()
    ev.accumulate()
    return ev


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--show-baseline', action='store_true',
                    help='also render the baseline model column (default: new model only)')
    p.add_argument('--no-raw', action='store_true', help='omit the plain raw-image column')
    p.add_argument('--no-gt', action='store_true', help='omit the GT-overlay column')
    return p.parse_args()


def main():
    args = parse_args()
    render_models = [BASE_NAME, NEW_NAME] if args.show_baseline else [NEW_NAME]
    show_raw, show_gt = not args.no_raw, not args.no_gt

    coco = load_coco(PERI_DATA / 'val', PERI_DATA / 'val/annotations.json', filename_contains='peri')
    print(f'real peri (full): {len(coco.imgs)} images, {len(coco.anns)} annotations\n')

    results = {}
    for name in (BASE_NAME, NEW_NAME):
        ev = run_coco_eval(coco, name)
        ap = coco_ap_ar(ev)
        pr = {}
        for iou_label, iou_idx in [('iou0.5', 0), ('iou0.75', 5)]:
            for score in (0.5, 0.3):
                pr[f'{iou_label}_score{score}'] = mean_precision_recall(ev, iou_idx, score)
        results[name] = dict(ap=ap, pr=pr)
        print(f'=== {name} ===')
        print(f"  segm AP={ap['AP']:.3f}  AP50={ap['AP50']:.3f}  AP75={ap['AP75']:.3f}  AR100={ap['AR100']:.3f}")
        for k, v in pr.items():
            print(f"  {k:>16s}  precision={v['precision']:.3f}  recall={v['recall']:.3f}")
        print()

    (ROOT / 'outputs/compare_peri_fusion/peri_subset_progress/precision_recall.json').write_text(
        json.dumps(results, indent=2))

    # --- extra renders spanning the dice range ---
    detections = {name: json.loads((PERI_OUT / f'{name}_predictions.json').read_text())
                  for name in (BASE_NAME, NEW_NAME)}
    rows = {name: {r['image_id']: r for r in json.loads((PERI_OUT / f'{name}_per_image.json').read_text())}
            for name in (BASE_NAME, NEW_NAME)}
    for tag, ids in EXTRA_RENDER_IDS:
        for image_id in ids:
            render(coco, detections, rows, render_models, image_id,
                   IMG_DIR / f'peri_{image_id}_ep106_{tag}.jpg', tag=f'(real peri, {tag})',
                   show_raw=show_raw, show_gt=show_gt)


if __name__ == '__main__':
    main()
