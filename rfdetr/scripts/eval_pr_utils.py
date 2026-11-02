"""Q4 helper: approximate fixed-score-threshold precision/recall from the
COCOeval precision/recall/scores arrays saved to <output_dir>/eval.pth by
`python -m rfdetr.cli.main --eval [--eval_xray_crop] ...`.

COCOeval itself only reports precision/recall swept over a fixed 101-point
recall grid, not at a fixed confidence score. This mirrors the approach
documented in docs/xray_periapical_eval.md section 2.2: for each class, take
the last recall-grid point whose score is still >= `score_threshold`, and
average precision/recall over classes that have any GT instances.

Run e.g.:
  PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/compare_teeth_roi.py
"""
import numpy as np
import torch


def coco_ap_ar(eval_pth_path):
    """Recomputes COCOeval's summarize() AP/AP50/AP75/AR100 (area=all, maxDets=100)
    directly from the saved precision/recall arrays, for the IoU metric (bbox or
    segm -- whichever rfdetr's eval loop last wrote to eval.pth) they were computed
    for. Matches pycocotools.cocoeval.COCOeval.summarize() exactly."""
    e = torch.load(eval_pth_path, map_location='cpu', weights_only=False)
    P, R = e['precision'], e['recall']

    def _mean(arr):
        arr = arr[arr > -1]
        return float(np.mean(arr)) if arr.size else -1.0

    return dict(
        AP=_mean(P[:, :, :, 0, 2]),
        AP50=_mean(P[0, :, :, 0, 2]),
        AP75=_mean(P[5, :, :, 0, 2]),
        AR100=_mean(R[:, :, 0, 2]),
    )


def mean_precision_recall(eval_pth_path, iou_index=0, score_threshold=0.5):
    """iou_index: 0 -> IoU 0.5, 5 -> IoU 0.75 (COCOeval's default 10-point IoU grid).
    area index fixed to 0 (all), maxDets index fixed to 2 (100), matching docs."""
    e = torch.load(eval_pth_path, map_location='cpu', weights_only=False)
    P, S, R = e['precision'], e['scores'], e['recall']  # [T,101,K,A,M], [T,101,K,A,M], [T,K,A,M]
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
