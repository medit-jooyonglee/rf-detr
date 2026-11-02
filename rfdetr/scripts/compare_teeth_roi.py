"""Q4: does the peri-data-fusion checkpoint improve precision/recall under
teeth_roi-mode (ROI crop, two-pass second-stage input) evaluation vs the
deployed baseline?

Unlike the other 3 comparisons, this reuses rfdetr's own built-in
--eval/--eval_xray_crop pipeline (same one documented in
docs/xray_periapical_eval.md) instead of rfdetr.scripts.compare_core, because
that pipeline already implements the exact train/eval-consistent teeth_roi
crop + GT rebuild logic (rfdetr.datasets.xraypanoramic._rebuild_crop_eval_coco)
end to end via the standard COCOeval path. This script only adds the
comparison/reporting layer on top of its output (<output_dir>/eval.pth).

Reproduce from scratch (pick free GPUs; each run takes a few minutes):
  python -m rfdetr.cli.main \\
    --pretrain_weights outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth \\
    --xray_crop_mode teeth_roi --xray_crop_prob 1.0 --batch_size 1 --device cuda:0 \\
    --output_dir outputs/eval_teeth_roi_baseline --eval --eval_xray_crop

  python -m rfdetr.cli.main \\
    --pretrain_weights outputs/peri_data_fusion/checkpoint_best_ema.pth \\
    --xray_crop_mode teeth_roi --xray_crop_prob 1.0 --batch_size 1 --device cuda:0 \\
    --output_dir outputs/eval_teeth_roi_new --eval --eval_xray_crop

Then:
  PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/compare_teeth_roi.py
"""
import json
from pathlib import Path

from rfdetr.scripts.compare_core import ROOT
from rfdetr.scripts.eval_pr_utils import coco_ap_ar, mean_precision_recall

OUT_DIR = ROOT / 'outputs/compare_peri_fusion/teeth_roi'
EVAL_DIRS = {
    'baseline_panoramic_overlapfix': ROOT / 'outputs/eval_teeth_roi_baseline',
    'peri_data_fusion_ep45_ema': ROOT / 'outputs/eval_teeth_roi_new',
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, eval_dir in EVAL_DIRS.items():
        eval_pth = eval_dir / 'eval.pth'
        if not eval_pth.exists():
            raise FileNotFoundError(
                f'{eval_pth} missing -- run the --eval_xray_crop teeth_roi command (see this file\'s '
                f'docstring) for "{name}" first')
        entry = dict(eval_pth=str(eval_pth), segm_coco=coco_ap_ar(eval_pth))
        for iou_name, iou_idx in [('iou0.5', 0), ('iou0.75', 5)]:
            for thr in [0.5, 0.3]:
                entry[f'{iou_name}_score{thr}'] = mean_precision_recall(eval_pth, iou_index=iou_idx, score_threshold=thr)
        summary[name] = entry

    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('=== Q4: teeth_roi mode -- segmentation COCO AP/AR (area=all, maxDets=100) ===')
    for name, entry in summary.items():
        m = entry['segm_coco']
        print(f"  {name:>30s}  AP={m['AP']:.3f}  AP50={m['AP50']:.3f}  AP75={m['AP75']:.3f}  AR100={m['AR100']:.3f}")

    print('\n=== Q4: approximate fixed-score-threshold precision/recall (docs/xray_periapical_eval.md §2.2 method) ===')
    for iou_name in ['iou0.5', 'iou0.75']:
        for thr in [0.5, 0.3]:
            key = f'{iou_name}_score{thr}'
            print(f'  [{key}]')
            for name, entry in summary.items():
                pr = entry[key]
                print(f"    {name:>30s}  precision={pr['precision']:.3f}  recall={pr['recall']:.3f}")

    print(f'\nsaved: {OUT_DIR}/summary.json')


if __name__ == '__main__':
    main()
