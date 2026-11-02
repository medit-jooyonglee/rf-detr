"""Q1 + Q2: does the peri-data-fusion checkpoint match (or beat) the deployed
baseline on ordinary full panoramic images, and does it find overlapping
(crowded/neighboring) teeth at least as well?

Dataset: own-data val split (/data1/.../cbct_ios_dcm_latest_1022/val), same
split tools/compare_own1022.py and tools/compare_overlapfix_vs_deployed.py use,
so results are directly comparable to the summaries already recorded under
outputs/compare_overlapfix_vs_deployed/.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_panorama_and_overlap.py
"""
import json
from pathlib import Path

from rfdetr.scripts.compare_core import ROOT, load_coco, summarize_models, nonmetal_adjacent_overlap

DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022')
OUT_DIR = ROOT / 'outputs/compare_peri_fusion/panorama_overlap'
MODELS = {
    'baseline_panoramic_overlapfix': str(ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth'),
    'peri_data_fusion_ep45_ema': str(ROOT / 'outputs/peri_data_fusion/checkpoint_best_ema.pth'),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = load_coco(DATA / 'val', DATA / 'val' / 'annotations.json')
    print(f'own1022 val: {len(coco.imgs)} images, {len(coco.anns)} annotations')

    detections, overall = summarize_models(MODELS, coco, DATA / 'val', OUT_DIR)
    overlap = nonmetal_adjacent_overlap(coco, detections)

    summary = dict(
        dataset=str(DATA / 'val'),
        num_images=len(coco.imgs),
        checkpoints={k: v for k, v in MODELS.items()},
        overall_metrics={name: {k: v for k, v in m.items() if k in (
            'instance_dice', 'instance_iou', 'overlap_iou', 'overlap_recall', 'overlap_precision',
            'pair_recall', 'boundary_f1', 'contact_f1', 'path', 'sha256')}
            for name, m in overall.items()},
        nonmetal_adjacent_overlap=overlap,
    )
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('\n=== Q1: panoramic parity (instance dice/iou, boundary/contact f1) ===')
    for name, m in overall.items():
        print(f"  {name:>30s}  dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
              f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}")

    print('\n=== Q2: non-metal same-arch adjacent-tooth overlap (pairwise recall/precision) ===')
    for name, m in overlap.items():
        print(f"  {name:>30s}  recall={m['recall']*100:5.1f}%  precision={m['precision']*100:5.1f}%  gt_px={m['pair_gt']}")

    print(f'\nsaved: {OUT_DIR}/summary.json')


if __name__ == '__main__':
    main()
