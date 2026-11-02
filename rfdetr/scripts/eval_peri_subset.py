"""Q3: does the peri-data-fusion checkpoint handle genuine periapical-style
photographs (not synthetically cropped) better than the deployed baseline?

Dataset: seoul_univ_inspect_COCO val split, filtered to file_name containing
"peri" (case-insensitive). Those images are real periapical-style shots mixed
into the same val folder as "pano" full-panoramic shots and implant-only
closeups; filenames with "peri" are excluded from the xray_crop_mode
(periapical/teeth_roi) training-time augmentation (see
rfdetr.datasets.xraypanoramic.XrayPnoaramicInstance._is_peri_filename), so this
is a clean, un-cropped real-world peri test set -- evaluated here as-is (no
synthetic crop applied), same letterbox-resize-only preprocessing as full
panoramic images.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_peri_subset.py
"""
import json
from pathlib import Path

from rfdetr.scripts.compare_core import ROOT, load_coco, summarize_models

DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_COCO')
OUT_DIR = ROOT / 'outputs/compare_peri_fusion/peri_subset'
MODELS = {
    'baseline_panoramic_overlapfix': str(ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth'),
    'peri_data_fusion_ep45_ema': str(ROOT / 'outputs/peri_data_fusion/checkpoint_best_ema.pth'),
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco = load_coco(DATA / 'val', DATA / 'val' / 'annotations.json', filename_contains='peri')
    print(f'seoul_univ_inspect_COCO val, "peri" filenames only: {len(coco.imgs)} images, {len(coco.anns)} annotations')

    detections, overall = summarize_models(MODELS, coco, DATA / 'val', OUT_DIR)

    summary = dict(
        dataset=str(DATA / 'val') + ' (filename contains "peri")',
        num_images=len(coco.imgs),
        num_annotations=len(coco.anns),
        checkpoints={k: v for k, v in MODELS.items()},
        overall_metrics={name: {k: v for k, v in m.items() if k in (
            'instance_dice', 'instance_iou', 'overlap_iou', 'overlap_recall', 'overlap_precision',
            'pair_recall', 'boundary_f1', 'contact_f1', 'bbox_AP', 'segm_AP', 'path', 'sha256')}
            for name, m in overall.items()},
    )
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    print('\n=== Q3: real (un-cropped) peri images -- instance dice/iou ===')
    for name, m in overall.items():
        print(f"  {name:>30s}  dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
              f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}")

    print(f'\nsaved: {OUT_DIR}/summary.json')


if __name__ == '__main__':
    main()
