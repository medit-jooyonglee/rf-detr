"""Render a handful of GT / baseline / new-model side-by-side comparison images
from the cached predictions produced by eval_panorama_and_overlap.py and
eval_peri_subset.py (rfdetr/scripts/compare_core.py). No GPU/model needed --
this only reads the cached *_predictions.json RLE masks.

Run: PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/render_comparison_samples.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[2]
FONT = '/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
BASE_NAME = 'baseline_panoramic_overlapfix'
NEW_NAME = 'peri_data_fusion_ep45_ema'
SCORE_THRESHOLD = 0.3


def color(k):
    hsv = np.uint8([[[int(k * 137.5) % 180, 200, 255]]])
    return tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]))


def load(run_dir):
    run_dir = Path(run_dir)
    preds = {}
    for name in [BASE_NAME, NEW_NAME]:
        preds[name] = json.loads((run_dir / f'{name}_predictions.json').read_text())
    rows = {r['image_id']: r for r in json.loads((run_dir / f'{BASE_NAME}_per_image.json').read_text())}
    rows_new = {r['image_id']: r for r in json.loads((run_dir / f'{NEW_NAME}_per_image.json').read_text())}
    return preds, rows, rows_new


def panel(rgb, masks, labels, title, target_w=620, target_h=330):
    h, w = rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = round(w * scale), round(h * scale)
    patch = cv2.resize(rgb, (nw, nh))
    for mask, label in zip(masks, labels):
        crop = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(patch, contours, -1, color(label), 2)
    canvas = Image.new('RGB', (target_w, target_h + 40), (16, 22, 30))
    canvas.paste(Image.fromarray(patch), ((target_w - nw) // 2, 40 + (target_h - nh) // 2))
    ImageDraw.Draw(canvas).text((10, 8), title, font=ImageFont.truetype(FONT, 16), fill='white')
    return canvas


def render_image(coco_images_by_id, dets_by_model, rows_base, rows_new, image_id, out_path,
                  label='', target_w=620, target_h=330):
    file_name = rows_base[image_id]['file_name']
    rgb = np.array(Image.open(file_name).convert('RGB'))
    anns_gt = coco_images_by_id[image_id]
    gt_masks = [mu.decode(a['segmentation']).astype(bool) for a in anns_gt]
    gt_labels = [a['category_id'] for a in anns_gt]

    sheet = Image.new('RGB', (target_w * 3, target_h + 40), (16, 22, 30))
    cols = [('GT', gt_masks, gt_labels, None)]
    for name in [BASE_NAME, NEW_NAME]:
        dd = [d for d in dets_by_model[name] if d['image_id'] == image_id and d['score'] >= SCORE_THRESHOLD]
        masks = [mu.decode(d['segmentation']).astype(bool) for d in dd]
        labels = [d['category_id'] for d in dd]
        r = rows_base[image_id] if name == BASE_NAME else rows_new[image_id]
        cols.append((name, masks, labels, r))

    for col, (name, masks, labels, r) in enumerate(cols):
        if r is None:
            title = f'GT | image {image_id} {label}'
        else:
            title = f'{name} | dice {r["mean_dice"]:.3f} overlap_iou {(r["overlap_iou"] or 0):.3f}'
        sheet.paste(panel(rgb, masks, labels, title, target_w, target_h), (col * target_w, 0))
    sheet.save(out_path, quality=92)
    print('wrote', out_path)


def build_gt_index(predictions_json_dir, coco_json_path):
    from pycocotools.coco import COCO
    coco = COCO(str(coco_json_path))
    out = {}
    for image_id, anns in coco.imgToAnns.items():
        out[image_id] = [dict(a, segmentation=coco.annToRLE(a)) for a in anns]
    return out


def main():
    out_dir = ROOT / 'outputs/compare_peri_fusion/images'
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- peri subset: dramatic-improvement examples ---
    peri_dir = ROOT / 'outputs/compare_peri_fusion/peri_subset'
    preds, rows_base, rows_new = load(peri_dir)
    gt_index = build_gt_index(peri_dir, Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_COCO/val/annotations.json'))
    # re-key gt_index by the *local* json ids used inside the cached predictions (they come
    # from merge_coco_directories's single-source pass-through, so ids are unchanged)
    for image_id in [163, 132, 103]:
        render_image(gt_index, preds, rows_base, rows_new, image_id,
                     out_dir / f'peri_{image_id}.jpg', label='(real peri photo)')

    # --- panorama/overlap: honest mix (one regression, one improvement, one typical) ---
    pano_dir = ROOT / 'outputs/compare_peri_fusion/panorama_overlap'
    preds2, rows_base2, rows_new2 = load(pano_dir)
    gt_index2 = build_gt_index(pano_dir, Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022/val/annotations.json'))
    for image_id, label in [(15, '(overlap: new slightly worse)'), (71, '(overlap: new better)'), (20, '(overlap: typical)')]:
        render_image(gt_index2, preds2, rows_base2, rows_new2, image_id,
                     out_dir / f'panorama_{image_id}.jpg', label=label)

    print('done')


if __name__ == '__main__':
    main()
