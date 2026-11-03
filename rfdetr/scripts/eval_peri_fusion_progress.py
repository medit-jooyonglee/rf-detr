"""Progress check for outputs/peri_data_fusion.

docs/peri_data_fusion_eval.md was written against checkpoint_best_ema.pth at
epoch 45 (training was interrupted then). Training has since resumed and is
still running (outputs/peri_data_fusion/checkpoint_best_ema.pth is now well
past epoch 100). This re-evaluates the *current* checkpoint_best_ema.pth vs
the same deployed baseline on:

  1. a handful of panoramic images -- the same 3 own1022 val images already
     rendered for the epoch-45 checkpoint in docs/peri_data_fusion_eval.md
     (image ids 15, 20, 71), so the before/after is directly comparable.
  2. the FULL real-peri subset -- seoul_univ_inspect_COCO val, filename
     contains "peri", 82 images (same set as eval_peri_subset.py).

Uses rfdetr/scripts/compare_core.py (same inference/metrics pipeline as
eval_panorama_and_overlap.py / eval_peri_subset.py), writing to fresh
"*_progress" cache dirs tagged with the checkpoint's current epoch number so
the original epoch-45 cache under outputs/compare_peri_fusion/ is untouched.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_peri_fusion_progress.py
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu

from rfdetr.datasets.xraypanoramic import universal_to_fdi
from rfdetr.scripts.compare_core import ROOT, load_coco, summarize_models

PANO_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022')
PERI_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_COCO')
BASE_CKPT = ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth'
NEW_CKPT = ROOT / 'outputs/peri_data_fusion/checkpoint_best_ema.pth'
BASE_NAME = 'baseline_panoramic_overlapfix'

PANO_SAMPLE_IDS = [15, 20, 71]
PERI_RENDER_IDS = [163, 132, 103]
SCORE_THRESHOLD = 0.3
FONT = '/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'

METRIC_KEYS = ('instance_dice', 'instance_iou', 'boundary_f1', 'contact_f1')


def epoch_of(ckpt_path):
    return torch.load(ckpt_path, map_location='cpu', weights_only=False)['epoch']


def filter_ids(coco, ids):
    keep = set(ids)
    coco.dataset['images'] = [im for im in coco.dataset['images'] if im['id'] in keep]
    coco.dataset['annotations'] = [a for a in coco.dataset['annotations'] if a['image_id'] in keep]
    coco.createIndex()
    return coco


def color(k):
    hsv = np.uint8([[[int(k * 137.5) % 180, 200, 255]]])
    return tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]))


def panel(rgb, masks, labels, title, target_w=620, target_h=330):
    h, w = rgb.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = round(w * scale), round(h * scale)
    patch = cv2.resize(rgb, (nw, nh))
    for mask, label in zip(masks, labels):
        fdi = int(universal_to_fdi[label]) if 0 <= label < len(universal_to_fdi) else int(label)
        crop = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = color(label)
        cv2.drawContours(patch, contours, -1, c, 2)
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            M = cv2.moments(biggest)
            if M['m00'] > 0:
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            else:
                x, y, bw, bh = cv2.boundingRect(biggest)
                cx, cy = x + bw // 2, y + bh // 2
            text = str(fdi)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            org = (cx - tw // 2, cy + th // 2)
            cv2.putText(patch, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(patch, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    canvas = Image.new('RGB', (target_w, target_h + 40), (16, 22, 30))
    canvas.paste(Image.fromarray(patch), ((target_w - nw) // 2, 40 + (target_h - nh) // 2))
    ImageDraw.Draw(canvas).text((10, 8), title, font=ImageFont.truetype(FONT, 16), fill='white')
    return canvas


def render(coco, detections, rows, model_names, image_id, out_path, tag='', show_raw=True, show_gt=True):
    """model_names: list of detection-model names to render as columns, in
    order (e.g. [new_name] or [BASE_NAME, new_name]). show_raw/show_gt toggle
    the plain-image and GT-overlay columns."""
    im = coco.imgs[image_id]
    rgb = np.array(Image.open(im['file_name']).convert('RGB'))
    anns = coco.imgToAnns[image_id]

    cols = []
    if show_raw:
        cols.append(('raw', [], [], 'raw'))
    if show_gt:
        gt_masks = [mu.decode(coco.annToRLE(a)).astype(bool) for a in anns]
        gt_labels = [a['category_id'] for a in anns]
        cols.append(('GT', gt_masks, gt_labels, None))
    for name in model_names:
        dd = [d for d in detections[name] if d['image_id'] == image_id and d['score'] >= SCORE_THRESHOLD]
        masks = [mu.decode(d['segmentation']).astype(bool) for d in dd]
        labels = [d['category_id'] for d in dd]
        cols.append((name, masks, labels, rows[name][image_id]))

    sheet = Image.new('RGB', (620 * len(cols), 330 + 40), (16, 22, 30))
    for col, (name, masks, labels, r) in enumerate(cols):
        if r == 'raw':
            title = f'image {image_id} {tag} (raw)'
        elif r is None:
            title = f'GT | image {image_id} {tag}'
        else:
            title = f'{name} | dice {r["mean_dice"]:.3f}'
        sheet.paste(panel(rgb, masks, labels, title), (col * 620, 0))
    sheet.save(out_path, quality=92)
    print('wrote', out_path)


def run_subset(models, coco, img_root, out_dir, label):
    print(f'{label}: {len(coco.imgs)} images, {len(coco.anns)} annotations')
    detections, overall = summarize_models(models, coco, img_root, out_dir)
    rows = {}
    for name in models:
        rows[name] = {r['image_id']: r for r in json.loads((out_dir / f'{name}_per_image.json').read_text())}
    print(f'\n=== {label} ===')
    for name, m in overall.items():
        print(f"  {name:>34s}  dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
              f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}")
    return detections, rows, overall


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--show-baseline', action='store_true',
                    help='also render the baseline model column (default: new model only)')
    p.add_argument('--no-raw', action='store_true', help='omit the plain raw-image column')
    p.add_argument('--no-gt', action='store_true', help='omit the GT-overlay column')
    return p.parse_args()


def main():
    args = parse_args()
    show_raw, show_gt = not args.no_raw, not args.no_gt

    epoch = epoch_of(NEW_CKPT)
    new_name = f'peri_data_fusion_ep{epoch}_ema'
    models = {BASE_NAME: str(BASE_CKPT), new_name: str(NEW_CKPT)}
    render_models = [BASE_NAME, new_name] if args.show_baseline else [new_name]
    print(f'checkpoint_best_ema.pth is currently at epoch {epoch}')

    img_dir = ROOT / 'outputs/compare_peri_fusion/images'
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. a handful of pano images (own1022 val, same 3 ids as the ep45 doc render)
    pano_out = ROOT / 'outputs/compare_peri_fusion/panorama_progress'
    pano_out.mkdir(parents=True, exist_ok=True)
    coco_pano = filter_ids(load_coco(PANO_DATA / 'val', PANO_DATA / 'val/annotations.json'), PANO_SAMPLE_IDS)
    dets_pano, rows_pano, overall_pano = run_subset(models, coco_pano, PANO_DATA / 'val', pano_out,
                                                      f'pano sample ({len(PANO_SAMPLE_IDS)} images)')
    for image_id in PANO_SAMPLE_IDS:
        render(coco_pano, dets_pano, rows_pano, render_models, image_id,
               img_dir / f'panorama_{image_id}_ep{epoch}.jpg', tag='(pano progress check)',
               show_raw=show_raw, show_gt=show_gt)

    # 2. ALL real peri images (seoul_univ val, filename contains "peri")
    peri_out = ROOT / 'outputs/compare_peri_fusion/peri_subset_progress'
    peri_out.mkdir(parents=True, exist_ok=True)
    coco_peri = load_coco(PERI_DATA / 'val', PERI_DATA / 'val/annotations.json', filename_contains='peri')
    dets_peri, rows_peri, overall_peri = run_subset(models, coco_peri, PERI_DATA / 'val', peri_out,
                                                      'real peri (full)')
    for image_id in PERI_RENDER_IDS:
        if image_id in coco_peri.imgs:
            render(coco_peri, dets_peri, rows_peri, render_models, image_id,
                   img_dir / f'peri_{image_id}_ep{epoch}.jpg', tag='(real peri photo)',
                   show_raw=show_raw, show_gt=show_gt)

    summary = dict(
        new_checkpoint=str(NEW_CKPT),
        new_checkpoint_epoch=epoch,
        pano_sample_ids=PANO_SAMPLE_IDS,
        pano_metrics={k: {kk: v[kk] for kk in METRIC_KEYS} for k, v in overall_pano.items()},
        peri_full_metrics={k: {kk: v[kk] for kk in METRIC_KEYS} for k, v in overall_peri.items()},
        peri_num_images=len(coco_peri.imgs),
        peri_num_annotations=len(coco_peri.anns),
    )
    out_path = ROOT / 'outputs/compare_peri_fusion/progress_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nsaved summary:', out_path)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
