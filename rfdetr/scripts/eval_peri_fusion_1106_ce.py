"""Evaluate outputs/peri_data_fusion_1106_ce (the "with ce" continuation,
trained on top of outputs/peri_data_fusion_1104_ce) against two reference
points:

  - baseline_panoramic_overlapfix  -- the existing deployed checkpoint
    (outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix).
  - peri_data_fusion_ema ("without ce")  -- outputs/peri_data_fusion,
    the pre-"_ce" checkpoint that outputs/peri_data_fusion_1104_ce /
    outputs/peri_data_fusion_1106_ce were continued-trained from.

on four held-out (1106 val split) subsets, 10 images each where available:

  1. cbct_pano   -- cbct_ios_dcm_latest_1106 val, large-resolution (panoramic) images (5)
  2. seoul_pano  -- seoul_univ_inspect_1106 val, filename contains "pano" (5)
     (cbct_pano + seoul_pano together = the 10-image "xray/pano" sample)
  3. peri_seoul  -- seoul_univ_inspect_1106 val, filename contains "peri" (10)
     (real, un-cropped periapical-style photographs)
  4. peri_cbct   -- cbct_ios_dcm_latest_1106 val, resolution <=1000x1000 (10)
     (small-resolution images that are periapical-like per their resolution,
     same heuristic as rfdetr.deploy.predict's cbct peri/pano split)

Saves a handful of 5-column (raw | GT | baseline | without_ce | with_ce)
comparison sheets per subset (not every image) plus a JSON metrics summary.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_peri_fusion_1106_ce.py
"""
import json
from pathlib import Path

import numpy as np

from rfdetr.scripts.compare_core import ROOT, load_coco, run_inference
from rfdetr.scripts.eval_peri_fusion_progress import filter_ids, render, epoch_of
from tools.compare_own1022 import aggregate

XRAY_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1106')
PERI_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_1106')

BASE_CKPT = ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth'
WITHOUT_CE_CKPT = ROOT / 'outputs/peri_data_fusion/checkpoint_best_ema.pth'
WITH_CE_CKPT = ROOT / 'outputs/peri_data_fusion_1106_ce/checkpoint_best_ema.pth'
BASE_NAME = 'baseline_panoramic_overlapfix'
WITHOUT_CE_NAME = 'peri_data_fusion_ema (without_ce)'
WITH_CE_NAME = 'peri_data_fusion_1106_ce_ema (with_ce)'

MODELS = {BASE_NAME: str(BASE_CKPT), WITHOUT_CE_NAME: str(WITHOUT_CE_CKPT), WITH_CE_NAME: str(WITH_CE_CKPT)}
RENDER_MODELS = [BASE_NAME, WITHOUT_CE_NAME, WITH_CE_NAME]

OUT_DIR = ROOT / 'outputs/compare_peri_fusion_1106_ce'
IMG_DIR = OUT_DIR / 'images'

SEED = 42
N_PANO_EACH = 5   # cbct_pano + seoul_pano = 10 "xray/pano" images
N_PERI_EACH = 10  # per peri type
N_RENDER = 3      # sample sheets saved per subset


def sample_ids(coco, n, seed=SEED):
    ids = sorted(im['id'] for im in coco.dataset['images'] if coco.imgToAnns.get(im['id']))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    return ids[:n]


def build_subset(img_root, annot_file, n, filename_contains=None, resolution_at_most=None, seed=SEED):
    coco = load_coco(img_root, annot_file, filename_contains=filename_contains)
    if resolution_at_most is not None:
        mw, mh = resolution_at_most
        keep = {im['id'] for im in coco.dataset['images'] if im['width'] <= mw and im['height'] <= mh}
        coco.dataset['images'] = [im for im in coco.dataset['images'] if im['id'] in keep]
        coco.dataset['annotations'] = [a for a in coco.dataset['annotations'] if a['image_id'] in keep]
        coco.createIndex()
    ids = sample_ids(coco, n, seed=seed)
    return filter_ids(coco, ids)


def run_subset(label, coco, img_root, out_dir, n_render):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n=== {label}: {len(coco.imgs)} images, {len(coco.anns)} annotations ===')
    detections, rows = {}, {}
    for name, ckpt in MODELS.items():
        dets, r = run_inference(name, ckpt, coco, img_root, out_dir)
        detections[name] = dets
        rows[name] = r
    overall = {name: aggregate(r) for name, r in rows.items()}
    for name, m in overall.items():
        print(f"  {name:>34s}  dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
              f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}")

    row_by_id = {name: {r['file_name']: r for r in rows[name]} for name in MODELS}
    rows_for_render = {name: {im['id']: row_by_id[name][im['file_name']] for im in coco.imgs.values()}
                       for name in MODELS}
    image_ids = list(coco.imgs.keys())[:n_render]
    for image_id in image_ids:
        render(coco, detections, rows_for_render, RENDER_MODELS, image_id,
               IMG_DIR / f'{label}_{image_id}.jpg', tag=f'({label})')
    return overall


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    print('with_ce checkpoint epoch:', epoch_of(WITH_CE_CKPT))
    print('without_ce checkpoint epoch:', epoch_of(WITHOUT_CE_CKPT))

    # cbct_pano: LARGE-res images only (inverse of the <=1000x1000 peri_cbct filter below)
    cbct_full = load_coco(XRAY_DATA / 'valid', XRAY_DATA / 'valid/annotations.json')
    large_ids = {im['id'] for im in cbct_full.dataset['images']
                 if not (im['width'] <= 1000 and im['height'] <= 1000)}
    cbct_full.dataset['images'] = [im for im in cbct_full.dataset['images'] if im['id'] in large_ids]
    cbct_full.dataset['annotations'] = [a for a in cbct_full.dataset['annotations'] if a['image_id'] in large_ids]
    cbct_full.createIndex()
    cbct_pano = filter_ids(cbct_full, sample_ids(cbct_full, N_PANO_EACH, seed=SEED))

    seoul_pano = build_subset(PERI_DATA / 'valid', PERI_DATA / 'valid/annotations.json',
                               N_PANO_EACH, filename_contains='pano', seed=SEED)
    peri_seoul = build_subset(PERI_DATA / 'valid', PERI_DATA / 'valid/annotations.json',
                               N_PERI_EACH, filename_contains='peri', seed=SEED)
    peri_cbct = build_subset(XRAY_DATA / 'valid', XRAY_DATA / 'valid/annotations.json',
                              N_PERI_EACH, resolution_at_most=(1000, 1000), seed=SEED)

    overall_cbct_pano = run_subset('cbct_pano', cbct_pano, XRAY_DATA / 'valid', OUT_DIR / 'cbct_pano', N_RENDER)
    overall_seoul_pano = run_subset('seoul_pano', seoul_pano, PERI_DATA / 'valid', OUT_DIR / 'seoul_pano', N_RENDER)
    overall_peri_seoul = run_subset('peri_seoul', peri_seoul, PERI_DATA / 'valid', OUT_DIR / 'peri_seoul', N_RENDER)
    overall_peri_cbct = run_subset('peri_cbct', peri_cbct, XRAY_DATA / 'valid', OUT_DIR / 'peri_cbct', N_RENDER)

    # combined "xray/pano" view (cbct_pano + seoul_pano rows merged)
    pano_rows = {}
    for name in MODELS:
        cbct_rows = json.loads((OUT_DIR / 'cbct_pano' / f'{name}_per_image.json').read_text())
        seoul_rows = json.loads((OUT_DIR / 'seoul_pano' / f'{name}_per_image.json').read_text())
        pano_rows[name] = cbct_rows + seoul_rows
    overall_pano_combined = {name: aggregate(rows) for name, rows in pano_rows.items()}

    print('\n=== combined xray/pano (cbct_pano + seoul_pano, n=%d) ===' % len(pano_rows[BASE_NAME]))
    for name, m in overall_pano_combined.items():
        print(f"  {name:>34s}  dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
              f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}")

    def trim(m):
        return {k: {kk: v[kk] for kk in ('instance_dice', 'instance_iou', 'boundary_f1', 'contact_f1')}
                for k, v in m.items()}

    summary = dict(
        checkpoints={BASE_NAME: str(BASE_CKPT), WITHOUT_CE_NAME: str(WITHOUT_CE_CKPT), WITH_CE_NAME: str(WITH_CE_CKPT)},
        with_ce_epoch=epoch_of(WITH_CE_CKPT),
        without_ce_epoch=epoch_of(WITHOUT_CE_CKPT),
        subsets=dict(
            cbct_pano=dict(n=len(cbct_pano.imgs), metrics=trim(overall_cbct_pano)),
            seoul_pano=dict(n=len(seoul_pano.imgs), metrics=trim(overall_seoul_pano)),
            xray_pano_combined=dict(n=len(pano_rows[BASE_NAME]), metrics=trim(overall_pano_combined)),
            peri_seoul=dict(n=len(peri_seoul.imgs), metrics=trim(overall_peri_seoul)),
            peri_cbct=dict(n=len(peri_cbct.imgs), metrics=trim(overall_peri_cbct)),
        ),
    )
    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nsaved summary:', out_path)
    print('saved sample sheets:', IMG_DIR)


if __name__ == '__main__':
    main()
