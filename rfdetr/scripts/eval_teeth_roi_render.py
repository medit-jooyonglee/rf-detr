"""teeth_roi mode: run both checkpoints on the own1022 val set with the exact same
deterministic teeth_roi crop used at train/eval time (XrayPnoaramicInstance's own
_sample_crop_plan / parse_item_coco pipeline -- same code path the official
`--eval --eval_xray_crop` numbers in docs/peri_data_fusion_eval.md §4 come from,
NOT the manual letterbox preprocessing in compare_core.py), then render a few
GT / baseline / new side-by-side comparison images.

Run: PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_teeth_roi_render.py
"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from rfdetr.datasets.xraypanoramic import XrayPnoaramicInstance
from rfdetr.models import build_model
from rfdetr.models.segmentation_head import expand_normalized_boxes, paste_masks_in_image
from rfdetr.util.box_ops import box_cxcywh_to_xyxy
from tools.compare_own1022 import rle, metrics, aggregate

ROOT = Path(__file__).resolve().parents[2]
FONT = '/opt/conda/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'
DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022')
CHECKPOINTS = {
    'baseline_panoramic_overlapfix': ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth',
    'peri_data_fusion_ep45_ema': ROOT / 'outputs/peri_data_fusion/checkpoint_best_ema.pth',
}
OUT_DIR = ROOT / 'outputs/compare_peri_fusion/teeth_roi'
IMG_DIR = ROOT / 'outputs/compare_peri_fusion/images'
SCORE_THRESHOLD = 0.3


def build_dataset():
    return XrayPnoaramicInstance(
        img_folder=[],
        coco_directories=[(str(DATA / 'val'), str(DATA / 'val/annotations.json'))],
        include_masks=True,
        name='val',
        num_classes=32,
        xray_crop_mode='teeth_roi',
        xray_crop_prob=1.0,
        eval_xray_crop=True,
        eval_xray_crop_seed=42,
    )


def load_model(checkpoint_path, device='cuda:0'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = copy.deepcopy(checkpoint['args'])
    args.device = device
    args.pretrain_weights = str(checkpoint_path)
    model = build_model(args)
    load_result = model.load_state_dict(checkpoint['model'], strict=True)
    print(checkpoint_path, 'STRICT LOAD', load_result, flush=True)
    return model.to(device).eval(), args


def run_all(name, checkpoint_path, dataset, device='cuda:0'):
    cache = OUT_DIR / f'{name}_teethroi_predictions.json'
    rowcache = OUT_DIR / f'{name}_teethroi_rows.json'
    imgcache = OUT_DIR / f'{name}_teethroi_images.npz'
    if cache.exists() and rowcache.exists() and imgcache.exists():
        return json.loads(cache.read_text()), json.loads(rowcache.read_text())

    model, args = load_model(checkpoint_path, device)
    rows, detections = [], []
    saved_rgb = {}
    for idx in range(len(dataset)):
        src, target = dataset.parse_item_coco(idx)
        if src.shape[0] == 1:
            src = np.repeat(src, 3, axis=0)
        h, w = src.shape[1:]
        x = torch.from_numpy(src.copy()).float().to(device)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            y = model([x])
            prob = y['pred_logits'][0].sigmoid()
            scores, indices = prob.flatten().topk(args.num_select)
            qidx = indices // prob.shape[-1]
            labels = indices % prob.shape[-1]
            b = box_cxcywh_to_xyxy(y['pred_boxes'][0, qidx])
            mb = box_cxcywh_to_xyxy(expand_normalized_boxes(
                y['pred_boxes'][0, qidx], getattr(args, 'segmentation_crop_box_scale', 1.15)))
            factor = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
            boxes = b * factor
            maskboxes = mb * factor
            masks = paste_masks_in_image(y['pred_masks'][0, qidx], maskboxes, (h, w)) > 0
        labels = labels.cpu().numpy()
        scores = scores.float().cpu().numpy()
        boxes = boxes.float().cpu().numpy()
        masks = masks.cpu().numpy()

        gt_masks = np.asarray(target['masks']).astype(bool)
        gt_boxes_norm = np.asarray(target['boxes'], dtype=np.float32).reshape(-1, 4)
        gt_boxes_xyxy = box_cxcywh_to_xyxy(torch.from_numpy(gt_boxes_norm)).numpy() * np.array([w, h, w, h])
        labels_gt = np.asarray(target['labels']).reshape(-1)
        anns = [dict(bbox=[float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                     category_id=int(lab), image_id=idx)
                for (x0, y0, x1, y1), lab in zip(gt_boxes_xyxy, labels_gt)]
        if len(anns) == 0:
            continue
        row = metrics(gt_masks, masks, labels, scores, boxes, anns, h, w)
        row['image_id'] = idx
        rows.append(row)
        for lab, score, box, mask in zip(labels, scores, boxes, masks):
            if lab == 0 or score < SCORE_THRESHOLD:
                continue
            x0, y0, x1, y1 = map(float, box)
            detections.append(dict(image_id=idx, category_id=int(lab), score=float(score),
                                    bbox=[x0, y0, x1 - x0, y1 - y0], segmentation=rle(mask)))
        rgb_u8 = (np.transpose(src, [1, 2, 0]) * 255).astype(np.uint8)
        saved_rgb[str(idx)] = rgb_u8
        if idx % 20 == 0:
            print(name, idx + 1, '/', len(dataset), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(detections))
    rowcache.write_text(json.dumps(rows, indent=2))
    np.savez_compressed(imgcache, **saved_rgb)
    del model
    torch.cuda.empty_cache()
    return detections, rows


def color(k):
    hsv = np.uint8([[[int(k * 137.5) % 180, 200, 255]]])
    return tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]))


def panel(rgb, masks, labels, title, target_w=520, target_h=330):
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
    ImageDraw.Draw(canvas).text((10, 8), title, font=ImageFont.truetype(FONT, 15), fill='white')
    return canvas


def main():
    dataset = build_dataset()
    print('teeth_roi val images:', len(dataset))

    detections, rows, imgs = {}, {}, {}
    for name, ckpt in CHECKPOINTS.items():
        dets, rr = run_all(name, ckpt, dataset)
        detections[name] = {d['image_id']: [] for d in []}
        by_img = {}
        for d in dets:
            by_img.setdefault(d['image_id'], []).append(d)
        detections[name] = by_img
        rows[name] = {r['image_id']: r for r in rr}
        npz = np.load(OUT_DIR / f'{name}_teethroi_images.npz')
        imgs[name] = {int(k): npz[k] for k in npz.files}

    names = list(CHECKPOINTS.keys())
    base, new = names
    common_ids = sorted(set(rows[base]) & set(rows[new]))
    deltas = sorted(common_ids, key=lambda i: rows[new][i]['mean_dice'] - rows[base][i]['mean_dice'])
    selected = [('improved', deltas[-1]), ('typical', deltas[len(deltas) // 2]), ('regressed', deltas[0])]

    from pycocotools import mask as mu
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    gt_cache_by_model = {}
    for tag, image_id in selected:
        rgb = imgs[new][image_id]
        gt_masks = []
        gt_labels = []
        # Recompute GT for this index (cheap, deterministic) to avoid caching masks separately.
        src, target = dataset.parse_item_coco(image_id)
        gt_masks = np.asarray(target['masks']).astype(bool)
        gt_labels = np.asarray(target['labels']).reshape(-1)

        sheet = Image.new('RGB', (520 * 3, 330 + 40), (16, 22, 30))
        cols = [('GT', gt_masks, gt_labels, None)]
        for name in names:
            dd = detections[name].get(image_id, [])
            masks = [mu.decode(d['segmentation']).astype(bool) for d in dd]
            labels = [d['category_id'] for d in dd]
            cols.append((name, masks, labels, rows[name][image_id]))
        for col, (name, masks, labels, r) in enumerate(cols):
            title = f'GT | teeth_roi crop, image {image_id} ({tag})' if r is None else \
                f'{name} | dice {r["mean_dice"]:.3f}'
            sheet.paste(panel(rgb, masks, labels, title), (col * 520, 0))
        out_path = IMG_DIR / f'teethroi_{image_id}_{tag}.jpg'
        sheet.save(out_path, quality=92)
        print('wrote', out_path)

    agg = {name: aggregate(list(rows[name].values())) for name in names}
    (OUT_DIR / 'teethroi_render_summary.json').write_text(json.dumps(
        {name: {k: v for k, v in a.items() if k in ('instance_dice', 'instance_iou')} for name, a in agg.items()},
        indent=2))
    print(json.dumps({name: agg[name]['instance_dice'] for name in names}, indent=2))


if __name__ == '__main__':
    main()
