"""Compare a retrained checkpoint against the deployed pretrained model on two
specific questions (not general mAP):
  1. Non-metal, same-arch neighboring-tooth overlap preservation (crowding/
     mesial-distal overlap). Upper-lower crossing overlap is excluded -- that's
     usually a bad-shot panoramic artifact, not genuine tooth-to-tooth overlap.
  2. Metal / bright-prosthesis region fill-in (gray>=240 inside GT is a
     brightness proxy for metal, not a real material label).

Reuses the own1022 val split and the metrics()/aggregate() helpers already
built in tools/compare_own1022.py, plus the dataset's metal_candidates.json
(bright-region crops), so results are directly comparable to past runs.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /opt/conda/bin/python tools/compare_overlapfix_vs_deployed.py \\
      --checkpoint outputs/fine_tune_own1022_v2_overlapfix/checkpoint_best_ema.pth \\
      --name overlapfix_v2

Re-run with a different --checkpoint/--name as training progresses (e.g. once
checkpoint_best_total.pth exists) without editing this file.
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mu

from rfdetr.models import build_model, PostProcess
from rfdetr.engine import non_max_suppression
from tools.compare_own1022 import DATA, metrics, aggregate, rle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/compare_overlapfix_vs_deployed'
DEFAULT_PRETRAINED = ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth'
SCORE_THRESHOLD = 0.3
BRIGHT_THRESH = 240


def is_adjacent_fdi(fdi_a, fdi_b):
    """같은 quadrant에서 번호가 1 차이 나거나 정중선(11-21, 31-41)을 사이에 둔 인접 치아인지."""
    qa, na = divmod(int(fdi_a), 10)
    qb, nb = divmod(int(fdi_b), 10)
    if qa == qb:
        return abs(na - nb) == 1
    return na == 1 and nb == 1 and {qa, qb} in ({1, 2}, {3, 4})


def run_inference(name, path, coco, out_dir):
    """Returns (detections list, per-image metrics rows). Caches to out_dir."""
    cache = out_dir / f'{name}_predictions.json'
    rowcache = out_dir / f'{name}_per_image.json'
    if cache.exists() and rowcache.exists():
        return json.loads(cache.read_text()), json.loads(rowcache.read_text())

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = copy.deepcopy(checkpoint['args'])
    args.device = 'cuda:0'
    args.pretrain_weights = str(path)
    model = build_model(args)
    load = model.load_state_dict(checkpoint['model'], strict=True)
    print(name, 'STRICT LOAD', load, flush=True)
    model = model.to('cuda:0').eval()
    post = PostProcess(args.num_select, args.segmentation_mode, getattr(args, 'segmentation_crop_box_scale', 1.15))
    del post  # not used directly; kept for parity with compare_own1022.py's import

    import math
    from rfdetr.models.segmentation_head import expand_normalized_boxes, paste_masks_in_image
    from rfdetr.util.box_ops import box_cxcywh_to_xyxy

    rows, detections = [], []
    for idx, im in enumerate(coco.imgs.values()):
        anns = coco.imgToAnns[im['id']]
        rgb = np.array(Image.open(DATA / 'val' / im['file_name']).convert('RGB'))
        h, w = rgb.shape[:2]
        ih = (int(math.ceil(h * 640 / w / 16)) * 16 // 64 + 1) * 64
        scale = min(704 / w, ih / h); nw, nh = round(w * scale), round(h * scale)
        ox, oy = (704 - nw) // 2, (ih - nh) // 2
        x = np.zeros((ih, 704, 3), dtype=np.uint8)
        x[oy:oy + nh, ox:ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div_(255).to('cuda:0')
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            y = model([x])
            prob = y['pred_logits'][0].sigmoid()
            scores, indices = prob.flatten().topk(args.num_select)
            qidx = indices // prob.shape[-1]; labels = indices % prob.shape[-1]
            b = box_cxcywh_to_xyxy(y['pred_boxes'][0, qidx])
            mb = box_cxcywh_to_xyxy(expand_normalized_boxes(y['pred_boxes'][0, qidx], getattr(args, 'segmentation_crop_box_scale', 1.15)))
            factor = torch.tensor([704, ih, 704, ih], device='cuda:0')
            offset = torch.tensor([ox, oy, ox, oy], device='cuda:0')
            boxes = (b * factor - offset) / scale
            maskboxes = (mb * factor - offset) / scale
            masks = paste_masks_in_image(y['pred_masks'][0, qidx], maskboxes, (h, w)) > 0
        labels = labels.cpu().numpy(); scores = scores.float().cpu().numpy()
        boxes = boxes.float().cpu().numpy(); masks = masks.cpu().numpy()
        gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])
        row = metrics(gt, masks, labels, scores, boxes, anns, h, w)
        row['file_name'] = im['file_name']; rows.append(row)
        for lab, score, box, mask in zip(labels, scores, boxes, masks):
            if lab == 0:
                continue
            x0, y0, x1, y1 = map(float, box)
            detections.append(dict(image_id=im['id'], category_id=int(lab), score=float(score),
                                    bbox=[x0, y0, x1 - x0, y1 - y0], segmentation=rle(mask)))
        if idx % 20 == 0:
            print(name, idx + 1, '/', len(coco.imgs), flush=True)
    cache.write_text(json.dumps(detections))
    rowcache.write_text(json.dumps(rows, indent=2))
    del model
    torch.cuda.empty_cache()
    return detections, rows


def nonmetal_adjacent_overlap(coco, models_detections):
    """Pair-wise overlap recall/precision restricted to non-metal, same-arch
    neighboring-tooth pairs (excludes upper/lower crossing + bright/metal px)."""
    indexed = {name: defaultdict(list) for name in models_detections}
    for name, dets in models_detections.items():
        for d in dets:
            if d['score'] >= SCORE_THRESHOLD:
                indexed[name][d['image_id']].append(d)

    agg = {name: dict(pair_tp=0, pair_gt=0, pair_pred=0) for name in models_detections}
    for i, im in coco.imgs.items():
        anns = coco.imgToAnns[i]
        h, w = im['height'], im['width']
        gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])
        fdis = [a['category_id'] for a in anns]
        rgb = np.array(Image.open(DATA / 'val' / im['file_name'].replace('\\', '/')).convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bright = gray >= BRIGHT_THRESH

        pairs = []
        for a in range(len(anns)):
            for b in range(a):
                if not (gt[a] & gt[b]).any():
                    continue
                if is_adjacent_fdi(fdis[a], fdis[b]):
                    pairs.append((a, b))
        if not pairs:
            continue

        for name in models_detections:
            dd = indexed[name][i]
            labels = np.array([d['category_id'] for d in dd]) if dd else np.array([])
            scores = np.array([d['score'] for d in dd]) if dd else np.array([])
            pred_masks = [mu.decode(d['segmentation']).astype(bool) for d in dd]
            matched = [None] * len(anns)
            for category in {a['category_id'] for a in anns}:
                gi = [k for k, a in enumerate(anns) if a['category_id'] == category]
                cand = [k for k, l in enumerate(labels) if l == category]
                if not cand or len(gi) != 1:
                    continue
                best = max(cand, key=lambda k: scores[k])
                matched[gi[0]] = pred_masks[best]
            pair_tp = pair_total = pair_pred = 0
            for a, b in pairs:
                ov = gt[a] & gt[b] & ~bright  # exclude metal pixels from this question
                n = int(ov.sum())
                if n == 0:
                    continue
                pair_total += n
                ma, mb = matched[a], matched[b]
                if ma is not None and mb is not None:
                    pp = ma & mb
                    pair_tp += int((pp & ov).sum())
                    pair_pred += int((pp & ~bright).sum())
            agg[name]['pair_tp'] += pair_tp
            agg[name]['pair_gt'] += pair_total
            agg[name]['pair_pred'] += pair_pred

    result = {}
    for name, a in agg.items():
        result[name] = dict(
            pair_tp=a['pair_tp'], pair_gt=a['pair_gt'], pair_pred=a['pair_pred'],
            recall=a['pair_tp'] / max(1, a['pair_gt']),
            precision=a['pair_tp'] / max(1, a['pair_pred']),
        )
    return result


def metal_fill(coco, models_detections, metal_candidates):
    indexed = {name: defaultdict(list) for name in models_detections}
    for name, dets in models_detections.items():
        for d in dets:
            if d['score'] > .5:
                indexed[name][d['image_id']].append(d)

    totals = {name: dict(bright_gt_px=0, missing_px=0) for name in models_detections}
    per_image = []
    for item in metal_candidates:
        i = item['image_id']; im = coco.imgs[i]
        rgb = np.array(Image.open(DATA / 'val' / im['file_name'].replace('\\', '/')).convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gt = np.any([coco.annToMask(a).astype(bool) for a in coco.imgToAnns[i]], axis=0)
        x0, y0, x1, y1 = item['crop_box']
        roi = np.zeros(gt.shape, bool); roi[y0:y1, x0:x1] = True
        bright = (gray >= BRIGHT_THRESH) & gt & roi
        row = dict(image_id=i, bright_gt_px=int(bright.sum()), models={})
        for name in models_detections:
            dd = indexed[name][i]
            boxes = np.array([d['bbox'] for d in dd]).reshape(-1, 4); boxes[:, 2:] += boxes[:, :2]
            keep = non_max_suppression(boxes, np.array([d['score'] for d in dd]), threshold=.4) if len(dd) else []
            pred = np.zeros(gt.shape, bool)
            for j in keep:
                pred |= mu.decode(dd[j]['segmentation']).astype(bool)
            miss = int((bright & ~pred).sum())
            row['models'][name] = dict(missing_px=miss, missing_ratio=miss / max(1, int(bright.sum())))
            totals[name]['bright_gt_px'] += int(bright.sum())
            totals[name]['missing_px'] += miss
        per_image.append(row)

    result = {name: dict(bright_gt_px=t['bright_gt_px'], missing_px=t['missing_px'],
                          missing_ratio=t['missing_px'] / max(1, t['bright_gt_px']))
              for name, t in totals.items()}
    return result, per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=str, required=True, help='retrained checkpoint to evaluate')
    parser.add_argument('--name', type=str, required=True, help='short label for output files/report')
    parser.add_argument('--pretrained', type=str, default=str(DEFAULT_PRETRAINED), help='deployed model checkpoint')
    parser.add_argument('--out', type=str, default=str(DEFAULT_OUT), help='output directory')
    args = parser.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    models = {'pretrained': Path(args.pretrained), args.name: Path(args.checkpoint)}

    coco = COCO(str(DATA / 'val/annotations.json'))
    coco.dataset.setdefault('info', {})
    for x in coco.dataset['images']:
        x['file_name'] = x['file_name'].replace('\\', '/')

    detections = {}
    overall = {}
    for name, path in models.items():
        print('=== MODEL', name, '===', flush=True)
        dets, rows = run_inference(name, path, coco, out_dir)
        detections[name] = dets
        overall[name] = aggregate(rows)
        overall[name]['path'] = str(path)

    metal_candidates = json.loads((ROOT / 'outputs/compare_own1022/metal_candidates.json').read_text())

    nonmetal = nonmetal_adjacent_overlap(coco, detections)
    metal, metal_per_image = metal_fill(coco, detections, metal_candidates)

    summary = dict(
        checkpoints={name: str(p) for name, p in models.items()},
        overall_metrics={name: {k: v for k, v in m.items() if k in (
            'instance_dice', 'instance_iou', 'overlap_iou', 'overlap_recall',
            'boundary_f1', 'contact_f1', 'bbox_AP', 'segm_AP', 'path')}
            for name, m in overall.items()},
        nonmetal_adjacent_overlap=nonmetal,
        metal_fill=metal,
    )
    (out_dir / f'summary_{args.name}.json').write_text(json.dumps(summary, indent=2))

    print('\n=== non-metal, same-arch neighboring-tooth overlap (recall = GT overlap px correctly covered by both correct-FDI predictions) ===')
    for name, m in nonmetal.items():
        print(f'  {name:>15s}  recall={m["recall"]*100:5.1f}%  precision={m["precision"]*100:5.1f}%  gt_px={m["pair_gt"]}')

    print('\n=== metal / bright-prosthesis region: missing ratio (lower = better fill) ===')
    for name, m in metal.items():
        print(f'  {name:>15s}  missing={m["missing_ratio"]*100:5.2f}%  gt_px={m["bright_gt_px"]}')

    print(f'\nsaved: {out_dir}/summary_{args.name}.json')


if __name__ == '__main__':
    main()
