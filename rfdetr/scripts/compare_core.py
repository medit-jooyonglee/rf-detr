"""Core, dataset-agnostic comparison logic shared by the rfdetr/scripts/eval_*.py
comparison scripts.

This generalizes the inference/metrics pipeline originally written (and still
usable standalone) in tools/compare_own1022.py and tools/compare_overlapfix_vs_deployed.py
so it can run against an arbitrary (img_root, coco annotations) pair instead of
the single hardcoded own1022 dataset. The per-image metric definitions (dice,
iou, overlap iou/recall, pairwise overlap, boundary/contact f1) and the
nonmetal_adjacent_overlap pairwise-overlap-recall metric are imported directly
from those modules rather than re-implemented, so results stay comparable with
past runs recorded under outputs/compare_own1022/ and outputs/compare_overlapfix_vs_deployed/.

Run scripts with PYTHONPATH=. (same convention as tools/*.py), e.g.:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /opt/conda/bin/python rfdetr/scripts/eval_panorama_and_overlap.py
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

from rfdetr.models import build_model, PostProcess
from rfdetr.models.segmentation_head import expand_normalized_boxes, paste_masks_in_image
from rfdetr.util.box_ops import box_cxcywh_to_xyxy
from rfdetr.datasets.xraypanoramic import merge_coco_directories

from tools.compare_own1022 import rle, metrics, aggregate
from tools.compare_overlapfix_vs_deployed import nonmetal_adjacent_overlap, is_adjacent_fdi

ROOT = Path(__file__).resolve().parents[2]


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_coco(img_root, annot_file, filename_contains: Optional[str] = None) -> COCO:
    """Load a COCO val set the same way XrayPnoaramicInstance does: rewrite
    file_name to an absolute path, drop images missing on disk, and optionally
    keep only images whose file_name contains `filename_contains` (case-insensitive).

    Reuses merge_coco_directories (single-source call) so the missing-file
    filtering logic stays in exactly one place in the codebase.
    """
    api = merge_coco_directories([(str(img_root), str(annot_file))], include_masks=False)
    coco = api.coco
    coco.dataset.setdefault('info', {})
    for x in coco.dataset['images']:
        x['file_name'] = x['file_name'].replace('\\', '/')

    if filename_contains:
        needle = filename_contains.casefold()
        keep_ids = {im['id'] for im in coco.dataset['images']
                    if needle in os.path.basename(im['file_name']).casefold()}
        coco.dataset['images'] = [im for im in coco.dataset['images'] if im['id'] in keep_ids]
        coco.dataset['annotations'] = [a for a in coco.dataset['annotations'] if a['image_id'] in keep_ids]
        coco.createIndex()

    return coco


def load_model(checkpoint_path, device='cuda:0'):
    """Load a model exactly as its own training args describe it (same
    convention as tools/compare_own1022.py): never hand-construct ModelConfig."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = copy.deepcopy(checkpoint['args'])
    args.device = device
    args.pretrain_weights = str(checkpoint_path)
    model = build_model(args)
    load_result = model.load_state_dict(checkpoint['model'], strict=True)
    print(checkpoint_path, 'STRICT LOAD', load_result, flush=True)
    model = model.to(device).eval()
    return model, args, checkpoint


def run_inference(name, checkpoint_path, coco: COCO, img_root, out_dir, device='cuda:0'):
    """Runs a model over every image in `coco` (absolute file_name already
    rewritten by load_coco). Caches detections + per-image metric rows to
    out_dir/{name}_predictions.json / {name}_per_image.json so repeat runs
    (e.g. adding a new comparison axis) don't re-run inference.

    Returns (detections, rows). `detections` entries: dict(image_id,
    category_id, score, bbox[xywh], segmentation=RLE). `rows`: per-image dict
    from tools.compare_own1022.metrics(), plus 'file_name'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f'{name}_predictions.json'
    rowcache = out_dir / f'{name}_per_image.json'
    if cache.exists() and rowcache.exists():
        return json.loads(cache.read_text()), json.loads(rowcache.read_text())

    model, args, _ = load_model(checkpoint_path, device)
    img_root = Path(img_root)

    rows, detections = [], []
    images = list(coco.imgs.values())
    for idx, im in enumerate(images):
        anns = coco.imgToAnns[im['id']]
        if not anns:
            continue
        file_path = im['file_name']
        if not os.path.isabs(file_path):
            file_path = img_root / file_path
        rgb = np.array(Image.open(file_path).convert('RGB'))
        h, w = rgb.shape[:2]
        ih = (int(math.ceil(h * 640 / w / 16)) * 16 // 64 + 1) * 64
        scale = min(704 / w, ih / h)
        nw, nh = round(w * scale), round(h * scale)
        ox, oy = (704 - nw) // 2, (ih - nh) // 2
        x = np.zeros((ih, 704, 3), dtype=np.uint8)
        x[oy:oy + nh, ox:ox + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div_(255).to(device)
        with torch.inference_mode(), torch.autocast('cuda', enabled=False):
            y = model([x])
            prob = y['pred_logits'][0].sigmoid()
            scores, indices = prob.flatten().topk(args.num_select)
            qidx = indices // prob.shape[-1]
            labels = indices % prob.shape[-1]
            b = box_cxcywh_to_xyxy(y['pred_boxes'][0, qidx])
            mb = box_cxcywh_to_xyxy(expand_normalized_boxes(
                y['pred_boxes'][0, qidx], getattr(args, 'segmentation_crop_box_scale', 1.15)))
            factor = torch.tensor([704, ih, 704, ih], device=device)
            offset = torch.tensor([ox, oy, ox, oy], device=device)
            boxes = (b * factor - offset) / scale
            maskboxes = (mb * factor - offset) / scale
            masks = paste_masks_in_image(y['pred_masks'][0, qidx], maskboxes, (h, w)) > 0
        labels = labels.cpu().numpy()
        scores = scores.float().cpu().numpy()
        boxes = boxes.float().cpu().numpy()
        masks = masks.cpu().numpy()
        gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])
        row = metrics(gt, masks, labels, scores, boxes, anns, h, w)
        row['file_name'] = im['file_name']
        rows.append(row)
        for lab, score, box, mask in zip(labels, scores, boxes, masks):
            if lab == 0:
                continue
            x0, y0, x1, y1 = map(float, box)
            detections.append(dict(image_id=im['id'], category_id=int(lab), score=float(score),
                                    bbox=[x0, y0, x1 - x0, y1 - y0], segmentation=rle(mask)))
        if idx % 20 == 0:
            print(name, idx + 1, '/', len(images), flush=True)

    cache.write_text(json.dumps(detections))
    rowcache.write_text(json.dumps(rows, indent=2))
    del model
    torch.cuda.empty_cache()
    return detections, rows


def summarize_models(models: Dict[str, str], coco: COCO, img_root, out_dir, device='cuda:0'):
    """Runs run_inference()+aggregate() for every (name -> checkpoint_path) pair.
    Returns (detections_by_name, overall_metrics_by_name)."""
    detections, overall = {}, {}
    for name, path in models.items():
        print('=== MODEL', name, path, '===', flush=True)
        dets, rows = run_inference(name, path, coco, img_root, out_dir, device=device)
        detections[name] = dets
        overall[name] = aggregate(rows)
        overall[name]['path'] = str(path)
        try:
            overall[name]['sha256'] = sha256_of(path)
        except OSError:
            pass
    return detections, overall
