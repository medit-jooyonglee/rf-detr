"""Correctness + speed comparisons around mask pasting
(``rfdetr/models/segmentation_head.py``), using the real call arguments
captured from one ``draw_preditions_boxes`` pass on the same 28-tooth
seoul_pano sample used by ``profile_two_pass.py``.

Two things are compared here:
  1. ``paste_masks_in_image`` (GPU, returns raw logits) vs the numba
     reimplementation ``paste_masks_in_image_by_numba`` -- kept only as a
     reference/comparison; NOT used in production (numba was slower: CPU
     round-trip cost > the GPU op's own ~3-6ms).
  2. The actual production path: old
     ``to_numpy(paste_masks_in_image(...) > threshold)`` (transfers the full
     (K, H, W) canvas, ~65-85ms for this sample) vs the current
     ``paste_masks_to_numpy(...)`` (transfers only each instance's own small
     box crop, canvas is built on CPU after transfer) -- this is what
     `rfdetr/engine.py` actually calls now.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/profile_paste_masks.py
"""
import time

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import resize_image
from rfdetr.engine import draw_preditions_boxes
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.models import segmentation_head
from trainer import image_utils, torch_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']
SAMPLE_FILE_SUFFIX = '000000_957466 pano 3.jpg'
MASK_LOGIT_THRESHOLD = 0.0  # math.log(0.5 / (1 - 0.5)) -- matches production default


def timeit(label, fn, n=20):
    for _ in range(3):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f'{label}: {(time.perf_counter() - t0) / n * 1000:.2f} ms')


def main():
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    param = next(model.parameters())
    convert_options = {'device': param.device, 'dtype': param.dtype}

    datasets = load_val_coco()
    file_name = None
    for source_name, coco in datasets:
        for im in coco.imgs.values():
            if im['file_name'].endswith(SAMPLE_FILE_SUFFIX):
                file_name = im['file_name']
        if file_name:
            break

    img = image_utils.cv2_imread(file_name, flags=cv2.IMREAD_GRAYSCALE)
    rsz_img, content_box = resize_image(img, stride=64)
    img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
    img_tensor = torch_utils.data_convert(img2[None], **convert_options)

    # Capture the exact (masks, boxes_xyxy, image_size, threshold) engine.py
    # passes to paste_masks_to_numpy (the function it actually calls now),
    # instead of hand-rolling synthetic boxes.
    captured = {}
    original_fn = segmentation_head.paste_masks_to_numpy

    def capture(masks, boxes_xyxy, image_size, mask_logit_threshold):
        captured['masks'] = masks.detach().clone()
        captured['boxes'] = boxes_xyxy.detach().clone()
        captured['image_size'] = image_size
        captured['threshold'] = mask_logit_threshold
        return original_fn(masks, boxes_xyxy, image_size, mask_logit_threshold)

    segmentation_head.paste_masks_to_numpy = capture
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            if not isinstance(outputs, dict):
                outputs = dict(zip(('pred_boxes', 'pred_logits', 'pred_masks'), outputs))
            draw_preditions_boxes(
                img_tensor, outputs, save=False, origin_size=img.shape[:2],
                segmentation_mode='crop_and_resize', mask_probability_threshold=0.5,
                segmentation_crop_box_scale=1.15, with_source_concat=False,
                paste_masks_at_original_size=True, input_content_box=content_box,
                drawing_mode='polygon', instance_results=[], fdi_refinement=True,
                confidence_threshold=0.5, render=False,
            )
    finally:
        segmentation_head.paste_masks_to_numpy = original_fn

    masks, boxes, image_size, threshold = (
        captured['masks'], captured['boxes'], captured['image_size'], captured['threshold'])
    print(f'captured: masks={tuple(masks.shape)} boxes={tuple(boxes.shape)} '
          f'image_size={image_size} threshold={threshold}')

    # --- 1. paste_masks_in_image vs its numba reimplementation (reference only) ---
    out_orig = segmentation_head.paste_masks_in_image(masks, boxes, image_size).cpu().float()
    out_numba = segmentation_head.paste_masks_in_image_by_numba(masks, boxes, image_size).float()
    diff = (out_orig - out_numba).abs()
    print(f'[1] logit max abs diff (orig vs numba): {diff.max().item():.6f}')
    mismatch = ((out_orig > 0.0) != (out_numba > 0.0)).sum().item()
    print(f'[1] boolean mask mismatch pixels (orig vs numba): {mismatch} / {out_orig.numel()}')
    timeit('[1] paste_masks_in_image (original, GPU)',
           lambda: segmentation_head.paste_masks_in_image(masks, boxes, image_size))
    timeit('[1] paste_masks_in_image_by_numba (warm, NOT used in production)',
           lambda: segmentation_head.paste_masks_in_image_by_numba(masks, boxes, image_size))

    # --- 2. old (full-canvas transfer) vs new (paste_masks_to_numpy) production path ---
    def old_path():
        pasted = segmentation_head.paste_masks_in_image(masks, boxes, image_size)
        return torch_utils.to_numpy(pasted > threshold)

    out_old = old_path()
    out_new = original_fn(masks, boxes, image_size, threshold)
    mismatch2 = (out_old != out_new).sum()
    print(f'[2] old-path vs paste_masks_to_numpy mismatch pixels: {mismatch2} / {out_old.size}  '
          f'dtype_match={out_old.dtype == out_new.dtype}')
    timeit('[2] OLD: paste_masks_in_image + threshold + to_numpy (full-canvas transfer)', old_path)
    timeit('[2] NEW: paste_masks_to_numpy (small-crop transfer, CPU-side scatter)',
           lambda: original_fn(masks, boxes, image_size, threshold))


if __name__ == '__main__':
    main()
