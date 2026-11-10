"""Correctness + speed comparison of the (uint8-dtype-fixed) numpy
``label_image`` merge vs the numba box-restricted replacement
``merge_label_image_by_numba`` (``rfdetr/datasets/visualize.py``), using the
real render_detection() call args captured from one forward pass on the same
28-tooth seoul_pano sample used by profile_render_bottleneck.py.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/profile_merge_label_image.py
"""
import time

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import resize_image
from rfdetr.engine import draw_preditions_boxes
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.datasets import visualize as visualize_mod
from trainer import image_utils, torch_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']
SAMPLE_FILE_SUFFIX = '000000_957466 pano 3.jpg'


def timeit(label, fn, n=20):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    print(f'{label}: {(time.perf_counter() - t0) / n * 1000:.3f} ms')


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

    captured = {}
    original_render_detection = visualize_mod.render_detection

    def capture(image, boxes_xy, labels, colors, **kwargs):
        if 'masks' not in captured:
            captured['boxes_xy'] = boxes_xy.copy()
            captured['labels'] = np.asarray(labels).copy()
            captured['masks'] = kwargs.get('masks').copy() if kwargs.get('masks') is not None else None
            mask_boxes_xy = kwargs.get('mask_boxes_xy')
            captured['mask_boxes_xy'] = mask_boxes_xy.copy() if mask_boxes_xy is not None else None
            captured['image_shape'] = image.shape[:2]
        return original_render_detection(image, boxes_xy, labels, colors, **kwargs)

    visualize_mod.render_detection = capture
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
                confidence_threshold=0.5, render=True,
            )
    finally:
        visualize_mod.render_detection = original_render_detection

    boxes_xy = captured['boxes_xy']
    labels = captured['labels']
    masks = captured['masks']
    mask_boxes_xy = captured['mask_boxes_xy']
    image_shape = captured['image_shape']
    print(f'captured: boxes={boxes_xy.shape} mask_boxes={None if mask_boxes_xy is None else mask_boxes_xy.shape} '
          f'labels={labels.shape} masks={masks.shape} ({masks.dtype}) image_shape={image_shape}')

    def numpy_fixed():
        return np.max(
            labels.astype(np.uint8)[:, None, None] * masks, axis=0
        ).astype(labels.dtype)

    def numba_box_restricted():
        return visualize_mod.merge_label_image_by_numba(labels, masks, mask_boxes_xy, image_shape)

    ref = numpy_fixed()
    out = numba_box_restricted()
    print(f'correctness: max abs diff = {np.abs(ref.astype(np.int64) - out.astype(np.int64)).max()}  '
          f'dtype match = {ref.dtype == out.dtype}')

    timeit('label_image merge (numpy, uint8-fixed, full-canvas scan)', numpy_fixed)
    timeit('label_image merge (numba, box-restricted)', numba_box_restricted)


if __name__ == '__main__':
    main()
