"""Break down draw_preditions_boxes' post-processing cost, since
profile_paste_masks.py showed paste_masks_in_image itself is only ~2.8ms (GPU),
not the ~136ms gap seen earlier -- that gap must be elsewhere. This times
each candidate CPU-side step with the REAL captured render_detection() call
args (image, boxes_xy, labels, colors, masks) from one forward pass on the
same 28-tooth seoul_pano sample.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python /tmp/claude-0/-workspace-repo-rf-detr/6df29afc-dd07-463d-950c-0057a0d16acb/scratchpad/profile_render_bottleneck.py
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


def timeit(label, fn, n=10):
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

    # Capture render_detection's real call args (same real masks/boxes/labels
    # draw_preditions_boxes would pass in production).
    captured = {}
    original_render_detection = visualize_mod.render_detection

    def capture(image, boxes_xy, labels, colors, **kwargs):
        if 'masks' not in captured:
            captured['image'] = image.copy()
            captured['boxes_xy'] = boxes_xy.copy()
            captured['labels'] = np.asarray(labels).copy()
            captured['colors'] = colors.copy()
            captured['masks'] = kwargs.get('masks').copy() if kwargs.get('masks') is not None else None
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

    image = captured['image']
    boxes_xy = captured['boxes_xy']
    labels = captured['labels']
    colors = captured['colors']
    masks = captured['masks']
    print(f'captured: image={image.shape} boxes={boxes_xy.shape} labels={labels.shape} '
          f'masks={None if masks is None else masks.shape} ({masks.dtype})')

    # 1) full render_detection, polygon mode (what two_pass/predict.py actually uses)
    timeit('render_detection (full, polygon)',
           lambda: original_render_detection(image, boxes_xy, labels, colors, masks=masks,
                                              drawing_mode='polygon'))
    # 2) full render_detection, mask mode
    timeit('render_detection (full, mask)',
           lambda: original_render_detection(image, boxes_xy, labels, colors, masks=masks,
                                              drawing_mode='mask'))
    # 3) render_detection with masks=None -> isolates draw_bboxes + FDI labels only
    timeit('render_detection (masks=None: draw_bboxes+labels only)',
           lambda: original_render_detection(image, boxes_xy, labels, colors, masks=None))
    # 4) isolate label_image merge: original int64 formula vs the uint8-cast fix
    #    (rfdetr/datasets/visualize.py), plus a correctness check between them.
    ref = np.max(labels[:, None, None] * masks, axis=0)
    fixed = np.max(labels.astype(np.uint8)[:, None, None] * masks, axis=0).astype(labels.dtype)
    print(f'label_image fix correctness: max abs diff = {np.abs(ref.astype(np.int64) - fixed.astype(np.int64)).max()}')
    timeit('label_image merge (ORIGINAL int64 formula)',
           lambda: np.max(labels[:, None, None] * masks, axis=0))
    timeit('label_image merge (FIXED uint8 formula)',
           lambda: np.max(labels.astype(np.uint8)[:, None, None] * masks, axis=0).astype(labels.dtype))
    # 5) isolate findContours + polylines loop (polygon mode's extra cost over label_image)
    thickness = 2 * max(1, int(round(image.shape[1] / 1000)))
    canvas = image.copy()

    def contours_and_polylines():
        c = canvas.copy()
        for instance_mask, instance_label in zip(masks, labels):
            contours, _ = cv2.findContours(
                np.ascontiguousarray(instance_mask, dtype=np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            color = tuple(int(x) for x in colors[instance_label])
            cv2.polylines(c, contours, True, color, thickness, lineType=cv2.LINE_AA)
        return c
    timeit('findContours+polylines loop only (28 instances, full-res)', contours_and_polylines)

    # 6) isolate just findContours (no polylines)
    def contours_only():
        out = []
        for instance_mask in masks:
            contours, _ = cv2.findContours(
                np.ascontiguousarray(instance_mask, dtype=np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            out.append(contours)
        return out
    timeit('findContours only (28 instances, full-res)', contours_only)

    # 7) isolate apply_blending_mask (mask mode's extra cost)
    from trainer import utils_numpy
    label_image = np.max(labels[:, None, None] * masks, axis=0)
    color_label_image = colors[label_image]
    timeit('apply_blending_mask only (mask mode)',
           lambda: utils_numpy.apply_blending_mask(image.copy(), color_label_image))

    # 8) isolate the restored_instance_masks stack (engine.py:794-798, already-restored case)
    posit_masks_u8 = [np.ascontiguousarray(m, dtype=np.uint8) for m in masks]
    timeit('restored_instance_masks stack+astype(bool) (engine.py already-restored case)',
           lambda: np.stack(posit_masks_u8).astype(bool))

    # 9) isolate FDI label putText loop
    from rfdetr.datasets.visualize import _draw_fdi_labels
    timeit('_draw_fdi_labels only',
           lambda: _draw_fdi_labels(image.copy(), boxes_xy, labels))


if __name__ == '__main__':
    main()
