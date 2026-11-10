"""Stage-by-stage timing of two_pass_predict() end-to-end (519.7ms total per
measure_predict_latency.py) on the same 28-tooth seoul_pano sample. Replicates
two_pass_predict's exact control flow (rfdetr/deploy/predict.py) with timers
around each stage, instead of guessing from the earlier per-function profiles.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/profile_two_pass_stages.py
"""
import time

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import (
    _predict_single_pass, resize_image, teeth_roi_box_center_aware,
    DEFAULT_TWO_PASS_ROI_MARGIN, DEFAULT_TWO_PASS_SHRINK_RATIO,
    DEFAULT_TWO_PASS_MIN_SCALE_GAIN, DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD,
    MASK_PROBABILITY_THRESHOLD,
)
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from trainer import image_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']
SAMPLE_FILE_SUFFIX = '000000_957466 pano 3.jpg'


class Stopwatch:
    def __init__(self):
        self.rows = []

    def lap(self, label, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        self.rows.append((label, dt))
        return out

    def report(self):
        total = sum(dt for _, dt in self.rows)
        print()
        for label, dt in self.rows:
            print(f'  {label:<55s} {dt:8.2f} ms  ({dt / total * 100:5.1f}%)')
        print(f'  {"TOTAL":<55s} {total:8.2f} ms')


def run_two_pass_instrumented(img, model, convert_options, sw):
    h, w = img.shape[:2]
    drawing_mode = 'polygon'
    roi_margin = DEFAULT_TWO_PASS_ROI_MARGIN
    roi_shrink_ratio = DEFAULT_TWO_PASS_SHRINK_RATIO
    min_scale_gain = DEFAULT_TWO_PASS_MIN_SCALE_GAIN
    pass2_confidence_threshold = DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD

    def pass1_kwargs(render):
        return dict(
            drawing_mode=drawing_mode, mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
            segmentation_crop_box_scale=1.15, with_source_concat=False, fdi_refinement=True,
            confidence_threshold=0.5, render=render)

    _, _, inst1 = sw.lap(
        '1. pass1 forward+postprocess (render=False)',
        lambda: _predict_single_pass(img, model, convert_options, **pass1_kwargs(render=False)),
    )

    crop_box = sw.lap(
        '2. ROI calc (teeth_roi_box_center_aware)',
        lambda: teeth_roi_box_center_aware(inst1['boxes'], inst1['fdi'], (h, w),
                                            cfg={'eval_margin': roi_margin}),
    )
    x0, y0, x1, y1 = crop_box
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    crop_w, crop_h = x1i - x0i, y1i - y0i
    print(f'  crop region: ({x0i},{y0i},{x1i},{y1i}) size={crop_w}x{crop_h} '
          f'vs full {w}x{h} (area ratio {crop_w*crop_h/(w*h):.2f})')

    def scale_gain_calc():
        crop_img = img[y0i:y1i, x0i:x1i]
        _, content_box_full = resize_image(img, stride=64)
        scale1 = (content_box_full[2] - content_box_full[0]) / w
        _, content_box_crop = resize_image(crop_img, stride=64)
        scale2 = (content_box_crop[2] - content_box_crop[0]) / crop_w
        return crop_img, scale2 / scale1

    crop_img, gain = sw.lap('3. scale_gain calc (2x resize_image calls)', scale_gain_calc)
    print(f'  scale_gain={gain:.3f} (>= {min_scale_gain} -> two-pass proceeds)')

    full_thickness = 2 * max(1, int(round(w / 1000)))
    full_font_scale = max(0.5, w / 1600.0)
    color2_crop, mask2_crop, inst2 = sw.lap(
        '4. pass2 forward+postprocess (render=True, ON CROP -- now reused, not discarded)',
        lambda: _predict_single_pass(
            crop_img, model, convert_options, drawing_mode=drawing_mode,
            mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
            segmentation_crop_box_scale=1.15, with_source_concat=False, fdi_refinement=True,
            confidence_threshold=pass2_confidence_threshold,
            polygon_thickness=full_thickness, label_font_scale=full_font_scale),
    )

    def offset_and_paste():
        offset = np.array([x0i, y0i, x0i, y0i], dtype=inst2['boxes'].dtype)
        inst2['boxes'] = inst2['boxes'] + offset
        full_masks = np.zeros((inst2['masks'].shape[0], h, w), dtype=bool)
        full_masks[:, y0i:y1i, x0i:x1i] = inst2['masks']
        inst2['masks'] = full_masks
        return inst2

    inst2 = sw.lap('5. offset translate + full-canvas mask zero-init/paste', offset_and_paste)

    def composite():
        raw_image = image_utils.to_magnitude_images(img)
        display_image = raw_image.copy()
        display_image[y0i:y1i, x0i:x1i] = color2_crop
        mask2 = np.zeros((h, w), dtype=mask2_crop.dtype)
        mask2[y0i:y1i, x0i:x1i] = mask2_crop
        return display_image, mask2

    sw.lap(
        '6. composite: to_magnitude_images(img) + paste already-rendered crop patch '
        '(replaces the old full-canvas render_detection re-render)',
        composite,
    )


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
    print(f'sample: {file_name} shape={img.shape}')

    # warm-up (cudnn autotune + numba jit)
    sw_warm = Stopwatch()
    run_two_pass_instrumented(img, model, convert_options, sw_warm)

    sw = Stopwatch()
    run_two_pass_instrumented(img, model, convert_options, sw)
    sw.report()


if __name__ == '__main__':
    main()
