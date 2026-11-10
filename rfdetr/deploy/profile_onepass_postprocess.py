"""One-pass draw_preditions_boxes() broken down by named post-processing
module (NMS, fdi_refinement, mask paste, base-image conversion,
render_detection), on the same 28-tooth seoul_pano sample. Unlike
profile_render_bottleneck.py (which only covered render_detection's internals),
this also covers the NMS/fdi_refinement/mask-paste stage that happens before
render_detection -- the ~184ms "render=False" baseline from profile_two_pass.py
that was never broken down further.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/profile_onepass_postprocess.py
"""
import time
from collections import defaultdict
from functools import wraps

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import resize_image
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from trainer import image_utils, torch_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']
SAMPLE_FILE_SUFFIX = '000000_957466 pano 3.jpg'

TIMES = defaultdict(float)
COUNTS = defaultdict(int)


def tracked(label, fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        TIMES[label] += (time.perf_counter() - t0) * 1000
        COUNTS[label] += 1
        return out
    return wrapper


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

    # Patch the named module boundaries engine.py calls into. These are read
    # from the module attribute at call time (plain `import`/`from X import Y`
    # inside draw_preditions_boxes' body executes fresh each call), so
    # patching here takes effect without touching engine.py.
    import rfdetr.engine as engine_mod
    import rfdetr.util.fdi_refinement as fdi_mod
    import rfdetr.models.segmentation_head as seg_mod
    import rfdetr.datasets.visualize as viz_mod

    originals = {
        'non_max_suppression': engine_mod.non_max_suppression,
        'refine_fdi_labels': fdi_mod.refine_fdi_labels,
        'paste_masks_in_image': seg_mod.paste_masks_in_image,
        'to_magnitude_images': image_utils.to_magnitude_images,
        'render_detection': viz_mod.render_detection,
        'to_numpy': torch_utils.to_numpy,
        'cv2_resize': cv2.resize,
    }
    engine_mod.non_max_suppression = tracked('1. non_max_suppression (NMS)', originals['non_max_suppression'])
    fdi_mod.refine_fdi_labels = tracked('2. refine_fdi_labels (fdi_refinement)', originals['refine_fdi_labels'])
    seg_mod.paste_masks_in_image = tracked('3. paste_masks_in_image', originals['paste_masks_in_image'])
    image_utils.to_magnitude_images = tracked('4. to_magnitude_images (base image conv.)', originals['to_magnitude_images'])
    viz_mod.render_detection = tracked('5. render_detection (draw+contours+labels)', originals['render_detection'])

    # engine.py does `from trainer import torch_utils` / `import cv2` INSIDE
    # draw_preditions_boxes' body (executed fresh each call), but that just
    # binds a local name to the SAME shared module object -- patching the
    # module's own attribute here (trainer.torch_utils.to_numpy, cv2.resize)
    # affects it too. Both are called many times with small tensors (NMS
    # inputs, logits, etc.), so label each call by its own shape instead of
    # a single bucket, to spot the one big call (mask threshold tensor,
    # base-image upscale) instead of averaging it away with the tiny ones.
    orig_to_numpy = originals['to_numpy']

    def to_numpy_dispatch(t, *a, **kw):
        label = f'X. torch_utils.to_numpy (shape={tuple(t.shape)}, dtype={t.dtype})'
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_to_numpy(t, *a, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        TIMES[label] += (time.perf_counter() - t0) * 1000
        COUNTS[label] += 1
        return out
    torch_utils.to_numpy = to_numpy_dispatch

    orig_cv2_resize = originals['cv2_resize']

    def cv2_resize_dispatch(src, dsize, *a, **kw):
        label = f'X. cv2.resize (src={src.shape} -> {dsize})'
        t0 = time.perf_counter()
        out = orig_cv2_resize(src, dsize, *a, **kw)
        TIMES[label] += (time.perf_counter() - t0) * 1000
        COUNTS[label] += 1
        return out
    cv2.resize = cv2_resize_dispatch

    # restored_instance_masks construction (engine.py:794-798) is inline code
    # (np.stack over a per-instance np.ascontiguousarray list comprehension),
    # not a function call we can patch at its own name -- patch np.stack/
    # np.ascontiguousarray themselves for the duration of this one profiled
    # call (safe here: draw_preditions_boxes is only invoked once per run).
    orig_np_stack = np.stack
    orig_np_ascontig = np.ascontiguousarray

    def np_stack_dispatch(arrays, *a, **kw):
        arrays = list(arrays)
        label = f'X. np.stack (n={len(arrays)}, shape={arrays[0].shape if arrays else None})'
        t0 = time.perf_counter()
        out = orig_np_stack(arrays, *a, **kw)
        TIMES[label] += (time.perf_counter() - t0) * 1000
        COUNTS[label] += 1
        return out

    def np_ascontig_dispatch(a, *args, **kw):
        label = f'X. np.ascontiguousarray (shape={getattr(a, "shape", None)})'
        t0 = time.perf_counter()
        out = orig_np_ascontig(a, *args, **kw)
        TIMES[label] += (time.perf_counter() - t0) * 1000
        COUNTS[label] += 1
        return out
    np.stack = np_stack_dispatch
    np.ascontiguousarray = np_ascontig_dispatch

    from rfdetr.engine import draw_preditions_boxes

    def run_once():
        TIMES.clear()
        COUNTS.clear()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(img_tensor)
            if not isinstance(outputs, dict):
                outputs = dict(zip(('pred_boxes', 'pred_logits', 'pred_masks'), outputs))
            torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            draw_preditions_boxes(
                img_tensor, outputs, save=False, origin_size=img.shape[:2],
                segmentation_mode='crop_and_resize', mask_probability_threshold=0.5,
                segmentation_crop_box_scale=1.15, with_source_concat=False,
                paste_masks_at_original_size=True, input_content_box=content_box,
                drawing_mode='polygon', instance_results=[], fdi_refinement=True,
                confidence_threshold=0.5, render=True,
            )
        t_end = time.perf_counter()
        return (t_fwd - t0) * 1000, (t_end - t_fwd) * 1000

    # warm-up (cudnn autotune)
    run_once()
    fwd_ms, draw_ms = run_once()

    named_total = sum(TIMES.values())
    print(f'\n  model forward                                         {fwd_ms:8.2f} ms')
    for label in sorted(TIMES):
        print(f'  {label:<55s} {TIMES[label]:8.2f} ms  (x{COUNTS[label]})')
    residual = draw_ms - named_total
    print(f'  {"residual (mask threshold+transfer, restored_instance_masks stack, loop overhead, etc.)":<55s} {residual:8.2f} ms')
    print(f'  {"draw_preditions_boxes TOTAL":<55s} {draw_ms:8.2f} ms')
    print(f'  {"forward + draw TOTAL":<55s} {fwd_ms + draw_ms:8.2f} ms')


if __name__ == '__main__':
    main()
