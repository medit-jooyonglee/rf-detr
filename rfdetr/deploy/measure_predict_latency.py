"""Measures predict_wrapper() end-to-end latency + peak VRAM for the two
reference samples in docs/performance.md's "측정 환경 / 추론 속도 / VRAM" table:
pano (two-pass, gating passes) and peri (gating auto-falls-back to one-pass).
Same methodology: 1 warm-up + 5-iteration average, torch.cuda.synchronize(),
reset_peak_memory_stats()/max_memory_allocated().

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/measure_predict_latency.py
"""
import time

import cv2
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import predict_wrapper
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from trainer import image_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']
# (label, file suffix, two_pass override -- None means production default/gating)
CASES = [
    ('pano, one-pass (two_pass=False, forced)', '000000_957466 pano 3.jpg', False),
    ('pano, two-pass (two_pass=True, gating passes)', '000000_957466 pano 3.jpg', True),
    ('peri, one-pass (two_pass=True, gating auto-falls-back)', '000008_242419 peri 1-2.jpg', True),
]


def find_file(datasets, suffix):
    for source_name, coco in datasets:
        for im in coco.imgs.values():
            if im['file_name'].endswith(suffix):
                return im['file_name']
    return None


def measure(label, img, model, two_pass, n=5):
    kwargs = dict(return_two_pass_info=True, return_instance_results=True, two_pass=two_pass)
    predict_wrapper(img, model=model, **kwargs)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    info = None
    for _ in range(n):
        *_, info = predict_wrapper(img, model=model, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n * 1000
    peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f'{label}: {elapsed_ms:.1f} ms  peak_vram={peak_mib:.0f} MiB  '
          f'used_two_pass={info.get("used_two_pass")}')


def main():
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    datasets = load_val_coco()
    cache = {}
    for label, suffix, two_pass in CASES:
        if suffix not in cache:
            file_name = find_file(datasets, suffix)
            cache[suffix] = image_utils.cv2_imread(file_name, flags=cv2.IMREAD_GRAYSCALE)
            print(f'  file={file_name} shape={cache[suffix].shape}')
        measure(label, cache[suffix], model, two_pass=two_pass)


if __name__ == '__main__':
    main()
