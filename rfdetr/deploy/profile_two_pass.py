"""two-pass(``two_pass_predict``)가 느린 원인이 모델 forward인지 ``draw_preditions_boxes``
후처리(원본 해상도 mask paste + polygon contour)인지 분리 측정한다. seoul_pano 샘플(치아 28개)
실측 결과(2026-11-10, peri_data_fusion_1106_ce_ema 체크포인트):

    model forward only                                            :  35.4 ms
    draw_preditions_boxes (full, polygon+fdi_refine+paste@orig_size): 496.3 ms
    draw_preditions_boxes (fdi_refinement=False)                    : 484.1 ms  (DP 재배열은 범인 아님)
    draw_preditions_boxes (render=False)                            : 178.4 ms  (그리기만으로 ~318ms)

이 결과로 ``two_pass_predict``가 1차를 ``render=False``로 먼저 돌려 2차 진행 여부만 가볍게
판단하고, 2차가 실제로 진행되면(pano) 1차 전체 렌더링을 생략하도록 바꿨다(`rfdetr/deploy/
predict.py`) -- pano 두-pass 평균 지연 1069.7ms -> 650.0ms (`docs/performance.md` 참고).

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/profile_two_pass.py
"""
import time

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import resize_image
from rfdetr.engine import draw_preditions_boxes
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from trainer import image_utils, torch_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']
SAMPLE_FILE_SUFFIX = '000000_957466 pano 3.jpg'


def timeit(label, fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f'{label}: {(time.perf_counter() - t0) / n * 1000:.1f} ms')


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

    with torch.no_grad():
        timeit('model forward only', lambda: model(img_tensor))

        outputs = model(img_tensor)
        if not isinstance(outputs, dict):
            outputs = dict(zip(('pred_boxes', 'pred_logits', 'pred_masks'), outputs))

        def draw(fdi_refinement=True, render=True):
            draw_preditions_boxes(
                img_tensor, outputs, save=False, origin_size=img.shape[:2],
                segmentation_mode='crop_and_resize', mask_probability_threshold=0.5,
                segmentation_crop_box_scale=1.15, with_source_concat=False,
                paste_masks_at_original_size=True, input_content_box=content_box,
                drawing_mode='polygon', instance_results=[], fdi_refinement=fdi_refinement,
                confidence_threshold=0.5, render=render,
            )
        timeit('draw_preditions_boxes (full, polygon+fdi_refine+paste@orig_size)', draw)
        timeit('draw_preditions_boxes (fdi_refinement=False)', lambda: draw(fdi_refinement=False))
        timeit('draw_preditions_boxes (render=False)', lambda: draw(render=False))


if __name__ == '__main__':
    main()
