"""Evaluate rfdetr/deploy/predict.py's actual deployed inference pipeline --
init_and_get_model() + resize_image() + draw_preditions_boxes(), the exact
same calls predict.py::main() makes per file -- against COCO val GT, to
measure refine_fdi_labels()'s (fdi_refinement=True/False) effect on
duplicate/outlier FDI numbers ("중복 치아 추론").

예외처리 필요(참고만, 여기서는 다루지 않음): refine_fdi_labels()
(rfdetr/util/fdi_refinement.py)는 영구치 32개(arch당 16개) 전제로 동작한다.
유치(deciduous teeth)는 32개보다 훨씬 많은 별도 FDI 대역(51~85)을 쓰므로, 모델
클래스 공간에 유치가 추가되면 이 보정을 그대로 쓸 수 없고 별도 분기가 반드시
필요하다 -- 정의부(rfdetr/util/fdi_refinement.py)와 호출부
(rfdetr/engine.py::draw_preditions_boxes 내부) 양쪽에 동일한 주석을 남겨뒀다.

Compares, for each of {baseline_panoramic_overlapfix, peri_data_fusion_1106_ce}:
  - fdi_refinement=False (NMS 직후 raw 결과)
  - fdi_refinement=True  (refine_fdi_labels 적용)
on the full 1106 valid split of both xray_panoramic data sources
(cbct_ios_dcm_latest_1106/valid, seoul_univ_inspect_1106/valid).

outputs/peri_data_fusion ("without_ce")는 의도적으로 제외 -- 이미
outputs/compare_peri_fusion_1106_ce/ 에서 충분히 평가됨.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/evaluate.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import resize_image, MASK_PROBABILITY_THRESHOLD
from rfdetr.engine import draw_preditions_boxes
from rfdetr.datasets.xraypanoramic import label2fdi
from rfdetr.scripts.compare_core import ROOT, load_coco
from tools.compare_own1022 import metrics, aggregate
from trainer import image_utils, torch_utils

XRAY_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1106')
PERI_DATA = Path('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_1106')

MODELS = {
    'baseline_panoramic_overlapfix': str(ROOT / 'outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth'),
    'peri_data_fusion_1106_ce_ema': str(ROOT / 'outputs/peri_data_fusion_1106_ce/checkpoint_best_ema.pth'),
}
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement'
METRIC_KEYS = ('instance_dice', 'instance_iou', 'boundary_f1', 'contact_f1')


def load_val_coco():
    cbct = load_coco(XRAY_DATA / 'valid', XRAY_DATA / 'valid/annotations.json')
    seoul = load_coco(PERI_DATA / 'valid', PERI_DATA / 'valid/annotations.json')
    return [('cbct_ios_dcm_latest_1106', cbct), ('seoul_univ_inspect_1106', seoul)]


def count_duplicate_teeth(universal_labels):
    """같은 arch(상/하악) 안에 동일 universal label(=치아번호)이 2회 이상 검출된
    '중복 치아' 개수. (label 1~16 상악, 17~32 하악; 0/배경은 제외)"""
    if universal_labels.size == 0:
        return 0
    dup = 0
    for lo, hi in ((1, 16), (17, 32)):
        arch = universal_labels[(universal_labels >= lo) & (universal_labels <= hi)]
        if arch.size == 0:
            continue
        _, counts = np.unique(arch, return_counts=True)
        dup += int(np.clip(counts - 1, 0, None).sum())
    return dup


def run_model(name, checkpoint_path, datasets, device='cuda'):
    # init_and_get_model()은 모듈 전역(g_rfdetr_model)에 모델을 캐시해서, pretrain_weights가
    # 달라도 두 번째 호출은 첫 모델을 그대로 반환한다 -- 체크포인트를 바꿀 때마다 리셋 필요.
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=checkpoint_path, device=device)

    rows = {False: [], True: []}
    dup_counts = {False: 0, True: 0}
    n_images = 0
    for source_name, coco in datasets:
        images = list(coco.imgs.values())
        for im in images:
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            # 아래 5줄은 predict.py::main()의 추론 루프와 동일한 전처리/forward 호출이다.
            rsz_img, content_box = resize_image(img, stride=64)
            img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
            img_tensor = torch_utils.data_convert(img2[None])
            with torch.no_grad():
                outputs = model(img_tensor)
                if not isinstance(outputs, dict):
                    outputs = dict(zip(('pred_boxes', 'pred_logits', 'pred_masks'), outputs))

            gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])
            for refine in (False, True):
                instance_results = []
                draw_preditions_boxes(
                    img_tensor, outputs, save=False, origin_size=img.shape[:2],
                    segmentation_mode='crop_and_resize',
                    mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                    segmentation_crop_box_scale=1.15,
                    paste_masks_at_original_size=True,
                    input_content_box=content_box,
                    drawing_mode='polygon',
                    instance_results=instance_results,
                    fdi_refinement=refine,
                )
                r = instance_results[0]
                universal_labels = label2fdi[r['fdi']]
                dup_counts[refine] += count_duplicate_teeth(universal_labels)
                row = metrics(gt, r['masks'], universal_labels, r['scores'], r['boxes'], anns, h, w)
                row['file_name'] = im['file_name']
                rows[refine].append(row)
            n_images += 1
            if n_images % 20 == 0:
                print(name, source_name, n_images, flush=True)

    overall = {refine: aggregate(rows[refine]) for refine in (False, True)}
    return overall, dup_counts, n_images


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_val_coco()

    summary = {}
    for name, ckpt in MODELS.items():
        print(f'\n=== {name} ({ckpt}) ===')
        overall, dup_counts, n_images = run_model(name, ckpt, datasets)
        print(f'  images evaluated: {n_images}')
        for refine in (False, True):
            m = overall[refine]
            print(f"  fdi_refinement={refine!s:<5}  dice={m['instance_dice']:.4f}  iou={m['instance_iou']:.4f}  "
                  f"boundary_f1={m['boundary_f1']:.4f}  contact_f1={m['contact_f1']:.4f}  "
                  f"duplicate_teeth={dup_counts[refine]}")
        summary[name] = dict(
            checkpoint=ckpt,
            n_images=n_images,
            no_refinement=dict({k: overall[False][k] for k in METRIC_KEYS}, duplicate_teeth=dup_counts[False]),
            with_refinement=dict({k: overall[True][k] for k in METRIC_KEYS}, duplicate_teeth=dup_counts[True]),
        )

    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nsaved summary:', out_path)


if __name__ == '__main__':
    main()
