"""two-pass(crop+재추론)이 one-pass(전체 이미지 640-longside resize) 대비 실제로 치아당
유효 해상도를 얼마나 올려주는지(scale_gain) pano/peri 카테고리별로 통계 낸다.

scale1 = 원본 전체 이미지를 resize_image()(긴 변->640, stride64)로 리사이즈할 때 스케일
         (= content_box 너비 / 원본 너비 -- letterbox 패딩 제외, 실제 내용물 스케일 기준)
scale2 = teeth_roi_box_center_aware() crop 영역을 같은 resize_image()로 리사이즈할 때 스케일
scale_gain = scale2 / scale1  (1에 가까우면 crop해도 유효 해상도 이득이 없다는 뜻 -> one-pass와
             동일하게 둬도 됨. pano처럼 크게 축소됐던 원본을 crop하면 이 값이 커진다.)

기존 roi_shrink_ratio 기준(crop_area / 원본_area)과 나란히 출력해서 두 기준의 분별력을 비교.
이 결과로 ``two_pass_predict``의 ``min_scale_gain`` 기본값(1.3)을 정했다 -- 아래 실측(2026-11-10,
peri_data_fusion_1106_ce_ema 체크포인트)에서 cbct_pano/seoul_pano의 scale_gain 최소값이
1.315/1.518로 1.3 이상인 반면, cbct_peri/seoul_peri는 중앙값이 1.23/1.21로 뚜렷이 낮아
(최댓값만 1.3~2.0대로 일부 겹침) 1.3을 기준으로 두면 거의 모든 pano는 통과하고 대부분의
peri는 자동으로 one-pass로 떨어진다:

    cbct_pano   n=42  scale_gain mean=1.851 median=1.855 min=1.315 max=2.387
    cbct_peri   n=37  scale_gain mean=1.239 median=1.227 min=1.000 max=1.965
    seoul_pano  n=40  scale_gain mean=1.841 median=1.806 min=1.518 max=2.658
    seoul_peri  n=53  scale_gain mean=1.212 median=1.213 min=1.000 max=1.604

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/scale_gain_stats.py
"""
import cv2
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import _predict_single_pass, resize_image
from rfdetr.datasets.xray_crop_aug import teeth_roi_box_center_aware
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.deploy.evaluate_by_type import categorize
from trainer import image_utils

MODEL_CKPT = MODELS['peri_data_fusion_1106_ce_ema']


def main():
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    param = next(model.parameters())
    convert_options = {'device': param.device, 'dtype': param.dtype}

    datasets = load_val_coco()
    rows = {}
    n = 0
    for source_name, coco in datasets:
        for im in coco.imgs.values():
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            cat = categorize(source_name, im['file_name'], im['width'], im['height'])
            img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]

            _, _, inst1 = _predict_single_pass(img, model, convert_options, drawing_mode='polygon')
            if inst1['boxes'].shape[0] == 0:
                continue
            crop_box = teeth_roi_box_center_aware(inst1['boxes'], inst1['fdi'], (h, w))
            if crop_box is None:
                continue
            x0, y0, x1, y1 = crop_box
            x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
            crop_w, crop_h = x1i - x0i, y1i - y0i
            if crop_w <= 0 or crop_h <= 0:
                continue

            _, content_box_full = resize_image(img, stride=64)
            scale1 = (content_box_full[2] - content_box_full[0]) / w

            crop_img = img[y0i:y1i, x0i:x1i]
            _, content_box_crop = resize_image(crop_img, stride=64)
            scale2 = (content_box_crop[2] - content_box_crop[0]) / crop_w

            area_ratio = (crop_w * crop_h) / (w * h)
            rows.setdefault(cat, []).append((scale1, scale2, scale2 / scale1, area_ratio))

            n += 1
            if n % 20 == 0:
                print(n, flush=True)

    print()
    for cat in sorted(rows):
        arr = np.array(rows[cat])
        print(f"{cat}  n={len(arr)}")
        print(f"  scale1(1-pass, full->640)      mean={arr[:,0].mean():.3f}  median={np.median(arr[:,0]):.3f}")
        print(f"  scale2(2-pass, crop->640)      mean={arr[:,1].mean():.3f}  median={np.median(arr[:,1]):.3f}")
        print(f"  scale_gain(scale2/scale1)      mean={arr[:,2].mean():.3f}  median={np.median(arr[:,2]):.3f}  "
              f"min={arr[:,2].min():.3f}  max={arr[:,2].max():.3f}")
        print(f"  area_ratio(crop_area/orig_area) mean={arr[:,3].mean():.3f}  median={np.median(arr[:,3]):.3f}")


if __name__ == '__main__':
    main()
