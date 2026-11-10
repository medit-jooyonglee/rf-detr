"""최종(two-pass 기본 활성, scale_gain 게이팅 적용) predict_wrapper()로 pano 5장 + peri 5장
대표 결과 이미지를 4분할(raw | GT | one-pass | two_pass_predict 결과)로 저장한다.

peri는 게이팅상 대부분 2차가 생략돼 '2-pass' 패널도 실질적으로 one-pass와 동일하게 나오는 게
정상이다(그 자체가 이번 결론의 확인 포인트).

색상은 ``trainer.vtk_utils.get_teeth_color_table(normalize=False)``를 그대로 사용 -- FDI
첫째자리(quadrant)에 따라 밝기가 갈리고(1,4분면은 기본색, 2,3분면은 어둡게) 같은 분면 내
위치(1~8)는 색상(hue)으로 구분되는 기존 체계를 그대로 반영한다(수정 아님, 재사용).

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/render_final_samples.py
"""
from pathlib import Path

import cv2
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import _predict_single_pass, two_pass_predict
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import universal_to_fdi
from trainer import image_utils, vtk_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement/final_result_samples'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]
N_PER_CAT = 5

COLORS = vtk_utils.get_teeth_color_table(normalize=False)
COLORS[0, :] = 0


def color(fdi):
    return tuple(int(c) for c in COLORS[fdi % len(COLORS)])


def panel(gray, fdi, masks, title):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(len(fdi)):
        f = int(fdi[i])
        c = color(f)
        m = masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, c, 4)
        M = cv2.moments(m)
        if M['m00'] > 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            cv2.putText(img, str(f), (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_val_coco()
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    # cbct/seoul 소스가 섞이도록 카테고리별로 N_PER_CAT 중 절반씩 배정 (cbct가 먼저 순회되면
    # 전부 cbct로만 채워지는 걸 방지 -- 이번 조사에서 다룬 seoul_pano 대표 파일 우선 포함).
    PRIORITY_FILES = ('000004_993944 pano 1.jpg', '000006_158114 pano 2.jpg', '000000_957466 pano 3.jpg')
    quota = {
        'cbct_pano': (N_PER_CAT + 1) // 2, 'seoul_pano': N_PER_CAT // 2,
        'cbct_peri': (N_PER_CAT + 1) // 2, 'seoul_peri': N_PER_CAT // 2,
    }
    by_cat = {}
    for source_name, coco in datasets:
        for im in coco.imgs.values():
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            cat = categorize(source_name, im['file_name'], im['width'], im['height'])
            by_cat.setdefault(cat, []).append((im, coco))

    picks = {'pano': [], 'peri': []}
    for cat, cap in quota.items():
        items = by_cat.get(cat, [])
        if cat == 'seoul_pano':
            items = sorted(items, key=lambda x: not any(x[0]['file_name'].endswith(p) for p in PRIORITY_FILES))
        group = 'peri' if cat.endswith('_peri') else 'pano'
        for im, coco in items[:cap]:
            picks[group].append((cat, im, coco))

    for group, items in picks.items():
        for idx, (cat, im, coco) in enumerate(items):
            anns = coco.imgToAnns[im['id']]
            img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
            gt_masks = np.stack([coco.annToMask(a).astype(bool) for a in anns])
            gt_fdi = universal_to_fdi[np.array([a['category_id'] for a in anns])]

            _, _, inst1 = _predict_single_pass(img, model, convert_options)
            _, _, inst2, used2, _ = two_pass_predict(img, model, convert_options=convert_options)

            p_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(p_raw, 'raw', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            p_gt = panel(img, gt_fdi, gt_masks, 'GT')
            p_one = panel(img, inst1['fdi'], inst1['masks'], 'one-pass')
            p_two = panel(img, inst2['fdi'], inst2['masks'], f'predict_wrapper default (two_pass={used2})')

            h = img.shape[0]
            scale = 480 / h
            tiles = [cv2.resize(p, None, fx=scale, fy=scale) for p in (p_raw, p_gt, p_one, p_two)]
            combined = np.concatenate(tiles, axis=1)
            out_path = OUT_DIR / f"{group}_{idx}_{cat}.jpg"
            # vtk_utils 색상 테이블은 RGB 순서(rfdetr/datasets/visualize.py의 관례와 동일) --
            # cv2.imwrite는 BGR로 저장하므로 끝에서 한 번만 채널을 뒤집는다.
            cv2.imwrite(str(out_path), combined[..., ::-1])
            print(f'saved: {out_path}  cat={cat} used_two_pass={used2} file={im["file_name"]}')


if __name__ == '__main__':
    main()
