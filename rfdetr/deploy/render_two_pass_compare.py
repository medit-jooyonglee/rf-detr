"""현재(center-aware ROI + pass1/pass2 threshold 분리) two_pass_predict() 기준으로
raw | GT | one-pass | two-pass 4분할 비교 이미지를 다시 그린다. 이전(naive crop + threshold 0.5)
버전의 결과(outputs/eval_fdi_refinement/two_pass_debug/seoul_pano_{0,3,4,5,6,7}.jpg)와 비교해
육안 검증하기 위한 용도 -- 파일명에 _v2를 붙여 이전 결과를 덮어쓰지 않는다.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/render_two_pass_compare.py
"""
from pathlib import Path

import cv2
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import _predict_single_pass, two_pass_predict
from rfdetr.deploy.evaluate import MODELS, load_val_coco
from rfdetr.datasets.xraypanoramic import universal_to_fdi
from trainer import image_utils, vtk_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement/two_pass_debug'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]
TARGET_FILES = {
    'seoul_pano_4': '000004_993944 pano 1.jpg',  # 이전: 31/32/41/42 누락
    'seoul_pano_6': '000006_158114 pano 2.jpg',  # 이전: 18<->47 혼동
    'seoul_pano_0': '000000_957466 pano 3.jpg',  # 이전: 거의 동일 (베이스라인 삼아 재확인)
}
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
    coco = dict(datasets)['seoul_univ_inspect_1106']
    predict_mod.g_rfdetr_model = None
    model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    model_parameter = next(model.parameters())
    convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}

    for tag, fname in TARGET_FILES.items():
        im = next(x for x in coco.imgs.values() if x['file_name'].endswith(fname))
        anns = coco.imgToAnns[im['id']]
        img = image_utils.cv2_imread(im['file_name'], flags=cv2.IMREAD_GRAYSCALE)
        gt_masks = np.stack([coco.annToMask(a).astype(bool) for a in anns])
        gt_fdi = universal_to_fdi[np.array([a['category_id'] for a in anns])]

        _, _, inst1 = _predict_single_pass(img, model, convert_options)
        _, _, inst2, used2, _ = two_pass_predict(img, model, convert_options=convert_options)

        print(fname, 'used_two_pass=', used2)
        print('  one-pass fdi:', sorted(inst1['fdi'].tolist()))
        print('  two-pass fdi:', sorted(inst2['fdi'].tolist()))

        p_raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(p_raw, 'raw', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        p_gt = panel(img, gt_fdi, gt_masks, 'GT')
        p_one = panel(img, inst1['fdi'], inst1['masks'], 'one-pass')
        p_two = panel(img, inst2['fdi'], inst2['masks'], f'two-pass(v2, used={used2})')

        h = img.shape[0]
        scale = 480 / h
        tiles = [cv2.resize(p, None, fx=scale, fy=scale) for p in (p_raw, p_gt, p_one, p_two)]
        combined = np.concatenate(tiles, axis=1)
        out_path = OUT_DIR / f"{tag}_v2.jpg"
        # vtk_utils 색상 테이블은 RGB 순서(rfdetr/datasets/visualize.py의 관례와 동일) --
        # cv2.imwrite는 BGR로 저장하므로 끝에서 한 번만 채널을 뒤집는다.
        cv2.imwrite(str(out_path), combined[..., ::-1])
        print('saved:', out_path)


if __name__ == '__main__':
    main()
