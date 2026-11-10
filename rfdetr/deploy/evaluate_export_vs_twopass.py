"""실제 배포 후보 두 경로를 '수출(export)된 결과물' 기준으로 비교한다.

  A. one-pass-export: torch.jit(libtorch) export된 고정 입력 크기 모델로 단일 추론.
     export는 학습 때와 달리 가변 해상도를 못 받으므로(고정 shape 1개), pano/peri 각각
     실제 이미지 크기 분포(아래 '모양 선택' 참고)에 맞는 별도 고정 shape로 따로 export한다.
  B. two-pass: 기존 ``two_pass_predict`` (가변 eager 모델, 1차 전체 추론 -> teeth_roi
     center-aware crop -> 2차 고해상도 재추론). crop 영역이 이미지마다 달라 export(고정 크기)로는
     구현 불가 -- peri류는 crop이 거의 안 줄어 자동으로 1-pass(eager, dynamic resize)로 끝남
     (roi_shrink_ratio).

둘 다 "진짜 배포했을 때" 쓰일 경로 그대로이고, (A)는 fp16 export 모델(``export_libtorch`` 기본
dtype)이라 정밀도 손실까지 포함한다.

모양 선택 (``rfdetr/datasets/xraypanoramic.py``의 ``resize_policy: long_side_keep_ratio_letterbox_stride64``와
동일한 "긴 변 -> 640" 규칙을 고정 shape 하나로 최대한 손실 없이 재현하도록 실제 검증셋 크기 분포로
산출 -- 아래 두 숫자는 본 스크립트 작성 시점에 1회성으로 조사한 값, 데이터셋이 바뀌면 재산출 필요):
  - PANO_EXPORT_SHAPE=(384, 640): cbct_pano(aspect 2.1~2.25), seoul_pano 대다수(aspect
    ~1.5~2.25)는 W=640이 긴 변 기준(학습과 동일 배율)로 꽉 차고 H<=384라 손실 없음. 단
    aspect<1.67인 소수(seoul_pano 52장 중 12장 -- 실제로는 전악 pano가 아니라
    'implant_*_lower/upper.jpg' 류 반악/부분 촬영 이미지로, evaluate_by_type.categorize()가
    경로에 'panoramic'이 포함돼 있어 파일명에 'peri'가 없으면 전부 seoul_pano로 묶이는 바람에
    섞여 들어간 것 -- 진짜 분류 버그라기보다 categorize()의 느슨한 휴리스틱 한계)는 학습 대비
    추가로 더 눌려 들어간다 (아래 PANO_PARTIAL_ARCH_FILES로 따로 집계).
  - PERI_EXPORT_SHAPE=(512, 640): 처음엔 "어떤 aspect든 손실 없이"를 목표로 (640, 640)을 썼으나
    실측 결과 eager 모델에 동일한 (640,640) 고정 캔버스만 줘도(= export 여부와 무관) dice가
    0.72->0.66으로 떨어지는 걸 확인 (``rfdetr/deploy/evaluate_two_pass.py``의 동적 1-pass와
    비교). 즉 "자르지만 않으면 된다"가 아니라 "학습 때 보던 수준 이상으로 여백(padding)이
    늘어나는 것 자체가 분포 이탈"임을 실측으로 확인한 것 -- peri/teeth_roi 학습 crop은 치아가
    프레임을 거의 꽉 채우도록 만들어지므로, 동적 처리의 중앙값 타깃(640x512)보다 훨씬 여유로운
    640x640 캔버스는 학습 때 못 보던 분포. 중앙값에 맞춘 (512, 640)으로 줄였더니 eager 고정
    캔버스 기준 dice=0.742/iou=0.702로 동적 베이스라인(~0.72)을 회복(오히려 소폭 상회) --
    aspect<640/512=1.25인 소수(= PANO_PARTIAL_ARCH과 달리 여기선 진짜 peri 중 극히 일부, 예:
    cbct_peri 514x514 근처)는 학습 대비 추가로 더 눌려 들어가는 트레이드오프를 감수.

Run (pick a free GPU):
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /opt/conda/bin/python rfdetr/deploy/evaluate_export_vs_twopass.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

import rfdetr.deploy.predict as predict_mod
from rfdetr.deploy.predict import (
    _predict_single_pass, two_pass_predict, export_libtorch, load_export_model,
    DEFAULT_TWO_PASS_ROI_MARGIN, DEFAULT_TWO_PASS_SHRINK_RATIO,
    DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD,
)
from rfdetr.deploy.evaluate import MODELS, load_val_coco, METRIC_KEYS
from rfdetr.deploy.evaluate_by_type import categorize
from rfdetr.datasets.xraypanoramic import label2fdi
from tools.compare_own1022 import metrics, aggregate
from trainer import image_utils

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'outputs/eval_fdi_refinement'
EXPORT_DIR = ROOT / 'outputs/export/pano_peri_1106_ce'
MODEL_NAME = 'peri_data_fusion_1106_ce_ema'
MODEL_CKPT = MODELS[MODEL_NAME]

PANO_EXPORT_SHAPE = (384, 640)  # (H, W)
PERI_EXPORT_SHAPE = (512, 640)

# evaluate_by_type.categorize()가 경로에 'panoramic' 문자열이 있어 'pano'로 묶는 바람에
# seoul_pano에 섞여 들어간 반악/부분 촬영 이미지 (진짜 전악 pano 아님) -- 참고용 집계 분리.
PANO_PARTIAL_ARCH_KEYWORDS = ('_lower.jpg', '_upper.jpg')
# GT 자체가 잘못 레이블링된 것으로 확인된 케이스 (docs/todolist.md 참고: 12번에 11번/임플란트가
# 합쳐져 레이블링됨). 이 파일의 one-pass/two-pass 비교는 모델 품질 지표로 신뢰하지 않는다.
KNOWN_GT_MISLABELED = ('000004_993944 pano 1.jpg',)


def is_partial_arch(file_name):
    name = file_name.lower()
    return name.endswith(PANO_PARTIAL_ARCH_KEYWORDS) and 'implant' in name or 'healthreport' in name or 'restoration' in name


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_val_coco()

    print('exporting pano/peri fixed-shape libtorch models...')
    export_libtorch(str(EXPORT_DIR / 'pano_model_libtorch.pt'), shape=PANO_EXPORT_SHAPE,
                     pretrain_weights=MODEL_CKPT, use_ema=False)
    export_libtorch(str(EXPORT_DIR / 'peri_model_libtorch.pt'), shape=PERI_EXPORT_SHAPE,
                     pretrain_weights=MODEL_CKPT, use_ema=False)

    pano_model, pano_shape = load_export_model(str(EXPORT_DIR / 'pano_model_libtorch.pt'),
                                                export_shape=PANO_EXPORT_SHAPE)
    peri_model, peri_shape = load_export_model(str(EXPORT_DIR / 'peri_model_libtorch.pt'),
                                                export_shape=PERI_EXPORT_SHAPE)
    pano_param = next(pano_model.parameters())
    peri_param = next(peri_model.parameters())
    pano_convert = {'device': pano_param.device, 'dtype': pano_param.dtype}
    peri_convert = {'device': peri_param.device, 'dtype': peri_param.dtype}

    predict_mod.g_rfdetr_model = None
    eager_model = predict_mod.init_and_get_model(use_ema=False, pretrain_weights=MODEL_CKPT, device='cuda')
    eager_param = next(eager_model.parameters())
    eager_convert = {'device': eager_param.device, 'dtype': eager_param.dtype}

    rows = {'one_pass_export': {}, 'two_pass': {}}
    rows_partial_arch_excluded = {'one_pass_export': {}, 'two_pass': {}}
    two_pass_used = {}
    n_by_cat = {}
    n_images = 0
    partial_arch_files = []
    gt_mislabeled_files = []

    for source_name, coco in datasets:
        for im in coco.imgs.values():
            anns = coco.imgToAnns[im['id']]
            if not anns:
                continue
            file_name = im['file_name']
            cat = categorize(source_name, file_name, im['width'], im['height'])
            img = image_utils.cv2_imread(file_name, flags=cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            gt = np.stack([coco.annToMask(a).astype(bool) for a in anns])

            if cat.endswith('_peri'):
                exp_model, exp_resize_hw, exp_convert = peri_model, peri_shape, peri_convert
            else:
                exp_model, exp_resize_hw, exp_convert = pano_model, pano_shape, pano_convert

            _, _, inst_exp = _predict_single_pass(img, exp_model, exp_convert, drawing_mode='polygon',
                                                   resize_hw=exp_resize_hw)
            u_exp = label2fdi[inst_exp['fdi']]
            row_exp = metrics(gt, inst_exp['masks'], u_exp, inst_exp['scores'], inst_exp['boxes'], anns, h, w)
            rows['one_pass_export'].setdefault(cat, []).append(row_exp)

            _, _, inst2, used_two_pass, _ = two_pass_predict(
                img, eager_model, drawing_mode='polygon', convert_options=eager_convert,
                roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN, roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO,
                pass2_confidence_threshold=DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD)
            u2 = label2fdi[inst2['fdi']]
            row2 = metrics(gt, inst2['masks'], u2, inst2['scores'], inst2['boxes'], anns, h, w)
            rows['two_pass'].setdefault(cat, []).append(row2)

            is_partial = cat == 'seoul_pano' and is_partial_arch(Path(file_name).name)
            is_gt_bad = Path(file_name).name in KNOWN_GT_MISLABELED
            if is_partial:
                partial_arch_files.append(file_name)
            if is_gt_bad:
                gt_mislabeled_files.append(file_name)
            if not is_partial and not is_gt_bad:
                rows_partial_arch_excluded['one_pass_export'].setdefault(cat, []).append(row_exp)
                rows_partial_arch_excluded['two_pass'].setdefault(cat, []).append(row2)

            two_pass_used[cat] = two_pass_used.get(cat, 0) + int(used_two_pass)
            n_by_cat[cat] = n_by_cat.get(cat, 0) + 1
            n_images += 1
            if n_images % 20 == 0:
                print(source_name, n_images, flush=True)

    def build_summary(rows_dict, label):
        summary = {}
        print(f'\n=== {MODEL_NAME}: one-pass(export) vs two-pass [{label}] ===')
        all_cats = sorted(rows_dict['one_pass_export'])
        for cat in all_cats + ['ALL']:
            if cat == 'ALL':
                r1 = [r for rs in rows_dict['one_pass_export'].values() for r in rs]
                r2 = [r for rs in rows_dict['two_pass'].values() for r in rs]
                n = sum(len(v) for v in rows_dict['one_pass_export'].values())
            else:
                r1, r2 = rows_dict['one_pass_export'][cat], rows_dict['two_pass'][cat]
                n = len(r1)
            if not r1:
                continue
            m1, m2 = aggregate(r1), aggregate(r2)
            print(f"  {cat:<12s} n={n}")
            print(f"    one-pass-export  dice={m1['instance_dice']:.4f}  iou={m1['instance_iou']:.4f}  "
                  f"boundary_f1={m1['boundary_f1']:.4f}  contact_f1={m1['contact_f1']:.4f}")
            print(f"    two-pass         dice={m2['instance_dice']:.4f}  iou={m2['instance_iou']:.4f}  "
                  f"boundary_f1={m2['boundary_f1']:.4f}  contact_f1={m2['contact_f1']:.4f}")
            summary[cat] = dict(
                n_images=n,
                one_pass_export={k: m1[k] for k in METRIC_KEYS},
                two_pass={k: m2[k] for k in METRIC_KEYS},
            )
        return summary

    summary_all = build_summary(rows, 'raw, 전체')
    summary_clean = build_summary(rows_partial_arch_excluded, 'seoul_pano 반악/부분촬영 + GT 오류 파일 제외')

    out = dict(
        model=MODEL_NAME, pano_export_shape=list(PANO_EXPORT_SHAPE), peri_export_shape=list(PERI_EXPORT_SHAPE),
        two_pass_used_by_cat=two_pass_used, n_by_cat=n_by_cat,
        partial_arch_files_excluded_from_clean=partial_arch_files,
        gt_mislabeled_files_excluded_from_clean=gt_mislabeled_files,
        summary_raw=summary_all, summary_clean=summary_clean,
    )
    out_path = OUT_DIR / 'summary_export_vs_twopass.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print('\nsaved summary:', out_path)
    print('partial-arch files excluded in clean view:', len(partial_arch_files))
    print('GT-mislabeled files excluded in clean view:', len(gt_mislabeled_files))


if __name__ == '__main__':
    main()
