# peri 데이터 합류(`peri_data_fusion`) 평가 결과

`seoul_univ_inspect_COCO` 데이터셋(peri/pano 혼합)을 학습에 합류시키고 `overlap_loss_coef`를 켠 상태로 fine-tuning한 체크포인트가, 기존 배포 모델 대비 (1) 기존 panoramic 성능을 유지하는지, (2) 겹친 치아(overlap) 검출이 좋아지는지, (3) 실제 peri(치근단) 사진에서 잘 동작하는지, (4) teeth_roi crop 모드에서 precision/recall이 오르는지 확인한다.

- 평가일: 2026-11-02
- 비교 코드: `rfdetr/scripts/compare_core.py`, `eval_panorama_and_overlap.py`, `eval_peri_subset.py`, `eval_pr_utils.py`, `compare_teeth_roi.py`
- 원본 JSON: `outputs/compare_peri_fusion/{panorama_overlap,peri_subset,teeth_roi}/summary.json`

## 0. 비교 대상 / 중요 전제

| | 체크포인트 | epoch | sha256(앞 16자리) |
|---|---|---|---|
| 배포 모델(baseline) | `outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix/checkpoint_best_total.pth` | - | `61fe051a5fe5d5f5` |
| 신규(peri 합류) | `outputs/peri_data_fusion/checkpoint_best_ema.pth` | **45** | `e990e1bf99256f05` |

**⚠️ 신규 체크포인트는 500 epoch 중 45~58 epoch에서 중단된 중간 결과다** (`nohup.out`에 `KeyboardInterrupt` — 수동 중단, `checkpoint.pth`는 epoch 58, best는 epoch 45). 아래 1·2번 결과의 소폭 하락은 "기능이 잘못됐다"가 아니라 "아직 덜 수렴했다"로 읽어야 한다. baseline도 동일한 `xray_crop_mode=['periapical','teeth_roi']`, `overlap_loss_coef=4.0` 설정으로 이미 학습된 모델이라(체크포인트 `args`에 저장된 값), 2번 overlap 질문은 "overlap loss 효과 유무"가 아니라 "peri 데이터 합류가 기존 overlap 성능을 깨지 않는지"를 보는 것에 가깝다. overlap loss 자체의 효과(precision/recall trade-off)는 `outputs/compare_overlapfix_vs_deployed/summary_overlapfix_v2_ep44.json`에 이미 기록되어 있다.

카테고리 스킴: `cbct_ios_dcm_latest_1022`와 `seoul_univ_inspect_COCO` 둘 다 동일한 32-class "universal 번호" 체계를 쓴다(직접 비교 확인, id/name 완전 일치). 결측 이미지(`seoul_univ_inspect_COCO`: train 49장, val 10장 json에는 있으나 디스크에 없음)는 `rfdetr.datasets.xraypanoramic.merge_coco_directories`가 자동으로 걸러낸 뒤 비교에 사용했다(이번 세션에서 추가한 수정, `rfdetr/scripts/compare_core.py::load_coco`가 동일 함수를 재사용).

## 1. 기존 panoramic 데이터에서 동등 이상인가?

데이터: `cbct_ios_dcm_latest_1022/val` 85장 (own-data val, `tools/compare_own1022.py`/`compare_overlapfix_vs_deployed.py`와 동일 split). crop 없이 원본 전체 영상 평가.

| 지표 | baseline | 신규(ep45) | 차이 |
|---|---|---|---|
| instance dice | **0.9165** | 0.9026 | −0.014 |
| instance iou | **0.8520** | 0.8367 | −0.015 |
| boundary F1 | **0.8531** | 0.8305 | −0.023 |
| contact F1 | **0.8122** | 0.7954 | −0.017 |

**결론: 소폭 하락 (동등 미달).** 차이는 크지 않지만(dice −1.5%p) 전 지표에서 일관되게 baseline이 앞선다. epoch 45/500이라는 초기 중단 지점을 감안하면, peri 데이터 합류 + crop 증강 재도입이 아직 기존 panoramic 분포에 완전히 재적응하지 못한 상태로 보인다. 재현: `PYTHONPATH=. python rfdetr/scripts/eval_panorama_and_overlap.py`.

## 2. 겹친 치아(overlap) 영역을 잘 찾는가?

같은 val 85장에서, 금속/보철(밝기≥240, GT 내부) 픽셀을 제외한 같은 악궁 인접 치아(FDI 번호 1 차이 또는 정중선 사이) 쌍의 overlap 영역 기준.

| | baseline | 신규(ep45) | 차이 |
|---|---|---|---|
| pairwise recall | **76.1%** | 70.5% | −5.6%p |
| pairwise precision | **21.9%** | 18.7% | −3.2%p |
| overlap GT px | 94,943 | 94,943 | - |

**결론: 소폭 하락.** 두 모델 다 precision이 낮은 편인데(baseline도 21.9%), 이는 overlap loss 자체가 "겹침 과다예측" 경향을 유발하기 때문으로 보인다(`outputs/compare_overlapfix_vs_deployed/summary_overlapfix_v2_ep44.json`에서 overlap loss 도입 전/후 precision 49.6%→21.2%로 이미 떨어진 바 있음 — baseline이 그 결과물). 신규 체크포인트는 그 baseline보다도 recall/precision이 살짝 더 낮아, 1번과 같은 "미수렴" 설명이 유력하다. 재현: 1번과 동일 스크립트(한 번에 같이 계산됨).

**시각 확인:** 실제로 보면 차이는 전체적으로 미묘하다 — "가장 더 나빠진" 사례도 전체 윤곽은 거의 동일하고 경계 픽셀 수준의 차이다. 1·2번 결론이 수치상 하락이라도 육안 품질 저하로 체감되는 수준은 아니다.

| 이미지 | GT / baseline / 신규(ep45) |
|---|---|
| image 15 (overlap 소폭 하락 사례) | ![image 15](../outputs/compare_peri_fusion/images/panorama_15.jpg) |
| image 20 (전형적 사례) | ![image 20](../outputs/compare_peri_fusion/images/panorama_20.jpg) |
| image 71 (overlap 개선 사례) | ![image 71](../outputs/compare_peri_fusion/images/panorama_71.jpg) |

## 3. 실제 peri(치근단) 이미지에서 잘 동작하는가?

데이터: `seoul_univ_inspect_COCO/val`에서 파일명에 `peri`가 포함된 82장(annotation 314개) — 패노라마에서 합성한 crop이 아니라 **실제로 촬영된 치근단 사진**. (이 파일들은 이번 세션에서 추가한 `_is_peri_filename` 로직에 의해 학습 시 crop 증강 대상에서 제외되어, 원본 그대로 사용된 실사 peri 테스트셋이다.)

| 지표 | baseline | 신규(ep45) | 차이 |
|---|---|---|---|
| instance dice | 0.0677 | **0.6315** | **+0.564** |
| instance iou | 0.0613 | **0.5892** | **+0.528** |
| boundary F1 | 0.0300 | **0.2620** | **+0.232** |
| contact F1 | 0.0270 | **0.1927** | **+0.166** |
| GT 인스턴스 중 매칭 성공 | 69 / 314 (22%) | **254 / 314 (81%)** | **+59%p** |
| dice=0 (완전 실패) 이미지 수 | 55 / 82 (67%) | **6 / 82 (7%)** | |

**결론: 압도적으로 개선.** baseline은 실제 peri 사진에서 거의 작동하지 않는다(치아 10개 중 2개꼴 매칭, 이미지 3장 중 2장은 아예 매칭되는 치아가 0개). 패노라마 원본에서 합성한 periapical crop 증강만으로는 실사 치근단 사진의 도메인(질감/대비/프레이밍)을 전혀 커버하지 못했다는 뜻이다. 신규 모델은 epoch 45라는 초기 단계인데도 real peri 검출이 극적으로 좋아졌다 — **이번 데이터 합류 작업의 핵심 성과**. 다만 dice 0.63/boundary F1 0.26 자체는 아직 절대적으로 높은 수준은 아니라 추가 학습 여지가 크다. 재현: `PYTHONPATH=. python rfdetr/scripts/eval_peri_subset.py`.

**시각 확인 (3장 모두 동일 패턴 재현):** baseline의 dice≈0은 "마스크를 못 찾아서"가 아니라, **치아 윤곽 자체는 거의 맞는데 FDI 번호(클래스)를 잘못 매긴다**(치아 색이 GT와 다름 → class-constrained 매칭에서 전부 miss 처리). 패노라마 맥락(치열 전체에서의 위치) 없이 peri 크롭 한 장만 보면, pretrained 모델은 "몇 번 치아인지" 추론을 못 하는 것으로 보인다. 신규 모델은 윤곽도 더 정확해졌을 뿐 아니라 번호 매김도 GT와 거의 정확히 일치한다.

| 이미지 | GT / baseline / 신규(ep45) |
|---|---|
| image 103 | ![image 103](../outputs/compare_peri_fusion/images/peri_103.jpg) |
| image 132 | ![image 132](../outputs/compare_peri_fusion/images/peri_132.jpg) |
| image 163 | ![image 163](../outputs/compare_peri_fusion/images/peri_163.jpg) |

(렌더링 스크립트: `rfdetr/scripts/render_comparison_samples.py`, 원본: `outputs/compare_peri_fusion/images/`)

## 4. teeth_roi 모드에서 precision/recall이 향상되는가?

`rfdetr/cli/main.py --eval --eval_xray_crop --xray_crop_mode teeth_roi --xray_crop_prob 1.0` (docs/xray_periapical_eval.md와 동일 방식, 학습/평가 crop·GT 로직 재사용)으로 own-data(259장, peri 파일명 제외) + Kaggle/Tufts(80장) 병합 val set 339장 전체에 teeth_roi crop을 적용해 평가했다(단, peri 파일명 이미지는 이번 세션 수정으로 crop 제외 → 원본 그대로 섞여 들어간다. 순수 teeth_roi 입력만 보고 싶다면 own1022 kaggle/Tufts 쪽(80장, crop 100% 적용)만 따로 봐야 하지만, 이번 수치는 전체 혼합 기준이다).

### COCO segm AP/AR (area=all, maxDets=100)

| | baseline | 신규(ep45) | 차이 |
|---|---|---|---|
| AP | 0.595 | **0.636** | **+0.041** |
| AP50 | 0.876 | **0.918** | **+0.042** |
| AP75 | 0.751 | **0.803** | **+0.052** |
| AR@100 | 0.662 | **0.702** | **+0.040** |

### 고정 score 기준 근사 precision/recall (docs/xray_periapical_eval.md §2.2 방식)

| mask IoU 기준 | score 기준 | baseline P / R | 신규(ep45) P / R |
|---|---|---|---|
| 0.5 | 0.5 | 0.918 / 0.840 | 0.922 / **0.875** |
| 0.5 | 0.3 | 0.854 / 0.880 | 0.850 / **0.922** |
| 0.75 | 0.5 | 0.833 / 0.759 | **0.846** / **0.803** |
| 0.75 | 0.3 | 0.765 / 0.792 | **0.779** / **0.842** |

**결론: 뚜렷하게 개선.** AP/AP50/AP75/AR 전부 +4~5%p 상승, 4가지 (IoU, score) 조합 모두에서 recall이 +3.5~6.2%p 오르고 precision은 거의 변화 없거나 소폭 상승(0.5 IoU·score 0.3 조합만 -0.004로 사실상 동률). bbox 쪽도 동일 경향(AP 0.644→0.691, AR 0.714→0.761; 로그: `logs/eval_teeth_roi_{baseline,new}.log`). teeth_roi 모드는 1·2번과 반대로 **epoch 45 시점에도 이미 개선**되어 있다. 재현: 위 두 `--eval` 명령 실행 후 `PYTHONPATH=. python rfdetr/scripts/compare_teeth_roi.py`.

## 5. 종합

| 질문 | 결과 |
|---|---|
| 1. 기존 panoramic 동등 이상? | △ 소폭 하락 (dice −1.5%p) — 미수렴 추정 |
| 2. overlap 검출 유지/개선? | △ 소폭 하락 (recall −5.6%p) — 미수렴 추정, baseline도 이미 overlap loss 적용 모델 |
| 3. 실제 peri 이미지 동작? | ✅ **압도적 개선** (dice 0.07→0.63, 매칭률 22%→81%) |
| 4. teeth_roi precision/recall? | ✅ **뚜렷한 개선** (AP +4.1%p, recall 전 구간 +3.5~6.2%p) |

peri 데이터 합류의 1차 목적(peri/ROI 입력 대응)은 중간 체크포인트(epoch 45/500)만으로도 이미 분명히 달성되고 있다. 반면 기존 panoramic/overlap 성능은 아직 baseline을 완전히 따라잡지 못했으므로, 학습을 끝까지(또는 최소 100~150 epoch 이상) 진행한 뒤 1·2번을 재평가하는 것을 권장한다. 또한 panoramic 하락 폭이 학습 재개 후에도 좁혀지지 않는다면 peri/pano 데이터 비율이나 `xray_crop_prob` 조정을 고려할 필요가 있다.

## 6. 재현 명령 요약

```bash
# 1, 2: 기존 panoramic + overlap (own1022 val 85장)
CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=. python rfdetr/scripts/eval_panorama_and_overlap.py

# 3: 실제 peri 이미지 (seoul_univ val, 파일명 "peri" 포함 82장)
CUDA_VISIBLE_DEVICES=<free> PYTHONPATH=. python rfdetr/scripts/eval_peri_subset.py

# 4: teeth_roi 모드 (cli --eval 재사용, 체크포인트별로 각각 실행)
python -m rfdetr.cli.main --pretrain_weights <ckpt> \
  --xray_crop_mode teeth_roi --xray_crop_prob 1.0 --batch_size 1 --device cuda:<free> \
  --output_dir outputs/eval_teeth_roi_<name> --eval --eval_xray_crop
PYTHONPATH=. python rfdetr/scripts/compare_teeth_roi.py
```

결과 파일:
- `outputs/compare_peri_fusion/panorama_overlap/summary.json` (+ `*_predictions.json`, `*_per_image.json` 캐시)
- `outputs/compare_peri_fusion/peri_subset/summary.json`
- `outputs/compare_peri_fusion/teeth_roi/summary.json`
- `outputs/eval_teeth_roi_{baseline,new}/eval.pth`, `logs/eval_teeth_roi_{baseline,new}.log`
