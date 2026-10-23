# PA(치근단) crop 학습 · 평가 결과

파노라마 xray로 학습한 모델이 치근단 방사선(PA) 형태의 입력에서 얼마나 동작하는지 평가한 결과와, PA crop 학습 방식을 정리한다.

- 평가일: 2026-10-22
- 체크포인트: `outputs/fine_tune_pa_teeth_roi_padding_fixed/checkpoint_best_regular.pth`
  - `outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth`에서 periapical + teeth_roi crop 증강으로 fine-tuning한 모델
- 평가 데이터: val 80장 (① Kaggle + ② Tufts, 파일명 정렬 기준 0.85~0.9 구간). ③ 자체 데이터는 평가에 포함되지 않는다.

## 1. PA crop 학습 방식

코드: `rfdetr/datasets/xray_crop_aug.py` (`periapical_box`, `sample_crop`), `rfdetr/datasets/xraypanoramic.py` (`parse_item_coco`, `_read_coco_image`)

### 1.1 crop 생성

파노라마 원본 해상도에서 PA 영상처럼 잘라낸 뒤, 기존 증강·리사이즈 파이프라인을 그대로 적용한다.

| 항목 | 기본값 (`DEFAULT_PERIAPICAL`) | 설명 |
|---|---|---|
| 치아 수 | 3~5개 | 같은 악궁에서 x 방향으로 이웃한 치아 |
| 악궁 | 상악 0.4 / 하악 0.4 / 상하악 0.2 | 상하악이면 같은 x 범위의 하악 치아까지 세로로 포함 |
| 비율 | 가로형 w/h ≈ 1.39 (±8%) | 전치부는 70%, 구치부는 10% 확률로 세로형(1/1.39) |
| 여유 | 좌우 0~15%, 치근 쪽 5~25%, 반대 악궁 쪽 0~25% | 이웃 치아·반대 악궁 치관이 일부 보이게 |
| 모서리 | 사선(chamfer) 0.7 / 둥근(round) 0.3, 짧은 변의 6~14% | 모서리 영역은 영상·마스크 모두 0 |
| 입력 크기 | 긴 변 640, stride 64 배수 | |

### 1.2 crop 경계에서 잘린 치아 처리

| 상황 | 처리 |
|---|---|
| 치아 **bbox 면적의 25% 이상**이 crop 안 (`min_visible=0.25`) | GT 유지. 마스크를 crop 경계·모서리로 자르고, bbox는 **보이는 마스크 기준으로 재계산**. 클래스는 원래 치아 번호 |
| bbox 면적의 25% 미만 | **GT에서 제외**. 영상 픽셀은 그대로 남는다 (배경으로 학습됨) |
| 리사이즈 후 마스크 16px 미만 | GT에서 제외 |
| PA 모서리 영역 | 영상을 0으로 칠하고 마스크에서도 제외 |

절반 이상 잘린 치아(25~50% 보임)도 보이는 부분만 정답으로 학습한다.

### 1.3 학습 / 평가 명령

```bash
# 학습 (PA 40%, ROI 40%, 전체 영상 20%)
python -m rfdetr.cli.main \
  --pretrain_weights <체크포인트> \
  --xray_crop_mode periapical teeth_roi \
  --xray_crop_prob 0.4 0.4 \
  --batch_size 4 --device cuda:5 \
  --output_dir <출력 폴더>

# PA만 평가 (val 100% PA crop, 이미지별 고정 seed 42)
python -m rfdetr.cli.main \
  --pretrain_weights outputs/fine_tune_pa_teeth_roi_padding_fixed/checkpoint_best_regular.pth \
  --xray_crop_mode periapical --xray_crop_prob 1.0 \
  --batch_size 1 --device cuda:5 \
  --output_dir outputs/eval_periapical_only \
  --eval --eval_save --eval_xray_crop

# 전체 영상 평가 (crop 없음)
python -m rfdetr.cli.main \
  --pretrain_weights outputs/fine_tune_pa_teeth_roi_padding_fixed/checkpoint_best_regular.pth \
  --batch_size 1 --device cuda:5 \
  --output_dir outputs/eval_full_image \
  --eval
```

- `--xray_crop_prob`는 mode 순서대로 들어간다. 합이 1보다 작으면 나머지는 전체 영상이다. `periapical 0.5`만 주면 절반은 전체 영상으로 평가된다.
- `--output_dir`를 주지 않으면 `eval.pth`와 예측 이미지가 기본 학습 폴더(`output/xray_teeth33_dinov2tiny_small_seg_crop_regularization`)에 저장된다.
- `--eval_xray_crop`의 평가 GT도 1.2절과 같은 규칙으로 만든다(25% 미만 치아는 GT에서 제외).

## 2. 결과: 전체 영상 vs PA-only

### 2.1 COCO 지표

| 지표 | 전체 영상 | PA-only | 차이 |
|---|---|---|---|
| bbox AP@[.5:.95] | 0.587 | **0.621** | +0.034 |
| bbox AP50 | **0.951** | 0.862 | −0.089 |
| bbox AP75 | 0.665 | **0.747** | +0.082 |
| bbox AR@100 | 0.655 | **0.721** | +0.066 |
| segm AP@[.5:.95] | **0.601** | 0.568 | −0.033 |
| segm AP50 | **0.949** | 0.860 | −0.089 |
| segm AP75 | **0.743** | 0.685 | −0.058 |
| segm AP small | — | 0.196 | |
| segm AP medium | **0.579** | 0.558 | −0.021 |
| segm AP large | 0.639 | **0.654** | +0.015 |
| segm AR@100 | 0.651 | **0.664** | +0.013 |

전체 영상에는 COCO small 영역(32² px 미만)에 해당하는 치아가 없어 −1로 나온다.

### 2.2 precision / recall (segm)

COCO 출력에는 고정 score 기준의 precision/recall이 없다. 그래서 `eval.pth`의 PR 곡선(recall 101구간)에서 score가 기준 이상인 마지막 지점을 읽어 32개 클래스 평균을 냈다. 근사값이다.

| 마스크 IoU 기준 | score 기준 | 전체 영상 P / R | PA-only P / R |
|---|---|---|---|
| 0.5 | 0.5 | 0.960 / **0.925** | 0.947 / **0.717** |
| 0.5 | 0.3 | 0.950 / 0.939 | 0.806 / 0.852 |
| 0.75 | 0.5 | 0.833 / 0.794 | 0.841 / 0.633 |
| 0.75 | 0.3 | 0.825 / 0.802 | 0.718 / 0.726 |

<details>
<summary>계산 스크립트</summary>

```python
import torch, numpy as np
e = torch.load('outputs/eval_periapical_only/eval.pth', weights_only=False)
P, S, R = e['precision'], e['scores'], e['recall']   # P, S: [T,101,K,A,M], R: [T,K,A,M]
rec_thr = np.linspace(0, 1, 101)
ti, thr = 0, 0.5                                     # ti=0: IoU 0.5, ti=5: IoU 0.75
ps, rs = [], []
for k in range(P.shape[2]):
    if R[ti, k, 0, 2] < 0:
        continue
    p, s = P[ti, :, k, 0, 2], S[ti, :, k, 0, 2]
    ok = np.where((s >= thr) & (p > 0))[0]
    if len(ok) == 0:
        ps.append(0); rs.append(0); continue
    ps.append(p[ok[-1]]); rs.append(rec_thr[ok[-1]])
print(np.mean(ps), np.mean(rs))
```
</details>

## 3. 분석

- **precision은 거의 같고 recall이 크게 떨어진다** (score 0.5, IoU 0.5 기준 0.925 → 0.717). 찾은 치아는 정확하지만 놓치는 치아가 많다.
- **PA에서는 score가 애매한 예측이 많다.** score 기준을 0.3으로 낮추면 PA recall이 0.85로 회복되지만 precision이 0.81로 떨어진다. 전체 영상은 기준을 바꿔도 거의 변하지 않는다.
- **PA에만 small 영역이 생기고 성능이 낮다** (segm AP small 0.196). crop 가장자리에서 일부만 보이는 치아가 여기에 해당한다. recall 하락의 주원인으로 추정한다. 잘린 정도별 recall은 아직 측정하지 않았다.
- **bbox AP75는 PA가 높고, segm AP75는 PA가 낮다.** PA에서는 치아가 픽셀상 커서 박스는 정밀해진다. 하지만 refinement head가 큰 치아를 128×64로 줄여 처리하면서 마스크 경계 정보를 잃는 것으로 보인다(`docs/xray_teeth_modifications.md` 2.6절).

### 3.1 잘린 치아 처리에서 의심되는 문제

1. **25% 기준 근처의 라벨이 모순된다.** 30% 보이는 치아는 "치아", 20% 보이는 치아는 영상에 보이는데도 "배경"으로 학습한다. 가장자리에서 score가 낮아지는 원인일 수 있다.
2. **기준이 bbox 면적이다.** 치아는 bbox를 다 채우지 않으므로, 같은 25%라도 치관이 걸렸는지 치근 끝이 걸렸는지에 따라 실제로 보이는 치아 양이 다르다.
3. **평가 GT도 같은 규칙을 쓴다.** 25% 미만 치아를 모델이 찾으면 FP로 잡혀 precision이 깎인다.

## 4. 개선 후보 (미적용)

| 후보 | 기대 효과 |
|---|---|
| 보이는 비율 10~25% 치아는 영상에서 가리거나 loss에서 제외(ignore) | 기준 근처 라벨 모순 제거 → 가장자리 score·recall 개선 |
| `min_visible` 기준을 bbox 면적 → 마스크 면적으로 변경 | 치아 부위에 따른 편차 제거 |
| PA 입력의 추론 score 기준 0.35~0.4 | 재학습 없이 recall 회복 (precision과 trade-off) |
| `segmentation_crop_size`를 `(192, 96)`으로 키우고 fine-tuning | PA에서 큰 치아의 마스크 경계(segm AP75) 개선 |
| 잘린 정도별(25~50%, 50~100%) recall 측정 | recall 하락 원인 확정 |

## 5. 주의

- val이 80장뿐이라 수치 변동이 크다.
- 전체 영상 평가 로그에 `_v2.json not found` 경고가 찍혔지만, 서버의 `xray_coco_33_seg.json`이 이미 원본 좌표(v2 내용) 버전이라 수치는 유효하다(2026-10-22 확인, 이후 경고 제거). 이 json은 annotation id가 0부터 시작해서, id 0인 GT 1개에 매칭된 예측은 FP로 계산된다(41,319개 중 1개, 영향 미미).
- 전체 영상과 PA-only는 GT가 다르다. PA는 crop된 GT, 전체 영상은 원본 GT(`xray_coco_33_seg_v2.json`)다. 같은 이미지라도 치아 수와 크기 분포가 달라 AP를 직접 비교할 때 주의한다.
- 결과 파일
  - `outputs/eval_periapical_only/`: `eval.log`, `eval.pth`(segm), `eval/periapical/`(예측 이미지, 원본 | 예측 polygon)
  - `outputs/eval_full_image/`: `eval.log`, `eval.pth`(segm)
