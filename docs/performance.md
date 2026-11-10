## Performance Per versioin

### Version:

- 모델: `outputs/peri_data_fusion_1106_ce` (`checkpoint_best_ema.pth`, epoch 171)
- 평가일: 2026-11-09
- 평가 파이프라인: `rfdetr/deploy/evaluate_by_type.py` -- 배포 코드 `rfdetr/deploy/predict.py`의
  실제 추론 경로(`init_and_get_model` + `resize_image` letterbox + `draw_preditions_boxes`,
  `fdi_refinement=True` 기본값)를 그대로 재사용
- 비교 대상(baseline): `outputs/pretrained/segmentation/xray_teeth_seg/panoramic_overlapfix`
  (`checkpoint_best_total.pth`, 기존 배포 모델)
- 데이터: `cbct_ios_dcm_latest_1106` + `seoul_univ_inspect_1106`의 valid split 전체(174장,
  GT annotation 있는 이미지 기준), 유형별 세분화
- raw 결과: `outputs/eval_fdi_refinement/summary_by_type.json`

| 유형 | n | baseline dice | **with_ce dice** | baseline iou | **with_ce iou** | baseline boundary_f1 | **with_ce boundary_f1** | baseline contact_f1 | **with_ce contact_f1** |
|---|---|---|---|---|---|---|---|---|---|
| 전체(ALL) | 174 | 0.620 | **0.834** | 0.555 | **0.767** | 0.457 | **0.616** | 0.502 | **0.670** |
| cbct_pano (대형 panorama) | 42 | 0.782 | **0.867** | 0.707 | **0.794** | 0.670 | **0.755** | 0.659 | **0.746** |
| cbct_peri (저해상도, ≤1000x1000) | 37 | 0.013 | **0.725** | 0.013 | **0.692** | 0.017 | **0.529** | 0.005 | **0.347** |
| seoul_pano (파일명에 "pano") | 40 | 0.648 | **0.836** | 0.573 | **0.765** | 0.465 | **0.701** | 0.453 | **0.697** |
| seoul_peri (파일명에 "peri") | 55 | 0.009 | **0.722** | 0.008 | **0.684** | 0.001 | **0.319** | 0.003 | **0.261** |

- 중복 FDI 번호(`fdi_refinement` 적용 후): baseline/with_ce 모두 전 유형 0건.
- 결론: 4개 유형 전부에서 baseline 대비 개선(특히 peri 계열은 baseline이 거의 작동하지
  않던 수준에서 실사용 가능 수준으로 향상). 배포 모델 교체 권장. 단, peri 계열의
  `contact_f1`(인접 치아 경계 정확도)은 dice 대비 상대적으로 낮아 추후 개선 여지로 남겨둠.

## 추론 메커니즘: one-pass / two-pass (`rfdetr/deploy/predict.py`)

### 공통 전처리

모든 경로(one-pass/two-pass, eager/export)가 같은 letterbox 규칙을 쓴다(`resize_image()`):
긴 변을 기준 크기(기본 640)로 종횡비 유지 축소 → 각 축을 stride(64) 배수로 올림 → 그 캔버스
중앙에 배치하고 남는 부분은 0으로 padding. 학습 전처리(`resize_policy:
long_side_keep_ratio_letterbox_stride64`, `rfdetr/datasets/xraypanoramic.py`)와 동일한 규칙.

### one-pass

전체(또는 crop) 이미지 1회 추론(`_predict_single_pass`). `predict_wrapper`/`main`의 기본 경로이자
`two_pass_predict`가 내부적으로 생략될 때 최종 결과가 되는 경로이기도 하다.

### two-pass (`two_pass_predict`, 기본 활성 `two_pass=True`)

1. **1차**: 전체 이미지 one-pass 추론(`pass1_confidence_threshold=0.5`)으로 1차 치아 bbox/FDI
   확보.
2. **ROI 산출**: `teeth_roi_box_center_aware()` — 11/21(상악 중절치, 정중선) 중심을 x 원점으로
   고정하고, 그 원점 기준 좌/우 중 더 넓은 쪽 폭으로 대칭 crop을 구성(+margin). 11/21을 둘 다
   못 찾으면 bbox 합집합 중심으로 폴백.
3. **2차 실행 여부 게이팅** (아래 둘 다 통과해야 2차 실행, 하나라도 걸리면 1차 결과를 그대로
   반환 -- 파일명/해상도 기반 pano·peri 분류 없이 "crop이 실제로 도움되는지"만으로 결정):
   - `roi_shrink_ratio`(기본 0.85): crop 면적이 원본 면적의 85% 이상(거의 안 줄어듦)이면 생략.
   - `min_scale_gain`(기본 **1.3**): crop이 1차 대비 유효 해상도를 충분히 올려주는지. 전체
     이미지와 crop 각각을 `resize_image()`로 리사이즈했을 때의 content_box(letterbox padding
     제외) 스케일 비율 `scale2/scale1`이 이 값 미만이면 생략. 실측(`rfdetr/deploy/
     scale_gain_stats.py`, peri_data_fusion_1106_ce_ema 기준):
     | 카테고리 | scale_gain mean | median | min | max |
     |---|---|---|---|---|
     | cbct_pano | 1.851 | 1.855 | 1.315 | 2.387 |
     | seoul_pano | 1.841 | 1.806 | 1.518 | 2.658 |
     | cbct_peri | 1.239 | 1.227 | 1.000 | 1.965 |
     | seoul_peri | 1.212 | 1.213 | 1.000 | 1.604 |
     pano는 전체 프레임이 640 longside로 크게 축소돼 있어 치아 영역만 crop하면 유효 해상도가
     1.3~2.4배 커지는 반면, peri는 원본 자체가 이미 치아로 꽉 차 있어 crop해도 거의 그대로다.
     1.3을 기준으로 두면 pano는 사실상 전부 통과하고, peri 대다수는 자동으로 1-pass로 끝난다
     (실측: cbct_pano/seoul_pano 42/42, 40/40 전부 2차 실행, cbct_peri 10/37, seoul_peri
     18/55만 2차 실행 -- 나머지는 scale_gain<1.3이라 1-pass로 종료).
4. **2차**: crop 영역만 다시 letterbox해 고해상도로 재추론(`pass2_confidence_threshold=0.35`).
   crop으로 전체 이미지 context(상/하악 배치 등)가 사라지며 confidence 캘리브레이션이 1차와
   달라지기 때문에 1차보다 낮은 threshold를 쓴다(PR curve F1-최적점 0.338, dice sweep 하한
   0.3의 중간값).
5. 2차 결과가 1차를 전부 대체(좌표를 원본으로 역변환, mask를 원본 크기 캔버스에 paste).

### export(torch.jit/libtorch) 배포 경로의 제약

export는 고정 입력 크기 하나로 trace되므로 two-pass의 가변 crop을 지원할 수 없다
(`predict_wrapper`가 `torch.jit.ScriptModule`이면 자동으로 1-pass로 폴백). 따라서 export 모델은
**pano용/peri용 고정 shape를 따로 export**해야 한다 -- 하나로 퉁치면 한쪽이 손해:

| 대상 | shape (H, W) | 근거 |
|---|---|---|
| pano | (384, 640) | 실측 종횡비(2.0~2.25 위주)에서 긴 변(W)=640, 짧은 변(H) 중앙값(320)+여유. 학습과 거의 동일 배율. |
| peri | (512, 640) | **중요**: 처음엔 "안 잘리기만 하면 된다"고 보고 (640,640)을 썼으나, eager 모델에 동일한 (640,640) 고정 캔버스만 줘도(export 여부 무관) dice가 0.72대→0.66대로 급락하는 걸 확인 -- export 버그가 아니라 **학습 때보다 과도한 여백(padding) 자체가 분포 이탈**임을 실측으로 확인. 실측 중앙값(640x512)에 맞춰 (512,640)으로 줄이자 베이스라인을 회복(오히려 소폭 상회: dice 0.742/iou 0.702). 즉 추론 전처리는 "자르지만 않으면 되는" letterbox가 아니라 학습 때 보던 수준의 padding 비율을 맞춰야 한다. |

## one-pass vs two-pass 최종 성능 비교 (2026-11-10)

- 모델: `outputs/peri_data_fusion_1106_ce/checkpoint_best_ema.pth` (위와 동일)
- one-pass 쪽은 **export(torch.jit, fp16, 위 고정 shape)** 모델 기준 -- 실제 배포 조건과
  동일하게 맞추기 위함(동적 eager 모델 기준과는 다를 수 있음에 주의).
- two-pass 쪽은 위 게이팅(`roi_shrink_ratio=0.85`, `min_scale_gain=1.3`)이 적용된
  `two_pass_predict()` 기본값(현재 production 기본값) 그대로.
- raw 결과: `outputs/eval_fdi_refinement/summary_export_vs_twopass.json`
  (`rfdetr/deploy/evaluate_export_vs_twopass.py`)

| 카테고리 | n | 2차 실행 비율 | one-pass(export) dice/iou/boundary_f1/contact_f1 | two-pass(게이팅 적용, 기본값) 동일 지표 |
|---|---|---|---|---|
| cbct_pano | 42 | 42/42 | 0.865/0.790/0.739/0.733 | **0.888/0.826/0.828/0.820** (전 지표 승) |
| seoul_pano | 39(GT오류 1건 제외) | 39/39 | **0.830/0.756**/0.672/0.671 | 0.808/0.744/**0.728/0.738** (트레이드오프: dice·iou는 one-pass, boundary·contact는 two-pass) |
| cbct_peri | 37 | 10/37 | **0.781/0.742/0.540/0.383** | 0.637/0.605/0.484/0.326 (one-pass 승) |
| seoul_peri | 55 | 18/55 | **0.712/0.672/0.300**/0.249 | 0.673/0.639/0.324/**0.272** (one-pass 우세, contact만 근소하게 two-pass) |
| **ALL** | 173 | - | 0.834/0.764/0.595/0.650 | 0.824/0.765/**0.651/0.720** |

- **결론/현재 배포 설정**: `two_pass=True`가 기본값(`predict_wrapper`/`main`). pano는 게이팅이
  사실상 전부 2차를 실행시켜 cbct_pano는 확실한 이득, seoul_pano는 "전체 겹침(dice/iou) 소폭
  손해 vs 치아 경계/인접 정확도(boundary_f1/contact_f1, two-pass를 만든 원래 목적) 뚜렷한 이득"의
  트레이드오프 -- 후자를 우선한다는 전제로 기본 활성 결정.
- **잔여 한계(peri)**: `min_scale_gain=1.3` 게이팅으로 2차 실행 비율을 cbct_peri 27/37→10/37,
  seoul_peri 33/55→18/55로 크게 줄였지만, scale_gain 분포 자체가 pano/peri 간에 완전히
  분리되지 않아(cbct_peri 최댓값 1.965, seoul_peri 최댓값 1.604가 cbct_pano 최솟값 1.315보다
  큼) 일부 peri 이미지는 여전히 2차가 실행되고, 그 경우도 여전히 순손실로 측정됨(카테고리
  집계에 남아 있음). 더 엄격한 임계값이나 peri 전용 예외 처리는 추후 과제로 남겨둠.

## 측정 환경 / 추론 속도 / VRAM (최종, 2026-11-10)

- **PC 스펙**: GPU NVIDIA RTX A6000 (48GB VRAM, driver 595.91.07) / CPU AMD EPYC 9354 32-Core
  Processor x2소켓(64 physical core, 128 thread) / RAM 1.0TiB / PyTorch 2.10.0+cu128, CUDA 12.8.
  단일 GPU(배치=1) 기준, eager(비-export) 모델로 측정.
- 측정 방법: `predict_wrapper()`(production 기본값, `two_pass=True` + 위 게이팅) 1회 warm-up 후
  5회 반복 평균. `torch.cuda.synchronize()` + `reset_peak_memory_stats()`/`max_memory_allocated()`
  로 측정. 스크립트: `rfdetr/deploy/measure_predict_latency.py`.

| 시나리오 | 샘플 | 원본 해상도(H x W) | 2차 실행 여부 | **평균 지연(최종)** | peak VRAM |
|---|---|---|---|---|---|
| two-pass (pano, 게이팅 통과) | seoul_pano `000000_957466 pano 3.jpg` | 1468x2988 | 실행됨(1차+2차 forward 2회) | **319.9 ms** (최초 1069.7ms 대비 -70.1%, 3.34배) | **409 MiB** (최초 677 MiB 대비 -40%) |
| one-pass (peri, 게이팅으로 자동 1-pass) | seoul_peri `000008_242419 peri 1-2.jpg` | 1404x1876 | 생략됨(forward 1회) | **128.6 ms** (최초 109.4ms 대비 +17.6%) | **421 MiB** |

- pano가 peri 대비 약 2.5배 느린 건 1차(전체 이미지) + 2차(crop 재추론) 두 번의 forward pass가
  돌기 때문. VRAM은 이번 수정(아래 5단계)으로 mask를 GPU에서 전체 캔버스로 안 만들게 되면서
  오히려 peri와 비슷한 수준으로 줄어듦.
- peri는 최적화 이후에도 최초값(109.4ms)보다 소폭 느리다 — "1차 렌더링 생략" 최적화가 2차를
  생략하는 경로(대부분의 peri)에서는 가벼운 1차 호출 + 전체 렌더링 재호출, 두 번을 하게 만드는
  구조적 트레이드오프라서다(아래 최적화 히스토리 참고). pano를 1초 가까이에서 끌어내리는 게
  우선이라 받아들인 트레이드오프이며, peri 자체는 애초에 128.6ms로 충분히 빠르다.
- 48GB급 A6000 기준이며, 더 작은 VRAM(예: 8~16GB급) GPU에서의 여유분이나, 배치 처리/export
  (fp16, 고정 shape) 조합 시의 속도 변화는 이번 측정 범위에 포함하지 않음 -- 필요 시 추가 측정.

### 최적화 히스토리

아래 다섯 가지를 순서대로 적용해 pano 1069.7ms → 319.9ms, peri 109.4ms → 128.6ms가 됐다
(각 단계 측정 스크립트는 하위 항목 참고).

| 단계 | pano | peri | 핵심 원인 |
|---|---|---|---|
| 0. 최초 측정 | 1069.7 ms | 109.4 ms | - |
| 1. 1차 렌더링 생략 | 650.0 ms | 131.6 ms | 2차 진행 시 버려질 1차 렌더링을 매번 전부 그리고 있었음 |
| 2. `label_image` dtype(uint8) | 559.9 ms | 127.9 ms | `label_image` 병합이 int64 임시배열(~977MB)을 만듦 |
| 3. `label_image` box-restriction(numba) | 519.7 ms | 117.5 ms | 병합을 전체 캔버스가 아닌 인스턴스 박스 영역만 순회 |
| 4. 2차 렌더링 재사용(합성) | 379.7 ms | 122.2 ms | 2차가 crop 해상도로 이미 그려둔 결과를 버리고 전체 해상도에서 또 그리고 있었음 |
| 5. mask paste GPU→CPU 전송 순서 변경 | **319.9 ms** | **128.6 ms** | `(N,H,W)` 전체 캔버스를 GPU에서 만들어 통째로 전송(~65~85ms, ~123MB) 하고 있었음 |

#### 1. 1차 렌더링 생략 (2026-11-10)

- **원인 분석**(`rfdetr/deploy/profile_two_pass.py`, 위 pano 샘플 기준): 모델 forward 자체는
  **35.4ms**로 빠른데, `draw_preditions_boxes`의 전체 후처리(원본 해상도 캔버스에 치아별 mask
  paste + polygon contour 그리기)가 **496.3ms**로 대부분을 차지. `render=False`로 그리기 단계만
  꺼도 박스/FDI/점수 추출 자체는 **178.4ms** 남음(NMS, fdi_refinement 등) -- 즉 "그리기"만으로
  약 **318ms**를 쓰고 있었다.
- two-pass에서 1차의 렌더링 결과(color/mask 이미지)는 **2차가 실제로 진행되면 통째로 버려지는
  값**이다(crop_box/게이팅 판단에는 box/fdi/score만 필요). 그런데도 기존 코드는 1차마다 매번
  전체 렌더링을 돌리고 있었음.
- **수정**(`two_pass_predict`): 1차를 먼저 `render=False`로 가볍게 호출해 box/fdi/score만
  얻어 ROI/게이팅을 판단한다. 2차가 생략되는 경로(peri류)에서만 1차를 `render=True`로 다시
  호출해 최종 렌더링을 만든다 -- 즉 pano(2차 진행)는 1차 전체 렌더링을 아예 하지 않고,
  peri(2차 생략)는 대신 "가벼운 호출 + 전체 렌더링" 둘 다 하게 돼 약간 더 비싸진다.
- **효과**: pano(2차 진행) **1069.7ms → 650.0ms**(약 39% 단축). peri(2차 생략, 원래도 빠름)는
  109.4ms → 131.6ms로 소폭 증가(가벼운 1차 호출이 추가로 한 번 더 들어가기 때문) -- 체감상
  pano 쪽이 압도적으로 느렸던 문제를 해소하는 게 우선이라 수용 가능한 트레이드오프로 판단.

#### 2. `render_detection`의 `label_image` 병합 dtype (2026-11-10)

- **"paste_masks_in_image이 느린 거 아니냐"는 의심으로 재프로파일링**
  (`rfdetr/deploy/profile_paste_masks.py`): 실제로 GPU `paste_masks_in_image` 자체는 28개
  인스턴스 기준 **2.8ms**로 전혀 느리지 않았다(numba로 재구현해 비교해봐도 정확도는 동일하고
  오히려 CPU 왕복 때문에 12ms로 더 느려짐 -- numba 버전은 비교용으로만 남기고 채택 안 함,
  `paste_masks_in_image_by_numba`). 그럼 위에서 줄인 "렌더링 318ms"는 어디서 나는지 다시
  단계별로 쪼개서 측정(`rfdetr/deploy/profile_render_bottleneck.py`).
- **원인**: `rfdetr/datasets/visualize.py`의 `render_detection()`이 ``drawing_mode``(mask/polygon)
  와 무관하게 항상 계산하는 `label_image = np.max(labels[:, None, None] * masks, axis=0)`.
  `labels`가 int64라 bool `masks`와 곱하면 `(N, H, W)` int64 임시 배열이 생기는데, 28개 인스턴스
  x 1468x2988 기준 **약 977MB** 임시 배열 할당+스캔 — 이게 **160ms**로 `render_detection` 전체
  (polygon 모드 214ms)의 대부분을 차지하고 있었다. FDI 라벨은 항상 48 이하라 uint8로 충분.
- **수정**: `labels.astype(np.uint8)[:, None, None] * masks`로 곱한 뒤 결과만 원래 dtype으로
  되돌림(반환 dtype/값 동일, 임시 배열만 8배 작아짐 — 정확도 동일 확인, max abs diff 0).
- **효과**(같은 pano 샘플, `rfdetr/deploy/profile_render_bottleneck.py`):
  `label_image` 병합 160.0ms → **25.6ms**(6.2배), `render_detection`(polygon, 실사용 모드)
  214.4ms → **76.2ms**(2.8배, -138ms). End-to-end `predict_wrapper` 기준 pano
  650.0ms → **559.9ms**, peri 131.6ms → **127.9ms**(`rfdetr/deploy/measure_predict_latency.py`).
- 잔여: `drawing_mode='mask'`는 `apply_blending_mask`가 별도로 186ms 걸려(이번 범위 밖, mask 모드는
  현재 production에서 안 씀) 여전히 느림. `restored_instance_masks` 스택(`engine.py:794-798`)도
  33ms 정도 남아있음 — 다음에 더 줄이려면 이쪽.

#### 3. `label_image` 병합의 numba box-restriction (2026-11-10)

- **지적**: "`masks`의 bbox 정보는 이미 알고 있으니 유효 영역에만 처리하면 되지 않나" --
  `masks`는 `paste_masks_in_image`가 박스 영역에만 값을 쓰고 나머지는 전부 `False`로 들어오므로,
  `np.max(...)`가 매번 `(N, H, W)` 전체를 스캔하는 건 여전히 낭비. 박스 영역만 순회하는 numba
  커널로 교체(`rfdetr/datasets/visualize.py::merge_label_image_by_numba` /
  `_merge_label_image_kernel`, 인스턴스별 순차 루프 -- 겹치는 영역에 동시에 쓰는 걸 피하려고
  `prange`는 안 씀).
- **첫 시도에서 버그 발견**: 박스 경계로 `boxes_xy`(그려지는 detection box)를 썼더니
  `max abs diff = 47`로 틀림. 원인: segmentation은 `mask_boxes =
  expand_normalized_boxes(pred_boxes, segmentation_crop_box_scale=1.15)`로 **detection box보다
  15% 더 큰 박스**를 crop/paste 기준으로 쓰는데(`rfdetr/engine.py:547-555`), `render_detection`에는
  이 확장된 박스가 전달되지 않고 있었다 -- 즉 마스크의 실제 non-zero 영역이 `boxes_xy`보다 넓어서
  box-restriction이 그 바깥쪽 유효 픽셀을 잘라먹었다.
- **수정**: `engine.py`가 `paste_masks_in_image`에 실제로 넘긴 (render 좌표로 복원된) 박스를
  `mask_boxes_render`로 따로 추적해 `render_detection(..., mask_boxes_xy=...)`로 전달하고,
  `render_detection`은 이게 주어지면 numba box-restricted 경로를, 없으면(예: GT 시각화,
  박스에 제한 안 되는 `segmentation_mode='full_image'`) 항상-정확한 전체 스캔으로 fallback한다.
  수정 후 재검증: max abs diff = 0 (`rfdetr/deploy/profile_merge_label_image.py`).
- **효과**: `label_image` 병합 26.7ms → **6.1ms**(4.3배, uint8 dtype 수정판 대비). End-to-end
  `predict_wrapper` 기준 pano 559.9ms → **519.7ms**, peri 127.9ms → **117.5ms**.

#### 4. 2차 렌더링 재사용 — 전체 해상도 재-렌더링 제거 (2026-11-10)

- **단계별 재측정**(`rfdetr/deploy/profile_two_pass_stages.py`, `two_pass_predict()`를 단계별로
  그대로 재현해 측정, 같은 pano 샘플): 1차(render=False) 183ms, 2차 forward+postprocess
  (render=True, **crop 해상도**) 146ms, offset/paste 19ms까지는 합리적인데, 그 뒤에 **원본 전체
  해상도에서 `render_detection`을 한 번 더 호출하는 데만 75ms**(16.9%)가 또 들어가고 있었다.
- **원인**: `two_pass_predict()`는 2차를 `crop_img`에 대해 `render=True`로 돌려(그래야 2차 자체의
  `instance_results`/박스가 나온다) **crop 해상도로 이미 contour/label까지 그린 이미지를 만들어
  놓고도 그 결과(`color_image`)를 버린 뒤**, 좌표를 원본으로 옮긴 `inst2`만 가지고 원본 전체
  해상도 캔버스에서 `render_detection`을 처음부터 다시 호출하고 있었다(`predict.py:990-993`,
  수정 전). 이 두 번째 호출은 `mask_boxes_xy`를 넘기지 않아 (3)의 box-restriction도 못 받고,
  무엇보다 **`cv2.findContours`/`cv2.polylines`는애초에 박스 복원과 무관하게 인스턴스 마스크
  배열 전체(`image_h x image_w`) 크기를 스캔**하므로 (2)/(3) 어떤 최적화로도 못 줄이는 비용이었다
  (crop이 아닌 전체 pano 크기라 더 크다).
- **수정**: 2차의 crop 해상도 렌더링 결과(`color2_crop`, contour/label이 이미 그려진 이미지)를
  버리지 않고, 원본 전체 해상도 베이스 이미지(`to_magnitude_images(img)`)의 crop 위치에 그대로
  붙여넣는다(`display_image[y0i:y1i, x0i:x1i] = color2_crop`) — 두 번째 `render_detection` 호출
  자체를 없앴다. 픽셀 좌표계가 동일해(2차도 같은 crop 영역 기준으로 그렸으므로) 단순 대입으로
  충분하다. 부작용 하나: `render_detection`의 선 두께/폰트 크기는 그려지는 이미지의 폭 기준
  자동 스케일이라, crop(좁음) 기준으로 그리면 원본 기준보다 얇아진다 — `_predict_single_pass`에
  `polygon_thickness`/`label_font_scale` 인자를 추가해 원본 전체 폭 기준값을 명시적으로 넘겨
  시각적으로 동일하게 유지했다.
- **효과**: 이 스크립트 기준 443ms → 395ms. End-to-end `predict_wrapper` 기준 pano
  519.7ms → **379.7ms**(최초 1069.7ms 대비 누적 -64.5%, 2.82배). peri는 이 경로(2차)를 안 타므로
  변화 없음(122.2ms, 노이즈 수준).

#### 5. mask paste GPU→CPU 전송 순서 변경 (2026-11-10)

- **"그래도 320~380ms면 아직 느린데?"는 지적으로 1-pass draw_preditions_boxes를 모듈별로
  전부 쪼개서 측정**(`rfdetr/deploy/profile_onepass_postprocess.py`, 주요 함수
  monkeypatch + `np.stack`/`np.ascontiguousarray`/`torch_utils.to_numpy`/`cv2.resize`까지
  호출별로 추적): NMS(<1ms)/fdi_refinement(<1ms)/`paste_masks_in_image` 자체(3~6ms)는 전부
  안 느림 — 그런데 바로 다음 줄의 `torch_utils.to_numpy(pasted_masks > mask_logit_threshold)`
  호출 하나가 **65~85ms**로, `render_detection`(75~92ms)과 맞먹는 두 번째로 큰 비용이었다.
  지금까지 아무도 지목 안 한 지점.
- **원인**: `paste_masks_in_image`가 GPU에서 각 인스턴스를 박스 크기로 resize해 **이미 큰
  (K, image_h, image_w) 캔버스에 paste**해두고, 그 캔버스 전체를 CPU로 전송한다(28개 인스턴스
  x 1468x2988 기준 **~123MB**). 각 인스턴스의 실제 내용은 자기 박스 안에만 있는데
  (`pasted_masks_in_image_by_numba`를 만들 때도 확인했던 것과 같은 패턴), 그 바깥의 0 픽셀까지
  통째로 전송하고 있었다. 직접 검증: 같은 크기의 bool 텐서 `.cpu().numpy()`만 단독으로 재도
  65~87ms로 재현됨(pinned memory로는 4.7ms, 박스 크기만 전송하면 0.7ms로 100배 차이 -- 즉
  compute가 아니라 **전송량 자체가 병목**).
  - pinned memory는 검토했으나 기각: 버퍼를 매번 새로 할당하면 page-locking 자체가 똑같이
    ~67ms 걸리고, 같은 크기를 재사용할 때만 빨라짐(PyTorch 내부 캐시) -- 이미지 크기가 매번
    달라지는 실사용 환경에서는 전역 pinned 버퍼 풀을 직접 관리해야 해서 복잡도 대비 이득이
    불확실함. 반면 전송량 자체를 줄이면 pinned 여부와 무관하게 빠르다(위 0.7ms는 일반
    `.cpu()`로 측정한 것).
- **수정**(`rfdetr/models/segmentation_head.py::paste_masks_to_numpy`, 신규): GPU에서는
  인스턴스별로 박스 크기로 resize + threshold까지만 하고(여전히 작음), **큰 캔버스에는 paste하지
  않는다**. 모든 인스턴스의 작은 결과를 하나의 flat 버퍼로 합쳐 **1회만** CPU로 전송한 뒤, 이미
  0으로 채워둔 `(K, image_h, image_w)` numpy 배열에 인스턴스별로 슬라이스 대입해 캔버스를
  CPU에서 만든다. 최종 shape/dtype/값은 기존과 완전히 동일(`instance_results['masks']` 등
  외부 계약 불변, reverse registration 등 다른 소비자 영향 없음) -- `paste_masks_in_image`
  자체는 그대로 두고 다른 호출부들(`tools/compare_own1022.py` 등)은 안 건드림.
  검증: 기존 경로와 비교해 **mismatch 0 픽셀**(`rfdetr/deploy/profile_paste_masks.py`).
- **효과**: 이 전송 단계만 82.1ms → **13.2ms**(6.2배). End-to-end `predict_wrapper` 기준 pano
  one-pass 316.8ms → **221.2ms**, two-pass 419.7ms → **319.9ms**(최초 1069.7ms 대비 누적
  **-70.1%, 3.34배**), peri one-pass 142.5ms → **128.6ms**. peak VRAM도 pano 677MiB →
  **409MiB**(-40%, 더 이상 GPU에 큰 캔버스를 만들지 않으므로).
- 잔여: `render_detection`(75~92ms, `cv2.findContours`/`polylines`가 여전히 박스가 아닌
  `image_h x image_w` 전체 배열을 스캔)은 이번 수정 범위 밖 -- 같은 철학(박스 단위로만 처리)을
  적용하려면 contour-finding 자체를 박스-로컬로 바꾸고 좌표를 offset해야 해서 더 큰 리팩토링이
  필요함(`instance_results['masks']` 계약을 유지한 채로 해야 함). 추후 과제로 남겨둠.
