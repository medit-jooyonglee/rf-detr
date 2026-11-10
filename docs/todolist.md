# TODO

## two-pass 추론 (teeth_roi crop 후 재추론) — 배포 보류 (2026-11-09)

- 배경: pano 이미지에서 1차 전체이미지 추론 → 1차 검출 bbox 기준 crop → 고해상도 재추론하는
  two-pass 옵션을 `predict.py`에 구현(`two_pass_predict`, `--two-pass`). peri류는 crop이 거의
  안 줄어들면 자동으로 1-pass로 끝남(`roi_shrink_ratio`).
- 평가(`outputs/eval_fdi_refinement/`): cbct_pano는 one-pass 대비 전 지표 개선. 반면
  seoul_pano/peri 계열은 dice/iou가 오히려 one-pass보다 낮아짐.
- 원인: ROI 좌표 버그는 아님(GT-box oracle crop으로 재평가해도 seoul_pano는 여전히 열세 —
  `summary_gt_crop_vs_pred_crop.json`). 실제 원인 두 가지를 특정함:
  1. 2차(crop 후 재추론)는 1차와 confidence 캘리브레이션이 달라져, 정확히 맞춘 치아가 공용
     threshold(0.5)에 걸려 사라짐 (`000004_993944 pano 1.jpg`: 31/41번 사례).
  2. 1차 검출이 한쪽 구치부를 놓치면 ROI 중심(x)이 그쪽으로 쏠리는 불안정성 확인.
- 적용한 완화책(코드 반영 완료, `rfdetr/deploy/predict.py` + `rfdetr/datasets/xray_crop_aug.py`):
  1. pass1/pass2 confidence threshold 분리 (pass1=0.5 고정, pass2 기본 0.35).
  2. ROI를 11/21(상악 중절치) 중심 대칭 crop으로 변경(`teeth_roi_box_center_aware`).
  - 효과: seoul_pano/seoul_peri/cbct_peri 전 지표 개선, cbct_pano는 noise 수준 유지
    (`summary_two_pass.json`).
- 잔여 이슈: 완화책 적용 후에도 seoul_pano는 box+FDI exact-match 기준 AP가 여전히 one-pass보다
  낮음(0.858 vs 0.920 — `pr_curve_by_category_summary.json`). crop+리사이즈로 인한 모델 자체의
  분류/스케일 불안정성으로, 후처리 튜닝만으로는 완전히 못 닫는 격차.
- **결정: two-pass 배포 적용 보류.** 코드는 `--two-pass` 옵션으로 남겨두되(기본 off), 프로덕션
  기본 파이프라인은 one-pass 유지. 육안 검증(`render_two_pass_compare.py` → `*_v2.jpg`)상으로는
  teeth-roi(two-pass) 방식 자체의 방향성은 더 낫다고 판단 — 다만 아래 재개 조건을 거쳐 데이터
  기준으로 더 면밀히 재평가한 뒤 **배포 기본 옵션으로 승격**하는 것을 목표로 한다.
- 재개 조건(다음에 할 일):
  1. `teeth_roi_box_center_aware`를 학습용(`train=True`)으로 확장: 11/21 중심 앵커는 유지하되
     margin/shift는 기존처럼 랜덤화하고, 추가로 center(cx) 자체도 jitter해서 "1차 추론이 11/21을
     놓치거나 틀리는" 불완전 검출 상황까지 학습 분포에 포함시킨다. 즉 teeth_roi 학습
     augmentation을 "GT 완전 합집합"이 아니라 "실제 1차 추론 분포(불완전 검출 + center 불안정)"에
     맞춰 재설계.
  2. 위 augmentation으로 재학습 → `pr_curve_two_pass_by_category.py`로 category별 재평가, 특히
     seoul_pano box+FDI AP(현재 0.858 vs one-pass 0.920) 격차가 좁혀지는지 확인.
  3. 단, 이 변경은 "crop 중심 불안정성"만 겨냥한 것이라 image_6에서 확인된 18↔47(사랑니) 순수
     FDI 오분류처럼 center-aware와 무관한 오류는 별도 원인 분석이 필요할 수 있음 — 재평가 시 함께
     육안 확인.
  4. 재평가 결과가 one-pass 대비 전 카테고리에서 동등 이상이면 `--two-pass` 기본값을 on으로
     바꿔 배포 기본 파이프라인으로 승격.
- 관련 산출물: `rfdetr/deploy/evaluate_two_pass.py`, `debug_two_pass_roi.py`,
  `debug_two_pass_center.py`, `pr_curve_two_pass.py`, `pr_curve_two_pass_by_category.py`,
  `render_two_pass_compare.py` / `outputs/eval_fdi_refinement/*.json`, `*.png`,
  `two_pass_debug/*.jpg`.

## 후속: export(실배포) 기준 one-pass vs two-pass 재평가 (2026-11-10)

- 배경: 위 재평가는 전부 one-pass를 eager 모델 + 동적(가변 해상도) 전처리로 측정했는데, 실제
  배포는 `torch.jit`로 **고정 입력 크기** export해서 쓴다(`export_libtorch`,
  `used_export`/`EXPORT_FIX_SIZE`). "학습 전처리(긴 변 640 등방 + stride64 pad)와 동일하게 만든
  export 결과물"로 다시 비교해야 실제 배포 성능과 맞다는 지적에 따라 재평가.
  (`rfdetr/deploy/evaluate_export_vs_twopass.py`, 체크포인트
  `peri_data_fusion_1106_ce/checkpoint_best_ema.pth`)
- **export 고정 shape는 pano/peri 따로**: export는 가변 해상도를 못 받아 고정 shape 1개가
  필요한데, pano(가로로 긴 전악)와 peri(거의 정사각/세로)는 원본 종횡비가 완전히 달라 하나로
  퉁치면 한쪽이 손해.
  - PANO_EXPORT_SHAPE=(H384, W640): 실측 종횡비(2.0~2.25 위주)에서 긴 변(W)=640, 짧은 변(H)
    실측 중앙값(320) + 여유. 학습과 거의 동일 배율.
  - PERI_EXPORT_SHAPE=(H512, W640): **중요 발견** — 처음엔 "어떤 종횡비든 안 잘리게" (640,640)으로
    잡았더니 cbct_peri/seoul_peri dice가 오히려 동적 처리 대비 ~5~8%p 급락. eager 모델에
    export와 똑같은 (640,640) 고정 캔버스만 줘도 동일하게 떨어지는 걸 확인해 **export 버그가
    아니라 "학습 때 안 보던 과도한 여백(padding) 자체가 분포 이탈"** 임을 실측으로 확인. 실측
    중앙값(640x512)에 맞춰 (512,640)으로 줄이니 동적 베이스라인을 회복(오히려 소폭 상회:
    peri 전체 dice 0.742 vs 0.72대).
  - 시사점: 추론 전처리를 "자르지만 않으면 되는" letterbox로 안일하게 접근하면 안 되고, 학습
    분포와 비슷한 수준의 padding 비율을 맞춰야 한다 — **현재 프로덕션 export 기본값
    `EXPORT_FIX_SIZE=(384,704)`가 실제로 peri류 추론에도 쓰이고 있다면, 위와 같은 이유로 peri
    정확도가 저평가되고 있을 가능성이 있다** (384x704는 peri 실측 중앙값 640x512보다 더 작고
    종횡비도 다름 -- 별도 확인 필요, 이번 범위에는 포함 안 함).
- **재평가 결과** (`outputs/eval_fdi_refinement/summary_export_vs_twopass.json`, dice/iou/
  boundary_f1/contact_f1, one-pass는 위 export 모델 기준):

  | 카테고리 | n | one-pass(export) | two-pass |
  |---|---|---|---|
  | cbct_pano | 42 | 0.865/0.790/0.739/0.733 | **0.888/0.826/0.828/0.820** (two-pass 전 지표 승) |
  | seoul_pano | 39(GT오류 1건 제외) | **0.830/0.756**/0.672/0.671 | 0.808/0.744/**0.728/0.738** (dice·iou는 one-pass, boundary·contact는 two-pass — 트레이드오프) |
  | cbct_peri | 37 | **0.781/0.742/0.540/0.383** | 0.623/0.591/0.473/0.336 (one-pass 전 지표 승) |
  | seoul_peri | 55 | **0.712/0.672/0.300**/0.249 | 0.653/0.619/0.319/**0.262** (one-pass 3개 승, contact만 two-pass) |

- **해석**:
  1. **two-pass가 전 지표에서 확실히 이기는 건 cbct_pano뿐.**
  2. seoul_pano는 진짜 트레이드오프 — two-pass는 전체 영역 겹침(dice/iou)은 약간 손해, 대신
     치아 경계/인접 치아 분리 정확도(boundary_f1/contact_f1,애초에 two-pass를 만든 목적과
     직결)는 뚜렷이 개선. GT 오류 1건(000004) 포함/제외해도 결론 동일.
  3. **peri(cbct/seoul 둘 다)는 이번에 교정된 export 기준으로는 one-pass가 명확히 우위.**
     `roi_shrink_ratio`(0.85)가 걸러내지 못하고 two-pass를 실행한 비율이 cbct_peri 27/37,
     seoul_peri 33/55로 꽤 높은데, 그 2차 재추론이 실제로는 손해였다는 뜻 — 이전
     `summary_two_pass.json`에서 "peri도 noise 수준"이라고 봤던 건 one-pass 쪽 측정(동적 eager)이
     export 배포 조건과 달라서 생긴 착시였다.
- **결론/권고**: two-pass를 배포 기본값으로 올린다면 **pano 입력에만 한정**하는 게 데이터에
  부합한다(cbct_pano는 확실한 이득, seoul_pano는 경계정확도 우선이면 이득). peri는 교정된
  one-pass-export가 모든 면에서 우위이므로 전면 승격은 데이터와 맞지 않음 — `roi_shrink_ratio`
  휴리스틱만으로는 peri를 충분히 걸러내지 못하므로, 승격 시 카테고리 기반(또는 더 보수적인
  shrink 기준) 분기 추가가 필요.
- 관련 산출물: `rfdetr/deploy/evaluate_export_vs_twopass.py`,
  `outputs/eval_fdi_refinement/summary_export_vs_twopass.json`,
  `outputs/export/pano_peri_1106_ce/{pano,peri}_model_libtorch{,_info}.{pt,json}`.

## 후속: two-pass 기본 승격 + 속도 최적화 (2026-11-10)

- **배포 결정**: `predict.py`의 `two_pass_predict()`에 `min_scale_gain`(기본 1.3) 게이팅 추가
  (`roi_shrink_ratio`와 함께 둘 다 통과해야 2차 실행). pano는 scale_gain이 항상 1.3 이상(실측
  cbct_pano/seoul_pano 최솟값 1.315/1.518)이라 거의 100% 2차 실행, peri는 중앙값이 1.2대라
  대부분 자동으로 1-pass로 떨어짐(cbct_peri 27/37→10/37, seoul_peri 33/55→18/55). 이에 따라
  `two_pass=True`를 `predict_wrapper`/`main`의 기본값으로 승격.
- **잔여 한계**: scale_gain 분포가 pano/peri 간 완전히 분리되지 않아(겹치는 구간 있음) 일부
  peri는 여전히 2차가 실행되고 그 경우 순손실로 측정됨 -- 더 엄격한 임계값/peri 전용 예외는
  추후 과제.
- **속도 최적화**: two-pass가 유난히 느리다(~1초)는 지적에 프로파일링(`rfdetr/deploy/
  profile_two_pass.py`)해보니 모델 forward는 35ms인데 `draw_preditions_boxes`의 원본 해상도
  mask paste+contour 그리기가 ~500ms. 2차가 실제로 진행되면 1차의 렌더링 결과는 버려지는데도
  매번 전체 렌더링을 하고 있던 게 원인 -- 1차를 `render=False`로 먼저 돌려 box/fdi만 얻고,
  2차가 생략될 때만 1차를 다시 `render=True`로 재호출하도록 수정. pano 두-pass 평균 지연
  1069.7ms→650.0ms (peri는 109.4ms→131.6ms로 소폭 증가, 수용 가능한 트레이드오프).
- 상세 수치/방법론: `docs/performance.md`.
- 관련 산출물: `rfdetr/deploy/scale_gain_stats.py`, `profile_two_pass.py`,
  `render_final_samples.py` / `outputs/eval_fdi_refinement/final_result_samples/*.jpg`.

## 후속: 후처리 추가 병목 — `render_detection`의 `label_image` 병합 dtype (2026-11-10)

- "`paste_masks_in_image`이 느린 거 아니냐"는 의심(treq-pass 그리기 1초 가까이 걸리는 문제
  재지적)으로 `paste_masks_in_image`만 따로 재프로파일링(`rfdetr/deploy/profile_paste_masks.py`,
  numba 재구현 `paste_masks_in_image_by_numba`도 비교용으로 추가) → 실제론 GPU에서 2.8ms밖에
  안 걸림, numba는 오히려 더 느려(12ms, CPU 왕복 비용) 채택 안 함. 즉 이전에 "paste 끄면 136ms
  절약"으로 봤던 건 다른 걸 같이 바꾼 착시였음.
- 단계별 재분해(`rfdetr/deploy/profile_render_bottleneck.py`)로 진짜 원인 특정: `visualize.py`의
  `render_detection()`이 drawing_mode와 무관하게 항상 계산하는
  `label_image = np.max(labels[:,None,None]*masks, axis=0)`가 int64 dtype 때문에 28인스턴스
  x 1468x2988 기준 ~977MB 임시 배열을 만들어 160ms를 씀(render_detection 전체 214ms의 대부분).
- 수정: FDI 라벨(≤48)은 uint8로 충분 → 곱셈만 uint8로 하고 결과를 원래 dtype으로 되돌림(반환값
  동일, 정확도 동일 확인). `render_detection` 214ms→76ms, end-to-end `predict_wrapper` pano
  650.0ms→**559.9ms**, peri 131.6ms→**127.9ms**.
- 관련 산출물: `rfdetr/deploy/profile_paste_masks.py`, `profile_render_bottleneck.py`,
  `measure_predict_latency.py`, 수정은 `rfdetr/datasets/visualize.py`.

## 후속: `label_image` 병합 numba box-restriction + 버그 수정 (2026-11-10)

- "masks의 bbox 정보는 이미 아니까 유효 영역만 처리하면 되지 않냐"는 지적으로
  `label_image = np.max(...)`를 전체 캔버스 대신 각 인스턴스 박스 영역만 순회하는 numba 커널로
  교체(`rfdetr/datasets/visualize.py::merge_label_image_by_numba`).
- **첫 구현은 틀렸음**(max abs diff 47): 박스 경계로 `boxes_xy`(detection box)를 썼는데, 실제
  mask가 차지하는 영역은 `mask_boxes = expand_normalized_boxes(pred_boxes,
  segmentation_crop_box_scale=1.15)`로 **15% 더 큰 박스** 기준이라(`engine.py:547-555`) 마스크
  일부가 잘렸음. `engine.py`가 `paste_masks_in_image`에 실제로 넘긴 박스를
  `mask_boxes_render`로 추적해 `render_detection(..., mask_boxes_xy=...)`로 전달하도록 수정 후
  재검증(max abs diff 0, `rfdetr/deploy/profile_merge_label_image.py`).
- **효과**: `label_image` 병합 26.7ms→6.1ms. End-to-end pano 559.9ms→**519.7ms**,
  peri 127.9ms→**117.5ms**(`docs/performance.md`).
- 교훈: segmentation crop box(`mask_boxes`)와 detection box(`boxes`)가 다르다는 걸 깜빡하면
  "박스 안에서만 처리" 류 최적화는 조용히 정확도를 깎는다 -- 박스 기반 최적화를 또 넣을 땐
  어느 박스(denorm 전/후, scale 전/후)를 기준으로 하는지부터 확인할 것.

## 후속: two-pass 2차 렌더링 재사용 — 전체 해상도 재-렌더링 제거 (2026-11-10)

- "pano 519.7ms 아직도 느린데 어디가 병목이냐"는 지적으로 `two_pass_predict()`를 단계별로
  그대로 재현해 측정(`rfdetr/deploy/profile_two_pass_stages.py`): 1차/2차 forward+postprocess는
  합리적인데, 그 뒤 **원본 전체 해상도에서 `render_detection`을 한 번 더 부르는 데만 75ms**가
  더 들어감.
- 원인: 2차를 `crop_img`에 `render=True`로 돌려 **crop 해상도로 이미 contour/label까지 그려놓고
  그 결과를 버린 뒤**, 원본 좌표로 옮긴 박스/마스크만 가지고 원본 전체 해상도 캔버스에서
  `render_detection`을 처음부터 다시 호출하고 있었다(`predict.py`, 수정 전). 이 두 번째 호출은
  `cv2.findContours`/`polylines`가 (3)의 box-restriction과 무관하게 애초에 캔버스 전체 크기
  배열을 스캔하므로, 이전 최적화들로는 전혀 줄지 않는 비용이었다.
- 수정: 2차가 이미 그려둔 crop 해상도 렌더링 결과(`color2_crop`)를 버리지 않고 원본 전체 해상도
  베이스 이미지의 crop 위치에 그대로 붙여넣는 것으로 교체 — 두 번째 `render_detection` 호출
  자체를 없앴다. 선 두께/폰트가 그려지는 이미지 폭 기준 자동 스케일이라 crop 기준으로 그리면
  얇아지는 부작용이 있어, `_predict_single_pass`에 `polygon_thickness`/`label_font_scale`을
  추가해 원본 전체 폭 기준값을 명시적으로 넘기도록 함(시각적 동일성 유지, 육안 확인 완료).
- 효과: End-to-end pano **519.7ms → 379.7ms**(최초 1069.7ms 대비 누적 -64.5%, 2.82배). peri는
  이 경로를 안 타므로 변화 없음.
- 관련 산출물: `rfdetr/deploy/profile_two_pass_stages.py`, 수정은 `rfdetr/deploy/predict.py`
  (`_predict_single_pass`, `two_pass_predict`).

## 후속: mask paste GPU→CPU 전송 순서 변경 (2026-11-10)

- "pano 320~380ms도 아직 느린데?" 지적으로 1-pass 후처리를 모듈별로 전부 쪼개서 측정
  (`rfdetr/deploy/profile_onepass_postprocess.py`): NMS/fdi_refinement/`paste_masks_in_image`
  전부 안 느렸는데, 바로 다음 `torch_utils.to_numpy(pasted_masks > threshold)` 한 줄이
  **65~85ms**로 `render_detection`(75~92ms)과 맞먹는 두 번째로 큰 비용이었음 — 지금까지 아무도
  안 지목한 지점.
- 원인: `paste_masks_in_image`가 GPU에서 인스턴스별 박스 크기로 resize한 뒤 이미 큰
  `(K, image_h, image_w)` 캔버스에 paste해두고, 그 전체(28인스턴스 x 1468x2988 ≈ 123MB)를
  CPU로 전송. 각 인스턴스 내용은 자기 박스 안에만 있는데 바깥 0 픽셀까지 통째로 전송 중이었음.
  pinned memory도 검토했으나(동일 크기 재사용 시만 빠르고 이미지 크기가 매번 바뀌는 환경이라
  기각) 전송량 자체를 줄이는 쪽으로 결정.
- 수정: `paste_masks_to_numpy`(신규, `rfdetr/models/segmentation_head.py`) — GPU에서는 박스
  크기로 resize+threshold까지만(여전히 작음) 하고, 전체 인스턴스를 flat 버퍼 하나로 합쳐 1회만
  전송한 뒤 CPU에서 0-배열에 슬라이스 대입으로 캔버스를 만든다. 최종 shape/dtype/값은 완전
  동일(`instance_results['masks']` 등 외부 계약 불변) — 기존 `paste_masks_in_image`은
  그대로 두고 다른 호출부(`tools/compare_own1022.py` 등)는 안 건드림. mismatch 0 픽셀 검증
  (`rfdetr/deploy/profile_paste_masks.py`).
- 효과: 전송 단계 82.1ms→13.2ms(6.2배). End-to-end pano one-pass 316.8ms→**221.2ms**,
  two-pass 419.7ms→**319.9ms**(최초 1069.7ms 대비 누적 -70.1%, 3.34배), peri
  142.5ms→**128.6ms**. peak VRAM pano 677MiB→**409MiB**(-40%).
- 테스트: `test_draw_predictions_pastes_only_positive_nms_queries`가
  `paste_masks_in_image`를 monkeypatch하고 있어서 `paste_masks_to_numpy`로 갱신(함수 변경에
  따른 당연한 수정). `test_crop_loss_includes_outer_false_positive_ring`은 이번 변경과 무관한
  기존 실패(`KeyError: 'size'`, lwdetr.py) — 변경 전 코드에서도 동일하게 실패함을 확인.
- 잔여(다음 과제): `render_detection`의 `cv2.findContours`/`polylines`(75~92ms)는 여전히
  박스가 아닌 전체 캔버스를 스캔 — 같은 철학을 적용하려면 contour-finding을 박스-로컬로 바꾸고
  좌표 offset하는 더 큰 리팩토링 필요(`instance_results['masks']` 계약 유지하면서).
- 관련 산출물: `rfdetr/deploy/profile_onepass_postprocess.py`, `profile_paste_masks.py`(갱신),
  `measure_predict_latency.py`(갱신, one-pass/two-pass 동일 이미지 비교 케이스 추가).
