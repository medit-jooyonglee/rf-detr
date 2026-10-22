# vanilla RF-DETR 대비 변경 사항 (xray 치아 instance segmentation)

- 기준: upstream `roboflow/rf-detr` `0c1e833` (2025-11-13, PR #453 merge) 이후 fork
- 대상 학습 경로: `rfdetr/cli/main.py::train_from_xray_teeth_dir` (`dataset_file='xray_teeth'`, `segmentation_head=True`)
- 사소한 수정(경로, 로그, 오타, 방어 코드 일부)은 제외하고 구조·학습·추론 결과에 영향을 주는 변경만 정리한다.

## 요약

| 구분 | vanilla RF-DETR | 이 저장소 | 주요 커밋 |
|---|---|---|---|
| 백본 | DINOv2 windowed small(384ch, 6 head) | **DINOv2 windowed tiny(192ch, 3 head)** 추가 | `1feab29` |
| 마스크 예측 | query × 1/4 해상도 feature 내적 (`full_image`) | `full_image` + **crop-and-resize refinement head** (`crop_and_resize`) | `202f1fd`, `87a4093` |
| 마스크 loss | point-sampled CE + Dice | + **boundary BCE**, refinement용 CE/Dice/boundary/**outer ring** | `2c6964c`, `202f1fd` |
| 입력 | 정사각 resize | 너비 640, stride 64 배수의 **비정사각** 입력 (`square_resize=False`) | `902838e`, `d023e73` |
| 백본 버그 | windowed attention에서 W 패치 수를 H로 계산 | 수정 (비정사각 입력에서만 드러남) | `9b5f1d0` |
| export | ONNX 중심 | **CoreML / libtorch / FP16** 대응 | `391a56e`, `c467d52`, `3587bfc` |
| 데이터 | COCO / Roboflow | 파노라마 xray 3종 + 금속 아티팩트 증강 + PA/ROI crop 증강 | `2144498`, `2c6964c`, `5f93518` |

## 1. 백본 경량화: `dinov2_windowed_tiny`

커밋 `1feab29` (dino tiny추가), `9b5f1d0` (num patch h, w 버그 수정)

- `rfdetr/models/backbone/dinov2_configs/dinov2_tiny.json` 추가. `hidden_size=192`, `num_attention_heads=3`, `num_hidden_layers=12`, `mlp_ratio=4`.
- `size_to_width["tiny"] = 192`. 인코더 파라미터는 small 대비 약 1/4이다(ViT 블록당 약 `12·d²`: 384² → 192²).
- 학습 설정(`main.py`): `RFDETRSmall` 골격에 `encoder='dinov2_windowed_tiny'`, `patch_size=16`, `num_windows=4`(small 기본 2), `num_queries=100`(기본 300), `num_select=30~35`.
- **주의: tiny는 공개 DINOv2 가중치가 없다.** `patch_size=16`이라 hub 가중치 로딩도 꺼진다. 인코더는 처음부터 xray 데이터로 학습되고, 이후 학습은 자체 체크포인트(`outputs/pretrained/.../checkpoint.pth`)에서 이어간다. COCO 사전학습 RF-DETR 가중치는 인코더 shape이 달라 적용되지 않는다.
- `dinov2_with_windowed_attn.py` 버그 수정: window 분할 reshape의 마지막 축이 `num_h_patches_per_window`로 되어 있었다. 정사각 입력에서는 H=W라 문제가 없지만, 비정사각 입력(너비 640 기준 가변 높이)에서는 window가 잘못 잘린다. `num_w_patches_per_window`로 수정했다.

## 2. Crop-and-resize segmentation refinement

커밋 `202f1fd` (crop and resize head 추가, coarse hint 옵션), `87a4093`, `3587bfc`(FP16), `abaaaf3`(jit 지연 로딩)

vanilla RF-DETR에는 없는 2단계 마스크 예측이다. `segmentation_mode='crop_and_resize'`일 때만 동작하고, 기본값 `full_image`는 vanilla와 같다.

### 2.1 구조

```
image ─► backbone/decoder ─► boxes, query feature
                               │
                               ▼
               SegmentationHead (vanilla, 1/4 해상도) ─► coarse mask  (pred_masks_coarse)
                               │
  box ×1.15 확장 ─► 입력 영상 crop (128×64, RGB)  ┐
                 └► coarse 확률 crop (1ch, hint) ┴─► CropAndResizeSegmentationHead (U-Net, 4ch)
                                                         │
                                                         ▼
                                 ROI mask logit (128×64) ─► 박스 위치에 paste ─► threshold
```

- `CropAndResizeSegmentationHead` (`rfdetr/models/segmentation_head.py`): 32-64-128 채널의 2단 U-Net이다. 입력은 4채널(영상 3 + coarse hint 1), 출력은 1채널 binary logit이다. 완전 합성곱이라 가중치 shape은 crop 크기와 무관하다.
- **refinement head는 백본 feature를 쓰지 않는다.** 입력 영상 텐서(`samples.tensors`)에서 직접 crop한 픽셀과 coarse 확률만 본다. coarse mask는 `detach()`되므로 refinement loss는 백본/디코더로 역전파되지 않는다.
- crop은 **모델 입력 텐서**(너비 640으로 resize된 영상)에서 `grid_sample`로 뽑는다. 원본 해상도 영상에서 뽑는 것이 아니다.

### 2.2 파라미터 (`ModelConfig` / `TrainConfig`)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `segmentation_mode` | `full_image` | `crop_and_resize`로 켜야 refinement 사용 |
| `segmentation_crop_size` | `(128, 64)` | ROI crop 크기 (H, W). 치아가 세로로 길어서 2:1. 학습·추론에서 같아야 한다 |
| `segmentation_crop_box_scale` | `1.15` | crop 전에 박스를 확장하는 비율. 박스 오차와 경계 바깥 문맥을 확보한다 |
| `coarse_hint_scale` | `0.35` | coarse 확률 `p`를 `(2p-1)·0.35`로 넣는다. 힌트를 약하게 줘서 영상 경계를 보게 한다 |
| `coarse_hint_dropout` | `0.30` | 학습 시 ROI 단위로 힌트를 0으로 지운다. 힌트를 그대로 베끼는 것을 막는다 |
| `refine_outer_boundary_loss_coef` | `4.0` | refinement outer-ring loss 가중치 |

### 2.3 학습 / 추론 경로 차이

| | 학습 (`_predict_target_crop_masks`) | 추론 (`_predict_crop_masks`) |
|---|---|---|
| crop 박스 | **GT 박스** × 1.15 | **예측 박스** × 1.15 |
| coarse hint | GT 박스와 IoU + 0.25·class score가 가장 높은 query의 coarse mask | 해당 query의 coarse mask |
| 출력 | `pred_mask_crops` (loss용) | `pred_masks` (ROI logit) |

학습은 GT 박스 기준이라 추론 시 예측 박스 오차는 `1.15` 확장 여유로만 흡수된다. 박스가 실제 치아보다 작게 나오면 마스크도 확장 박스 밖으로 나갈 수 없다(paste 범위가 확장 박스).

### 2.4 후처리

- `PostProcess`(COCO 평가)와 `engine.py::draw_preditions_boxes`(시각화/predict)는 ROI logit을 확장 박스 크기로 bilinear resize해서 캔버스에 붙여넣은 뒤(`paste_masks_in_image`) threshold한다.
- `paste_masks_at_original_size=True`면 원본 해상도 캔버스에 바로 붙여넣는다. 이진화된 label map을 다시 resize할 때 생기는 계단 현상을 피할 수 있다.

### 2.5 경계 정확도가 오르는 이유

1. vanilla 마스크는 1/4 해상도 feature에서 나오므로, 입력 640 기준으로도 4px 단위로 뭉개진다.
2. refinement는 치아 하나를 고정 크기(128×64)로 다시 보므로, 작은 치아일수록 상대 해상도가 높아진다.
3. boundary·outer-ring loss가 경계 픽셀에 학습을 집중시킨다(3절).

### 2.6 `segmentation_crop_size` 조정 시 고려 사항

- crop은 입력 텐서(너비 640)에서 뽑으므로, ROI의 **입력 픽셀 수보다 crop 크기가 크면** 업샘플링일 뿐이라 새 정보가 없다.
  - 파노라마 전체 영상: 치아 박스가 입력 기준 대략 폭 30~50px, 높이 100~150px 수준이다. 128×64가 이미 업샘플링 영역이라 크기를 키워도 이득이 작다.
  - PA / teeth_roi crop 입력: 치아 3~5개가 640 안에 들어와 치아당 폭 100px 이상, 높이 300px 이상이 된다. 128×64는 다운샘플링이 되므로 **크기를 키우면 경계 이득이 있을 수 있다.**
- U-Net이 max-pool을 2번 하므로 H, W는 4의 배수로 둔다. 세로:가로 2:1을 유지하려면 `(192, 96)`, `(256, 128)` 순으로 올린다.
- refinement 연산량은 픽셀 수에 비례한다. `(160, 96)`은 약 1.9배, `(192, 96)`은 약 2.3배다.
- 가중치는 그대로 로드되지만 학습 스케일이 달라지므로, 크기를 바꾸면 fine-tuning이 필요하다.

## 3. 마스크 loss 추가

커밋 `2c6964c` (boundary loss 추가, 메탈 아티팩트 augmentation), `202f1fd`

| loss | 대상 | 정의 | 가중치 |
|---|---|---|---|
| `loss_mask_boundary` | coarse mask (vanilla head) | point sample BCE × (1 + 4·boundary). boundary = GT dilate(5×5) − erode(5×5) | `mask_boundary_loss_coef=2.0` |
| `loss_mask_refine_ce` | refinement ROI | 픽셀 BCE 평균 | `mask_ce_loss_coef` |
| `loss_mask_refine_dice` | refinement ROI | soft Dice | `mask_dice_loss_coef` |
| `loss_mask_refine_boundary` | refinement ROI | BCE × (1 + 4·boundary) | `mask_boundary_loss_coef` |
| `loss_mask_refine_outer` | refinement ROI | GT 바로 바깥 ring(dilate − GT)의 BCE 평균 | `refine_outer_boundary_loss_coef=4.0` |

- boundary loss는 금속·보철 아티팩트처럼 경계 대비가 낮은 경우를 겨냥했다. 일반 CE/Dice는 쉬운 내부 픽셀에 지배된다.
- outer-ring loss는 마스크가 바깥으로 번지는 것을 억제한다.
- **부작용:** 이웃 치아와 겹치는 영역이 GT에서 빠져 있으면(라벨이 상호 배타적이면) 그 영역이 outer ring에 들어가 강하게 억제된다. 파노라마 중첩 영역 예측이 잘리는 원인 중 하나다. 데이터셋별 중첩 라벨 비율은 README의 "중첩 라벨 방식" 항목을 참고한다.
- Hungarian matcher의 mask cost와 기존 `loss_masks`는 `pred_masks_coarse`를 쓴다. refinement는 매칭에 관여하지 않는다.

## 4. 입력 / 데이터 파이프라인

커밋 `902838e`, `2144498`, `c257702`, `5f93518`, `0e18941`

- 데이터셋: `rfdetr/datasets/xraypanoramic.py` (`XrayPnoaramicInstance`, `XrayPnoaramicInstanceCoco`)
  - ① Kaggle Teeth Segmentation, ② Tufts Dental 2022(`util/dataloader.py::tufts_to_coco`로 변환), ③ 자체 labelme→COCO 데이터. ③은 train 전용이다.
  - 클래스: universal 번호 1~32 + 배경 (`num_classes=32`). 시각화/출력은 FDI로 매핑한다(`label_to_fdi`).
  - 입력은 너비 640, 높이 stride 64 배수로 비율을 유지한다(`square_resize=False`). 평가 GT json은 원본 좌표 기준이다(`*_v2.json`, `docs/troubleshoots/xray_coco_json_segm_eval.md`).
- 증강 (`create_albu_transform`, `rfdetr/datasets/aug_albumentations.py`). 아래에서 3개를 골라 적용한다(`A.SomeOf`).
  - `MaskawareImageAug`: 임의 치아 polygon 영역을 erode → feather blur → 밝기 증가 + blur. **금속/보철 아티팩트**(경계가 흐린 고휘도 영역)를 모사한다.
  - `CoarseDropout`(밝은 패치), `RandomBrightnessContrast`, `Rotate(±8°)`.
- crop 증강 (`rfdetr/datasets/xray_crop_aug.py`, `--xray_crop_mode`, `--xray_crop_prob`). 원본 해상도에서 crop한 뒤 위 파이프라인을 그대로 적용한다. train split에만 적용된다.
  - `periapical`: 이웃 치아 3~5개, 상악/하악/상하악, 사선 또는 둥근 모서리로 PA 영상을 모사한다.
  - `teeth_roi`: 전체 치아 bbox 합집합 + margin으로 crop한다. two-pass 2차 입력용이고, 1차 박스 오차를 shift로 모사한다.
  - 평가: `--eval_xray_crop`이면 val에도 고정 seed로 crop을 적용해 polygon 시각화를 저장한다(`3e8b8bc`).

## 5. 사전학습 가중치 로딩

`rfdetr/main.py::Model.__init__`

- 체크포인트와 클래스 수가 다르면 head를 체크포인트 크기로 맞춰 로드한 뒤 요청한 클래스 수로 다시 초기화한다(로드된 가중치를 warm start로 사용).
- `load_state_dict(strict=False)`를 `try/except`로 감쌌다. shape이 맞지 않는 키는 건너뛰고 나머지만 로드하며, 실패해도 학습은 계속된다. **로그에서 `Failed to load pretrain weights`가 찍히는지 확인해야 한다.**

## 6. Export (CoreML / libtorch / FP16)

커밋 `391a56e`, `c467d52`, `9db56af`, `3587bfc`, `abaaaf3`

- `MSDeformAttn`: export 모드에서는 `ms_deform_attn_core_pytorch_export`를 쓴다. CoreML이 지원하지 않는 rank-6 텐서를 만들지 않도록 level/point 축을 합친 구현이다. 학습 경로(`vanila_forward`)는 그대로다.
- DINOv2 positional embedding 보간: vanilla의 `bicubic + antialias`를 `interp_mode`/`antialias` 옵션으로 뺐다. 기본값은 CoreML 호환을 위해 **`bilinear`**로 바꿨다. vanilla 사전학습 가중치와 보간 방식이 다르다는 점에 유의한다.
- crop/paste 연산은 입력 dtype으로 grid를 만들어 FP16 export에서도 동작한다.
- `dice_loss` / `sigmoid_ce_loss`의 `torch.jit.script`를 첫 호출 시점으로 미뤘다(frozen 패키징에서 소스가 없어 import 시 실패하던 문제).
- 추론/export 진입점: `rfdetr/deploy/predict.py` (`export_libtorch`, `coreml_export_main`, `predict_wrapper`).

## 7. 추론 후처리 (`engine.py::draw_preditions_boxes`)

- score > 0.5, 배경 제외, box NMS(IoU 0.4)로 필터링한 뒤 남은 query만 마스크를 paste한다.
- `drawing_mode='mask'`: label map을 `np.max(label·mask)`로 병합하므로 **중첩 픽셀은 FDI 번호가 큰 치아에 귀속된다**(중첩이 보이지 않음). 반환되는 `mask_images`도 이 방식이다.
- `drawing_mode='polygon'`: 인스턴스별 독립 마스크의 외곽선을 그린다. 후처리로 중첩을 제거하지 않는다.
- `instance_results`: 인스턴스별 bool 마스크를 반환한다. predict의 치아 간 중첩 리포트(`compute_tooth_overlaps`)에 쓰인다.
- NMS IoU 0.4는 크게 겹친 인접 치아 박스를 통째로 제거할 수 있다.

## 관련 문서

- `docs/xray_dataset_conversion.md`: 학습 라벨(`ann/*.json`) 포맷과 데이터셋별 변환 과정
- `docs/xray_periapical_eval.md`: PA crop 학습 방식, 전체 영상 대비 PA-only 평가 결과
- `README.md` 하단: 데이터셋 목록, 평가 json 빌드, PA/ROI 학습 명령
- `docs/troubleshoots/xray_coco_json_segm_eval.md`: segm 평가 0 문제 원인 분석
