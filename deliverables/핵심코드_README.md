# 핵심 Python 코드 패키지

이 ZIP은 모델 weight·dataset·출력 파일을 제외하고 발표 내용을 뒷받침하는 소스만 모았습니다.

## 권장 읽기 순서

### RF-DETR / X-ray teeth numbering

1. `rf_detr/rfdetr/datasets/xraypanoramic.py` — FDI mapping, COCO/LabelMe/Kaggle 통합, box/mask target 생성
2. `rf_detr/rfdetr/cli/main.py` — DINOv2 Tiny 기반 치아 학습 구성 예
3. `rf_detr/rfdetr/models/backbone/dinov2.py` — tiny encoder 선택과 positional embedding 보간
4. `rf_detr/rfdetr/models/backbone/dinov2_with_windowed_attn.py` — window/global attention 구현
5. `rf_detr/rfdetr/models/transformer.py` — two-stage proposal과 deformable decoder
6. `rf_detr/rfdetr/models/lwdetr.py` — 모델 조립, loss, postprocess
7. `rf_detr/rfdetr/deploy/predict.py` — 원본 크기 복원 및 추론/저장

### Reverse-registration

1. `reverse_registration/reversereg/datasets/cocoimage2d.py` — 2D 등록 학습 데이터 생성
2. `reverse_registration/reversereg/models/similarity2d.py` — coarse similarity + arch coefficient
3. `reverse_registration/reversereg/models/deformation2d.py` — dense/anchor non-rigid field
4. `reverse_registration/reversereg/models/encoder2d.py` — anchor feature sampling/attention pooling
5. `reverse_registration/pcregmodel/data/cococustom.py` — arc-length 보존 치열 변형
6. `reverse_registration/reversereg/preproc/sampler.py` — panoramic 2D ↔ world 3D 좌표 변환
7. `reverse_registration/reversereg/scripts/register_ai_process.py` — 실제 AI 등록 파이프라인
8. `reverse_registration/reversereg/scripts/registration_xray_ios.py` — X-ray/IOS 전체 호출 흐름

## 범위

- RF-DETR 기준: `develop...detr_teeth`
- Reverse-registration 기준: `origin/main...nonrigid2d`와 현재 working tree
- DINOv2 Tiny 경량화는 RF-DETR에 포함되며 reverse-registration에는 포함되지 않습니다.
