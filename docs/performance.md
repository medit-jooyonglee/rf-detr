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

