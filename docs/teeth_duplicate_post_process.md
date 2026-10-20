E:\dataset\reverse_tomosynthesis\TEST\test01\Raymond Chow 원본 이미지에서 16, 17인데, 17, 17번으로 추론된다. rfdetr\deploy\predict.py  치아 번호 확률값만 가지고 빠르게 중복 fdi번호만 제거하는 / 또는 진짜 outlier fdi번호만 보정하는 로직을 추가하고자한다. 참고 문헌, 기술 https://www.nature.com/articles/s41598-019-40414-y 


결과 이미지
temp\resultsxray\Raymond Chow\pano_jan_24_overlap.JPG

teeth-seg-3d도 비슷하게 dental-arch 3d로 sorting하고 처리햇음

다른 저작소 로직인데. 해당 코드 참고해라. 
D:\workspace\repositories\teeth-segmentation-3d\

models\cores\voxelpostprocess.py::PostProcessingFdinumbering


xray에서 upper & lower 인식은 잘되므로

upper / lower 나눠서 중복 치아 제거하는 후처리 로직 구현

xray에서 upper ,lwer arch 순으로 sotring은 매우매우 간단하므로. x좌표값 sorting 

---

## 구현 요약

- 코드: `rfdetr/util/fdi_refinement.py::refine_fdi_labels`
- 호출: `rfdetr/engine.py::draw_preditions_boxes(..., fdi_refinement=True)` — NMS 직후, `nms_refinement=True`일 때만 동작 (기본 on, `fdi_refinement=False`로 끔)

### 알고리즘
1. 예측 label 기준으로 upper(1~16) / lower(17~32) 분리
2. 각 arch 내 박스 x-center로 오름차순 정렬
   - universal label 기준 upper는 x↑ → label↑ (18…11, 21…28)
   - lower는 x↑ → label↓ (48…41, 31…38) 이므로 label 순서를 뒤집어 맞춤
3. "x 순서대로 치아 번호가 strictly increasing" 제약 하에서 Σ log(prob) 최대 배정을 DP로 계산
   - teeth-seg-3d `PostProcessingFdinumbering`의 combination template matching과 동일한 최적해를 O(m·16)으로 구함 (전수 조합 탐색과 결과 일치 검증)
4. 보정된 label의 확률로 `pred_scores` 갱신

### 특성
- argmax 결과가 이미 중복 없이 순서가 맞으면 그 자체가 최적해 → **변경 없음**. 중복/순서 outlier가 있을 때만 바뀜
- arch당 검출 수 > 16 (과검출)이면 보정 skip
- 변경 시 로그 출력: `[fdi-refine] upper: [18, 17, 17] -> [18, 17, 16]`

### 한계
- upper/lower 판정은 argmax label을 그대로 신뢰 (상·하악 오분류는 보정 안 함)
- 결손 치아(빈 자리)는 확률로만 판단하므로 인접 치아 확률이 애매하면 한 칸씩 밀릴 수 있음
