# xray 학습 데이터 변환 (`ann/*.json` 포맷)

학습 로더 `rfdetr/datasets/xraypanoramic.py::XrayPnoaramicInstance`가 읽는 라벨 포맷과, 데이터셋별로 그 포맷을 만드는 과정을 정리한다.
평가용 COCO json(`xray_coco_33_seg_v2.json`) 빌드는 README의 "xray 데이터셋 변환 · 평가 COCO json 빌드"와 `docs/troubleshoots/xray_coco_json_segm_eval.md`를 참고한다.

## 1. 전체 흐름

| 데이터 | 원본 포맷 | 변환 | 학습 로더 입력 | 로더 함수 |
|---|---|---|---|---|
| ① Kaggle Teeth Segmentation (`xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2`) | supervisely `ann/*.json` + `img/*.jpg` | **불필요** | `d2/ann/*.json` | `search_kaggle_data01` |
| ② Tufts Dental 2022 (`kaggle_2222`) | Labelbox export `Segmentation/teeth_polygon.pkl` + `Radiographs/*` | **`tufts_to_coco`로 `ann/*.json` 생성** | `kaggle_2222/ann/*.json` | `search_kaggle_data02` |
| ③ 자체 데이터 (`cbct_ios_dcm_latest_0824`) | labelme | labelme repo `labelme2coco.py`로 COCO 변환 | `annotations.json` (COCO) | `_read_coco_image` (`coco_directories`) |

①②는 같은 supervisely 형식(`ann/*.json`)으로 맞춰서 같은 파서(`parse_item_coco` 안의 `extract_annotation_info`)로 읽는다. ③만 COCO 형식이고 train 전용이다.

## 2. 공통 라벨 포맷: supervisely `ann/*.json`

파일명은 **이미지 파일명 + `.json`**이다(예: `0001.JPG` → `ann/0001.JPG.json`). 로더는 이 규칙으로 이미지와 짝을 짓는다.

```json
{
  "objects": [
    {
      "classTitle": "19",
      "classId": "19",
      "bbox": [...],
      "points": { "exterior": [[x1, y1], [x2, y2], ...] }
    }
  ]
}
```

| 필드 | 의미 | 로더가 쓰는 방식 |
|---|---|---|
| `classTitle` | **Universal 번호 문자열** (1~16 상악, 17~32 하악) | `int()`로 변환해 그대로 클래스 id로 쓴다 (`num_classes=32`, 배경 0) |
| `classId` | Tufts 변환본은 `classTitle`과 같다. Kaggle 원본은 supervisely 내부 id | 사용 안 함 |
| `bbox` | Tufts 원본 bbox | **사용 안 함**. bbox는 `exterior`의 min/max로 다시 계산한다 |
| `points.exterior` | 원본 해상도 픽셀 좌표의 외곽 polygon 1개 | 마스크 생성(`fillPoly`), bbox 계산 |
| `points.interior` | 구멍 polygon (Kaggle 원본에 있을 수 있음) | **사용 안 함** |

- 좌표는 **원본 이미지 해상도** 기준이다. resize(너비 640, 높이 stride 64 배수)는 로더가 학습 시점에 한다.
- 치아마다 독립된 polygon이라 인접 치아 polygon끼리 겹칠 수 있다(중첩 라벨). 로더도 치아별로 독립된 마스크를 만들어 중첩을 유지한다.
- 시각화·출력은 Universal → FDI로 매핑한다(`xraypanoramic.py::universal_to_fdi`, `label_to_fdi`).

## 3. ① Kaggle: 변환 없음

- 폴더 구성: `Teeth Segmentation JSON/meta.json`, `d2/img/*.jpg`, `d2/ann/*.json`
- `deep_search_files`는 `{dataset_dir}/../meta.json`이 있으면 Kaggle 형식으로 판단한다.
- `meta.json`의 class title은 Universal 번호(1~32)다. FDI가 아니다.
- 이미지와 `ann` 개수가 다르면 파일명 교집합만 쓴다.

## 4. ② Tufts: `teeth_polygon.pkl` → `ann/*.json`

변환 코드: `rfdetr/util/dataloader.py::tufts_to_coco`, `create_annotation_info`
(함수 이름은 `to_coco`지만 실제 출력은 COCO가 아니라 supervisely `ann/*.json`이다.)

### 4.1 실행

```bash
# 권장: 변환 + 변환 점검(원본 대비 IoU) 후 종료
python -m rfdetr.cli.main --build_tufts

# 단독 실행
python -m rfdetr.util.dataloader --tufts_dir <kaggle_2222> [--out_dir <폴더>] [--limit N] [--close_kernel 0] [--drawing]
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `tufts_dir` | 모듈 상단 `file` 경로 기준 | `Segmentation/teeth_polygon.pkl`, `Radiographs/`가 있는 폴더 |
| `out_dir` | `{tufts_dir}/ann` | 학습 로더가 읽는 위치 |
| `backup` | `True` | 기존 `ann`에 json이 있으면 `ann_backup_{시각}`으로 옮기고 새로 만든다 |
| `close_kernel` | `0` | `>1`이면 `MORPH_CLOSE`. 3×3만 써도 면적이 약 4% 부풀어서 기본은 끈다 |
| `limit` | `-1` | 앞에서부터 N장만 변환 (시험용) |
| `drawing` | `False` | `results/{이미지명}.png`에 컬러 마스크 저장 |

### 4.2 입력 구조 (`teeth_polygon.pkl`)

```
[ { "External ID": "<이미지 파일명>",
    "Label": { "objects": [ { "title": "19" | "A".."T",
                              "bounding box": [...],
                              "polygons": [ [[x, y], ...], ... ] } ] } }, ... ]
```

- 이미지 경로: `{tufts_dir}/Radiographs/{External ID}`. 없으면 assert로 중단한다.
- object 하나는 **외곽 polygon 1개 + 경계를 따라 붙은 픽셀 단위 잔조각 polygon 수십 개**로 되어 있다(마스크에서 추출한 형태).
- 인접 치아가 겹친 영역은 양쪽 치아에 모두 포함돼 있다(중첩 띠 두께 약 6~7px).

### 4.3 변환 절차 (`create_annotation_info`)

치아(object)마다 다음을 수행한다.

1. **번호 변환**
   - `title`이 숫자면 Universal 번호 그대로 쓴다.
   - `A`~`T`(유치)는 같은 위치의 영구치 Universal 번호로 매핑한다(`universal_to_permanent`). 예: `A→4`, `J→13`, `K→20`, `T→29`.
   - 알 수 없는 title은 그 object만 건너뛴다(이미지 전체를 버리지 않는다).
2. **치아별 바이너리 마스크 생성**: 점이 3개 이상인 polygon 조각을 **하나씩** `fillPoly`해서 합집합을 만든다.
3. (옵션) `close_kernel > 1`이면 `MORPH_CLOSE`를 적용한다.
4. **외곽 contour 추출**: `findContours(RETR_EXTERNAL)` 중 **면적이 가장 큰 것** 하나를 `points.exterior`로 저장한다. 떨어진 잔조각은 버린다.
5. `classTitle = classId = str(universal)`, `bbox`는 원본 값을 그대로 기록한다.

이미지 단위로 예외가 나면 그 이미지만 건너뛰고 실패 수를 출력한다.

### 4.4 이전 변환 방식의 문제와 수정 내용

| 이전 구현 | 문제 | 현재 |
|---|---|---|
| 조각 전체를 `fillPoly` 한 번에 호출 | even-odd 규칙으로 외곽 안의 잔조각이 **구멍**이 됨 | 조각을 하나씩 그려 합집합 |
| `MORPH_OPEN 9×9` | 구멍 뚫린 경계 띠가 깎여 **면적 평균 −12%**, 인접 치아 중첩이 77% → 0.3%로 사라짐 | OPEN 제거 (CLOSE는 옵션, 기본 끔) |
| `contours[0]` 저장 | OPEN 후 덩어리가 갈라지면 작은 조각만 남는 치아 발생 | 가장 큰 contour 저장 |
| `dataloader.py` import만 해도 변환 실행 | 학습 중 import로 `ann`이 덮어써질 수 있음 | `__main__` 또는 `--build_tufts`일 때만 실행 |

수정 후 원본 마스크 대비 IoU는 평균 약 0.993이다(`close_kernel=0`).

### 4.5 점검

```bash
python -m rfdetr.datasets.xray_overlap_stats --check_tufts_conversion [--ann_dir <폴더>]
```

- `iou(raw,converted)` 평균이 약 0.99이고 `images_missing_in_converted`가 0이면 정상이다.
- 데이터셋별 중첩 라벨 통계: `python -m rfdetr.datasets.xray_overlap_stats --num_samples 100`

### 4.6 알려진 한계

- **유치 번호 중복**: 유치를 영구치 위치 번호로 매핑하므로 혼합치열 이미지에서 같은 번호가 두 개 생긴다(100장 표본에서 40쌍).
- 외곽 contour 하나만 저장하므로 한 치아가 실제로 둘 이상의 덩어리로 라벨된 경우 작은 쪽은 사라진다.

## 5. ③ 자체 데이터: labelme → COCO

- 변환: 사내 labelme repo(`https://github.com/medit-AI/labelme.git`)의 `examples/instance_segmentation/labelme2coco.py`. 라벨을 수정했을 때만 다시 돌린다.
- 결과: `{cbct_ios_dcm_latest_0824}/annotations.json` (COCO, 원본 좌표). **category id = Universal 번호**다.
- 로딩: `main.py`의 `coco_directories=[(이미지 폴더, annotations.json)]` → `CocoDetection`. `file_name`의 `\`는 `/`로 바꾼다. split당 디렉터리는 1개만 지원한다.
- **split**: `cbct_ios_dcm_latest_1022`부터 `train/`, `val/` 폴더가 나뉘어 있다(`splits.json`, 338 / 85장).
  - train: `coco_directories` (`.../train/`)
  - val: `coco_directories_val` (`.../val/`). 지정하지 않으면 예전처럼 ③은 train 전용이다.
- **평가 GT**: val에서는 `XrayPnoaramicInstanceCoco._merge_own_coco_gt`가 ③ val의 `annotations.json`을 평가 json(①②)에 새 image/annotation id로 합친다. ③ json도 원본 좌표, category id = Universal 번호라 기준이 같다.
- 데이터셋 인덱스는 ③이 앞쪽, ①②가 뒤쪽이다.

## 6. 학습 로더에서의 처리 (참고)

- **split**: ①② 각각 파일명 정렬 순서로 train `0~0.85`, val `0.85~0.9`, test `0.9~1.0`이다. ③은 폴더(`train/`, `val/`)로 나뉜다. xray 데이터셋에서는 test도 val split으로 만든다(`rfdetr/main.py`).
- **영상**: grayscale로 읽고, 너비 640, 높이 stride 64 배수로 resize한다(비율 유지).
- **마스크**: `exterior` polygon을 resize 비율로 스케일한 뒤 치아별 채널에 `fillPoly`한다(치아 간 중첩 유지).
- **라벨 없는 이미지**: 학습 시 class 0의 임의 박스 3개를 넣는다(`is_dummy`). 평가 GT json에는 넣지 않는다.
- **crop 증강**(`--xray_crop_mode periapical teeth_roi`): resize 전에 원본 좌표에서 crop하고 polygon을 offset만큼 옮긴다. 상세는 `rfdetr/datasets/xray_crop_aug.py`.

## 7. 서버 반영 체크리스트

1. 로컬에서 `--build_tufts` 실행 후 IoU 점검 결과를 확인한다.
2. `kaggle_2222/ann` 폴더를 서버 `/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle_2222/`에 복사한다. json만 올리면 서버 학습은 예전 라벨을 계속 쓴다.
3. 평가 json도 새로 빌드했다면 `xray_coco_33_seg_v2.json`을 같이 복사한다.
4. 확인이 끝난 `ann_backup_*` 폴더는 지운다.
