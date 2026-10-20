# xray teeth 평가 json 오류: segm AP가 0으로 나오는 문제

- 작성일: 2026-10-12
- 대상: `rfdetr/cli/main.py::train_from_xray_teeth_dir` (`dataset_file='xray_teeth'`, `segmentation_head=True`)
- 관련 파일: `rfdetr/datasets/xraypanoramic.py`, `rfdetr/engine.py::evaluate`, `rfdetr/datasets/coco_eval.py`
- 결론: **평가 GT json을 원본 좌표 기준으로 다시 만든다 (재빌드).** 학습 코드는 수정할 필요가 없다.

---

## 1. 증상

`--eval`로 validate를 실행하면 bbox는 정상인데 segm은 모든 지표가 정확히 `0.000`으로 나온다.
예측 마스크를 시각화해 보면 품질은 좋다.

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
```

모델 성능이 낮은 경우에는 0.0xx처럼 작은 값이 나오지, 모든 값이 정확히 0이 되지는 않는다.
모든 값이 0이라는 건 **GT와 예측이 한 번도 매칭되지 않았다**는 뜻이고, 평가 쪽 문제로 봐야 한다.

---

## 2. 데이터셋 구성 (main.py 기준, 마지막 학습 환경)

| # | 데이터 | 로컬 PC 경로 | 원본 포맷 | 로더 | 사용 split | 비고|
|---|---|---|---|---|---|---|
| ① | Kaggle Teeth Segmentation (humansintheloop) | `E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2` | 이미지별 `ann/*.json` (supervisely) | `search_kaggle_data01` → `parse_item_coco` | train 0~0.85 / val 0.85~0.9 / test 0.9~1.0 ||
| ② | Tufts Dental 2022 | `E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222` | `rfdetr/util/dataloader.py::tufts_to_coco`로 `ann/*.json` 형태로 변환 | `search_kaggle_data02` → `parse_item_coco` | 위와 같음 ||
| ③ | 자체 구축 데이터 (`cbct_ios_dcm`) | `E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm` | `annotations.json` (원래부터 COCO는 아니고, labelme 로 구축후 cocoformat으로 변환. ) | labelme-https://github.com/medit-AI/labelme.git  / dataset(dropbox) Medit_AI Dropbox\Medit_AI\xray_panoramic\cbct_ios_dcm   / 변환 tool examples\instance_segmentation\labelme2coco.py (labelme)`CocoDetection` → `_read_coco_image` (`ResizeCocoSample`) | **train에만 사용** ||
| - | UFBA-425, A dual-labeled dataset | - | 변환 안 됨 | - | 사용 안 함 (README에만 있음) ||

- 평가 GT json(`xray_coco_33_seg.json`)에는 ①과 ②만 들어 있다. ①은 이미지 598장(치아 14,663개), ②는 1,000장(치아 24,055개).
- annotation 수(38,809개)는 이미지 수가 아니라 **치아 instance 수**다. 이미지당 평균 약 24개.
- ③은 train에만 쓰이고, 자기 COCO json의 width/height가 실제 이미지 크기와 일치하는 것을 샘플로 확인했다. 이번 문제와는 관계없다.

---

## 3. 원인

### 3.1 평가 흐름

```
model output (정규화 좌표 [0,1])
  → PostProcess(outputs, orig_size)         # rfdetr/engine.py::evaluate
      bbox: 원본 크기를 곱해 원본 좌표로
      mask: paste_masks_in_image → 원본 크기 (H, W) 마스크
  → CocoEvaluator: 예측 마스크를 원본 크기 RLE로 인코딩
  → COCOeval: GT json(base_ds)과 비교
      GT polygon → annToRLE(json의 images[].height, width 크기 캔버스)
```

`orig_size`는 `parse_item_coco`에서 `src.shape[:2]`, 즉 **리사이즈 전 원본 크기**다.
따라서 **GT json도 원본 좌표계여야 한다.**

### 3.2 현재 json의 실제 상태 (전체 annotation 검사 결과)

| 항목 | 들어가야 할 값 | 현재 json | 판정 |
|---|---|---|---|
| `images[].width` / `height` | 원본 크기 (예: 2041 × 1024) | **384 × 704** (리사이즈된 크기이고 w/h도 뒤바뀜) | ❌ |
| `annotations[].bbox` | 원본 `[x, y, w, h]` | 원본 좌표 | ✅ |
| `annotations[].segmentation` | 원본 polygon | 원본 좌표 (polygon 범위와 bbox가 정확히 일치) | ✅ |
| `annotations[].area` | 원본 마스크 면적 | 리사이즈 기준 (bbox 면적 대비 약 0.09~0.15) | ❌ |
| segmentation 없는 annotation | 0개 | 87개 | ⚠️ |
| bbox 면적이 0인 annotation | 0개 | 4개 | ⚠️ |
| image id | 0..N-1 연속 | 연속 | ✅ |

### 3.3 그래서 이렇게 된다

- **segm**: GT polygon(x 좌표가 최대 2000 정도)을 json에 적힌 크기인 704×384 캔버스에 그리면서 대부분이 잘려 나간다.
  예측 RLE는 1024×2041 크기라서 GT와 크기도 맞지 않는다. 결과적으로 IoU가 0이 되고 AP/AR도 0이 된다.
- **bbox**: IoU 계산에 이미지 크기가 쓰이지 않기 때문에 정상으로 나온다.
  다만 `area`가 작게 들어가 있어서 **area별(small/medium/large) 지표는 틀린 값**이다. large가 `-1`로 나오는 것도 이 때문이다.

### 3.4 근본 원인

json을 만드는 `coco_json_export`가 **학습용 resize 파이프라인(`parse_item_coco`)을 그대로 재사용**한다.
그 결과 원본 좌표와 리사이즈 좌표가 한 json 안에 섞였다.
데이터셋마다 포맷을 COCO로 변환하는 과정과 학습용 전처리가 같은 코드를 공유하면서 생긴 문제다.

### 3.5 학습에는 영향이 없다

학습 경로는 이 json을 **라벨로 쓰지 않는다.**
라벨은 ①②는 이미지별 원본 annotation에서, ③은 자체 `annotations.json`에서 직접 읽는다.

```
원본 이미지와 원본 polygon/bbox
  → resize (너비 640 기준, stride 64 → 704 등)
  → bbox / 리사이즈된 크기 → [0,1] 정규화 (cx, cy, w, h)
  → mask는 리사이즈된 해상도로 그림
```

`xray_coco_33_seg.json`이 학습에서 쓰이는 곳은 `XrayPnoaramicInstanceCoco`의 `file_name` → `image_id` 매핑뿐이다.
따라서 json을 원본 좌표로 다시 만들어도 학습 결과에는 영향이 없고, **기존 체크포인트도 그대로 사용할 수 있다.**

### 3.6 데이터셋별 재빌드 필요 여부

| 데이터 | 평가 json(`_v2`)에 포함 | 좌표 기준 | 재빌드 |
|---|---|---|---|
| ① Kaggle | 포함 | `--build_json`으로 원본 크기로 다시 만듦 | 필요 |
| ② Tufts | 포함 | `--build_json`으로 원본 크기로 다시 만듦 | 필요 |
| ③ 자체 데이터 (`cbct_ios_dcm_latest_0824`) | **제외** (train에만 사용) | 이미 원본 크기 | 불필요 |

**③은 평가에 들어가지 않는다.** `xraypanoramic.py::build()`에서 `image_set == 'train'`일 때만 `coco_directories`를 넘기고, val/test에서는 `None`으로 버린다.
③에는 split도 적용되지 않는다. `CocoDetection`이 `annotations.json`의 이미지 420장을 **전부 train으로** 읽는다.

```
train : ①② (0 ~ 0.85) + ③ 전체 420장
val   : ①② (0.85 ~ 0.9)          ← --eval은 이것만 평가
test  : ①② (0.9 ~ 1.0)
```

그래서 기존 json으로 `--eval`을 실행하면 segm이 **전부 0**으로 나온다. ③의 json에는 문제가 없지만, val에는 json이 깨진 ①②만 있기 때문이다.

**③의 json이 원본 크기인 이유**
- 변환 코드(labelme repo `examples/instance_segmentation/labelme2coco.py::convert_labelme_dirs_to_coco`)는 `width/height`를 실제 이미지 크기(`img.shape`)로 넣는다.
- `segmentation`은 labelme의 원본 polygon이고, `area/bbox`는 원본 크기 마스크에서 계산한다.
- resize 단계가 없어서, ①②처럼 원본 좌표와 리사이즈 좌표가 섞일 수 없다.

**③ `annotations.json` 전수 검사 결과 (2026-10-12)**
- 이미지 420장, annotation 11,359개. 대부분 2640×1256이다.
- json의 width/height와 실제 이미지 크기가 다른 이미지: 0장
- 이미지 밖으로 나간 polygon: 0개. area/bbox 면적 비율 중앙값: 0.647 (정상)
- category는 1~32로, ①②와 같은 universal 번호 체계다. 숫자가 아닌 라벨은 없다.
- image id는 0..N-1로 연속이다.

**③을 다시 변환하거나 평가에 넣어야 하는 경우**
- labelme에서 라벨을 추가하거나 수정했을 때: `convert_labelme_dirs_to_coco`를 새 출력 폴더로 다시 실행한다. 원본 크기 규칙은 그대로 유지된다.
- ③도 평가하고 싶을 때: 지금은 ③ 전체가 train이라 그대로 넣으면 학습한 이미지로 평가하게 된다. ③에도 train/val split을 추가해야 한다(미구현).

---

## 4. 해결 시나리오 (추천): 평가 json 재빌드

로컬 PC(`E:/dataset/reverse_tomosynthesis/kaggle_xrays`)에서 코드를 수정하고 json을 다시 만든 뒤 검증하고 서버에 반영한다.

### 4.1 현재 export 코드로 그냥 다시 만들면 안 되는 이유

`coco_json_export`와 `test_build_coco_json`을 지금 상태로 실행하면 지금보다 더 나빠진다.

1. `width/height`가 `image.shape[1:]`, 즉 리사이즈된 크기로 들어간다.
2. `segmentation`이 `rsz_polygons`(리사이즈 좌표)로 들어간다. bbox는 원본 좌표라서 둘이 어긋난다.
3. `name='train'`이고 `augment_en`의 기본값이 True라서 **랜덤 증강(회전 등)이 polygon에 적용된다.** GT가 실행할 때마다 달라진다.
4. `area`가 리사이즈된 마스크의 픽셀 수로 들어간다.
5. annotation이 없는 이미지는 `parse_item_coco`가 **랜덤 가짜 박스 3개(class 0)**를 만들고, 이게 GT로 export된다.
6. ③(`coco_apis`) 데이터가 포함되면 `segmentation`이 bool 마스크 배열이라 json으로 저장할 수 없다.
7. segmentation 개수가 bbox 개수와 다르면 그 이미지의 segmentation이 전부 빠진다. 현재 json의 87개가 이 경우다.

### 4.2 코드 수정 (`rfdetr/datasets/xraypanoramic.py`)

학습 경로의 동작은 바꾸지 않고, export에 필요한 정보만 추가로 넘겨준다.

**(1) `parse_item_coco`: 원본 polygon과 가짜 박스 여부를 target에 추가**

```python
# bboxes/class_labels가 비어 있어서 랜덤 박스를 만드는 분기에서
is_dummy = False
if bboxes.size == 0 and class_labels.size == 0:
    is_dummy = True
    ...  # 기존 랜덤 박스 생성 코드 그대로

# annot dict를 만드는 부분에서 (augmentation이 꺼져 있으면 kept_bbox_indices는 전체 인덱스)
annot['is_dummy'] = is_dummy
if self.include_masks and not is_dummy:
    annot['segmentation_orig'] = [
        np.asarray(annot_data[i]['segmentation'], dtype=np.float64).ravel().tolist()
        for i in kept_bbox_indices
    ]
```

- `segmentation`/`segmentation_orig` 키는 `XrayPnoaramicInstanceCoco.__getitem__`에서 `pop`되거나 쓰이지 않으므로 학습에는 영향이 없다.
  새로 추가한 `segmentation_orig`, `is_dummy`도 `__getitem__`에서 `segmentation`과 함께 `pop` 해 둔다. tensor 변환(`data_convert`)에서 문제가 생기지 않게 하기 위해서다.

**(2) `coco_json_export`: 원본 좌표 기준으로 다시 작성**

```python
def coco_json_export(self, base_dir='', debug_break=-1):
    import pycocotools.mask as mask_util

    base_dir = base_dir.replace('\\', '/')
    class_names = ['gum', *[str(v) for v in fdi_sort], 'unknown']
    categories = [{"id": i, "name": n, "supercategory": "tooth_type"} for i, n in enumerate(class_names)]
    coco_json = {"images": [], "annotations": [], "categories": categories}

    # GT에는 augmentation이 들어가면 안 된다
    prev_augment, self.augment_en = self.augment_en, False
    num_coco = len(self.coco_apis) if self.coco_apis is not None else 0
    skipped = {"dummy_image": 0, "no_segmentation": 0, "empty_box": 0}
    try:
        for idx in tqdm.tqdm(range(len(self))):
            if idx < num_coco:
                continue  # ③ 자체 COCO 데이터는 평가 json에 넣지 않는다
            _, target = self.parse_item_coco(idx, norm_bbox=False, box_format='xywh')

            filename = self.file_name(idx).replace('\\', '/')
            assert filename.startswith(base_dir), f"{filename} does not start with {base_dir}"
            h, w = [int(v) for v in target['orig_size']]       # 원본 (H, W)
            image_id = len(coco_json["images"])                  # 0..N-1 연속 유지
            coco_json["images"].append({
                "id": image_id,
                "file_name": os.path.relpath(filename, base_dir).replace('\\', '/'),
                "width": w,
                "height": h,
            })

            if target.get('is_dummy', False):
                skipped["dummy_image"] += 1
                continue

            segs = target.get('segmentation_orig', [])
            for box, label, seg in zip(target['boxes'], target['labels'], segs):
                x, y, bw, bh = [float(v) for v in box]
                # 원본 이미지 범위로 clip
                x0, y0 = max(x, 0.0), max(y, 0.0)
                x1, y1 = min(x + bw, w), min(y + bh, h)
                if x1 <= x0 or y1 <= y0:
                    skipped["empty_box"] += 1
                    continue
                if len(seg) < 6:  # 점 3개 미만
                    skipped["no_segmentation"] += 1
                    continue
                rle = mask_util.merge(mask_util.frPyObjects([seg], h, w))
                coco_json["annotations"].append({
                    "id": len(coco_json["annotations"]),
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "segmentation": [seg],
                    "area": float(mask_util.area(rle)),
                    "iscrowd": 0,
                })
            if 0 < debug_break <= len(coco_json["images"]):
                break
    finally:
        self.augment_en = prev_augment

    print(f"export: images={len(coco_json['images'])}, "
          f"annotations={len(coco_json['annotations'])}, skipped={skipped}")
    return coco_json
```

**(3) `XrayPnoaramicInstanceCoco.__init__`: 매핑 안전장치**

```python
# 현재: meta = [self.coco.loadImgs(i)[0] for i in range(len(self.coco.imgs))]
meta = self.coco.loadImgs(self.coco.getImgIds())          # id가 연속이 아니어도 동작
...
img_id_mapping = {...: v['id'] for v in meta}
image_files = list(img_id_mapping.keys())
...
matched_idx.append(img_id_mapping[image_files[matched]] if matched >= 0 else -1)
...
assert all(m >= 0 for m in matched_idx), 'some images are not found in annotation json'
```

현재 코드는 리스트 인덱스를 그대로 image id로 쓴다. id가 0..N-1로 연속일 때만 맞는 방식이라, id 값을 직접 쓰도록 바꾼다.

### 4.3 json 빌드 (로컬 PC)

> **구현 완료**: `python -m rfdetr.cli.main --build_json`
> - 학습과 같은 `dataset_dir`(①②)로 `xray_coco_33_seg_v2.json`을 빌드하고 `verify_xray_coco_json`으로 검증한다.
> - 평소 실행(학습/`--eval`)에서는 `_v2.json`이 있으면 그 파일을 쓴다.
> - 전체 빌드(1598장)에서 GT 자기 자신 평가 결과는 bbox/segm 모두 AP 1.0이다. 처음 0.999로 나온 원인은 annotation id가 0부터 시작한 것(pycocotools는 매칭된 GT id 0을 "매칭 안 됨"으로 해석)과, 면적 0인 일직선 polygon 1개(Kaggle 367.jpg 15번)였다. 둘 다 export에서 수정했다(id는 1부터, 면적 0 polygon 제외).

`test_build_coco_json`을 ①② 기준으로 수정한다.

```python
def test_build_coco_json():
    base_dir = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays'
    num_classes = 33
    dataset = XrayPnoaramicInstance(
        img_folder=[
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222',
        ],
        num_classes=num_classes,
        include_masks=True,
        coco_directories=None,          # ③ 제외
        name='train',
        splits={'train': (0, 1)},       # 전체 이미지 (val/test split은 로더가 file_name으로 매칭)
        augment_en=False,               # 증강 금지
    )
    res = dataset.coco_json_export(base_dir=base_dir)
    filename = os.path.join(base_dir, f'xray_coco_{num_classes}_seg_v2.json')
    with open(filename, 'w') as f:
        json.dump(res, f)
    print(f'coco json file saved to {filename}')
```

실행 (repo 루트에서):

```bash
# xraypanoramic.py의 __main__ 블록에서 test_build_coco_json()을 호출하도록 바꾼 뒤
python -m rfdetr.datasets.xraypanoramic
```

- `file_name`은 `base_dir` 기준 상대경로다(`xray_teeth_seg_kaggle/...`, `kaggle_2222/...`). 기존 json과 같은 규칙이라 서버(`/data1/jooyonglee/reverse_tomo/xray_panoramic/`)에서도 그대로 매칭된다.
- **기존 `xray_coco_33_seg.json`은 덮어쓰지 말고** 새 파일(`_v2`)로 만든다.

### 4.4 검증

**(1) json 정합성 체크**

```python
import json, os
import numpy as np
from PIL import Image

base = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays'
d = json.load(open(os.path.join(base, 'xray_coco_33_seg_v2.json')))
im = {i['id']: i for i in d['images']}

# 1) width/height == 실제 이미지 크기
bad = [i['file_name'] for i in d['images']
       if Image.open(os.path.join(base, i['file_name'])).size != (i['width'], i['height'])]
print('size mismatch:', len(bad))

# 2) polygon이 bbox/이미지 안에 있는지, area 비율이 정상인지
ratios = []
for a in d['annotations']:
    s = np.array(a['segmentation'][0]).reshape(-1, 2)
    x, y, w, h = a['bbox']
    img = im[a['image_id']]
    assert s[:, 0].max() <= img['width'] + 1 and s[:, 1].max() <= img['height'] + 1
    ratios.append(a['area'] / (w * h))
print('area/bbox_area median:', np.median(ratios))   # 0.6~0.8 정도면 정상 (기존 json은 약 0.1)
print('images:', len(d['images']), 'annotations:', len(d['annotations']))
```

**(2) GT 자기 자신 평가 (모델 없이 평가 파이프라인만 확인)**

GT를 그대로 예측 결과로 넣었을 때 bbox와 segm AP가 모두 1.0이 나와야 한다.

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gt = COCO(json_path)
dets = [{**a, 'score': 1.0} for a in gt.dataset['annotations']]
for iou_type in ['bbox', 'segm']:
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, iou_type)
    ev.evaluate(); ev.accumulate(); ev.summarize()   # AP = 1.000 이어야 정상
```

기존 json으로 같은 테스트를 하면 segm AP가 1.0이 나오지 않는다. 이 차이로 문제가 해결됐는지 바로 확인할 수 있다.

**(3) 실제 eval**

1. 새 json을 서버 `/data1/jooyonglee/reverse_tomo/xray_panoramic/`에 복사한다.
2. `rfdetr/cli/main.py`의 `annot_file`을 `.../xray_coco_33_seg_v2.json`으로 바꾼다.
3. `--eval`로 실행해서 segm AP가 0이 아닌 정상 범위로 나오는지, bbox의 area별 지표(large 등)가 `-1`이 아닌지 확인한다.
4. 학습 데이터 로딩에도 문제가 없는지 1 epoch 정도 짧게 돌려 본다. 매칭 assert가 통과하고 이미지 수가 기존과 같아야 한다.

### 4.5 체크리스트

- [ ] `parse_item_coco`에 `is_dummy`, `segmentation_orig` 추가
- [ ] `coco_json_export`를 원본 좌표 기준으로 다시 작성
- [ ] `XrayPnoaramicInstanceCoco`의 id 매핑과 assert 보강
- [ ] 로컬에서 `xray_coco_33_seg_v2.json` 빌드
- [ ] 정합성 체크 통과 (크기 불일치 0, area 비율 정상)
- [ ] GT 자기 자신 평가에서 bbox/segm AP 1.0
- [ ] 서버에 반영하고 `--eval`로 segm AP 확인
- [ ] 짧은 학습으로 로딩 정상 확인

---

## 5. 검토했지만 추천하지 않는 방식

### 5.1 기존 json만 보정 (width/height, area만 수정)

- 방법: 기존 json에서 `images[].width/height`를 실제 이미지 크기로 바꾸고, `area`를 원본 polygon으로 다시 계산한다. 코드는 수정하지 않는다.
- 장점: 가장 빠르다. 현재 bbox와 segmentation이 이미 원본 좌표라 바로 효과가 있다.
- **추천하지 않는 이유**
  - export 코드가 그대로 남아서, 데이터를 추가하거나 json을 다시 만드는 순간 같은 문제(또는 더 나쁜 문제, 4.1 참고)가 다시 생긴다.
  - segmentation이 없는 87개, bbox 면적이 0인 4개, 가짜 박스 같은 문제는 그대로 남는다.
- 쓸 만한 경우: 재빌드 전에 원인이 맞는지 빠르게 확인하는 임시 용도로만 쓴다.

### 5.2 평가를 리사이즈 좌표 기준으로 바꾸기

- 방법: `engine.py::evaluate`에서 PostProcess에 `orig_size` 대신 `size`(리사이즈 크기)를 넘기고, json 전체를 리사이즈 좌표로 맞춘다.
- **추천하지 않는 이유**
  - 평가 결과가 모델 입력 해상도(640/704, stride 등)에 묶인다. 해상도를 바꿀 때마다 json을 다시 만들어야 한다.
  - 실제 추론 결과(원본 크기)와 평가 기준이 달라진다.
  - COCO 표준(원본 이미지 기준)과 맞지 않고, upstream RF-DETR의 평가 코드와도 달라진다.
  - 이미지마다 리사이즈 크기가 달라서(704×384, 704×256 등) 관리하기 어렵다.

### 5.3 현재 `coco_json_export`의 일부만 고쳐서 다시 빌드

- 방법: `width, height = image.shape[1:]`의 순서만 바꾸거나 `orig_size`로 교체하고, 나머지는 그대로 둔다.
- **추천하지 않는 이유**
  - segmentation이 리사이즈 좌표(`rsz_polygons`)로 들어가는 문제와 augmentation이 섞이는 문제가 남는다. bbox와 polygon의 좌표계가 어긋나서 지금보다 더 나빠진다.
  - 학습용 resize 파이프라인을 export에 계속 공유하는 구조라, 학습 전처리를 바꿀 때마다 GT가 함께 바뀌는 위험이 남는다.

---

## 6. 참고: 확인할 때 쓴 명령

```python
# 현재 json의 이미지 크기와 실제 이미지 크기 비교
import json, os
from PIL import Image
base = '/data1/jooyonglee/reverse_tomo/xray_panoramic/'
d = json.load(open(base + 'xray_coco_33_seg.json'))
for i in d['images'][:3]:
    print(i['file_name'], (i['width'], i['height']),
          Image.open(os.path.join(base, i['file_name'].replace('\\', '/'))).size)
# → xray_teeth_seg_kaggle/.../img/1.jpg (384, 704) (2041, 1024)
```


----



데이터	평가 json(_v2)에 포함	좌표 기준	재빌드
① Kaggle	포함	--build_json으로 원본 크기로 다시 만듦	필요 (이번 수정)
② Tufts	포함	--build_json으로 원본 크기로 다시 만듦	필요 (이번 수정)
③ 자체 데이터 (cbct_ios_dcm_latest_0824)	제외 (train에만 사용)	이미 원본 크기	불필요