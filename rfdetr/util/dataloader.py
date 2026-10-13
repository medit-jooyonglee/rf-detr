import os
from trainer import vtk_utils, utils_numpy
import tqdm

import cv2
import numpy as np
import json

# path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_seg/training_data/training_data/quadrant'

# file = 'train_quadrant.json'

# filename = os.path.join(path, file)


# with open(filename, 'r') as f:
#     data = json.load(f)
    
    
# image_data = data['images']
# annotations_data = data['annotations']
# categories_data = data['categories']

# import cv2

# idx = 0
# image_meta = image_data[idx]
# image_id = image_meta['id']

# annotations = [ann for ann in annotations_data if ann['image_id'] == image_id]
# # annotations_dict = {ann['id']: ann for ann in annotations_data}

# image_file = os.path.join(path, 'xrays', image_meta['file_name'])
# assert os.path.exists(image_file), f"Image file {image_file} does not exist."

# image = cv2.imread(image_file, cv2.IMREAD_COLOR)

# seg_polys = [ann['segmentation'] for ann in annotations]
# import numpy as np

# for seg in seg_polys:
#     for poly in seg:
#         pts = np.array(poly).reshape(-1, 2).astype(np.int32)
#         cv2.fillPoly(image, [pts], color=(0, 255, 0))
        
        
# cv2.imwrite('temp.png', image)

from trainer import timefn
from rfdetr.datasets.teeth import draw_bboxes
file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222/Segmentation/teeth_polygon.json'


@timefn
def read_json():
    with open(file, 'r') as f:
        data = json.load(f)
    return data

import pickle

@timefn
def read_pickle():
    with open(file.replace('.json', '.pkl'), 'rb') as f:  
        data = pickle.load(f)
        
    return data


universal_to_fdi = np.array([
    0,  # Index 0
    18, 17, 16, 15, 14, 13, 12, 11,  # 1~8 (우상)
    21, 22, 23, 24, 25, 26, 27, 28,  # 9~16 (좌상)
    38, 37, 36, 35, 34, 33, 32, 31,  # 17~24 (좌하)
    41, 42, 43, 44, 45, 46, 47, 48   # 25~32 (우하)
])

universal_to_permanent = {
    "A": 4,
    "B": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "J": 13,
    
    "K": 20,
    "L": 21,
    "M": 22,
    "N": 23,
    "O": 24,
    
    "P": 25,
    "Q": 26,
    "R": 27,
    "S": 28,
    "T": 29,
}



def create_annotation_info(labels, image, drawing=False, mask=None, close_kernel=0):
    """Tufts Labelbox polygon -> supervisely 형식 objects.

    원본 object 하나 = 외곽 polygon 1개 + 경계를 따라 붙은 픽셀 단위 잔조각 polygon 수십 개 (마스크에서 추출된 형태).
    파노라마에서 인접 치아가 겹친 영역은 양쪽 치아에 모두 포함되어 있다(중첩 라벨, 두께 약 6~7px).

    이전 구현(fillPoly 한 번 호출 + MORPH_OPEN 9x9 + contours[0])의 문제:
      - fillPoly에 조각을 한 번에 넘기면 외곽 안 잔조각이 even-odd 규칙으로 구멍이 된다
      - 구멍 뚫린 경계 띠를 OPEN이 깎아서 면적이 평균 12% 줄고, 중첩 띠가 거의 다 사라졌다 (인접쌍 중첩 77% -> 0.3%)
      - OPEN 후 덩어리가 갈라지면 contours[0](첫 번째)을 저장해서 작은 조각만 남는 치아가 있었다
    수정:
      - 조각을 하나씩 그려 합집합. OPEN 제거 (close_kernel>1 이면 CLOSE, 기본 0: 3x3도 면적이 약 4% 부풀어서 생략)
      - 가장 큰 외곽 contour 저장
    점검: python -m rfdetr.datasets.xray_overlap_stats --check_tufts_conversion
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)) if close_kernel > 1 else None
    # _keys(['title', 'bounding box', 'polygons'])
    anno_data = dict(objects=[])
    bboxes = []
    segmentations = []
    for label in labels['objects']:
        title = label['title']
        # print(title)
        # if title == 'teeth':
        polygons = label['polygons']
        bbox = label['bounding box']
        bboxes.append(bbox)

        pts_list = []
        for poly in polygons:
            # 만약 poly가 바로 점들의 리스트라면:
            points = poly['points'] if isinstance(poly, dict) and 'points' in poly else poly

            # (N, 2) 형태의 정수형 NumPy 배열로 변환
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            # if title == '19':
                # print(pts.shape)
            # print(pts.shape)
            if len(pts) < 3:
                continue
            pts_list.append(pts)
        # try:
        try:
            universal_index = int(title)
        except ValueError:
            
            universal_index = universal_to_permanent.get(title)
            # print(f"Warning: Unable to convert title '{title}' to an integer. Skipping this label.")
            # continue
            # raise ValueError(f"Unable to convert title '{title}' to an integer.")
        if universal_index is None:
            # 알 수 없는 title은 이 object만 건너뛴다 (이전에는 예외로 이미지 전체가 빠졌다)
            continue

        # ex
        fdi_index = universal_to_fdi[universal_index]

        # 치아별로 독립된 바이너리 마스크를 만든다 (치아끼리 중첩 유지).
        # 조각을 하나씩 그려야 합집합이 된다. 한 번에 넘기면 겹친 조각이 구멍이 된다.
        tooth_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for pts in pts_list:
            cv2.fillPoly(tooth_mask, [pts], color=255)
        if kernel is not None:
            tooth_mask = cv2.morphologyEx(tooth_mask, cv2.MORPH_CLOSE, kernel)

        # 외곽 contour 중 가장 큰 것 (떨어진 잔조각은 버린다)
        contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        anno_data['objects'].append(dict(
            classTitle=str(universal_index),
            classId=str(universal_index),
            bbox=bbox,
            points={
                'exterior': contour.reshape(-1, 2).tolist()
            }
        ))
        
        
        if drawing and mask is not None:
            mask[tooth_mask > 0] = fdi_index
    return anno_data, bboxes

    
def tufts_to_coco(tufts_dir=None, out_dir=None, backup=True, drawing=False, limit=-1, close_kernel=0):
    """Tufts teeth_polygon.pkl -> {out_dir}/{이미지명}.json (supervisely 형식, 학습 로더 search_kaggle_data02가 읽음).

    tufts_dir: kaggle_2222 폴더 (Segmentation/teeth_polygon.pkl, Radiographs/ 포함). 기본값은 모듈 상단 file 기준
    out_dir:   기본값은 {tufts_dir}/ann (학습 로더가 읽는 위치)
    backup:    out_dir에 기존 json이 있으면 {out_dir}_backup_{시각} 으로 옮긴 뒤 새로 만든다
    drawing:   True면 results/ 에 확인용 컬러 마스크 png 저장
    limit:     >0 이면 앞에서부터 limit장만 변환 (시험용)
    """
    import glob
    import pickle
    import time

    tufts_dir = tufts_dir or os.path.join(os.path.dirname(file), '..')
    with open(os.path.join(tufts_dir, 'Segmentation', 'teeth_polygon.pkl'), 'rb') as f:
        data = pickle.load(f)
    if out_dir is None:
        out_dir = os.path.join(tufts_dir, 'ann')
    if backup and glob.glob(os.path.join(out_dir, '*.json')):
        backup_dir = f"{os.path.normpath(out_dir)}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        os.rename(out_dir, backup_dir)
        print(f'tufts_to_coco: backup {out_dir} -> {backup_dir}')
    os.makedirs(out_dir, exist_ok=True)
    if drawing:
        os.makedirs('results', exist_ok=True)
    num_failed = 0

    assert isinstance(data, list), "Expected a list of annotations in the JSON file."

    idx = 1

    # def extract_annotation_info(gt_data, idx):
        
        
                        
    #     extracted_data.append({
    #         "class_title": class_title,
    #         "class_id": class_id,
    #         "bbox": bbox,         # [x1, y1, x2, y2]
    #         "segmentation": polygon.tolist() # 리스트 형태의 폴리곤 좌표
    #     })
                
                
    universal_to_fdi = np.array([
        0,  # Index 0
        18, 17, 16, 15, 14, 13, 12, 11,  # 1~8 (우상)
        21, 22, 23, 24, 25, 26, 27, 28,  # 9~16 (좌상)
        38, 37, 36, 35, 34, 33, 32, 31,  # 17~24 (좌하)
        41, 42, 43, 44, 45, 46, 47, 48   # 25~32 (우하)
    ])

    num_items = len(data) if limit <= 0 else min(limit, len(data))
    for idx in tqdm.tqdm(range(num_items)):
        data0 = data[idx]
        # ict_keys(['Label', 'External ID'])          
        labels = data0['Label']

        img_file_name = data0['External ID']

        img_file = os.path.join(tufts_dir, 'Radiographs', img_file_name)

        assert os.path.exists(img_file), f"Image file {img_file} does not exist."

        
        
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)


        try:
            anno_data, bboxes = create_annotation_info(labels, image, drawing=drawing, mask=mask,
                                                       close_kernel=close_kernel)
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            num_failed += 1
            continue


        fname = os.path.basename(img_file)
        fname_json = fname + '.json'
        json_fname = os.path.join(out_dir, fname_json)
        with open(json_fname, 'w') as f:
            json.dump(anno_data, f, indent=4)
            
        # print(f"Saved annotation JSON to {json_fname}")
        if drawing:
            colors = vtk_utils.get_teeth_color_table(normalize=False)
            colors_image = colors[mask]
            
        # :
        
            bboxes = np.asarray(bboxes)
            if bboxes.size > 0:
                bboxes = bboxes[:, [1, 0, 3, 2]]  # Convert to (ymin, xmin, ymax, xmax) format
            # bboxes = np.array(bboxes, dtype=np.float32)

            colors_image = utils_numpy.apply_blending_mask(image, colors_image, alpha=0.5)
            # draw_bboxes(colors_image, bboxes, thickness=2)
            # colors_image[0] = 255
            ok = cv2.imwrite(f'results/{fname}.png', colors_image[..., ::-1])
    print(f'tufts_to_coco: converted {num_items - num_failed}/{num_items} images -> {out_dir} (failed {num_failed})')
    return out_dir


if __name__ == '__main__':
    # import 시 실행되어 ann/*.json을 덮어쓰지 않도록 __main__ 에서만 실행
    # 보통은 python -m rfdetr.cli.main --build_tufts 로 실행한다
    import argparse
    parser = argparse.ArgumentParser(description='Tufts teeth_polygon.pkl -> kaggle_2222/ann/*.json')
    parser.add_argument('--tufts_dir', type=str, default=None, help='kaggle_2222 폴더')
    parser.add_argument('--out_dir', type=str, default=None, help='기본: {tufts_dir}/ann (기존 json은 자동 백업)')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--close_kernel', type=int, default=0, help='0: 원본 그대로(IoU 0.993). 3: 경계가 약 4%% 부풀어짐')
    parser.add_argument('--drawing', action='store_true')
    args = parser.parse_args()
    tufts_to_coco(tufts_dir=args.tufts_dir, out_dir=args.out_dir, drawing=args.drawing, limit=args.limit,
                  close_kernel=args.close_kernel)