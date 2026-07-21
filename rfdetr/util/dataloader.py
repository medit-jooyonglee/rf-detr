import os

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
    with open(file.replace('.json', '.pkl'), 'wb') as f:  
        pickle.dump(data, f)
        
    return data


# data = read_json()
data = read_pickle()
    
    
    
    
assert isinstance(data, list), "Expected a list of annotations in the JSON file."

idx = 1

# def extract_annotation_info(gt_data, idx):
    
    
                    
#     extracted_data.append({
#         "class_title": class_title,
#         "class_id": class_id,
#         "bbox": bbox,         # [x1, y1, x2, y2]
#         "segmentation": polygon.tolist() # 리스트 형태의 폴리곤 좌표
#     })
                
for idx in range(10):
    data0 = data[idx]
    # ict_keys(['Label', 'External ID'])          
    labels = data0['Label']

    img_file_name = data0['External ID']

    img_file = os.path.join(os.path.dirname(file), '../Radiographs', img_file_name)

    assert os.path.exists(img_file), f"Image file {img_file} does not exist."


    # # for label in labels:
    #     title = label['title']
    #     print(title
    # 
    image = cv2.imread(img_file, cv2.IMREAD_COLOR)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    universal_to_fdi = np.array([
        0,  # Index 0
        18, 17, 16, 15, 14, 13, 12, 11,  # 1~8 (우상)
        21, 22, 23, 24, 25, 26, 27, 28,  # 9~16 (좌상)
        38, 37, 36, 35, 34, 33, 32, 31,  # 17~24 (좌하)
        41, 42, 43, 44, 45, 46, 47, 48   # 25~32 (우하)
    ])
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # _keys(['title', 'bounding box', 'polygons'])
    bboxes = []
    for label in labels['objects']:
        title = label['title']
        print(title)
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
            if title == '19':
                print(pts.shape)
            # print(pts.shape)
            if len(pts) < 3:
                continue
            pts_list.append(pts)

        universal_index = int(title)
        fdi_index = universal_to_fdi[universal_index]
        
        res = cv2.fillPoly(mask, pts_list, color=int(fdi_index))
        
        # concat = np.concatenate(pts_list, axis=0).reshape([-1, 2])
        # bmin, bmax = np.min(concat, axis=0), np.max(concat, axis=0)
        
        # submask = mask[bmin[1]:bmax[1], bmin[0]:bmax[0]]
        
        # res = cv2.fillPoly(mask, [concat], color=int(fdi_index))
        
        # mask = cv2.morphologyEx(submask == , cv2.MORPH_OPEN, kernel)
        
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    from trainer import vtk_utils, utils_numpy

    colors = vtk_utils.get_teeth_color_table(normalize=False)
    
    # colors[10:20, :] = np.array([255, 0, 0])
    # colors[20:30, :] = np.array([0, 255, 0])
    # colors[30:40, :] = np.array([0, 0, 255])
    # colors[40:50, :] = np.array([255, 255, 0])
    # 라벨링/트레이싱 노이즈로 생기는 가시 형태의 돌기만 제거 (치아 사이 홈은 보존해야 하므로 open만 적용, close/approxPolyDP는 실제 윤곽까지 뭉개서 제외)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    # colors[0, :] = 0
    colors_image = colors[mask]
    
    # (x, y)
    bboxes = np.asarray(bboxes)[:, [1, 0, 3, 2]]  # Convert to (ymin, xmin, ymax, xmax) format
    # bboxes = np.array(bboxes, dtype=np.float32)

    colors_image = utils_numpy.apply_blending_mask(image, colors_image, alpha=0.5)
    draw_bboxes(colors_image, bboxes, thickness=2)
    # colors_image[0] = 255
    ok = cv2.imwrite(f'results/temp_{idx}.png', colors_image[..., ::-1])

