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



def create_annotation_info(labels, image, drawing=False, mask=None):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
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
            
        # ex
        fdi_index = universal_to_fdi[universal_index]

        # 치아별로 독립된 바이너리 마스크에 fillPoly + open을 적용해야 한다.
        # 공유 mask(다중 라벨 값)에 바로 open을 걸면 서로 다른 치아의 값이 min/max 필터로 섞여버린다.
        tooth_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(tooth_mask, pts_list, color=255)
        tooth_mask = cv2.morphologyEx(tooth_mask, cv2.MORPH_OPEN, kernel)

        # 정제된(morphology 처리 후) 마스크에서 contour(segmentation) 좌표 추출
        contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anno_data['objects'].append(dict(
            classTitle=str(universal_index),
            classId=str(universal_index),
            bbox=bbox,
            points={
                'exterior': contours[0].reshape(-1, 2).tolist() 
            }
        ))
        
        
        if drawing and mask is not None:
            mask[tooth_mask > 0] = fdi_index
    return anno_data, bboxes

    
def tufts_to_coco():
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
                
                
    universal_to_fdi = np.array([
        0,  # Index 0
        18, 17, 16, 15, 14, 13, 12, 11,  # 1~8 (우상)
        21, 22, 23, 24, 25, 26, 27, 28,  # 9~16 (좌상)
        38, 37, 36, 35, 34, 33, 32, 31,  # 17~24 (좌하)
        41, 42, 43, 44, 45, 46, 47, 48   # 25~32 (우하)
    ])

    drawing = True


    # for idx in range(5):
    for idx in tqdm.tqdm(range(len(data))):
        data0 = data[idx]
        # ict_keys(['Label', 'External ID'])          
        labels = data0['Label']

        img_file_name = data0['External ID']

        img_file = os.path.join(os.path.dirname(file), '../Radiographs', img_file_name)

        assert os.path.exists(img_file), f"Image file {img_file} does not exist."

        
        
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)


        try:
            anno_data, bboxes = create_annotation_info(labels, image, drawing=drawing, mask=mask)
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            continue

        
        fname = os.path.basename(img_file)
        fname_json = fname + '.json'
        
        annot_save_dir = os.path.join(os.path.dirname(file), '../ann')
        os.makedirs(annot_save_dir, exist_ok=True)
        json_fname = os.path.join(annot_save_dir, fname_json)
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


tufts_to_coco()