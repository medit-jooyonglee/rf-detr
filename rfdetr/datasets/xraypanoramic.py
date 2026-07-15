# import torch.utils.data import Data
import tqdm
# import math
import glob
import json
import os
import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# from shapely import transform
from torch.utils.data.dataloader import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Literal
from trainer import diskmanager, get_logger, vtk_utils, timefn, image_utils
from trainer import vtk_utils, geometry_numpy, get_logger, time_strftime, utils_numpy, torch_utils
from trainer.image_utils import cv2_imread, cv2_imwrite
# from reversereg.preproc.sampler import cv2_imread, cv2_imwrite, to_rgba, blend_images
from rfdetr.datasets.coco import CocoDetection


# E:\temp\miccai_ct\img

"""https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-imagessummary_
"""
class XrayPnoramic(Dataset):
    def __init__(self, 
                 name='train',
                 splits={},
                 path_lists=[],
                 target_mode: Literal['edge'] = 'edge',
                 **kwargs):
        super(XrayPnoramic, self).__init__()
        
        self.meta_kaggle_mapping: np.ndarray = None
        self.name = name
        self.path_lists = path_lists
        self.splits = splits.get(name, (0, 1))
        
        
        self.target_width = 640
        self.source_files = []
        self.gt_files = []
        self.deep_search_files(path_lists)
        self.target_mode = target_mode
        
        self.mapping = np.zeros(256, dtype=np.int64)
            # upper teeth: 1-16, lower teeth: 17-32
        # fdi-> label

        

        

    def read_meta_kaggle_mapping(self, file):
                # file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/meta.json'
        with open(file, "r") as f:
            meta_data = json.load(f)

        # meta.json의 title은 FDI가 아니라 Universal Numbering System(1~32) 값이다.
        # self.mapping은 FDI 코드를 key로 하므로(마지막의 fdi_sort 참고), 여기서 미리
        # universal(1~32) -> FDI(11~48)로 변환해 둬야 self.mapping[self.meta_kaggle_mapping]가 맞게 동작한다.
        mask_to_fdi = {0: 0}
        color_table = []
        labels = []
        for idx, clsz in enumerate(meta_data["classes"]):
            mask_pixel_val = idx + 1
            title = clsz['title']
            if title.isdigit():
                pass
                color_hexa = clsz['color'][1:]
                rgb = tuple(int(color_hexa[i:i+2], 16) for i in (0, 2, 4))
                color_table.append(rgb)
                
                universal_num = int(title)
                
                
                # universal_num = int(title)
                labels.append(universal_num)
                # if 1 <= universal_num <= 32:
                    # mask_to_fdi[mask_pixel_val] = int(fdi_sort[universal_num - 1])
        color_table = np.array(color_table, dtype=np.uint8)
        color_table = np.concatenate([[(0, 0, 0)], color_table], axis=0)
        # color_table[:, ]
        # mapping = np.zeros_like)
        # k = np.array(list(mask_to_fdi.keys()))
        # v = np.array(list(mask_to_fdi.values()))

        # mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)
        # mapping_ar[k] = v
        return color_table
                
        
    def deep_search_files(self, path_lists):
        logger = get_logger()
        
        for path in path_lists:
            # meta_file = os.path.join(path, 'meta.json')
            # if os.path.exists(meta_file):
            # meta-file kaggle
            meta_file = os.path.join(path, '../meta.json')
            if os.path.exists(meta_file):
                # with open(meta_file, "r") as f:
                    # meta_data = json.load(f)
                    
                mapping = self.read_meta_kaggle_mapping(meta_file)
                self.meta_kaggle_mapping = mapping
                
                
                img_files = glob.glob(os.path.join(path, 'img/*.jpg'))
                # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
                annot_files = glob.glob(os.path.join(path, 'ann/*.json'))
                # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
                img_files = sorted(img_files)
                mask_files = sorted(annot_files)
                if len(img_files) == len(annot_files):
                    pass
                else:
                    logger.warning(f'number of image files and mask files do not match in path: {path}')
                    src_fname = [os.path.splitext(os.path.basename(name))[0] for name in img_files]
                    mask_fname = [os.path.splitext(os.path.basename(name))[0] for name in mask_files]
                    commons, args_a, args_b = np.intersect1d(src_fname, mask_fname, return_indices=True)
                    img_files = [img_files[i] for i in args_a]
                    mask_files = [mask_files[i] for i in args_b]
                start, end = self.splits
                start, end = int(len(img_files) * start), int(len(img_files) * end)
                
                self.source_files.extend(img_files[start:end])
            
                self.gt_files.extend(mask_files[start:end])
                
        
    def get_target_image_size(self, image_shape):

        size =  get_target_image_size(image_shape, self.target_width)
        if self.stride > 1:
            # assert self.stride in [2, 4, 8], 'only stride 2 or 4 is supported'
            size = (size[0] // self.stride + 1) * self.stride, (size[1] // self.stride  + 1 )* self.stride
        return size

    def __len__(self):
        return len(self.source_files)
        
    
    def preprocesing_target_edge_binary(self, target):
        # pass
        if self.target_mode == 'edge':
            # Canny edge 검출로 경계선 추출
            # 1. Sobel 계산 (기울기는 음수가 나올 수 있으므로 float64로 유지)
            mask_float = target.astype(np.float32)
            sobel_x = cv2.Sobel(mask_float, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(mask_float, cv2.CV_64F, 0, 1, ksize=3)

            # 2. 절대값을 취하고 8비트 정수로 변환 (곱 연산 없음)
            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)

            # 3. 단순히 더하기 (가중치를 주고 싶다면 cv2.addWeighted 사용)
            magnitude_approx = cv2.add(abs_sobel_x, abs_sobel_y)

            # 4. 이진화
            threshold_value = 0.5
            _, edge_mask = cv2.threshold(magnitude_approx, threshold_value, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            edge_mask_posit = edge_mask > 0
            #  * mapping_target

            
            res = edge_mask_posit * target
            res2 = cv2.dilate(res.astype(np.uint8), kernel, iterations=2, dst=edge_mask) # 경계선이 너무 얇아서 팽창 연산 추가

            res2 = self.mapping[res2]
            return res2
            
        else:
            pass
        return target

                 
    def _read_image(self, index):
        img_file = self.source_files[index]
        mask_file = self.gt_files[index]
        
        src = cv2_imread(img_file, flags=cv2.IMREAD_GRAYSCALE)
        target = cv2_imread(mask_file, flags=-1)
        target = target[..., 0] if target.ndim == 3 else target # instance id만 남기기 (0: 배경)
        
        target_img_size = self.get_target_image_size(src.shape[:2])
        cv_size = tuple(target_img_size[::-1])
        src = cv2.resize(src, cv_size, interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, cv_size, interpolation=cv2.INTER_NEAREST)
        src = (src / 255.).astype(np.float32)
        
        src = np.transpose(src, [2, 0, 1]) if src.ndim == 3 else src[None, ...]

        mapping_target = self.meta_kaggle_mapping[target]
        proc_target = self.preprocesing_target_edge_binary(mapping_target)
        if False:
            mapping = np.array([0, 128, 255], dtype=np.uint8)
            os.makedirs('outputs/results', exist_ok=True)
            cv2.imwrite(f'outputs/results/{time_strftime()}.png', mapping[proc_target])
        # print(np.unique(proc_target))
        # proc_target = (proc_target > 0).astype(np.int64)
        return src, proc_target
                
    def __getitem__(self, index):
        # return super().__getitem__(index)
        return self._read_image(index)

universal_to_fdi = np.array([
    0,  # Index 0
    18, 17, 16, 15, 14, 13, 12, 11,  # 1~8 (우상)
    21, 22, 23, 24, 25, 26, 27, 28,  # 9~16 (좌상)
    38, 37, 36, 35, 34, 33, 32, 31,  # 17~24 (좌하)
    41, 42, 43, 44, 45, 46, 47, 48   # 25~32 (우하)
])

# fdi_sort = np.concatenate([

#     np.arange(11, 19)[::-1],
#     np.arange(21, 29),
#     np.arange(31, 39)[::-1],
#     np.arange(41, 49),
# ])
fdi_sort = universal_to_fdi

label2fdi = np.zeros(256, dtype=np.int64)
label2fdi[np.arange(fdi_sort.size)] = fdi_sort
label2fdi2 = np.zeros_like(label2fdi)
label2fdi2[np.arange(fdi_sort.size)] = np.concatenate([
    [0], np.full([16], 1, dtype=np.int64), np.full([16], 2, dtype=np.int64)
])


def label_to_fdi(labels, num_classes:int):
    if num_classes == 3:
        return label2fdi2[labels]   
    elif num_classes == 33:
        # return labels
        return label2fdi[labels]


class XrayPnoaramicInstance(XrayPnoramic):
    def __init__(self, 
                 
                #  stride=1,
                #  include_masks=False,
                 
                 img_folder,
                 annot_file='',
                 transforms=None,
                 include_masks=False,
                 name='train',
                 debug=False,
                 split=None,
                 stride=64,
                 num_classes=3,
                 **kwargs):
        
        
        XrayPnoramic.__init__(self, 
                              path_lists=[img_folder],
                              
                              **kwargs)
        assert num_classes in [3, 33], 'upper & lower 3 or all intsnace classe 33'
        self.num_classes = num_classes
        
                #          name='train',
                #  splits={},
                #  path_lists=[],
                #  target_mode: Literal['edge'] = 'edge',
                 
        if os.path.exists(annot_file):
            CocoDetection.__init__(img_folder, annot_file, None, False)
        # params = {
        self.stride = stride
        self.include_masks = include_masks 
        # fdi_sort = [
        
        #     np.arange(11, 19)[::-1],
        self.mapping = np.arange(256, dtype=np.int64)
        # self.mapping[:] = 33
        self.mapping[0] = 0 
        # self.mapping[fdi_sort] = np.arange(1, 33)
        
        
    def processing_target_offset(self, target):
        # target
        
        target_posit = target > 0
        posit_inds = np.where(target_posit.ravel())[0]
        target_posit_value = target[target_posit]
        
        
        
        shape = target.shape[:2]
        # inds = np.where(target_posit.ravel())[0]
        uni_inds = utils_numpy.unique_indices(target_posit_value)
        # concatenate and yx->xy
        label_pose = {k: np.stack(np.unravel_index(posit_inds[v], target.shape[:2]), axis=-1)[:, ::-1] \
            for k, v in uni_inds.items()}
        
        
        offset_img = np.zeros([2, *shape], dtype=np.float32)
        scale_term = np.array(shape)[::-1]
        for k, v in label_pose.items():
            
            center = v.mean(axis=0)
            offset = (center - v) / scale_term
            # label_pose[k] = offset
            ix, iy = v[:, 0], v[:, 1]
            offset_img[:, iy, ix] = offset.T 
        
        return offset_img
        
    
    def stride_resize(self, target_label, target_offset):
        if self.stride > 1:
            assert self.stride in [2, 4, 8], 'only stride 2 or 4 is supported'
            stride_size = np.array(target_label.shape[::-1]) // self.stride
            target_label = cv2.resize(target_label, stride_size, interpolation=cv2.INTER_NEAREST).astype(target_label.dtype)
            target_offset = cv2.resize(target_offset.transpose(1, 2, 0), stride_size, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        return target_label, target_offset

    
    def parse_item_coco(self, index, norm_bbox=True, box_format: Literal['xcycwh', 'xywh']='xcycwh'):
        img_file = self.source_files[index]
        mask_file = self.gt_files[index]
        
        def extract_annotation_info(ann_file):
            with open(ann_file, "r") as f:
                data = json.load(f)
            
            extracted_data = []
            
            for obj in data["objects"]:
                class_title = obj["classTitle"]
                class_id = obj["classId"]
                
                # 1. Polygon 데이터 추출 (exterior points)
                # points는 [[x1, y1], [x2, y2], ...] 형태의 리스트
                polygon = np.array(obj["points"]["exterior"], dtype=np.int32)
                
                # 2. 바운딩 박스(BBox) 계산: x, y 각각의 min과 max 추출
                x_coords = polygon[:, 0]
                y_coords = polygon[:, 1]
                
                bbox = [
                    np.min(x_coords), # x_min
                    np.min(y_coords), # y_min
                    np.max(x_coords), # x_max
                    np.max(y_coords)  # y_max
                ]
                
                extracted_data.append({
                    "class_title": class_title,
                    "class_id": class_id,
                    "bbox": bbox,         # [x1, y1, x2, y2]
                    "segmentation": polygon.tolist() # 리스트 형태의 폴리곤 좌표
                })
                
            return extracted_data

        
        def draw_contour(image, polygons, color=(0, 255, 0), thickness=2):
            pts = np.array(polygons, dtype=np.int32).reshape((-1, 1, 2))
            
            # 윤곽선 그리기
            cv2.drawContours(image, [pts], -1, color, thickness)
            
            return image
        
        
        def draw_segmentation(image, polygons, color=(255, 255, 255)):
            """
            image: 그릴 대상 이미지
            polygons: [[x1, y1], [x2, y2], ...] 형태의 좌표 리스트 (배열)
            color: 채울 색상
            """
            # OpenCV는 좌표를 (N, 1, 2) 형태의 int32 배열로 요구합니다.
            pts = np.array(polygons, dtype=np.int32).reshape((-1, 1, 2))
            
            # 내부 채우기 (이미지 자체에 수정이 가해짐)
            cv2.fillPoly(image, [pts], color=color)
            
            return image

                
        annot_data = extract_annotation_info(mask_file)
        
        keys = ['class_title', 'class_id', 'bbox', 'segmentation']
        seg_polygos = [np.array([obj[key] for key in keys], dtype=object) for obj in annot_data]
        # *w, h, w, h format
        bboxes = np.array([obj['bbox'] for obj in annot_data], dtype=np.float32)
        # ()
        class_labels = np.array([int(obj['class_title']) for obj in annot_data], dtype=np.int64)
        
        class_labels = label_to_fdi(class_labels, self.num_classes)
                
        
        
        
        src = cv2_imread(img_file, flags=cv2.IMREAD_GRAYSCALE)
        # target = cv2_imread(mask_file, flags=-1)
        # target = target[..., 0] if target.ndim == 3 else target # instance id만 남기기 (0: 배경)
        # 
        target_img_size = self.get_target_image_size(src.shape[:2])
        cv_size = tuple(target_img_size[::-1])
        src_rsz = cv2.resize(src, cv_size, interpolation=cv2.INTER_LINEAR)
        scale_wh = np.array(cv_size) / np.array(src.shape[::-1])
        
        
        # src_rsz.shape[:2] * scale_wh
        # target_rsz = cv2.resize(target, cv_size, interpolation=cv2.INTER_NEAREST)
        
        # map_kaggle_to_label =self.mapping[self.meta_kaggle_mapping]
        
        # print('raw - label', np.unique(target_rsz))
        # mapping_target = map_kaggle_to_label[target_rsz]
        # # 
                # mapping_target = self.meta_kaggle_mapping[target]
        # proc_target = self.preprocesing_target_edge_binary(mapping_target)
        debug = False
        if debug:
            
            # target_fdi = label_to_fdi(target_rsz)
        
            from trainer import vtk_utils
            colors_fdis = vtk_utils.get_teeth_color_table(normalize=False)

            drawing = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            
            for annot in annot_data:
                # color = 
                fdi = label_to_fdi(int(annot['class_title']), self.num_classes)
                color = colors_fdis[fdi]
                
                
                # drawing = draw_contour(drawing, annot['segmentation'], np.random.randint(0, 255, 3).tolist(), thickness=2)
                drawing = draw_segmentation(drawing, annot['segmentation'], color.tolist())

            
            # res = utils_numpy.apply_blending_mask(drawing, target_color_img, alpha=0.5)
            # cv2.
            # mapping = np.array([0, 128, 255], dtype=np.uint8)
            os.makedirs('outputs/results', exist_ok=True)
            cv2.imwrite(f'outputs/results/{time_strftime()}.png', drawing[..., ::-1])
        
        # indices = utils_numpy.unique_indices(mapping_target.ravel())
        # indices.pop(0, None)
        
        # # indices
        # bboxes = []
        # labels = []
        # scale = 1.15
        # for k, v in indices.items():
        #     ij = np.unravel_index(v, mapping_target.shape)
        #     bbox = np.stack(ij, axis=-1)
            
        #     vmin, vmax = np.min(bbox, axis=0), np.max(bbox, axis=0)
        #     center = (vmin + vmax) / 2
        #     ext = (vmax - vmin) * scale / 2
        #     vmin0, vmax0 = center - ext, center + ext
        #     scale_bboxes = np.concatenate([vmin0, vmax0])
            
        #     bboxes.append(scale_bboxes)
        #     labels.append(k)
            
        # bboxes = np.array(bboxes)
        # labels = np.array(labels)
        if bboxes.size == 0:
            # num_negat = np.random.uniform()
            num_negat = 10
            ctr = np.random.uniform(0, src.shape[::2][::-1], [num_negat, 2])
            wh = np.random.uniform([10, 30], [20, 50], [num_negat, 2])
            vmin = ctr - wh / 2
            vmax = ctr + wh / 2
            bboxes = np.concatenate([vmin, vmax], axis=1)
            class_labels = np.zeros([num_negat], dtype=np.int64)

        # shape = input_img.shape[:2]
        # box_shape = np.array([w, h, w, h])
        
        if self.include_masks:
            raise NotImplementedError("Mask generation from polygons is not implemented yet.")
            # (num_instances, height, width)
            masks = labels.reshape(
                [labels.size, 1, 1]) == mapping_target[None]
        
            area = masks.reshape(masks.shape[0], -1).sum(axis=1) if masks.size > 0 else np.zeros([0])
        else:
            box_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            area = box_area
            
            
        if norm_bbox:
            norm_scale = scale_wh / np.array(cv_size)
            # box_shape = np.array(mapping_target.shape)
            bboxes = (bboxes.reshape([-1, 2]) * norm_scale).reshape(bboxes.shape)

        # if bboxes.size == 0:

            
            # raise ValueError(f"No bounding boxes found for index {index} in file {img_file}")
            
            # print(f'empty{index}')

        if box_format == 'xcycwh':
            # training & validate stage
            bboxes = xyxy_xcycwh(bboxes)
        elif box_format == 'xywh':
            # coco dataset format
            bboxes = xyxy_to_xywh(bboxes)
        
        # bboxes = utils_numpy.box_cxcywh_to_xyxy(bboxes)


            
        orig_shape = np.array(src.shape[:2])
        shape = np.array(src_rsz.shape[:2])
        annot = dict(
            boxes=bboxes,
            labels=class_labels,
            image_id=np.array([index]),
            area=area,
            is_crowd=np.zeros([bboxes.shape[0]], dtype=np.int64),
            # masks=masks,
            orig_size=orig_shape,
            size=shape,
        )
        if self.include_masks:
            annot['masks'] = masks
            
        # src_rsz_permute = np.transpose(src_rsz, [2, 0, 1]).copy()
        # on color
        src_rsz = (src_rsz / 255)[None]
        return src_rsz, annot

        
    def coco_json_export(self, base_dir='', debug_break=-1):
        
        class_names = [
            'gum', *[str(v) for v in fdi_sort], 'unknown'

        ]
        type_categories = [

        ]
        for i, name in enumerate(class_names):
            type_categories.append({
                "id": i,
                "name": name,
                'supercategory': 'tooth_type',
            })
        

        coco_json = {
            "images": [],
            "annotations": [],
            "categories": type_categories
        }
        
        
        annotations = coco_json["annotations"]
        images = coco_json["images"]
        annotation_id = 0
        for i in tqdm.tqdm(range(len(self))):
            try:
                item = self.parse_item_coco(i, norm_bbox=False, box_format='xywh')
            except ValueError as e:
                
                # print(f"Error processing index {i}: {e}")
                continue
            filename = self.source_files[i]
            relative_filename = os.path.relpath(filename, base_dir)
            # annotation_id = 1
            image, target = item
            height, width = image.shape[1:]
            image_id = i
            # for image_id, img in enumerate(custom_dataset, 1):?
            images.append({
                "id": image_id,
                "file_name": relative_filename,
                "width": height,
                "height": width,

            })
            keys = ['boxes', 'labels', 'area', 'is_crowd']
            bboxes, labels, areas, iscrowd = [target[key] for key in keys]

            for box, label, area, crowd in zip(bboxes, labels, areas, iscrowd):
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),  # [x, y, width, height]
                    "area": float(area),
                    "iscrowd": int(crowd),
                })
                annotation_id += 1
            if debug_break > 0 and debug_break > 20:
                # logg
                break

        return coco_json

        
    def __getitem__(self, index):
        item = self.parse_item_coco(index)
        # img, target = torch_utils.data_convert(item, device='cpu')
        img, target = torch_utils.data_convert(item, device='cpu')
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # if self._transforms is not None:
            # img, target = self._transforms(img, target)
        return img, target

            
def xyxy_xcycwh(in_bboxes):
    shape = in_bboxes.shape
    bboxes = in_bboxes.reshape([-1, 4])
    

    ctr = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    ext = (bboxes[:, 2:] - bboxes[:, :2]) 
    
    bboxes = np.concatenate([ctr, ext], axis=1)
    return bboxes.reshape(shape)


def xyxy_to_xywh(in_bboxes):
    shape = in_bboxes.shape
    bboxes = in_bboxes.reshape([-1, 4])
    
    wh = bboxes[:, 2:] - bboxes[:, :2]
    bboxes = np.concatenate([bboxes[:, :2], wh], axis=1)
    return bboxes.reshape(shape)


class XrayPnoaramicInstanceCoco(CocoDetection):
    def __init__(self, img_folder, annot_file, transforms=None, include_masks=False, **kwargs):
        CocoDetection.__init__(self, img_folder, annot_file, transforms, include_masks)
        self.include_masks = include_masks
        self.base_datset = XrayPnoaramicInstance(
            img_folder,
        )
        
    def __len__(self):
        return len(self.base_datset)
        # return 10
        
    def __getitem__(self, index):
        item = self.base_datset.parse_item_coco(index)
        # img, target = torch_utils.data_convert(item, device='cpu')
        img, target = torch_utils.data_convert(item, device='cpu')
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        img = np.repeat(img, 3, axis=0) if img.ndim == 3 and img.shape[0] == 1 else img
        # if self._transforms is not None:
            # img, target = self._transforms(img, target)
        return img, target
    
    
    
    
def get_target_image_size(image_shape, rererence_width:int = 640):
    ih, iw = image_shape[:2]
    multiple = 16
    
    scale = rererence_width / iw
    # else:
        # scale = 
    # ih = (ih * scale + multiple - 1) // multiple * multiple
    ih = int(np.ceil(ih * scale / multiple) * multiple)
    return (ih, rererence_width)
        



        

def main_xraypanoramic():
    pass

    dataset = XrayPnoramic(
        path_lists=[
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
        ],
        
    )
    
    # assert len
    save_path = 'e:/temp/reverse_registration/xray'
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(dataset)):
        
        src, target = dataset[i]
        src = np.transpose((src * 255).astype(np.uint8), [1, 2, 0])
        print(src.shape, target.shape)
        
        table = np.random.randint(0, 255, [256, 3]).astype(np.uint8)
        table[0] = 0
        target_color = table[target]
        # target_a = to_rgba(target)
        res = blend_images(src, target_color, 0.5)
        cv2.imwrite(os.path.join(save_path, f'res_{i}.png'), res.astype(np.uint8))
        
        
        
        
def main_xraypanoramic_instance():
    from trainer import torch_utils
    dataset = XrayPnoaramicInstance(
        path_lists=[
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
        ],
        stride=4,
    )
    
    def to_color_mapping(target_label, label_range):
        l_min, l_max = label_range 
        
        
        rainbow_color = vtk_utils.get_rainbow_color_table(l_max - l_min + 1, normalized=False)
        rainbow_color = np.array(rainbow_color)
        rainbow_color[0, :] = 0
        
        color_target = rainbow_color[target_label]
        os.makedirs('outputs/result', exist_ok=True)
        return color_target

    # assert len
    save_path = 'e:/temp/reverse_registration/xray/boundary'
    os.makedirs(save_path, exist_ok=True)
    for i in range(len(dataset)):
        
        # print(to)
        try:
            src, target = dataset[i]
        except ValueError as e:
            print(f"Error processing index {i}: {e}")
            continue
        print(torch_utils.get_shape([src, target]))
        src = np.transpose((src * 255).astype(np.uint8), [1, 2, 0])

        target_label, target_offset = target
        
        result = to_color_mapping(target_label, [0, 2])
        cv2.imwrite(os.path.join(save_path, f'label_{i}.png'), result.astype(np.uint8))
        
        restore_offset = target_offset * np.array(target_label.shape[:2][::-1])[:, None, None]
        center_posit = target_label == 1
        pose = np.argwhere(center_posit)[:, ::-1]
        offset_vec = restore_offset[:, center_posit]
        offset_pose = pose + offset_vec.T
        offset_pose_actor = vtk_utils.create_points_actor(offset_pose, point_size=5)
        vtk_utils.split_show([
        # pose,
            offset_pose_actor,pose
            
        
        ], [
            pose
        ])
        
        
def test_build_coco_json():
    
        
    img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # ann_file = ''
    dataset = XrayPnoaramicInstance(img_folder, '', None, True)
    
    dataset = XrayPnoaramicInstance(
        img_folder=
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
        ,
        stride=4,
    )
    assert len(dataset) > 0
    
    res = dataset.coco_json_export(base_dir=img_folder)
    
    # print(res)
    
    with open('xray_coco.json', 'w') as f:
        json.dump(res, f)


        
def test_load_coco_dataset():
    
        
    img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # ann_file = ''
    # dataset = XrayPnoaramicInstanceCoco(img_folder, '', None, True)
    path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle'
    # xray_coco.json'
    
    dataset = XrayPnoaramicInstanceCoco(
        
            # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
            os.path.join(path, 'Teeth Segmentation JSON/d2')
        ,
        os.path.join(path, 'xray_coco.json'),
        None,
        False
    )
    assert len(dataset) > 0
    
    for _ in range(len(dataset)):
        img, target = dataset[_]
        print(torch_utils.get_shape([img, target]))

        img, target = torch_utils.to_numpy([img, target])
        
        
        size = np.array(img.shape[1:])
        target_label = target['labels'] 
        
        target_bboxes = target['boxes']
        
        denorm_bboxes = boxes_to_xyxy(target_bboxes, size[::-1])
        
        drawing = cv2.cvtColor(img[0]*255, cv2.COLOR_GRAY2BGR)
        # denorm_bboxes = denorm_bboxes.reshape([-1, 2]).clip(0, size)
        denorm_bboxes_i = denorm_bboxes.astype(np.int32)
        
        target_fdi = label_to_fdi(target_label)
        
        from trainer import vtk_utils
        colors_fdis = vtk_utils.get_teeth_color_table(normalize=False)
        target_colors = colors_fdis[target_fdi]
        
        draw_bboxes(drawing, denorm_bboxes_i, target_colors, xy_format='xy')
        cv2.imwrite(f'outputs/result/xray_{_}.png', drawing[..., ::-1])
        # for box in denorm_bboxes_i:
        #     cv2.rectangle
        

def draw_bboxes(image, bboxes, colors=None, xy_format='yx'):
    bboxes = np.asarray(bboxes)
    bboxes = bboxes.reshape([-1, 4])
    if xy_format == 'yx':
        bboxes = bboxes[:, [1, 0, 3, 2]]
    
    for i, box in enumerate(bboxes):
        if colors is not None:
            color = colors[i]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(
            box[2]), int(box[3])), tuple(map(int, color)), 2)


        
    # res = dataset.coco_json_export(base_dir=img_folder)
def boxes_to_xyxy(boxes, size):
    ctr, hw = np.split(boxes, 2, axis=1)
    vmin = ctr - hw / 2
    vmax = ctr + hw / 2
    scale = np.asarray(size)
    scale = np.concatenate([scale, scale], axis=0)
    return np.concatenate([vmin, vmax], axis=1) * scale[None]
    # x0 = (x_c - 0.5 * w) * img_w
    
    # y0 = (y_c - 0.5 * h) * img_h
    # x1 = (x_c + 0.5 * w) * img_w
    # y1 = (y_c + 0.5 * h) * img_h
    # return np.concatenate([x0, y0, x1, y1], axis=1)



    # with open('xray_coco.json', 'w') as f:
    #     json.dump(res, f)



def build(image_set, args, resolution):
    img_folder = str(args.dataset_dir)
    # annotation file is builed by functoin . see teeth.py::test_build_teethdsata
    annot_file = os.path.join(img_folder, '../../xray_coco.json')
    args_dict = dict(args.__dict__)

    dataset = XrayPnoaramicInstanceCoco(
        img_folder=img_folder,
        annot_file=annot_file,
        transforms=None,
        name=image_set,
        **args_dict
    )
    return dataset


# def test_
        
if __name__ == '__main__':
    # main_xraypanoramic_instance()
    test_build_coco_json()
    
    # test_load_coco_dataset()
