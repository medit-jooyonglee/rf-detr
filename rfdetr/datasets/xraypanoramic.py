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
from trainer import vtk_utils, geometry_numpy, get_logger, time_strftime, utils_numpy, torch_utils, get_logger
from trainer.image_utils import cv2_imread, cv2_imwrite
# from reversereg.preproc.sampler import cv2_imread, cv2_imwrite, to_rgba, blend_images
from rfdetr.datasets.coco import CocoDetection


# E:/temp/miccai_ct\img

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
    
    
    def search_kaggle_data01(self, path, meta_file):
        logger = get_logger()
        
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
        return img_files[start:end], mask_files[start:end]
    
    
    def search_kaggle_data02(self, path):
        # pass
        logger = get_logger()
    
            
        img_files = glob.glob(os.path.join(path, 'Radiographs/*.*'))
        # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
        annot_files = glob.glob(os.path.join(path, 'ann/*.json'))
        if len(img_files) > 0 and len(img_files) > 0:
            # pass
        # if len(img_files) == len(annot_files):
            # pass
            # else:
            logger.warning(f'number of image files and mask files do not match in path: {path}')
            # src_fname = [os.path.splitext(os.path.basename(name))[0] for name in img_files]
            src_fname = [os.path.basename(name).lower() for name in img_files]
            annot_fname = [os.path.splitext(os.path.basename(name))[0].lower() for name in annot_files]
            commons, args_a, args_b = np.intersect1d(src_fname, annot_fname, return_indices=True)
            img_files = [img_files[i] for i in args_a]
            annot_files = [annot_files[i] for i in args_b]
            start, end = self.splits
            start, end = int(len(img_files) * start), int(len(img_files) * end)
            return img_files[start:end], annot_files[start:end]
        else:
            return [], []
        
        # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
        
        
    
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
                    
                # mapping = self.read_meta_kaggle_mapping(meta_file)
                # self.meta_kaggle_mapping = mapping
                
                
                # img_files = glob.glob(os.path.join(path, 'img/*.jpg'))
                # # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
                # annot_files = glob.glob(os.path.join(path, 'ann/*.json'))
                # # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
                # img_files = sorted(img_files)
                # mask_files = sorted(annot_files)
                # if len(img_files) == len(annot_files):
                #     pass
                # else:
                #     logger.warning(f'number of image files and mask files do not match in path: {path}')
                #     src_fname = [os.path.splitext(os.path.basename(name))[0] for name in img_files]
                #     mask_fname = [os.path.splitext(os.path.basename(name))[0] for name in mask_files]
                #     commons, args_a, args_b = np.intersect1d(src_fname, mask_fname, return_indices=True)
                #     img_files = [img_files[i] for i in args_a]
                #     mask_files = [mask_files[i] for i in args_b]
                # start, end = self.splits
                # start, end = int(len(img_files) * start), int(len(img_files) * end)
                img_files, mask_files = self.search_kaggle_data01(path, meta_file)
                # self.source_files.extend(img_files)
                # self.gt_files.extend(mask_files)
            else:
                img_files, mask_files = self.search_kaggle_data02(path)
            self.source_files.extend(img_files)
            self.gt_files.extend(mask_files)
            
                
        
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
label2fdi[fdi_sort] = np.arange(fdi_sort.size)
label2fdi2 = np.zeros_like(label2fdi)
label2fdi2[np.arange(fdi_sort.size)] = np.concatenate([
    [0], np.full([16], 1, dtype=np.int64), np.full([16], 2, dtype=np.int64)
])

def label_to_fdi(labels, num_classes:int=32):
    if num_classes in [32, 33]:
        mapping = np.arange(256, dtype=np.int64)
        mapping[np.arange(fdi_sort.size)] = fdi_sort
        return mapping[labels]
    else:
        return labels
        
        # return label2fdi2[labels]

def label_mapping(labels, num_classes:int):
    if num_classes == 3:
        return label2fdi2[labels]   
    elif num_classes == 33:
        # return labels
        return labels
    else:
        raise NotImplementedError(f'num_classes {num_classes} is not supported for label mapping')
        # return label2fdi[labels]


def create_albu_transform(max_rotate_degree=8, **params):
    import albumentations as A
    from rfdetr.datasets.aug_albumentations import RandomCropWithRoundedBorder, MaskawareImageAug
    
    return A.Compose([
        A.SomeOf(
            transforms=[
                MaskawareImageAug(
                    target_size=(512, 512),
                    crop_scale=(0.6, 0.8),
                    corner_radius=30,
                    border_thickness=4,
                    border_brightness_inc=100,
                    p=0.5,
                ),
                A.CoarseDropout(
                    num_holes_range=(4, 8),          # 지울 패치 개수 범위 (튜플 형태)
                    hole_height_range=(0.02, 0.05),  # 이미지 높이의 2% ~ 5 크기로 설정 (비율)
                    hole_width_range=(0.02, 0.05),      # 패치 최소 너비
                # fill_value=0,         # 채울 색상 (0: 검은색, 127: 회색 등 또는 'random')
                fill=200,        # 실행할 때마다 구멍마다 랜덤 색상이 알아서 채워짐
                p=0.5
            ),
                # A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=max_rotate_degree, p=0.5),
            ],
            n=3,  # 1개 또는 2개 선택
            p=1.0
        )
        ], 
        bbox_params=A.BboxParams(format='pascal_voc'),
        keypoint_params=A.KeypointParams(format='xy', )
    )
    
class XrayPnoaramicInstance(XrayPnoramic):
    def __init__(self, 
                 
                 img_folder,
                 annot_file='',
                 transforms=None,
                 include_masks=False,
                 name='train',
                 debug=False,
                 split=None,
                 stride=64,
                 num_classes=3,
                 splits=None,
                 bg_crop_prob=0.15,
                 augment_en=True,
                 **kwargs):

        default_splits = {
            'train': (0, 0.85),
            'val': (0.85, 0.9),
            'valid': (0.85, 0.9),
            'test': (0.9, 1.0),
        }
        self.augment_en = augment_en
        path_lists = [img_folder] if isinstance(img_folder, str) else img_folder
        XrayPnoramic.__init__(self,
                              name=name,
                              splits=splits if splits is not None else default_splits,
                              path_lists=path_lists,

                              **kwargs)
        self.bg_crop_prob = bg_crop_prob if name == 'train' else 0.0
        logger = get_logger()
        if num_classes in [2, 32]:
            logger.info(f'in case of RF-DETR. 0, background class is not included in num_classes {num_classes}. It will be added automatically.')
            # in case of rf-detr num_classes
            num_classes = num_classes + 1
        assert num_classes in [3, 33], 'upper & lower 3 or all intsnace classe 33'
        self.num_classes = num_classes
        
                #          name='train',
                #  splits={},
                #  path_lists=[],
                #  target_mode: Literal['edge'] = 'edge',
                 
        # if os.path.exists(annot_file):
        #     CocoDetection.__init__(img_folder, annot_file, None, False)
        # params = {
        self.stride = stride
        self.include_masks = include_masks 
        # fdi_sort = [
        
        #     np.arange(11, 19)[::-1],
        self.mapping = np.arange(256, dtype=np.int64)
        # self.mapping[:] = 33
        self.mapping[0] = 0
        
        self.albu_transform = create_albu_transform()
        # self.mapping[fdi_sort] = np.arange(1, 33)

    @staticmethod
    def _no_overlap(crop_box, bboxes_xyxy):
        """True if crop_box (x0,y0,x1,y1) has zero intersection with every box in bboxes_xyxy."""
        if bboxes_xyxy.shape[0] == 0:
            return True
        cx0, cy0, cx1, cy1 = crop_box
        ix0 = np.maximum(cx0, bboxes_xyxy[:, 0])
        iy0 = np.maximum(cy0, bboxes_xyxy[:, 1])
        ix1 = np.minimum(cx1, bboxes_xyxy[:, 2])
        iy1 = np.minimum(cy1, bboxes_xyxy[:, 3])
        inter_w = np.clip(ix1 - ix0, 0, None)
        inter_h = np.clip(iy1 - iy0, 0, None)
        return bool(np.all(inter_w * inter_h == 0))

    def _sample_background_crop(self, src, bboxes_xyxy, scale_range=(0.2, 0.5), max_tries=10):
        """Reject-sample a real sub-region of `src` that overlaps none of `bboxes_xyxy`.

        Returns the cropped image (a genuine background-only patch, not synthetic
        content) or None if no valid region was found within `max_tries`.
        """
        h, w = src.shape[:2]
        for _ in range(max_tries):
            crop_h = int(np.random.uniform(*scale_range) * h)
            crop_w = int(np.random.uniform(*scale_range) * w)
            if crop_h < 8 or crop_w < 8:
                continue
            y0 = np.random.randint(0, h - crop_h + 1)
            x0 = np.random.randint(0, w - crop_w + 1)
            crop_box = np.array([x0, y0, x0 + crop_w, y0 + crop_h], dtype=np.float32)
            if self._no_overlap(crop_box, bboxes_xyxy):
                return src[y0:y0 + crop_h, x0:x0 + crop_w]
        return None

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

    
    def parse_item_coco(self, index, norm_bbox=True, box_format: Literal['xcycwh', 'xywh']='xcycwh', return_raw_annotation=False):
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
        
        
        def draw_segmentation(image, polygons, color=(255, 255, 255), scale=None):
            """
            image: 그릴 대상 이미지
            polygons: [[x1, y1], [x2, y2], ...] 형태의 좌표 리스트 (배열)
            color: 채울 색상
            """
            # OpenCV는 좌표를 (N, 1, 2) 형태의 int32 배열로 요구합니다.
            pts = np.array(polygons, dtype=np.int32).reshape((-1, 1, 2))
            if scale is not None:
                pts = (pts * scale).astype(np.int32)
            # if image.ndim == 3:
                
            # 내부 채우기 (이미지 자체에 수정이 가해짐)
                cv2.fillPoly(image, [pts], color=color)
            elif image.ndim == 2:
                cv2.fillPoly(image, [pts], color=color[0])
                
            
            return image

                
        annot_data = extract_annotation_info(mask_file)
        src = cv2_imread(img_file, flags=cv2.IMREAD_GRAYSCALE)
        keys = ['class_title', 'class_id', 'bbox', 'segmentation']
        # seg_polygos = [np.array([obj[key] for key in keys], dtype=object) for obj in annot_data]
        # *w, h, w, h format
        bboxes = np.array([obj['bbox'] for obj in annot_data], dtype=np.float32).reshape(-1, 4)
        # ()
        class_labels = np.array([int(obj['class_title']) for obj in annot_data], dtype=np.int64)
        if bboxes.size == 0 and class_labels.size == 0:
            shape = np.array(src.shape[:2][::-1])
            num_gen = 3
            bmin = np.random.uniform(0, 0.8, size=(num_gen, 2))
            min_size = np.array([1/25, 1/6])
            
            size = np.random.uniform(min_size, min_size * 1.5, size=(num_gen, 2))
            bmax = bmin + size
            bboxes = np.concatenate([bmin * shape, bmax* shape], axis=-1).astype(np.int64)
            class_labels = np.zeros([  num_gen], dtype=np.int64)
            # pass
            # class_labels = np.zeros((0,), dtype=np.int64)
            
        class_labels = label_mapping(class_labels, self.num_classes)

        



        

        # Train-time only: occasionally replace this sample with a real
        # background-only crop (zero overlap with any GT box) so the model
        # gets genuine "no object" supervision instead of only ever seeing
        # images that contain teeth. This is real image content, not
        # synthetic boxes, so it doesn't corrupt the label distribution.
        # if self.bg_crop_prob > 0 and bboxes.shape[0] > 0 and np.random.uniform() < self.bg_crop_prob:
            # bg_crop = self._sample_background_crop(src, bboxes)
            # if bg_crop is not None:
            #     src = bg_crop
            #     bboxes = np.zeros((0, 4), dtype=np.float32)
            #     class_labels = np.zeros((0,), dtype=np.int64)
            #     annot_data = []
        # target = cv2_imread(mask_file, flags=-1)
        # target = target[..., 0] if target.ndim == 3 else target # instance id만 남기기 (0: 배경)
        # 
        target_img_size = self.get_target_image_size(src.shape[:2])
        cv_size = tuple(target_img_size[::-1])
        src_rsz = cv2.resize(src, cv_size, interpolation=cv2.INTER_LINEAR)
        src_rsz = np.repeat(src_rsz[..., None], 3, axis=-1) if src_rsz.ndim == 2 else src_rsz
        scale_wh = np.array(cv_size) / np.array(src.shape[::-1])
        
        rsz_bboxes = (bboxes.reshape([-1, 2]) * scale_wh).reshape(bboxes.shape)
        # rsz_polygons = 
        if self.include_masks:
            polygons = [annot['segmentation'] for annot in annot_data]
            rsz_polygons = [np.array(poly).reshape([-1, 2]) * scale_wh for poly in polygons]
            if rsz_polygons:
                rsz_polygons = np.concatenate(rsz_polygons, axis=0)
            else:
                rsz_polygons = np.zeros((0, 2), dtype=np.float32)
            polygons_indices = [len(poly) for poly in polygons]
            # rsz_polygons = np.concatenate([poly.reshape(-1, 2) for poly in rsz_polygons], axis=0) if len(rsz_polygons) > 0 else np.zeros((0, 2), dtype=np.float32)
        else:
            rsz_polygons = None
            polygons_indices = None
        
        # rsz_bboxes_labels = np.concatenate([rsz_bboxes, np.zeros([rsz_bboxes.shape[0], 1])], axis=-1)
        if self.name == 'train' and self.augment_en:
            transformed = self.albu_transform(
                image=src_rsz, 
                bboxes=rsz_bboxes, 
                # class_labels=class_labels,
                keypoints=rsz_polygons,
                polygons_indices=polygons_indices,
            )
            
            src_rsz = transformed['image']
            rsz_bboxes = transformed['bboxes']
            trans_rsz_polygons = transformed['keypoints']
        else:
            trans_rsz_polygons = rsz_polygons
            
        rsz_polygons = []
        start = 0
        for size in polygons_indices:
            rsz_polygons.append(trans_rsz_polygons[start:start+size])
            start += size
        
            
            

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
        
        
        # Note: bboxes/class_labels may legitimately be empty here (no teeth
        # annotated, or a background crop was substituted above). The
        # Hungarian matcher and set-based criterion handle zero-target
        # images natively (every query is assigned to "no object"), so we
        # intentionally do NOT fabricate boxes for the empty case.

        # shape = input_img.shape[:2]
        # box_shape = np.array([w, h, w, h])
        polygon_segmentation = []
        if self.include_masks:

            
            # for 
            if rsz_polygons is not None and len(rsz_polygons) > 0:
                # masks = np.zeros([bboxes.shape[0], *src_rsz.shape[:2]], dtype=np.uint8)
                masks = np.zeros([bboxes.shape[0], *src_rsz.shape[:2]], dtype=np.uint8)
                

                for i, poly in enumerate(rsz_polygons):
                    
                    polygon_segmentation.append(
                        np.array(poly).ravel().tolist()
                    )
                    draw_segmentation(masks[i], poly, color=(255, 255, 255), scale=None)
                    # print(i, np.sum(res > 0))
                
                masks = masks.astype(np.bool_)
                if debug:
                    for v in range(masks.shape[0]):
                        cv2.imwrite(f'outputs/results/mask_{v}.png', masks[v])
            else:
                masks = np.zeros([bboxes.shape[0], *src_rsz.shape[:2]], dtype=np.bool_)


            area = masks.reshape(masks.shape[0], -1).sum(axis=1) if masks.size > 0 else np.zeros([0])
        else:
            box_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            area = box_area
            
            
        if norm_bbox:
            # norm_scale = scale_wh / np.array(cv_size)
            norm_scale = 1 / np.array(cv_size)
            # box_shape = np.array(mapping_target.shape)
            bboxes = (rsz_bboxes.reshape([-1, 2]) * norm_scale).reshape(rsz_bboxes.shape)

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
            boxes=bboxes.astype(np.float32),
            labels=class_labels,
            image_id=np.array([index]),
            area=area,
            is_crowd=np.zeros([bboxes.shape[0]], dtype=np.int64),
            # masks=masks,
            segmentation=polygon_segmentation,
            orig_size=orig_shape,
            size=shape,
        )
        if self.include_masks:
            annot['masks'] = masks
            
        # src_rsz_permute = np.transpose(src_rsz, [2, 0, 1]).copy()
        # on color
        # src_rsz = (src_rsz / 255)[None].astype(np.float32)
        src_rsz = np.transpose(src_rsz, [2, 0, 1]).copy().astype(np.float32) / 255.0
        # if return_raw_annotation:
        return src_rsz, annot
        # else:
            # return src_rsz, annot

        
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
            keys = ['boxes', 'labels', 'area', 'is_crowd', 'segmentation']
            bboxes, labels, areas, iscrowd, segmentation = [target[key] for key in keys]

            for i, (box, label, area, crowd) in enumerate(zip(bboxes, labels, areas, iscrowd)):
                

                i_annot = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),  # [x, y, width, height]
                    "area": float(area),
                    "iscrowd": int(crowd),
                }
                if len(segmentation) == len(bboxes):
                    seg = segmentation[i]
                    i_annot["segmentation"] = [seg]
                
                annotations.append(i_annot)
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
            include_masks=include_masks,
            **kwargs,
                            #  include_masks=False,
        )
        
        self.index_coco_id = []
            
        # image id mapping    
        if len(self.coco.imgs) > 0:
            # assumption image id
            meta = [self.coco.loadImgs(i)[0] for i in range(len(self.coco.imgs))]
            img_id_mapping = {v['file_name'].replace('\\', '/').lower(): v['id'] for v in meta}
            image_files = list(img_id_mapping.keys())
            
            source_clean = [name.strip().replace('\\', '/').lower() for name in self.base_datset.source_files]
            
            matched_idx = []
            for abs_path in source_clean:
                for idx, relat in enumerate(image_files):
                    if abs_path.endswith(relat):
                        matched_idx.append(idx)
                        break
                    
            assert len(matched_idx) == len(source_clean), 'some images are not matched'
            self.index_coco_id = matched_idx

            
                          
        
    def __len__(self):
        return len(self.base_datset)
        # return 10
        
    def __getitem__(self, index):
        item = self.base_datset.parse_item_coco(index)
        # img, target = torch_utils.data_convert(item, device='cpu')
        img, target = item
        target.pop('segmentation', None)        
        if self.index_coco_id:
            target.update({'image_id': np.array(self.index_coco_id[index])})
        
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
    
        
    # img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    base_dir = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays'
    # ann_file = ''
    # dataset = XrayPnoaramicInstance(img_folder, '', None, True)
    
    num_classes = 33
    include_masks = True
    dataset = XrayPnoaramicInstance(
        img_folder= [
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222',
            
        ]
        ,
        num_classes=num_classes,
        # stride=4,
        include_masks=include_masks,
        name='train',
        splits={
            'train': (0, 1)
        }
    )
    assert len(dataset) > 0
    
    # num_classes = 3
    
    res = dataset.coco_json_export(base_dir=base_dir)
    
    # print(res)
    filename = f'xray_coco_{num_classes}.json'
    with open(filename, 'w') as f:
        json.dump(res, f)
        
    print(f'coco json file saved to {filename}')


        
def test_load_coco_dataset():
    
    from trainer import get_logger
    
    logger = get_logger()
    # img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle'
    kaggle_path2 = [
        'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222',
        'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
        '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2/',
        '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle_2222/',
    ]
    
    # ann_file = ''
    # ann_file = ''
    # dataset = XrayPnoaramicInstanceCoco(img_folder, '', None, True)
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle'
    # xray_coco.json'
    
    annot_file = '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_coco_33_seg.json'
    # annot_file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/xray_coco_33_seg.json'
    
    dataset = XrayPnoaramicInstanceCoco(
        
            # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
            # os.path.join(path, 'Teeth Segmentation JSON/d2')
            kaggle_path2,
            annot_file,
        # ,
        # os.path.join(path, 'xray_coco.json'),
        # None,
        None,
        include_masks=True,
        num_classes=32,
        name='train',
        # splits={
            # 'train': (0, 0.)
        # }
        
        
    )
    
    print(f'found dataset {len(dataset)}')
    assert len(dataset) > 0
    shape_stats = {
        'inputs': set(),
        'targets': set()
    }
    test_iter = 100
    for i in tqdm.tqdm(range(len(dataset))):
        img, target = dataset[i]
        # print(torch_utils.get_shape([img, target]))
        if i >= test_iter:
            break

        img, target = torch_utils.to_numpy([img, target])
        
        
        size = np.array(img.shape[1:])
        target_label = target['labels'] 
        
        target_bboxes = target['boxes']
        
        if target_bboxes.size == 0:
            logger.error(f"No bounding boxes found for index {i} in dataset.")
        
        segmentation = target.get('masks')
        if segmentation is not None:
            shape_stats['targets'].add(segmentation.shape[1:])
        shape_stats['inputs'].add(img.shape[1:])
        
        
        denorm_bboxes = boxes_to_xyxy(target_bboxes, size[::-1])
        
        drawing = cv2.cvtColor(img[0]*255, cv2.COLOR_GRAY2BGR)
        # denorm_bboxes = denorm_bboxes.reshape([-1, 2]).clip(0, size)
        denorm_bboxes_i = denorm_bboxes.astype(np.int32)
        
        target_fdi = label_to_fdi(target_label)
        
        from trainer import vtk_utils
        colors_fdis = vtk_utils.get_teeth_color_table(normalize=False)
        colors_fdis[0, :] = 0
        target_colors = colors_fdis[target_fdi]
        # target_colors[:, 0] = 0
        # label = fdi = label_to_fdi(pred_label)
        gt_visual = True
        if gt_visual:
            draw_bboxes(drawing, denorm_bboxes_i, target_colors, xy_format='xy')
            masks = target.get('masks')
            if masks is not None:
                target_fdi_mask = target_fdi[:, None, None] * masks
                target_fdi_color_image = colors_fdis[target_fdi_mask]
                target_fdi_color_image = np.max(target_fdi_color_image, axis=0)
                # cv2.imwrite('temp.png', target_fdi_color_image.astype(np.uint8))
                
                drawing = utils_numpy.apply_blending_mask(drawing, target_fdi_color_image, alpha=0.5)
            
        save_dir = 'outputs/result'
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/xray_{i}.png', drawing[..., ::-1])
        
            
        # print(f"Input shapes: {shape_stats['inputs']}")
        # print(f"Target shapes: {shape_stats['targets']}")
    
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
    img_folder = args.dataset_dir
    # annotation file is builed by functoin . see teeth.py::test_build_teethdsata
    annot_file = getattr(args, 'annot_file', '../../xray_coco.json')
    if os.path.exists(annot_file):
        pass
    else:
        annot_file = os.path.join(img_folder, annot_file)
        assert os.path.exists(annot_file), f'annotation file not found: {annot_file}'
    # annot_file = 
    # annot_file
    args_dict = dict(args.__dict__)

    dataset = XrayPnoaramicInstanceCoco(
        img_folder=img_folder,
        annot_file=annot_file,
        transforms=None,
        name=image_set,
        include_masks=args.segmentation_head,
        num_classes=getattr(args, 'num_classes', 2),
        bg_crop_prob=getattr(args, 'bg_crop_prob', 0.15),
        augment_en=False,
        # **args_dict
    )
    
    print(f'=====================Loaded dataset {image_set} with {len(dataset)} =====================')
    return dataset


# def test_
        
if __name__ == '__main__':
    # main_xraypanoramic_instance()
    # test_build_coco_json()
    # test_build_coco_json()
    test_load_coco_dataset()
