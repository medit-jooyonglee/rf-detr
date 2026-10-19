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
from PIL import Image

# image resolutsion
# \xray_teeth_seg_kaggle\Teeth Segmentation JSON\d2\img
# (2041, 1024) / (1852, 1024)

# (1615, 850)

# dr. baek
# (2640, 1256)

#  periapical / 치근단
# (645, 515) (66, 614)



# E:/temp/miccai_ct\img

"""https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-imagessummary_
    coco api + custom-datset merge@
    
    # index is as followed
    
    # https://github.com/medit-AI/labelme 
    # from labelme 학습 데이터셋 -> coco-api json으로 변환
    # examples\instance_segmentation\labelme2coco.py::convert_labelme_dirs_to_coco(...)
    
    
    
    1.first coco api 
    2. cumstom -datset
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
    
    
    def read_coco_image(self, index):
        # examples
        image_id = 123

        coco = self.coco_apis.coco
        # image 정보
        image_info = coco.loadImgs([image_id])[0]

        # 해당 image의 annotation id
        ann_ids = coco.getAnnIds(imgIds=[image_id])

        # annotation들
        annotations = coco.loadAnns(ann_ids)
        return 

        # mask_files = glob.glob(os.path.join(path, 'masks_machine/*.png'))
        
        
    
    def deep_search_files(self, path_lists):
        logger = get_logger()
        
        for path in path_lists:
            # meta_file = os.path.join(path, 'meta.json')
            # if os.path.exists(meta_file):
            # meta-file kaggle
            meta_file = os.path.join(path, '../meta.json')
            if os.path.exists(meta_file):

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
        if self.coco_apis is not None:
            return len(self.coco_apis) + len(self.source_files)
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
    
    def _read_coco_image(self, index, resizing=True):
        src, target = self.coco_apis[index]
        src = np.array(src)

        # xray_crop_mode: 원본 해상도 마스크를 직접 crop (③ 자체 데이터, category id = universal 번호)
        crop_plan = None
        if resizing:
            crop_plan = self._sample_crop_plan(
                np.asarray(target['boxes'], dtype=np.float32).reshape(-1, 4),
                np.asarray(target['labels']).reshape(-1),
                src.shape[:2],
                index=index,
            )
        if crop_plan is not None:
            from rfdetr.datasets.xray_crop_aug import corner_mask
            cx0, cy0, cx1, cy1 = [int(v) for v in crop_plan['box']]
            keep = crop_plan['keep']
            src = src[cy0:cy1, cx0:cx1].copy()
            labels = np.asarray(target['labels']).reshape(-1)[keep]
            iscrowd = np.asarray(target['iscrowd']).reshape(-1)[keep]
            valid = None
            if crop_plan['corner'] is not None:
                valid = corner_mask(src.shape[0], src.shape[1], *crop_plan['corner'])
                src = src * (valid[..., None] if src.ndim == 3 else valid)
            if target.get('masks') is not None:
                masks = np.asarray(target['masks'])[keep][:, cy0:cy1, cx0:cx1]
                if valid is not None:
                    masks = masks & valid.astype(bool)[None]
                boxes, area = self._boxes_from_masks(masks)
                visible = area >= 16
                masks, boxes, area = masks[visible], boxes[visible], area[visible]
                target = dict(target, masks=masks)
            else:
                boxes = np.asarray(target['boxes'], dtype=np.float32).reshape(-1, 4)[keep]
                boxes = boxes - np.array([cx0, cy0, cx0, cy0], dtype=np.float32)
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, src.shape[1])
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, src.shape[0])
                visible = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[visible]
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            target = dict(target, boxes=boxes, labels=labels[visible], area=area, iscrowd=iscrowd[visible],
                          orig_size=np.array(src.shape[:2]))

        # src = np.arr
        if resizing:


            # self.resize_image_and_target(src, target)
            target_rsz_img = self.get_target_image_size(src.shape[:2])
            if crop_plan is not None and crop_plan['target_hw'] is not None:
                target_rsz_img = crop_plan['target_hw']
            resizer = ResizeCocoSample(target_rsz_img)
            rsz_src, rsz_target = resizer(src, target)
            rsz_src = np.transpose(rsz_src, [2, 0, 1]) if rsz_src.ndim == 3 else rsz_src[None, ...]
            # return rsz_src, rsz_target
            #  = rsz_target
            src, target = rsz_src, rsz_target
            # return rsz_src, rsz_target

            # Match the RF-DETR target contract used by the JSON datasets:
            # normalized (cx, cy, width, height), not absolute xyxy pixels.
            boxes_xyxy = np.asarray(target['boxes'], dtype=np.float32).reshape(-1, 4)
            norm_xyxy = boxes_xyxy / np.array(
                [target_rsz_img[1], target_rsz_img[0]] * 2,
                dtype=np.float32,
            )
            target['boxes'] = xyxy_xcycwh(norm_xyxy)
        else:
            # c h, w 
            src = np.transpose(src, [2, 0, 1]) if src.ndim == 3 else src[None, ...]
            # return src, target
        
        # masks & segmentation key
        # is_crowd & iscrowd
        
        target['segmentation'] = target.get('masks')
        target['is_crowd'] = target.get('iscrowd')
        src = src / 255.0
        # self.resize_image_and_target(src, target)
        return src, target
                    


    def _read_image(self, index):
        if self.coco_apis is not None:
            num_len = len(self.coco_apis)
            src, target = self._read_coco_image(index) if index < num_len else self._read_custom_image(index - num_len)

            # return self.parse_item_coco(index)
            # coco read
            return src, target
        else:
            return self._read_custom_image(index)
            
        
             
    def _read_custom_image(self, index):
        
        
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
    elif num_classes in [4, 5]:
        mapping = np.arange(256, dtype=np.int64)
        mapping[np.arange(fdi_sort.size)] = fdi_sort // 10
        return mapping[labels]
    elif num_classes == [2, 3]:
        mapping = np.arange(256, dtype=np.int64)
        mapping[np.arange(fdi_sort.size)] = np.where(fdi_sort > 0,  fdi_sort // 20 + 1, 0) 
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
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_indices']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )
    
class XrayPnoaramicInstance(XrayPnoramic):
    def __init__(self, 
                 
                 img_folder,
                 include_masks=False,
                 name='train',
                 debug=False,
                 split=None,
                 stride=64,
                 num_classes=3,
                 splits=None,
                 bg_crop_prob=0.15,
                 augment_en=True,
                 coco_directories: List[Tuple[str, str]]=None,
                 xray_crop_mode: Union[None, str, List[str], Dict[str, float]] = None,
                 xray_crop_prob: Union[float, List[float]] = 1.0,
                 xray_crop_cfg: Optional[Dict] = None,
                 eval_xray_crop: bool = False,
                 eval_xray_crop_seed: int = 42,
                 **kwargs):
        """
        xray_crop_mode: 학습 샘플 crop 방식 (기본 None: 비활성). eval_xray_crop=True면 검증에도 적용.
            'periapical' : 치근단 방사선 모사 (이웃 치아 3~5개, 상악/하악/상하악 노출, 사선/둥근 모서리)
            'teeth_roi'  : two-pass 2차 입력 (전체 치아 영역 + margin, 비관심 영역 제거)
            여러 개 동시 사용: ['periapical', 'teeth_roi'], 'periapical,teeth_roi', {'periapical': 0.3, 'teeth_roi': 0.3}
            상세 규칙/파라미터는 rfdetr/datasets/xray_crop_aug.py
        xray_crop_prob: mode별 선택 확률. float면 모든 mode에 같은 값, list면 mode 순서대로.
            샘플마다 mode 하나를 확률대로 고르고 나머지(1 - 합)는 전체 영상. 합이 1을 넘으면 비율로 정규화
            (예: 두 mode 모두 1.0 -> 0.5 / 0.5, 전체 영상 0%. 두 mode 0.3 -> 0.3 / 0.3 / 전체 영상 0.4)
        xray_crop_cfg: xray_crop_aug.DEFAULT_PERIAPICAL / DEFAULT_TEETH_ROI 덮어쓸 값.
            {'periapical': {...}, 'teeth_roi': {...}} 로 mode별 지정 가능
        """

        default_splits = {
            'train': (0, 0.85),
            'val': (0.85, 0.9),
            'valid': (0.85, 0.9),
            'test': (0.9, 1.0),
        }
        self.coco_directories = coco_directories
        if coco_directories is not None and len(coco_directories) > 0:
            # TODO: complete merge of coco datasets
            assert len(coco_directories) == 1, 'we only support one coco directory for now. '
            coco_img_folder, coco_annot_file = coco_directories[0]
            self.coco_apis = CocoDetection(coco_img_folder, coco_annot_file, None, include_masks)
            
            # \\ -> / handling...
            coco = self.coco_apis.coco
            for image_info in coco.dataset.get("images", []):
                file_name = image_info.get("file_name")

                if isinstance(file_name, str):
                    image_info["file_name"] = file_name.replace("\\", "/")
            coco.createIndex()
        else:
            self.coco_apis = None
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

        from rfdetr.datasets.xray_crop_aug import normalize_crop_modes, split_crop_cfg
        # {mode: 샘플당 선택 확률}. 비어 있으면 crop 비활성
        self.xray_crop_modes = normalize_crop_modes(xray_crop_mode, xray_crop_prob)
        self.xray_crop_cfg = split_crop_cfg(xray_crop_cfg, self.xray_crop_modes)
        self.eval_xray_crop = eval_xray_crop
        self.eval_xray_crop_seed = eval_xray_crop_seed
        if self.xray_crop_modes and (name == 'train' or eval_xray_crop):
            full_prob = max(0.0, 1.0 - sum(self.xray_crop_modes.values()))
            logger.info(f'xray crop augmentation: probs={self.xray_crop_modes}, full_image={full_prob:.2f}, '
                        f'cfg={self.xray_crop_cfg}')

    def _sample_crop_plan(self, boxes_xyxy, labels, img_hw, index=0):
        """원본 해상도 기준 crop 계획 (xray_crop_aug.sample_crop). 비활성/전체 영상 선택/실패 시 None."""
        if not self.xray_crop_modes or (self.name != 'train' and not self.eval_xray_crop) or len(boxes_xyxy) == 0:
            return None
        from rfdetr.datasets.xray_crop_aug import pick_crop_mode, sample_crop
        # 검증에서는 GT 재빌드/worker/epoch와 무관하게 같은 이미지에 같은 crop 적용.
        rng = np.random if self.name == 'train' else np.random.RandomState(
            (self.eval_xray_crop_seed + index) % (2 ** 32)
        )
        mode = pick_crop_mode(self.xray_crop_modes, rng)
        if mode is None:
            return None
        plan = sample_crop(mode, boxes_xyxy, labels, img_hw, rng, train=self.name == 'train',
                           cfg=self.xray_crop_cfg[mode])
        if plan is None or not plan['keep'].any():
            return None
        return plan

    @staticmethod
    def _boxes_from_masks(masks):
        """[N, H, W] bool -> (xyxy [N, 4] float32, area [N])."""
        boxes = np.zeros((len(masks), 4), dtype=np.float32)
        area = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.zeros([0])
        for i, m in enumerate(masks):
            ys, xs = np.nonzero(m)
            if len(xs):
                boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return boxes, area

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

    
    def file_name(self, index):
        if self.coco_apis is not None:
            if index < len(self.coco_apis):
                coco_relative_file = self.coco_apis.coco.loadImgs([self.coco_apis.ids[index]])[0]['file_name']
                coco_base_directory = os.path.dirname(self.coco_directories[0][-1])
                full_path = os.path.join(coco_base_directory, coco_relative_file).replace('\\', '/')
                return full_path
            else:
                index = index - len(self.coco_apis)
                return self.source_files[index]
        else:
            return self.source_files[index]
    
    def parse_item_coco(self, index, norm_bbox=True, box_format: Literal['xcycwh', 'xywh']='xcycwh', return_raw_annotation=False):
        # 
        # self.coco_apis.coco.dataset.get('images')

        
        if self.coco_apis is not None:
            if index < len(self.coco_apis):
                return self._read_coco_image(index)
                # return self.coco_apis[index]
            else:
                index = index - len(self.coco_apis)
                
        else:
            index = index
        
        
        
        
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
            points = np.asarray(polygons)
            if image.size == 0 or points.size < 6:
                return image

            points = points.reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            if points.shape[0] < 3:
                return image

            pts = np.ascontiguousarray(points, dtype=np.int32).reshape((-1, 1, 2))
            if scale is not None:
                pts = np.ascontiguousarray(pts * scale, dtype=np.int32)
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

        # xray_crop_mode (periapical / teeth_roi): 원본 해상도에서 crop 후 이하 기존 파이프라인 그대로.
        # 라벨 매핑 전(universal 번호)이라 상악/하악 구분이 가능하다.
        crop_plan = self._sample_crop_plan(bboxes, class_labels, src.shape[:2], index=index)
        crop_valid = None  # PA 모서리 유효영역 (1: 영상, 0: 잘린 모서리)
        if crop_plan is not None:
            from rfdetr.datasets.xray_crop_aug import corner_mask
            cx0, cy0, cx1, cy1 = [int(v) for v in crop_plan['box']]
            keep = crop_plan['keep']
            src = src[cy0:cy1, cx0:cx1].copy()
            offset = np.array([cx0, cy0], dtype=np.float32)
            bboxes = bboxes[keep] - np.tile(offset, 2)
            class_labels = class_labels[keep]
            annot_data = [
                dict(annot, segmentation=(np.asarray(annot['segmentation'], dtype=np.float32) - offset).tolist())
                for annot, k in zip(annot_data, keep) if k
            ]
            if crop_plan['corner'] is not None:
                crop_valid = corner_mask(src.shape[0], src.shape[1], *crop_plan['corner'])
                src = src * crop_valid

        # annotation이 없는 이미지는 아래에서 랜덤 가짜 박스(class 0)를 만든다. 평가 GT json에는 넣으면 안 되므로 표시해 둔다.
        is_dummy = False
        if bboxes.size == 0 and class_labels.size == 0:
            is_dummy = True
            shape = np.array(src.shape[:2][::-1])
            num_gen = 3
            bmin = np.random.uniform(0, 0.8, size=(num_gen, 2))
            min_size = np.array([1/25, 1/6])
            
            size = np.random.uniform(min_size, min_size * 1.5, size=(num_gen, 2))
            # clamp so bmax never exceeds 1.0 (out-of-bounds boxes crash albumentations validation)
            bmax = np.minimum(bmin + size, 1.0)
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
        if crop_plan is not None and crop_plan['target_hw'] is not None:
            target_img_size = crop_plan['target_hw']  # PA: 긴 변 기준 크기 (가로형/세로형)
        cv_size = tuple(target_img_size[::-1])
        src_rsz = cv2.resize(src, cv_size, interpolation=cv2.INTER_LINEAR)
        src_rsz = np.repeat(src_rsz[..., None], 3, axis=-1) if src_rsz.ndim == 2 else src_rsz
        scale_wh = np.array(cv_size) / np.array(src.shape[::-1])
        rsz_valid = cv2.resize(crop_valid, cv_size, interpolation=cv2.INTER_NEAREST) if crop_valid is not None else None
        
        rsz_bboxes = (bboxes.reshape([-1, 2]) * scale_wh).reshape(bboxes.shape)

        # Clipping an annotation that lies completely outside the image can
        # collapse it to a zero-area box (for example y_min == y_max == 0).
        # Albumentations validates boxes before applying transforms and rejects
        # such boxes, so remove them together with their associated metadata.
        img_h, img_w = src_rsz.shape[:2]
        if rsz_bboxes.shape[0] > 0:
            rsz_bboxes[:, [0, 2]] = np.clip(rsz_bboxes[:, [0, 2]], 0, img_w)
            rsz_bboxes[:, [1, 3]] = np.clip(rsz_bboxes[:, [1, 3]], 0, img_h)
            valid_boxes = (
                np.isfinite(rsz_bboxes).all(axis=1)
                & (rsz_bboxes[:, 2] > rsz_bboxes[:, 0])
                & (rsz_bboxes[:, 3] > rsz_bboxes[:, 1])
            )
            if not valid_boxes.all():
                rsz_bboxes = rsz_bboxes[valid_boxes]
                bboxes = bboxes[valid_boxes]
                class_labels = class_labels[valid_boxes]
                if len(annot_data) == len(valid_boxes):
                    annot_data = [
                        annot for annot, valid in zip(annot_data, valid_boxes)
                        if valid
                    ]

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
        kept_bbox_indices = np.arange(rsz_bboxes.shape[0], dtype=np.int64)
        if self.name == 'train' and self.augment_en:
            extra_targets = {} if rsz_valid is None else {'mask': rsz_valid}  # 회전 등 기하 변환을 모서리 마스크에도 적용
            transformed = self.albu_transform(
                image=src_rsz,
                bboxes=rsz_bboxes,
                bbox_indices=kept_bbox_indices,
                # class_labels=class_labels,
                keypoints=rsz_polygons,
                polygons_indices=polygons_indices,
                **extra_targets,
            )

            src_rsz = transformed['image']
            if rsz_valid is not None:
                rsz_valid = transformed['mask']
            rsz_bboxes = np.asarray(transformed['bboxes'], dtype=np.float32).reshape(-1, 4)
            kept_bbox_indices = np.asarray(
                transformed['bbox_indices'], dtype=np.int64
            )
            bboxes = bboxes[kept_bbox_indices]
            class_labels = class_labels[kept_bbox_indices]
            trans_rsz_polygons = transformed['keypoints']
        else:
            trans_rsz_polygons = rsz_polygons

        if polygons_indices is not None:
            all_rsz_polygons = []
            start = 0
            for size in polygons_indices:
                all_rsz_polygons.append(trans_rsz_polygons[start:start+size])
                start += size
            rsz_polygons = [
                all_rsz_polygons[i]
                for i in kept_bbox_indices
                if i < len(all_rsz_polygons)
            ]
        else:
            rsz_polygons = None
        
            
            

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

            if crop_plan is not None and len(masks) == len(rsz_bboxes) and len(masks) > 0:
                # crop 경계/PA 모서리에서 잘린 부분을 마스크에서 빼고, bbox를 보이는 마스크 기준으로 다시 계산
                if rsz_valid is not None:
                    masks &= rsz_valid.astype(bool)[None]
                mask_boxes, mask_area = self._boxes_from_masks(masks)
                visible = mask_area >= 16  # 거의 안 보이는 치아(가장자리 몇 px)는 제외
                masks, rsz_bboxes = masks[visible], mask_boxes[visible]
                bboxes, class_labels = bboxes[visible], class_labels[visible]
                kept_bbox_indices = kept_bbox_indices[visible]
                polygon_segmentation = [p for p, v in zip(polygon_segmentation, visible) if v]

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
        # coco json export(평가 GT)용: 리사이즈/증강 전 원본 좌표 polygon. 학습 경로에서는 __getitem__에서 pop 된다.
        annot['is_dummy'] = is_dummy
        if self.include_masks and not is_dummy:
            annot['segmentation_orig'] = [
                np.asarray(annot_data[i]['segmentation'], dtype=np.float64).ravel().tolist()
                for i in kept_bbox_indices
                if i < len(annot_data)
            ]

        # src_rsz_permute = np.transpose(src_rsz, [2, 0, 1]).copy()
        # on color
        # src_rsz = (src_rsz / 255)[None].astype(np.float32)
        src_rsz = np.transpose(src_rsz, [2, 0, 1]).copy().astype(np.float32) / 255.0
        # if return_raw_annotation:
        return src_rsz, annot
        # else:
            # return src_rsz, annot

        


        
    def coco_json_export(self, base_dir: Union[str, List[str]] = '', debug_break=-1):
        """평가용 GT coco json을 원본 이미지 좌표 기준으로 만든다.

        학습 로더(parse_item_coco)로 읽되, 학습용 resize 결과는 쓰지 않는다.
        - images[].width/height : 원본 크기 (target['orig_size'])
        - bbox / segmentation   : 원본 좌표 (target['boxes'] with norm_bbox=False, target['segmentation_orig'])
        - area                  : 원본 polygon의 마스크 면적
        - augmentation 끄고, 가짜 박스(is_dummy) 이미지는 annotation 없이 image만 넣는다.
        - coco_apis(자체 구축 데이터 ③)는 train 전용이므로 넣지 않는다.

        base_dir: file_name을 상대경로로 만들 기준 디렉토리. 데이터셋 폴더들의 상위가 다르면 list로 여러 개 넘긴다.
        """
        import pycocotools.mask as mask_util

        base_dirs = [base_dir] if isinstance(base_dir, str) else list(base_dir)
        base_dirs = [b.replace('\\', '/').rstrip('/') + '/' for b in base_dirs]

        def to_relative(filename):
            filename = os.path.normpath(filename).replace('\\', '/')
            for b in base_dirs:
                if filename.lower().startswith(b.lower()):
                    return filename[len(b):]
            raise AssertionError(f"Filename {filename} does not start with base_dir {base_dirs}")

        class_names = ['gum', *[str(v) for v in fdi_sort], 'unknown']
        type_categories = [
            {"id": i, "name": name, "supercategory": "tooth_type"}
            for i, name in enumerate(class_names)
        ]
        coco_json = {
            "images": [],
            "annotations": [],
            "categories": type_categories
        }
        images = coco_json["images"]
        annotations = coco_json["annotations"]

        # GT에는 augmentation(회전 등)이 들어가면 안 된다
        prev_augment, self.augment_en = self.augment_en, False
        prev_crop, self.xray_crop_modes = self.xray_crop_modes, {}  # GT는 항상 원본 전체 영상 기준
        num_coco = len(self.coco_apis) if self.coco_apis is not None else 0
        skipped = {"coco_apis_image": 0, "dummy_image": 0, "no_segmentation": 0, "empty_box": 0}
        try:
            for idx in tqdm.tqdm(range(len(self))):
                if idx < num_coco:
                    # ③ 자체 COCO 데이터는 평가 json에 넣지 않는다 (segmentation이 bool mask라 json 저장도 불가)
                    skipped["coco_apis_image"] += 1
                    continue
                _, target = self.parse_item_coco(idx, norm_bbox=False, box_format='xywh')

                h, w = [int(v) for v in target['orig_size']]  # 원본 (H, W)
                image_id = len(images)  # 0..N-1 연속 유지
                images.append({
                    "id": image_id,
                    "file_name": to_relative(self.file_name(idx)),
                    "width": w,
                    "height": h,
                })

                if target.get('is_dummy', False):
                    skipped["dummy_image"] += 1
                    continue

                segs = target.get('segmentation_orig', None)
                if segs is not None and len(segs) != len(target['boxes']):
                    raise RuntimeError(f'segmentation/bbox count mismatch: {self.file_name(idx)}')

                for k, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
                    x, y, bw, bh = [float(v) for v in box]
                    # 원본 이미지 범위로 clip
                    x0, y0 = max(x, 0.0), max(y, 0.0)
                    x1, y1 = min(x + bw, float(w)), min(y + bh, float(h))
                    if x1 <= x0 or y1 <= y0:
                        skipped["empty_box"] += 1
                        continue
                    i_annot = {
                        # annotation id는 1부터. pycocotools COCOeval은 매칭 결과를 GT id로 저장하고
                        # 0을 "매칭 안 됨"으로 해석하므로, id 0인 GT에 매칭된 예측이 FP로 계산된다
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],  # [x, y, width, height]
                        "area": (x1 - x0) * (y1 - y0),
                        "iscrowd": 0,
                    }
                    if segs is not None:
                        seg = segs[k]
                        if len(seg) < 6:  # 점 3개 미만
                            skipped["no_segmentation"] += 1
                            continue
                        rle = mask_util.merge(mask_util.frPyObjects([seg], h, w))
                        mask_area = float(mask_util.area(rle))
                        if mask_area <= 0:
                            # 일직선/납작한 polygon (라벨 오류, 예: Kaggle 367.jpg 15번). IoU가 항상 0이라 제외
                            skipped["no_segmentation"] += 1
                            continue
                        i_annot["segmentation"] = [seg]
                        i_annot["area"] = mask_area
                    annotations.append(i_annot)

                if 0 < debug_break <= len(images):
                    break
        finally:
            self.augment_en = prev_augment
            self.xray_crop_modes = prev_crop

        print(f"coco_json_export: images={len(images)}, annotations={len(annotations)}, skipped={skipped}")
        return coco_json

        
    def __getitem__(self, index):
        item = self.parse_item_coco(index)
        for key in ('segmentation', 'segmentation_orig', 'is_dummy'):
            item[1].pop(key, None)
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
    def __init__(self, img_folder, annot_file, transforms=None, include_masks=False, coco_directories=None,**kwargs):
        CocoDetection.__init__(self, img_folder, annot_file, transforms, include_masks)
        self.include_masks = include_masks
        self.base_datset = XrayPnoaramicInstance(
            img_folder, 
            include_masks=include_masks,
            coco_directories=coco_directories,
            **kwargs,
                            #  include_masks=False,
        )
        
        self.index_coco_id = []
            
        # image id mapping    / rf-detr evaluatiopn. mapping....
        if len(self.coco.imgs) > 0:
            # image id 값을 직접 쓴다 (id가 0..N-1 연속이 아니어도 동작)
            meta = self.coco.loadImgs(self.coco.getImgIds())
            img_id_mapping = {v['file_name'].replace('\\', '/').lower(): v['id'] for v in meta}
            image_files = list(img_id_mapping.keys())

            # source_clean = [name.strip().replace('\\', '/').lower() for name in self.base_datset.source_files]
            source_clean  = [os.path.normpath(self.base_datset.file_name(i).strip()).replace('\\', '/').lower() for i in range(len(self.base_datset))]

            matched_idx = []
            for abs_path in source_clean:
                matched = -1
                for relat in image_files:
                    if abs_path.endswith('/' + relat) or abs_path == relat:
                        matched = img_id_mapping[relat]
                        break
                matched_idx.append(matched)

            # ③ 자체 coco 데이터(train 전용)는 평가 json에 없으므로 매칭 대상에서 제외
            num_coco = len(self.base_datset.coco_apis) if self.base_datset.coco_apis is not None else 0
            unmatched = [source_clean[i] for i in range(num_coco, len(source_clean)) if matched_idx[i] < 0]
            assert not unmatched, f'{len(unmatched)} images are not found in annotation json {annot_file}. e.g. {unmatched[:3]}'
            self.index_coco_id = matched_idx

        if (self.base_datset.name != 'train' and self.base_datset.eval_xray_crop
                and self.base_datset.xray_crop_modes):
            self._rebuild_crop_eval_coco()

    def _rebuild_crop_eval_coco(self):
        """원본 전체 영상 GT 대신 실제 검증 입력의 crop/mask를 평가 좌표로 변환한다."""
        from pycocotools.coco import COCO
        from pycocotools import mask as mask_util

        cropped = COCO()
        cropped.dataset = {
            'images': [], 'annotations': [],
            'categories': copy.deepcopy(self.coco.dataset['categories']),
        }
        for index in tqdm.tqdm(range(len(self)), desc='Building cropped validation GT'):
            _, target = self.base_datset.parse_item_coco(index)
            image_id = int(self.index_coco_id[index]) if self.index_coco_id else index
            height, width = map(int, target['orig_size'])
            cropped.dataset['images'].append(dict(id=image_id, height=height, width=width))
            if target.get('is_dummy', False):
                continue
            boxes = np.asarray(target['boxes']).reshape(-1, 4).copy()
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes *= np.array([width, height, width, height])
            for i, (box, label) in enumerate(zip(boxes, target['labels'])):
                annotation = dict(
                    id=len(cropped.dataset['annotations']) + 1,
                    image_id=image_id, category_id=int(label),
                    bbox=box.tolist(), area=float(box[2] * box[3]), iscrowd=0,
                )
                if self.include_masks:
                    # 로더가 실제로 사용한 마스크에는 PA 모서리 제거도 반영되어 있다.
                    mask = cv2.resize(target['masks'][i].astype(np.uint8),
                                      (width, height), interpolation=cv2.INTER_NEAREST)
                    rle = mask_util.encode(np.asfortranarray(mask))
                    annotation.update(segmentation=rle, area=float(mask_util.area(rle)))
                cropped.dataset['annotations'].append(annotation)
        cropped.createIndex()
        self.coco = cropped

            
                          
        
    def __len__(self):
        return len(self.base_datset)
        # return 10
        
    def __getitem__(self, index):
        item = self.base_datset.parse_item_coco(index)
        # img, target = torch_utils.data_convert(item, device='cpu')
        img, target = item
        for key in ('segmentation', 'segmentation_orig', 'is_dummy'):
            target.pop(key, None)
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
        

def get_size(image_shape, refenrece_width:int=640, stride:int=64):

    size =  get_target_image_size(image_shape, refenrece_width)
    if stride > 1:
        # assert stride in [2, 4, 8], 'only stride 2 or 4 is supported'
        size = (size[0] // stride + 1) * stride, (size[1] // stride  + 1 )* stride
    return size

        

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
        
        
def build_xray_coco_json(img_folder: Union[str, List[str]],
                         base_dir: Union[str, List[str]],
                         out_file: str,
                         num_classes: int = 33,
                         include_masks: bool = True,
                         debug_break: int = -1):
    """학습과 같은 데이터셋 로더(XrayPnoaramicInstance)로 평가용 GT coco json을 원본 좌표 기준으로 재빌드한다.

    - img_folder: 학습에 쓰는 dataset_dir 그대로 (①kaggle d2, ②kaggle_2222). ③ coco_directories는 넣지 않는다.
    - splits는 전체(0~1)로 만든다. train/val/test split은 로더가 file_name으로 매칭한다.
    - 기존 json을 덮어쓰지 않도록 out_file이 이미 있으면 에러.
    """
    assert not os.path.exists(out_file), f'{out_file} already exists. use a new file name (e.g. *_v2.json)'
    dataset = XrayPnoaramicInstance(
        img_folder=img_folder,
        num_classes=num_classes,
        include_masks=include_masks,
        coco_directories=None,          # ③ 제외
        name='train',
        splits={'train': (0, 1)},       # 전체 이미지
        augment_en=False,               # 증강 금지
    )
    assert len(dataset) > 0, f'no images found in {img_folder}'
    res = dataset.coco_json_export(base_dir=base_dir, debug_break=debug_break)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(res, f)
    print(f'coco json file saved to {out_file}')
    return out_file


def visualize_xray_coco_json(json_path: str, base_dir: str, vis_dir: str, num_vis: int = 20, image_ids=None):
    """json의 GT(polygon, bbox, category)를 원본 이미지 위에 그려서 저장한다.

    json 좌표계가 원본 이미지와 맞는지 눈으로 확인하는 용도. polygon/bbox가 치아 위에 정확히 겹쳐야 정상.
    - image_ids: 그릴 image id 목록. None이면 앞에서부터 num_vis장
    - 저장: {vis_dir}/{image_id:06d}_{파일명}.jpg
    """
    from pycocotools.coco import COCO
    colors_fdis = vtk_utils.get_teeth_color_table(normalize=False)

    coco = COCO(json_path)
    ids = list(image_ids) if image_ids is not None else coco.getImgIds()[:num_vis]
    os.makedirs(vis_dir, exist_ok=True)
    for image_id in tqdm.tqdm(ids, desc='visualize'):
        info = coco.loadImgs([image_id])[0]
        img = cv2_imread(os.path.join(base_dir, info['file_name']), flags=cv2.IMREAD_COLOR)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        overlay = img.copy()
        for a in anns:
            color = tuple(int(c) for c in colors_fdis[label_to_fdi(int(a['category_id']), 33)][::-1])  # RGB -> BGR
            for poly in a.get('segmentation', []):
                pts = np.asarray(poly, dtype=np.float64).reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        for a in anns:
            x, y, w, h = [int(round(v)) for v in a['bbox']]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.putText(img, str(label_to_fdi(int(a['category_id']), 33)), (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"id={image_id} {info['width']}x{info['height']} n={len(anns)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        name = os.path.splitext(os.path.basename(info['file_name']))[0]
        cv2_imwrite(os.path.join(vis_dir, f'{image_id:06d}_{name}.jpg'), img)
    print(f'saved {len(ids)} visualizations to {vis_dir}')


def verify_xray_coco_json(json_path: str, base_dir: str = None, check_image_size: bool = True,
                          vis_dir: str = None, num_vis: int = 20):
    """재빌드한 json 검증 (docs/troubleshoots/xray_coco_json_segm_eval.md 4.4).

    1) images[].width/height == 실제 이미지 크기
    2) polygon이 이미지 안에 있고, area/bbox_area 비율이 정상(약 0.6~0.8)인지
    3) GT를 그대로 예측으로 넣은 self-eval에서 bbox/segm AP == 1.0
    4) vis_dir를 주면 GT를 원본 이미지에 그린 결과를 num_vis장 저장 (base_dir 필요)
    """
    if vis_dir is not None:
        assert base_dir is not None, 'base_dir is required to save visualizations'
        visualize_xray_coco_json(json_path, base_dir, vis_dir, num_vis=num_vis)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(json_path, 'r') as f:
        d = json.load(f)
    im = {i['id']: i for i in d['images']}
    print(f"images: {len(d['images'])}, annotations: {len(d['annotations'])}")

    if check_image_size and base_dir is not None:
        bad = []
        for i in tqdm.tqdm(d['images'], desc='image size check'):
            with Image.open(os.path.join(base_dir, i['file_name'])) as img:
                if img.size != (i['width'], i['height']):
                    bad.append((i['file_name'], img.size, (i['width'], i['height'])))
        print(f'size mismatch: {len(bad)} {bad[:3]}')

    ratios, out_of_image = [], 0
    for a in d['annotations']:
        img = im[a['image_id']]
        x, y, w, h = a['bbox']
        if 'segmentation' in a:
            s = np.array(a['segmentation'][0]).reshape(-1, 2)
            if s[:, 0].max() > img['width'] + 1 or s[:, 1].max() > img['height'] + 1 or s.min() < -1:
                out_of_image += 1
        ratios.append(a['area'] / max(w * h, 1e-6))
    print(f'polygon out of image: {out_of_image}')
    print(f'area/bbox_area median: {np.median(ratios) if ratios else float("nan"):.3f} (0.6~0.8 정상, 기존 json은 약 0.1)')

    gt = COCO(json_path)
    stats = {}
    for iou_type in ['bbox', 'segm']:
        # loadRes는 bbox 키가 있으면 bbox 기준으로 처리하므로 iou_type별로 키를 나눠서 넣는다
        drop = 'segmentation' if iou_type == 'bbox' else 'bbox'
        dets = [{**{k: v for k, v in a.items() if k not in (drop, 'id', 'area')}, 'score': 1.0}
                for a in gt.dataset['annotations'] if iou_type == 'bbox' or 'segmentation' in a]
        if iou_type == 'segm':
            # 예측 결과처럼 RLE로 넣는다 (CocoEvaluator와 동일, loadRes는 결과 segmentation을 RLE로 기대)
            for det in dets:
                rle = gt.annToRLE(det)
                rle['counts'] = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                det['segmentation'] = rle
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        stats[iou_type] = ev.stats[0]
    # AR@1(maxDets=1)은 같은 이미지에 같은 치아번호가 2개 이상인 경우(유치 매핑, 중복 라벨) 때문에 1.0보다 낮은 게 정상
    print(f'self-eval AP: {stats} (둘 다 1.0 이어야 정상. 기존 json은 segm이 0)')
    return stats


def test_build_coco_json():
    base_dir = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays'
    num_classes = 33
    out_file = os.path.join(base_dir, f'xray_coco_{num_classes}_seg_v2.json')
    build_xray_coco_json(
        img_folder=[
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222',
        ],
        base_dir=base_dir,
        out_file=out_file,
        num_classes=num_classes,
    )
    verify_xray_coco_json(out_file, base_dir=base_dir)


        
def test_load_coco_dataset():
    
    from trainer import get_logger
    
    save_dir = 'outputs/result'
    
    logger = get_logger()
    # img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # img_folder = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle'
    kaggle_path2 = [
        # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222',
        # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
        '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2/',
        '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle_2222/',
    ]
    
    # ann_file = ''
    # ann_file = ''
    # dataset = XrayPnoaramicInstanceCoco(img_folder, '', None, True)
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle'
    # xray_coco.json'
    
    annot_file = '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_coco_33_seg.json'
    # annot_file = '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_coco_33_seg_v2.json'
    
    
    # annot_file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/xray_coco_33_seg.json'
    # annot_file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_coco_33_seg.json' 
    annot_file = None
    
    dataset = XrayPnoaramicInstanceCoco(
        
            # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
            # os.path.join(path, 'Teeth Segmentation JSON/d2')
            kaggle_path2,
            annot_file,
        # ,
        # os.path.join(path, 'xray_coco.json'),
        # None,
        None,
        xray_crop_mode=['periapical', 'teeth_roi'],  # None | 'periapical' | 'teeth_roi'
        xray_crop_prob=1.0,
        # xray_crop_mode
        include_masks=True,

        num_classes=32,
        name='train',
                coco_directories=[
            # ('E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm', \
                # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm/annotations.json')
            # ('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm/',
            #  '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm/annotations.json')
        ],
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
        # print(torch_utils.get_shape([img, target]))
        if i >= test_iter:
            break
        i = np.random.randint(0, len(dataset))
        img, target = dataset[i]

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
        # for table
        # visualize_number = 4
        # if 
        # denorm_bboxes = denorm_bboxes.reshape([-1, 2]).clip(0, size)
        denorm_bboxes_i = denorm_bboxes.astype(np.int32)
        test_vis_class = 32
        target_fdi = label_to_fdi(target_label, test_vis_class)
        
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
            
        os.makedirs(save_dir, exist_ok=True)
        save_name = f'{save_dir}/xray_{i}.png'
        cv2.imwrite(save_name, drawing[..., ::-1])
        logger.info(f"Saved visualization to: {save_name}")
        
            
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

import copy

def build(image_set, args, resolution):
    img_folder = args.dataset_dir
    # annotation file is builed by functoin . see teeth.py::test_build_teethdsata
    args_dict = copy.deepcopy(dict(args.__dict__))
    # args_dict = 
    annot_file = args_dict.pop('annot_file', '../../xray_coco.json')
    if os.path.exists(annot_file):
        pass
    else:
        annot_file = os.path.join(img_folder, annot_file)
        assert os.path.exists(annot_file), f'annotation file not found: {annot_file}'
    # annot_file = 
    # annot_file
    if image_set == 'train':
        # args_dict['augment_en'] = True
        coco_directories = args_dict.pop('coco_directories', None)
    else:
        args_dict.pop('coco_directories', None)
        coco_directories = None
        

    dataset = XrayPnoaramicInstanceCoco(
        img_folder=img_folder,
        annot_file=annot_file,
        transforms=None,
        name=image_set,
        include_masks=args.segmentation_head,
        # num_classes=getattr(args, 'num_classes', 2),
        # bg_crop_prob=getattr(args, 'bg_crop_prob', 0.15),
        augment_en=(image_set == 'train'),
        coco_directories=coco_directories,
        **args_dict
        # **args_dict
    )
    
    print(f'=====================Loaded dataset {image_set} with {len(dataset)} =====================')
    return dataset



class ResizeCocoSample:
    """
    Resize (image, target) pair.

    target format:
    {
        "boxes": np.ndarray[N, 4],       # xyxy
        "labels": np.ndarray[N],
        "masks": np.ndarray[N, H, W],    # optional
        ...
    }

    mask_resize_mode:
        "full" : full mask resize
        "roi"  : bbox 영역만 crop -> resize -> target canvas paste
    """

    def __init__(
        self,
        size: Tuple[int, int],
        keep_ratio: bool = True,
        mask_resize_mode: str = "roi",
    ):
        """
        size = (target_h, target_w)
        """
        self.target_h, self.target_w = size
        self.keep_ratio = keep_ratio
        self.mask_resize_mode = mask_resize_mode
        self.refer_width = 640

    def __call__(self, image: np.ndarray, target: dict):
        src_h, src_w = image.shape[:2]
        
        
        # (h, w)
        # size =  get_target_image_size([src_h, src_w], self.refer_width)

        if self.keep_ratio:
            image, scale_x, scale_y, offset_x, offset_y = \
                self._resize_letterbox(image)
        else:
            image = cv2.resize(
                image,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_LINEAR,
            )

            scale_x = self.target_w / src_w
            scale_y = self.target_h / src_h
            offset_x = 0
            offset_y = 0

        target = dict(target)

        boxes_src = np.asarray(
            target.get("boxes", []),
            dtype=np.float32,
        ).reshape(-1, 4)

        boxes_dst = self._resize_boxes(
            boxes_src,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        )

        target["boxes"] = boxes_dst

        if "masks" in target and target["masks"] is not None:
            masks = target["masks"]

            if self.mask_resize_mode == "roi":
                target["masks"] = self._resize_masks_roi(
                    masks,
                    boxes_src,
                    boxes_dst,
                )
            else:
                target["masks"] = self._resize_masks_full(
                    masks,
                    scale_x,
                    scale_y,
                    offset_x,
                    offset_y,
                )

        target["size"] = np.array(
            [self.target_h, self.target_w],
            dtype=np.int64,
        )

        return image, target

    # ---------------------------------------------------------
    # image
    # ---------------------------------------------------------

    def _resize_letterbox(self, image):
        src_h, src_w = image.shape[:2]

        scale = min(
            self.target_w / src_w,
            self.target_h / src_h,
        )

        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        offset_x = (self.target_w - new_w) // 2
        offset_y = (self.target_h - new_h) // 2

        if image.ndim == 3:
            canvas = np.zeros(
                (self.target_h, self.target_w, image.shape[2]),
                dtype=image.dtype,
            )
        else:
            canvas = np.zeros(
                (self.target_h, self.target_w),
                dtype=image.dtype,
            )

        canvas[
            offset_y:offset_y + new_h,
            offset_x:offset_x + new_w,
        ] = resized

        return (
            canvas,
            scale,
            scale,
            offset_x,
            offset_y,
        )

    # ---------------------------------------------------------
    # bbox
    # ---------------------------------------------------------

    def _resize_boxes(
        self,
        boxes,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
    ):
        if boxes.size == 0:
            return boxes.copy()

        boxes = boxes.copy()

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] += offset_x
        boxes[:, [1, 3]] += offset_y

        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            self.target_w,
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            self.target_h,
        )

        return boxes

    # ---------------------------------------------------------
    # mask: optimized ROI resize
    # ---------------------------------------------------------

    def _resize_masks_roi(
        self,
        masks,
        boxes_src,
        boxes_dst,
    ):
        """
        Full resolution mask를 resize하지 않고
        bbox 영역만 잘라서 resize.

        입력:
            masks: [N, src_h, src_w]

        출력:
            [N, target_h, target_w]
        """

        masks = np.asarray(masks)
        masks_dtype = masks.dtype
        n = len(masks)

        out_masks = np.zeros(
            (n, self.target_h, self.target_w),
            dtype=masks.dtype,
        )

        for i in range(n):
            sx1, sy1, sx2, sy2 = boxes_src[i]

            # bbox -> integer crop range
            sx1 = max(int(np.floor(sx1)), 0)
            sy1 = max(int(np.floor(sy1)), 0)
            sx2 = min(int(np.ceil(sx2)), masks.shape[2])
            sy2 = min(int(np.ceil(sy2)), masks.shape[1])

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            # -------------------------------------------
            # 핵심:
            # 4K full mask를 resize하지 않고 ROI만 추출
            # -------------------------------------------

            mask_crop = masks[
                i,
                sy1:sy2,
                sx1:sx2,
            ]

            dx1, dy1, dx2, dy2 = boxes_dst[i]

            dx1 = max(int(np.floor(dx1)), 0)
            dy1 = max(int(np.floor(dy1)), 0)
            dx2 = min(int(np.ceil(dx2)), self.target_w)
            dy2 = min(int(np.ceil(dy2)), self.target_h)

            dst_w = dx2 - dx1
            dst_h = dy2 - dy1

            if dst_w <= 0 or dst_h <= 0:
                continue

            mask_small = cv2.resize(
                mask_crop.astype(np.uint8),
                (dst_w, dst_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(masks_dtype)

            out_masks[
                i,
                dy1:dy2,
                dx1:dx2,
            ] = mask_small

        return out_masks

    # ---------------------------------------------------------
    # mask: simple/reference implementation
    # ---------------------------------------------------------

    def _resize_masks_full(
        self,
        masks,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
    ):
        masks = np.asarray(masks)

        out_masks = np.zeros(
            (
                len(masks),
                self.target_h,
                self.target_w,
            ),
            dtype=np.uint8,
        )

        for i, mask in enumerate(masks):
            src_h, src_w = mask.shape

            new_w = int(round(src_w * scale_x))
            new_h = int(round(src_h * scale_y))

            resized = cv2.resize(
                mask,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )

            out_masks[
                i,
                offset_y:offset_y + new_h,
                offset_x:offset_x + new_w,
            ] = resized

        return out_masks

# def test_
        
if __name__ == '__main__':
    # main_xraypanoramic_instance()
    # test_build_coco_json()
    
    test_load_coco_dataset()
