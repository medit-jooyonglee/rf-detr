import albumentations as A
import cv2
import numpy as np
import random
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.bbox_utils import denormalize_bboxes, normalize_bboxes
from pycocotools import mask


def create_augmentation_pipeline():
    """
    이미지, Bounding Box, Polygon을 동시에 처리할 수 있는 Albumentations 파이프라인 생성
    """
    transform = A.Compose(
        [
            # 1. 기하학적 변환 (Geometric Transforms)
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=15, p=0.5),
            # A.Resize(height=512, width=512),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['class_labels'],  # bbox별 클래스 레이블 (필요시 사용)
            min_area=0,
            min_visibility=0.3,
        ),
        keypoint_params=A.KeypointParams(
            format='xy',  # Polygon 점들을 keypoint로 변환하여 처리
            remove_invisible=False,
        ),
    )
    return transform


def apply_augmentations(image: np.ndarray, bboxes: np.ndarray, polygons: list, transform: A.Compose):
    """
    입력 데이터에 Albumentations 변환을 적용하는 함수

    Args:
        image: 2D/3D numpy array 이미지 (H, W, C)
        bboxes: numpy array of shape (N, 4) -> [x_min, y_min, x_max, y_max]
        polygons: list of (N, 2) 형태의 폴리곤 좌표 리스트
        transform: A.Compose 객체

    Returns:
        augmented_image, augmented_bboxes, augmented_polygons
    """
    # 1. Polygon(list of (N, 2))을 Albumentations Keypoints 형식으로 변환
    # Albumentations는 keypoint를 (x, y) 튜플 형태로 받습니다.
    flat_keypoints = []
    polygon_lens = []

    for poly in polygons:
        polygon_lens.append(len(poly))
        for pt in poly:
            flat_keypoints.append((float(pt[0]), float(pt[1])))

    # 더미 class_labels 생성 (bbox 개수만큼 필요)
    class_labels = [1] * len(bboxes)

    # 2. 변환 수행
    transformed = transform(
        image=image,
        bboxes=bboxes.tolist() if isinstance(bboxes, np.ndarray) else bboxes,
        class_labels=class_labels,
        keypoints=flat_keypoints,
    )

    # 3. 결과 추출 및 원복
    aug_image = transformed['image']
    aug_bboxes = np.array(transformed['bboxes'], dtype=np.float32)

    # Keypoints를 다시 원래 폴리곤 구조(list of (N, 2))로 재조립
    aug_keypoints = transformed['keypoints']
    aug_polygons = []
    idx = 0

    for length in polygon_lens:
        poly_pts = []
        for _ in range(length):
            if idx < len(aug_keypoints):
                poly_pts.append([aug_keypoints[idx][0], aug_keypoints[idx][1]])
            idx += 1
        aug_polygons.append(np.array(poly_pts, dtype=np.float32))

    return aug_image, aug_bboxes, aug_polygons


class RandomCropWithRoundedBorder(DualTransform):
    def __init__(
        self,
        target_size=(512, 512),
        crop_scale=(0.7, 0.9),
        corner_radius=20,
        border_thickness=3,
        border_brightness_inc=50,
        p=1.0
    ):
        super().__init__(p)
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.corner_radius = corner_radius
        self.border_thickness = border_thickness
        self.border_brightness_inc = border_brightness_inc

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
        """
        Image/keypoints 등 실제 입력 데이터에 의존하는 crop 파라미터를 계산.

        Keypoint가 주어지면 crop 영역이 keypoint 전체를 포함하도록(잘려나가지 않도록)
        crop 크기/위치를 keypoint 기준으로 산출하고, keypoint가 없으면 기존과 동일하게
        이미지 중앙을 기준으로 crop한다.
        """
        h, w = params["shape"][:2]
        target_h, target_w = self.target_size

        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])
        crop_h = int(min(h, target_h * scale))
        crop_w = int(min(w, target_w * scale))

        keypoints = data.get("keypoints")
        if keypoints is not None and len(keypoints) > 0:
            kp_arr = np.asarray(keypoints, dtype=np.float32)
            kp_x_min, kp_y_min = kp_arr[:, 0].min(), kp_arr[:, 1].min()
            kp_x_max, kp_y_max = kp_arr[:, 0].max(), kp_arr[:, 1].max()

            # keypoint 영역 전체(+여유 margin)가 crop 안에 들어오도록 crop 크기 확장
            margin_scale = 1.1
            needed_w = (kp_x_max - kp_x_min) * margin_scale
            needed_h = (kp_y_max - kp_y_min) * margin_scale
            crop_w = int(min(w, max(crop_w, needed_w)))
            crop_h = int(min(h, max(crop_h, needed_h)))

            # crop 중심을 keypoint 영역의 중심에 맞춤
            kp_center_x = (kp_x_min + kp_x_max) / 2
            kp_center_y = (kp_y_min + kp_y_max) / 2
            x_min = int(kp_center_x - crop_w / 2)
            y_min = int(kp_center_y - crop_h / 2)

            # crop 영역이 이미지 경계를 벗어나지 않도록 clamp
            x_min = max(0, min(x_min, w - crop_w))
            y_min = max(0, min(y_min, h - crop_h))
        else:
            y_min = max(0, (h - crop_h) // 2)
            x_min = max(0, (w - crop_w) // 2)

        return {"crop_info": (y_min, x_min, crop_h, crop_w)}

    def apply(self, img, crop_info, **params):
        target_h, target_w = self.target_size
        y_min, x_min, crop_h, crop_w = crop_info

        cropped_img = img[y_min:y_min+crop_h, x_min:x_min+crop_w]

        pad_buffer = self.corner_radius + self.border_thickness + 10
        inner_h = max(10, target_h - (pad_buffer * 2))
        inner_w = max(10, target_w - (pad_buffer * 2))

        res_h, res_w = self._get_resize_dims(crop_h, crop_w, inner_h, inner_w)
        resized_img = cv2.resize(cropped_img, (res_w, res_h), interpolation=cv2.INTER_LINEAR)

        # 1. 둥근 사각형 마스크 생성 및 이미지 크롭 적용
        mask = self._create_rounded_mask(res_w, res_h, self.corner_radius)
        final_fg = cv2.bitwise_and(resized_img, resized_img, mask=mask)

        # 2. [핵심] 둥근 모서리와 사방 변 전체에 OpenCV로 순백색 테두리 선 직접 그리기
        # 마스크의 외곽선(Contour)을 찾아 지정한 두께만큼 하얀색 라인을 덧그립니다.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_fg, contours, -1, (255, 255, 255), thickness=self.border_thickness)

        # 3. 최종 검은색(또는 원하는 배경) 캔버스 생성 및 중앙 배치
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8) 
        start_y = (target_h - res_h) // 2
        start_x = (target_w - res_w) // 2

        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # 테두리가 그려진 영역까지 포함하여 캔버스에 합성
        outer_radius = max(0, self.corner_radius + self.border_thickness)
        outer_mask = self._create_rounded_mask(res_w, res_h, outer_radius)
        color_outer_mask = cv2.cvtColor(outer_mask, cv2.COLOR_GRAY2BGR)
        color_mask_inv = cv2.bitwise_not(color_outer_mask)

        roi = canvas[start_y:start_y+res_h, start_x:start_x+res_w]
        
        if roi.shape[:2] == final_fg.shape[:2]:
            background = cv2.bitwise_and(roi, color_mask_inv)
            foreground = cv2.bitwise_and(final_fg, color_outer_mask)
            canvas[start_y:start_y+res_h, start_x:start_x+res_w] = cv2.add(background, foreground)

        return canvas

    def apply_to_bboxes(self, bboxes, crop_info, shape, **params):
        """BBox 리스트를 한 번에 안전하게 처리"""
        if bboxes is None or len(bboxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        y_min, x_min, crop_h, crop_w = crop_info
        orig_crop_x, orig_crop_y = x_min, y_min

        target_h, target_w = self.target_size
        pad_buffer = self.corner_radius + self.border_thickness + 10
        inner_h = target_h - (pad_buffer * 2)
        inner_w = target_w - (pad_buffer * 2)
        res_h, res_w = self._get_resize_dims(crop_h, crop_w, inner_h, inner_w)

        scale_x = res_w / crop_w
        scale_y = res_h / crop_h
        start_x = (target_w - res_w) // 2
        start_y = (target_h - res_h) // 2

        # Albumentations 내부적으로 bbox는 0~1 정규화 좌표로 전달되므로,
        # 원본 이미지 픽셀 좌표로 변환 후 crop/resize 연산을 수행한다.
        pixel_bboxes = denormalize_bboxes(np.asarray(bboxes, dtype=np.float32), shape[:2])

        transformed_bboxes = []
        for bbox in pixel_bboxes:
            bx_min, by_min, bx_max, by_max = bbox[:4]
            extra = bbox[4:]

            c_x_min = bx_min - orig_crop_x
            c_y_min = by_min - orig_crop_y
            c_x_max = bx_max - orig_crop_x
            c_y_max = by_max - orig_crop_y

            f_x_min = np.clip(c_x_min * scale_x + start_x, 0, target_w)
            f_y_min = np.clip(c_y_min * scale_y + start_y, 0, target_h)
            f_x_max = np.clip(c_x_max * scale_x + start_x, 0, target_w)
            f_y_max = np.clip(c_y_max * scale_y + start_y, 0, target_h)

            transformed_bboxes.append([f_x_min, f_y_min, f_x_max, f_y_max, *extra])

        transformed_bboxes = np.array(transformed_bboxes, dtype=np.float32)
        # crop 결과 캔버스(target_size) 기준으로 다시 정규화하여 반환
        return normalize_bboxes(transformed_bboxes, (target_h, target_w))

    def apply_to_keypoints(self, keypoints, crop_info, **params):
            """키포인트(폴리곤 점들)를 numpy array로 일괄 처리"""
            if len(keypoints) == 0:
                return np.empty((0, 2), dtype=np.float32)

            y_min, x_min, crop_h, crop_w = crop_info

            target_h, target_w = self.target_size
            pad_buffer = self.corner_radius + self.border_thickness + 10
            inner_h = target_h - (pad_buffer * 2)
            inner_w = target_w - (pad_buffer * 2)
            res_h, res_w = self._get_resize_dims(crop_h, crop_w, inner_h, inner_w)
            
            scale_x = res_w / crop_w
            scale_y = res_h / crop_h
            start_x = (target_w - res_w) // 2
            start_y = (target_h - res_h) // 2

            kpts_arr = np.asarray(keypoints, dtype=np.float32)
            # 벡터 연산으로 x, y 좌표 일괄 변환 (추가 속성 보존)
            kpts_arr[:, 0] = (kpts_arr[:, 0] - x_min) * scale_x + start_x
            kpts_arr[:, 1] = (kpts_arr[:, 1] - y_min) * scale_y + start_y

            return kpts_arr

    def _get_resize_dims(self, crop_h, crop_w, target_h, target_w):
        crop_ratio = crop_w / crop_h
        target_ratio = target_w / target_h
        if crop_ratio > target_ratio:
            new_w = target_w
            new_h = int(new_w / crop_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * crop_ratio)
        return max(1, new_h), max(1, new_w)

    def _create_rounded_mask(self, w, h, r):
        mask = np.zeros((h, w), dtype=np.uint8)
        r = min(r, w // 2, h // 2)
        cv2.rectangle(mask, (r, 0), (w-r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h-r), 255, -1)
        cv2.circle(mask, (r, r), r, 255, -1)
        cv2.circle(mask, (w-r, r), r, 255, -1)
        cv2.circle(mask, (r, h-r), r, 255, -1)
        cv2.circle(mask, (w-r, h-r), r, 255, -1)
        return mask

    def get_transform_init_args_names(self):
        return ("target_size", "crop_scale", "corner_radius", "border_thickness", "border_brightness_inc")
    
    


class MaskawareImageAug(DualTransform):
    def __init__(
        self,
        target_size=(512, 512),
        crop_scale=(0.7, 0.9),
        corner_radius=20,
        border_thickness=3,
        border_brightness_inc=50,
        p=1.0,
        **kwargs  #
    ):
        super().__init__(p=p, **kwargs)
        self.target_size = target_size
        self.crop_scale = crop_scale
        self.corner_radius = corner_radius
        self.border_thickness = border_thickness
        self.border_brightness_inc = border_brightness_inc

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_data(self, params, data):
       
        return dict(
            segment_polygons=data.get("keypoints"),
            polygons_indices=data.get("polygons_indices", []), # 값이 없을 경우 빈 리스트 반환
        )

    def apply(self, img, segment_polygons, **params):
        if segment_polygons is None or len(segment_polygons) == 0:
            return img
        target_h, target_w = self.target_size
        polygons = segment_polygons[:, :2]
        
        
        polygons_indices = params['polygons_indices']
        
        # split_polygons = np.split(polygons, np.cumsum(polygons_indices)[:-1], axis=0)
        split_polys = []
        start_idx = 0
        for i in range(len(polygons_indices)):
            end_idx = start_idx + polygons_indices[i]
            split_polys.append(polygons[start_idx:end_idx])
            start_idx = end_idx
            
        arg = np.random.choice(np.arange(len(split_polys)), size=np.random.randint(1, len(split_polys)+1), replace=False)
        
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for i0 in arg:
            poly = split_polys[i0]
            cv2.fillPoly(mask, [poly.astype(np.int32)], color=255)
            
        
        # kernel_size = max(3, self.border_thickness * 2 + 1)
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 원본 마스크에서 안쪽으로 침식된 마스크 생성
        eroded_mask = cv2.erode(mask, kernel, iterations=np.random.randint(1, 4))

        # feather(블러) 처리하여 하드-엣지 대신 주변 조직으로 서서히 번지는
        # 밝기 증가를 만든다 -> 금속/보철 아티팩트에서 흔히 보이는
        # "경계가 불분명한(edge가 washed-out)" 외형을 모사한다.
        feather_ksize = int(np.random.choice([9, 15, 21]))
        feather_mask = cv2.GaussianBlur(
            eroded_mask.astype(np.float32), (feather_ksize, feather_ksize), 0
        ) / 255.0
        feather_mask = feather_mask[..., None]

        brightness_inc = np.random.uniform(60, 255)
        img_f = img.astype(np.float32)
        brightened = np.clip(img_f + brightness_inc, 0, 255)

        # 아티팩트 영역 자체도 블러 처리해서 치아 경계의 대비를 낮춘다.
        blur_ksize = int(np.random.choice([5, 9, 13]))
        blurred = cv2.GaussianBlur(brightened, (blur_ksize, blur_ksize), 0)
        if blurred.ndim < img_f.ndim:
            blurred = blurred[..., None]

        out = img_f * (1 - feather_mask) + blurred * feather_mask
        return np.clip(out, 0, 255).astype(img.dtype)
        

    def apply_to_bboxes(self, bboxes, **params):
        return bboxes
        
    def apply_to_keypoints(self, keypoints, **params):
        polygons_indices = params['polygons_indices']
        
        return keypoints
            


    def get_transform_init_args_names(self):
        return ("target_size", "crop_scale", "corner_radius", "border_thickness", "border_brightness_inc")