import cv2
import albumentations as A
import numpy as np
from rfdetr.datasets.aug_albumentations import RandomCropWithRoundedBorder, MaskawareImageAug
import matplotlib
matplotlib.use('qtagg')  # 백엔드 설정 (GUI 없는 환경에서도 동작)
import matplotlib.pyplot as plt


def visualize(image, bboxes=None, keypoints=None, show=False):
    """결과 시각화 함수"""
    image = image.copy()
    if bboxes is not None:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:4]
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    
    # if keypoints:
    if keypoints is not None:
        for kp in keypoints:
        # cv2.polylines(image, [keypoints.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.fillPoly(image, [kp.astype(np.int32)], color=(0, 0, 255))
        # for kp in keypoints:
        #     cv2.circle(image, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
            
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # plt.axis('off')
        plt.show()
    return image

def main():
    # 1. 가상 이미지 및 레이블 데이터 생성
    # image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image = cv2.imread('sample.jpg', cv2.IMREAD_COLOR)

    # BBox 데이터 (N, 4) + 라벨 필드 설정용
    dummy_bboxes = [
        [200, 200, 400, 400, 0],
        [500, 500, 700, 700, 1]
    ] # pascal_voc format + label
    class_labels = [0, 1]
    dummy_bboxes = np.array(dummy_bboxes)[:, :4]

    # Polygon 데이터 (list of (N, 2)) -> Keypoints로 변환하여 처리
    polygon1 = np.array([[300, 300], [350, 300], [350, 350], [300, 350]])
    polygon2 = polygon1 + np.array([600, 600])  # 임의의 다각형 생성
    polygons = [polygon1, polygon2]
    polygons_indices = [len(poly) for poly in polygons]
    polygons_concat = np.concatenate(polygons, axis=0)
    # flat_keypoints = [pt for poly in [polygon1] for pt in poly]

    # 2. 파이프라인 정의
    transform = A.Compose([
        A.SomeOf(
            transforms=[
                MaskawareImageAug(
                    target_size=(512, 512),
                    crop_scale=(0.6, 0.8),
                    corner_radius=30,
                    border_thickness=4,
                    border_brightness_inc=100,
                    p=1.0,
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
                A.Rotate(limit=10, p=0.5),
            ],
            n=3,  # 1개 또는 2개 선택
            p=1.0
        )
        ], 
        bbox_params=A.BboxParams(format='pascal_voc'),
        keypoint_params=A.KeypointParams(format='xy', )
    )
    # # 3. 변환 수행
    # augmented = transform(
    #     image=image, 
    #     bboxes=dummy_bboxes, 
    #     class_labels=class_labels,
    #     keypoints=flat_keypoints
    # )

    src = visualize(image, dummy_bboxes, polygons)
    cv2.imwrite('results/original_image.png', src)
    for i in range(10):
    # 3. 변형 적용

    # 3. 변환 수행
        transformed = transform(
            image=image, 
            bboxes=np.array(dummy_bboxes)[:, :4],
            # class_labels=class_labels,
            keypoints=polygons_concat,
            polygons_indices=polygons_indices,
            # polygons_concat=polygons_concat,
        )
        # transformed = transform(image=image, bboxes=gt_bbox, keypoints=poly_pts)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_keypoints = transformed['keypoints']

        # 4. 결과 확인
        print("원본 이미지 크기:", image.shape[:2])
        print("변환 후 이미지 크기:", transformed_image.shape[:2])
        print("원본 BBox:", dummy_bboxes)
        print("변환 후 BBox:", transformed_bboxes)
        
        
        res = visualize(transformed_image, transformed_bboxes, None)
        cv2.imwrite(f'results/transformed_image_{i}.png', res)
        # visualize(image, )
    # visualize(transformed_image, transformed_bboxes, transformed_keypoints)
    
if __name__ == "__main__":
    main()