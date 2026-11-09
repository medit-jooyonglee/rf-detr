# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""데이터셋 시각화.

1. ``render_detection`` / ``save_rendered_image``: 박스+마스크를 영상에 그리고 저장하는 core 로직.
   원래 ``rfdetr/engine.py::draw_preditions_boxes``(모델 추론 결과 시각화) 안에 있던 순수 렌더링
   부분을 분리한 것으로, engine.py는 모델 출력(logits/NMS/FDI 보정/mask paste 등) 후처리만 하고
   최종 박스/라벨/마스크를 이 모듈에 넘겨 그리기만 위임한다.
2. ``visualize_dataset``: 모델 추론 없이 데이터셋 로더(GT)만으로 같은 core 로직을 재사용해 그린다.
   dataset __getitem__이 돌려주는 target 포맷(정규화 cxcywh boxes, 원시 class id label)이
   모델 pred_boxes/pred_label 포맷과 동일하므로, label_to_fdi 변환 + denorm만 거치면 바로
   render_detection에 넘길 수 있다 (NMS/FDI-refinement는 GT에는 필요 없음).

CLI: ``python -m rfdetr.datasets.visualize --amount all|<count>|<ratio>``
"""
import argparse
import os

import numpy as np


def _draw_fdi_labels(image, boxes_xy, labels, font_scale=None):
    """각 박스 좌상단에 FDI 번호를 적는다. 검은 외곽선 + 흰 글씨 이중 strokes로 어떤 배경(마스크
    색/엑스레이 명암)에서도 읽히게 한다. font_scale은 None이면 영상 너비 기준 자동 스케일."""
    import cv2

    font_scale = font_scale or max(0.5, image.shape[1] / 1600.0)
    thickness = max(1, int(round(font_scale * 2)))
    for box, label in zip(boxes_xy, labels):
        x0 = max(0, int(round(box[0])))
        y0 = max(0, int(round(box[1])))
        text = str(int(label))
        (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        org = (x0 + 2, y0 + th + 2)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def render_detection(
    image,
    boxes_xy,
    labels,
    colors,
    masks=None,
    drawing_mode='polygon',
    polygon_thickness=None,
    with_source_concat=False,
    draw_labels=True,
    label_font_scale=None,
):
    """한 장의 영상에 박스(+마스크+FDI 번호)를 그린다. 모델 추론/GT 어느 쪽에서 왔는지 모르는,
    이미 최종 확정된(denorm·NMS/FDI 변환 끝난) 결과만 받는 순수 렌더링 함수.

    Args:
        image: ``(H, W, C)`` 배열. 이미 이 함수가 그릴 해상도로 리사이즈/크롭되어 있어야 한다.
        boxes_xy: ``(N, 4)`` xyxy, ``image`` 픽셀 좌표계.
        labels: ``(N,)`` int. ``colors``/``colors[label]``로 색상을 찾을 FDI(또는 클래스) id.
        colors: ``(num_colors, 3)`` 색상 테이블.
        masks: 선택. ``(N, H, W)`` bool/uint8 instance mask, ``image``와 같은 해상도로 이미
            복원되어 있어야 한다. ``None``이면 마스크를 그리지 않는다(= label_image도 None 반환).
        drawing_mode: ``'polygon'``(기본)은 인스턴스별 외곽선만 그려 인접 치아 중첩 영역이 보이게
            한다. ``'mask'``는 겹치는 영역을 큰 FDI 라벨로 병합해 블렌딩한다.
        polygon_thickness: ``'polygon'`` 모드 외곽선 두께. ``None``이면 영상 너비 기준 자동.
        with_source_concat: True면 ``[그리기 전 원본 | 그린 결과]``를 가로로 이어붙인다
            (``masks``가 주어졌을 때만 동작 -- draw_preditions_boxes의 기존 동작과 동일).
        draw_labels: 기본 True. 각 박스 위에 FDI 번호 텍스트를 적는다.
        label_font_scale: FDI 번호 폰트 크기(cv2.putText fontScale). ``None``이면 영상 너비
            기준 자동 스케일(좁은 periapical crop과 넓은 panoramic 모두에서 읽히는 크기).

    Returns:
        (rendered_image, label_image): ``label_image``는 ``masks``가 None이면 None, 아니면
        인스턴스를 픽셀별로 병합한 ``(H, W)`` int 라벨 맵(겹치는 픽셀은 더 큰 라벨 값을 가짐).
    """
    if drawing_mode not in ('mask', 'polygon'):
        raise ValueError(f"drawing_mode must be 'mask' or 'polygon', got {drawing_mode!r}")
    from rfdetr.datasets.teeth import draw_bboxes
    import cv2

    labels = np.asarray(labels)
    src_image = image.copy()
    draw_bboxes(image, boxes_xy, colors=colors[labels])

    label_image = None
    if masks is not None:
        if len(masks) == 0:
            label_image = np.zeros(image.shape[:2], dtype=np.int64)
        else:
            label_image = np.max(labels[:, None, None] * masks, axis=0)

        if drawing_mode == 'polygon':
            # 치아별 외곽선을 따로 그려서 인접 치아 사이 중첩 영역이 보이게 한다 (채우지 않음)
            thickness = polygon_thickness or 2 * max(1, int(round(image.shape[1] / 1000)))
            for instance_mask, instance_label in zip(masks, labels):
                contours, _ = cv2.findContours(
                    np.ascontiguousarray(instance_mask, dtype=np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                )
                color = tuple(int(c) for c in colors[instance_label])
                cv2.polylines(image, contours, True, color, thickness, lineType=cv2.LINE_AA)
        else:
            from trainer import utils_numpy
            color_label_image = colors[label_image]
            image = utils_numpy.apply_blending_mask(image, color_label_image)

    if draw_labels and len(boxes_xy) > 0:
        _draw_fdi_labels(image, boxes_xy, labels, font_scale=label_font_scale)

    if masks is not None and with_source_concat:
        image = np.concatenate([src_image, image], axis=1)

    return image, label_image


def save_rendered_image(image, save_dir='results', fname=''):
    """``render_detection``의 결과를 ``<save_dir>/<YYYYMMDD>/<fname>``에 저장한다."""
    import time
    from trainer import image_utils, time_strftime

    dated_save_dir = os.path.join(save_dir, time.strftime('%Y%m%d'))
    if fname:
        save_name = os.path.join(dated_save_dir, fname)
    else:
        save_name = f'{dated_save_dir}/bounding_draw_{time_strftime()}.png'
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    image_utils.cv2_imwrite(save_name, image.astype(np.uint8)[..., ::-1])
    print("Saved image with bounding boxes to: ", save_name)
    return save_name


# ---------------------------------------------------------------------------
# 데이터셋(GT)만으로 시각화 -- 추론 없이 rfdetr/cli/main.py의 xray_teeth 데이터 로더를 그대로 사용.
# ---------------------------------------------------------------------------

# cli/main.py::train_from_xray_teeth_dir (Linux, data_mode='xray')의 ③ 자체 데이터 기본값과 동일.
DEFAULT_DATASET_DIR = []
DEFAULT_COCO_DIRECTORIES = [
    ('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1106/train/',
     '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1106/train/annotations.json'),
    ('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_1106/train/',
     '/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_1106/train/annotations.json'),
]
DEFAULT_COCO_DIRECTORIES_VAL = [
    ('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1104/valid/',
     '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1104/valid/annotations.json'),
    ('/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_1106/valid/',
     '/data1/jooyonglee/reverse_tomo/xray_panoramic/seoul_univ_inspect_1106/valid/annotations.json'),
]
DEFAULT_NUM_CLASSES = 32


def build_gt_dataset(
    image_set='train',
    dataset_dir=None,
    coco_directories=None,
    coco_directories_val=None,
    annot_file=None,
    num_classes=DEFAULT_NUM_CLASSES,
    segmentation_head=True,
    **extra_args,
):
    """학습과 같은 로더(rfdetr.datasets.build_dataset, dataset_file='xray_teeth')로 데이터셋을
    만든다. 모델/학습 관련 인자는 전부 무시되므로(dataset 쪽 ``**kwargs``), 데이터 경로만 신경쓰면
    된다. annot_file=None이면 (자체 coco_directories만 쓸 때처럼) 사전에 빌드해둔 평가용 결합
    coco json 없이도 동작한다 (rfdetr/datasets/xraypanoramic.py::XrayPnoaramicInstanceCoco 참고)."""
    from rfdetr.main import populate_args
    from rfdetr.datasets import build_dataset

    args = populate_args(
        dataset_file='xray_teeth',
        dataset_dir=DEFAULT_DATASET_DIR if dataset_dir is None else dataset_dir,
        coco_directories=DEFAULT_COCO_DIRECTORIES if coco_directories is None else coco_directories,
        coco_directories_val=(
            DEFAULT_COCO_DIRECTORIES_VAL if coco_directories_val is None else coco_directories_val
        ),
        annot_file=annot_file,
        num_classes=num_classes,
        segmentation_head=segmentation_head,
        **extra_args,
    )
    return build_dataset(image_set, args, resolution=args.resolution)


def select_indices(num_items, amount='all', seed=0):
    """``amount``: ``'all'``(전부) | int(개수) | 0과 1 사이 float(비율)."""
    if amount == 'all':
        return list(range(num_items))
    if isinstance(amount, float) or (isinstance(amount, str) and '.' in amount):
        ratio = float(amount)
        if not 0 < ratio <= 1:
            raise ValueError(f'ratio amount must be in (0, 1], got {ratio}')
        count = max(1, round(num_items * ratio))
    else:
        count = min(int(amount), num_items)
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(num_items, size=count, replace=False).tolist())


def visualize_dataset(
    dataset,
    amount='all',
    save_dir='outputs/dataset_visualize',
    drawing_mode='polygon',
    polygon_thickness=None,
    with_source_concat=False,
    num_classes=DEFAULT_NUM_CLASSES,
    draw_masks=True,
    draw_labels=True,
    label_font_scale=None,
    seed=0,
):
    """``dataset[idx]``가 돌려주는 GT만으로(추론 없이) ``render_detection``을 호출해 박스/마스크를
    그려 저장한다. ``dataset``은 ``build_gt_dataset()`` 등으로 만든, ``(img, target)``을 돌려주는
    어떤 데이터셋이든 된다 (``target``에 정규화 cxcywh ``boxes``, 원시 class id ``labels``, 선택적
    ``masks``가 있어야 한다 -- rfdetr의 xray_teeth 데이터셋 포맷)."""
    from trainer import torch_utils, vtk_utils, image_utils
    from rfdetr.datasets.xraypanoramic import label_to_fdi

    colors = vtk_utils.get_teeth_color_table(normalize=False)
    colors[0, :] = 0

    indices = select_indices(len(dataset), amount, seed=seed)
    os.makedirs(save_dir, exist_ok=True)

    for idx in indices:
        img, target = dataset[idx]
        image = torch_utils.to_numpy(img)
        image = np.transpose(image, (1, 2, 0))
        image = image_utils.to_magnitude_images(image)
        render_h, render_w = image.shape[:2]

        boxes = torch_utils.to_numpy(target['boxes']).reshape(-1, 4)
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes_xy = np.stack([
            (cx - bw / 2) * render_w, (cy - bh / 2) * render_h,
            (cx + bw / 2) * render_w, (cy + bh / 2) * render_h,
        ], axis=1)
        labels = label_to_fdi(
            torch_utils.to_numpy(target['labels']).reshape(-1).astype(np.int64), num_classes
        )

        masks = None
        if draw_masks and target.get('masks') is not None:
            masks = torch_utils.to_numpy(target['masks']).reshape(-1, render_h, render_w).astype(bool)

        rendered, _ = render_detection(
            image, boxes_xy, labels, colors,
            masks=masks,
            drawing_mode=drawing_mode,
            polygon_thickness=polygon_thickness,
            with_source_concat=with_source_concat,
            draw_labels=draw_labels,
            label_font_scale=label_font_scale,
        )
        save_rendered_image(rendered, save_dir=save_dir, fname=f'{idx:06d}.jpg')

    print(f'[visualize_dataset] saved {len(indices)}/{len(dataset)} GT visualizations to {save_dir}')
    return save_dir


def _parse_amount(value):
    if value == 'all':
        return 'all'
    if '.' in value:
        return float(value)
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image_set', type=str, default='train', choices=['train', 'val'],
                        help='train: coco_directories, val: coco_directories_val')
    parser.add_argument('--amount', type=_parse_amount, default='all',
                        help="'all' | 개수(int, 예: 50) | 비율(float, 예: 0.1)")
    parser.add_argument('--save_dir', type=str, default='outputs/dataset_visualize')
    parser.add_argument('--drawing_mode', type=str, default='polygon', choices=['mask', 'polygon'])
    parser.add_argument('--no_masks', action='store_true', help='박스만 그리고 마스크는 생략')
    parser.add_argument('--no_labels', action='store_true', help='FDI 번호 텍스트를 생략')
    parser.add_argument('--label_font_scale', type=float, default=None,
                        help='FDI 번호 폰트 크기(cv2 fontScale). 기본값 None(영상 너비 기준 자동)')
    parser.add_argument('--with_source_concat', action='store_true', help='원본과 결과를 나란히 저장')
    parser.add_argument('--seed', type=int, default=0, help='--amount가 all이 아닐 때 샘플링 시드')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument('--dataset_dir', type=str, nargs='*', default=None,
                        help='공개 kaggle 데이터 폴더(①②). 기본값: 없음(자체 coco_directories만 사용)')
    parser.add_argument('--annot_file', type=str, default=None,
                        help='평가용 결합 coco json. 기본값 None(coco_directories만으로 동작)')
    args = parser.parse_args()

    dataset = build_gt_dataset(
        image_set=args.image_set,
        dataset_dir=args.dataset_dir,
        annot_file=args.annot_file,
        num_classes=args.num_classes,
    )
    visualize_dataset(
        dataset,
        amount=args.amount,
        save_dir=args.save_dir,
        drawing_mode=args.drawing_mode,
        draw_masks=not args.no_masks,
        draw_labels=not args.no_labels,
        label_font_scale=args.label_font_scale,
        with_source_concat=args.with_source_concat,
        num_classes=args.num_classes,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
