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
from numba import njit


@njit(cache=True)
def _merge_label_image_kernel(masks, boxes, labels, out):
    """Per-instance label merge restricted to each instance's own box region,
    instead of scanning the full (N, H, W) canvas -- masks are already False
    outside their box (paste_masks_in_image only wrote inside it), so the
    full-canvas scan was pure waste. Sequential over instances (not prange):
    N is small (tens of teeth) and writes to `out` can overlap between
    instances, so parallelizing would race."""
    for k in range(masks.shape[0]):
        x0, y0, x1, y1 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3]
        lbl = labels[k]
        for y in range(y0, y1):
            for x in range(x0, x1):
                if masks[k, y, x] and lbl > out[y, x]:
                    out[y, x] = lbl


def merge_label_image_by_numba(labels, masks, boxes_xy, image_shape):
    """numba replacement for ``np.max(labels[:,None,None]*masks, axis=0)``:
    only visits each instance's own box region (``boxes_xy``), not the full
    image, since everywhere outside it is already False. See
    rfdetr/deploy/profile_merge_label_image.py for the correctness/speed
    comparison against the (now uint8-dtype-fixed) numpy version."""
    h, w = image_shape
    out = np.zeros((h, w), dtype=np.uint8)
    if masks.shape[0] == 0:
        return out.astype(labels.dtype)
    boxes_np = np.asarray(boxes_xy, dtype=np.float64)
    x0 = np.clip(np.floor(boxes_np[:, 0]), 0, w)
    y0 = np.clip(np.floor(boxes_np[:, 1]), 0, h)
    x1 = np.clip(np.ceil(boxes_np[:, 2]), 0, w)
    y1 = np.clip(np.ceil(boxes_np[:, 3]), 0, h)
    int_boxes = np.ascontiguousarray(
        np.stack([x0, y0, x1, y1], axis=1).astype(np.int64)
    )
    labels_u8 = np.ascontiguousarray(labels.astype(np.uint8))
    masks_arr = np.ascontiguousarray(masks)
    _merge_label_image_kernel(masks_arr, int_boxes, labels_u8, out)
    return out.astype(labels.dtype)


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
    mask_boxes_xy=None,
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
        boxes_xy: ``(N, 4)`` xyxy, ``image`` 픽셀 좌표계. 그려지는 detection box -- segmentation이
            쓰는 (보통 더 큰) crop box와는 다를 수 있으니 ``label_image`` 병합 경계로는 쓰지 않는다
            (``mask_boxes_xy`` 참고).
        labels: ``(N,)`` int. ``colors``/``colors[label]``로 색상을 찾을 FDI(또는 클래스) id.
        colors: ``(num_colors, 3)`` 색상 테이블.
        masks: 선택. ``(N, H, W)`` bool/uint8 instance mask, ``image``와 같은 해상도로 이미
            복원되어 있어야 한다. ``None``이면 마스크를 그리지 않는다(= label_image도 None 반환).
        mask_boxes_xy: 선택. ``(N, 4)`` xyxy, 각 인스턴스 마스크가 실제로 값을 가질 수 있는
            (``paste_masks_in_image``가 썼던) 영역 -- 보통 ``boxes_xy``보다 크다
            (``segmentation_crop_box_scale``만큼 확장된 crop box). 주어지면 label_image 병합을
            이 영역으로만 제한해 전체 캔버스를 스캔하지 않는 numba 경로
            (``merge_label_image_by_numba``)를 쓴다 -- ``None``이면(예: GT 시각화, 또는
            segmentation_mode가 박스에 제한되지 않는 full_image일 때) 항상-정확한 전체 스캔으로
            fallback한다.
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
        elif mask_boxes_xy is not None:
            # Masks are already False everywhere outside their own
            # mask_boxes_xy region (paste_masks_in_image only wrote inside
            # it) -- scan just that region per instance instead of the full
            # (N, H, W) canvas. ~5.6x faster on a 28-tooth 1468x2988 image
            # (rfdetr/deploy/profile_merge_label_image.py).
            label_image = merge_label_image_by_numba(
                labels, masks, mask_boxes_xy, image.shape[:2]
            )
        else:
            # FDI labels are always <256 (max 48): multiplying in the original
            # (int64) dtype allocates/scans one (N, H, W) int64 temp array per
            # call -- ~977MB and ~175ms for a 28-tooth 1468x2988 image, the
            # single largest cost in render_detection (measured via
            # rfdetr/deploy/profile_render_bottleneck.py). uint8 keeps the same
            # temp 8x smaller; cast back so the returned dtype is unchanged.
            label_image = np.max(
                labels.astype(np.uint8)[:, None, None] * masks, axis=0
            ).astype(labels.dtype)

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
