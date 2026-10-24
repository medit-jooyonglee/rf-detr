import os
import glob
import cv2
import ast
import random
import argparse
import shutil
import subprocess
import torch.nn as nn
from pathlib import Path
import time
from collections import defaultdict

import onnx
import torch
# import onnxsim
import numpy as np
from PIL import Image

import rfdetr.util.misc as utils
from rfdetr.util.utils import clean_state_dict
import rfdetr.datasets.transforms as T
from rfdetr.models import build_model
from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall
from trainer import torch_utils

from rfdetr.config import (
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRSegPreviewConfig,
    TrainConfig,
    SegmentationTrainConfig,
    ModelConfig
)

from rfdetr.engine import draw_preditions_boxes

g_rfdetr_model = None
MASK_PROBABILITY_THRESHOLD = 0.5
EXPORT_FIX_SIZE = (384, 704)
TEMP_EXCLUDED_FILENAME_KEYWORDS = ('peri',)


def is_inference_filename_excluded(file_path):
    """Temporary filename filter used by the batch prediction command."""
    filename = os.path.basename(os.fspath(file_path)).casefold()
    return any(
        keyword.casefold() in filename
        for keyword in TEMP_EXCLUDED_FILENAME_KEYWORDS
    )

def get_target_image_size(image_shape, rererence_width:int = 640):
    ih, iw = image_shape[:2]
    multiple = 16
    
    scale = rererence_width / iw
    # else:
        # scale = 
    # ih = (ih * scale + multiple - 1) // multiple * multiple
    ih = int(np.ceil(ih * scale / multiple) * multiple)
    return (ih, rererence_width)
    

def get_image_size(shape, stride=64):
    h0, w0 = shape
    size = (h0 // stride + 1) * stride, (w0 // stride + 1) * stride
    return size


def resize_image(image, stride=64, target_width=640, resize_hw=None):
    """Letterbox an image, optionally to an exact export size in (H, W)."""
    if resize_hw is None:
        size = get_target_image_size(
            image.shape[:2], rererence_width=target_width
        )
        target_h, target_w = get_image_size(size, stride=stride)
    else:
        if len(resize_hw) != 2:
            raise ValueError("resize_hw must contain exactly (height, width).")
        target_h, target_w = (int(value) for value in resize_hw)
        if target_h <= 0 or target_w <= 0:
            raise ValueError("resize_hw dimensions must be positive.")

    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = cv2.resize(
        image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas_shape = (target_h, target_w, *image.shape[2:])
    canvas = np.zeros(canvas_shape, dtype=image.dtype)
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized
    content_box = (
        offset_x,
        offset_y,
        offset_x + resized_w,
        offset_y + resized_h,
    )
    return canvas, content_box



def is_adjacent_fdi(fdi_a, fdi_b):
    """같은 quadrant에서 번호가 1 차이 나거나 정중선(11-21, 31-41)을 사이에 둔 인접 치아인지."""
    qa, na = divmod(int(fdi_a), 10)
    qb, nb = divmod(int(fdi_b), 10)
    if qa == qb:
        return abs(na - nb) == 1
    return na == 1 and nb == 1 and {qa, qb} in ({1, 2}, {3, 4})


def compute_tooth_overlaps(instance_result, file_key=''):
    """인스턴스(치아) 마스크 간 중첩 영역을 계산한다.

    Returns:
        tooth_rows: 치아별 면적/중첩 면적/중첩 비율
        pair_rows: 중첩이 있는 치아 쌍별 교집합 면적, IoU, 작은 치아 대비 비율
        overlap_map: 2개 이상 치아가 겹친 픽셀 (H, W) bool
    """
    masks = instance_result['masks']
    fdis = instance_result['fdi']
    scores = instance_result['scores']
    if len(masks) == 0:
        return [], [], None
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    count_map = masks.sum(axis=0, dtype=np.uint8)
    overlap_map = count_map >= 2

    # 마스크 bbox 로 후보 쌍을 거른 뒤 교집합 계산
    mask_boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        mask_boxes.append(
            (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None
        )
    pair_rows = []
    pair_px = defaultdict(dict)
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            box_a, box_b = mask_boxes[a], mask_boxes[b]
            if box_a is None or box_b is None:
                continue
            x0, y0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            x1, y1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            if x0 >= x1 or y0 >= y1:
                continue
            inter = int(np.logical_and(masks[a, y0:y1, x0:x1], masks[b, y0:y1, x0:x1]).sum())
            if inter == 0:
                continue
            union = int(areas[a] + areas[b] - inter)
            fdi_a, fdi_b = sorted((int(fdis[a]), int(fdis[b])))
            pair_px[a][b] = pair_px[b][a] = inter
            pair_rows.append({
                'file': file_key,
                'fdi_a': fdi_a,
                'fdi_b': fdi_b,
                'adjacent': is_adjacent_fdi(fdi_a, fdi_b),
                'inter_px': inter,
                'iou': inter / max(union, 1),
                'inter_over_min_area': inter / max(min(areas[a], areas[b]), 1),
            })

    tooth_rows = []
    for k in range(len(masks)):
        area = int(areas[k])
        overlap_px = int(np.logical_and(masks[k], overlap_map).sum())
        neighbors = pair_px.get(k, {})
        top = max(neighbors.items(), key=lambda kv: kv[1]) if neighbors else None
        tooth_rows.append({
            'file': file_key,
            'fdi': int(fdis[k]),
            'score': float(scores[k]),
            'area_px': area,
            'overlap_px': overlap_px,
            'overlap_ratio': overlap_px / max(area, 1),
            'num_overlap_neighbors': len(neighbors),
            'neighbors': ' '.join(
                f'{int(fdis[j])}:{px}' for j, px in sorted(neighbors.items(), key=lambda kv: -kv[1])
            ),
            'max_neighbor_fdi': int(fdis[top[0]]) if top else '',
            'max_neighbor_px': top[1] if top else 0,
        })
    return tooth_rows, pair_rows, overlap_map


def draw_overlap_highlight(color_image, overlap_map, color=(255, 0, 0), alpha=0.6):
    """polygon 렌더 위에 중첩 픽셀을 반투명 색으로 칠한다 (RGB)."""
    image = color_image.astype(np.float32).copy()
    image[overlap_map] = image[overlap_map] * (1 - alpha) + np.array(color, np.float32) * alpha
    return image.astype(np.uint8)


def write_overlap_report(tooth_rows, pair_rows, save_dir, image_count):
    import csv
    os.makedirs(save_dir, exist_ok=True)

    def write_csv(name, rows):
        path = os.path.join(save_dir, name)
        if rows:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        return path

    write_csv('overlap_per_tooth.csv', tooth_rows)
    write_csv('overlap_pairs.csv', pair_rows)

    # FDI 별 집계
    by_fdi = defaultdict(list)
    for row in tooth_rows:
        by_fdi[row['fdi']].append(row)
    fdi_rows = []
    for fdi in sorted(by_fdi):
        rows = by_fdi[fdi]
        ratios = np.array([r['overlap_ratio'] for r in rows])
        overlap_px = np.array([r['overlap_px'] for r in rows])
        fdi_rows.append({
            'fdi': fdi,
            'n_detected': len(rows),
            'n_with_overlap': int((overlap_px > 0).sum()),
            'pct_with_overlap': 100.0 * (overlap_px > 0).mean(),
            'mean_area_px': float(np.mean([r['area_px'] for r in rows])),
            'mean_overlap_px': float(overlap_px.mean()),
            'mean_overlap_ratio_pct': 100.0 * ratios.mean(),
            'median_overlap_ratio_pct': 100.0 * np.median(ratios),
            'p95_overlap_ratio_pct': 100.0 * np.percentile(ratios, 95),
            'max_overlap_ratio_pct': 100.0 * ratios.max(),
        })
    write_csv('overlap_summary_by_fdi.csv', fdi_rows)

    by_pair = defaultdict(list)
    for row in pair_rows:
        by_pair[(row['fdi_a'], row['fdi_b'])].append(row)
    pair_summary = []
    for (fa, fb), rows in by_pair.items():
        pair_summary.append({
            'fdi_a': fa,
            'fdi_b': fb,
            'adjacent': rows[0]['adjacent'],
            'n_images': len(rows),
            'mean_inter_px': float(np.mean([r['inter_px'] for r in rows])),
            'max_inter_px': int(np.max([r['inter_px'] for r in rows])),
            'mean_iou_pct': 100.0 * float(np.mean([r['iou'] for r in rows])),
            'mean_inter_over_min_pct': 100.0 * float(np.mean([r['inter_over_min_area'] for r in rows])),
        })
    pair_summary.sort(key=lambda r: (-r['n_images'], -r['mean_inter_px']))
    write_csv('overlap_summary_by_pair.csv', pair_summary)

    total_area = sum(r['area_px'] for r in tooth_rows)
    total_overlap = sum(r['overlap_px'] for r in tooth_rows)
    n_non_adjacent = sum(1 for r in pair_rows if not r['adjacent'])
    lines = [
        '# Per-tooth overlap report',
        '',
        f'- images: {image_count}',
        f'- detected teeth: {len(tooth_rows)}',
        f'- teeth with overlap: {sum(1 for r in tooth_rows if r["overlap_px"] > 0)}',
        f'- overlapping pairs: {len(pair_rows)} (non-adjacent: {n_non_adjacent})',
        f'- overlapped tooth pixels / total tooth pixels: {100.0 * total_overlap / max(total_area, 1):.2f}%',
        '',
        '## By FDI',
        '',
        '| FDI | n | overlap>0 (%) | mean ratio (%) | median (%) | p95 (%) | max (%) |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in fdi_rows:
        lines.append(
            f"| {r['fdi']} | {r['n_detected']} | {r['pct_with_overlap']:.1f} | "
            f"{r['mean_overlap_ratio_pct']:.2f} | {r['median_overlap_ratio_pct']:.2f} | "
            f"{r['p95_overlap_ratio_pct']:.2f} | {r['max_overlap_ratio_pct']:.2f} |"
        )
    lines += [
        '',
        '## Top pairs (by frequency)',
        '',
        '| pair | adjacent | n images | mean px | max px | mean IoU (%) | mean inter/min (%) |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in pair_summary[:30]:
        lines.append(
            f"| {r['fdi_a']}-{r['fdi_b']} | {r['adjacent']} | {r['n_images']} | "
            f"{r['mean_inter_px']:.0f} | {r['max_inter_px']} | {r['mean_iou_pct']:.2f} | "
            f"{r['mean_inter_over_min_pct']:.2f} |"
        )
    top_teeth = sorted(tooth_rows, key=lambda r: -r['overlap_ratio'])[:20]
    lines += [
        '',
        '## Worst teeth (highest overlap ratio)',
        '',
        '| file | FDI | area px | overlap px | ratio (%) | neighbors (fdi:px) |',
        '|---|---|---|---|---|---|',
    ]
    for r in top_teeth:
        lines.append(
            f"| {r['file']} | {r['fdi']} | {r['area_px']} | {r['overlap_px']} | "
            f"{100.0 * r['overlap_ratio']:.2f} | {r['neighbors']} |"
        )
    report_path = os.path.join(save_dir, 'overlap_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved overlap report to: {report_path}')
    return report_path


DEFAULT_FACE_MODEL = 'thirds/face_detector/models/face_landmarker.task'


def load_face_cropper(face_model=DEFAULT_FACE_MODEL):
    """MediaPipe Face Landmarker 기반 MouthCropper (thirds/face_detector)."""
    import sys
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from thirds.face_detector.mouth_crop import MouthCropper, ensure_landmarker_model
    return MouthCropper(ensure_landmarker_model(Path(face_model)))


def restore_instance_result(instance_result, roi_box, image_hw):
    """crop 좌표계 instance 결과(boxes/masks)를 원본 이미지 좌표로 복원."""
    x0, y0, x1, y1 = roi_box
    masks = instance_result['masks']
    full_masks = np.zeros((len(masks), *image_hw), dtype=bool)
    full_masks[:, y0:y1, x0:x1] = masks
    boxes = np.asarray(instance_result['boxes'], dtype=np.float32).reshape(-1, 4)
    return dict(
        instance_result,
        boxes=boxes + np.array([x0, y0, x0, y0], dtype=np.float32),
        masks=full_masks,
    )


def instance_label_map(instance_result, image_hw):
    """instance mask -> FDI label map (겹치면 큰 FDI, draw_preditions_boxes와 동일 규칙)."""
    label_map = np.zeros(image_hw, dtype=np.uint8)
    for fdi, mask in sorted(zip(instance_result['fdi'], instance_result['masks']), key=lambda v: int(v[0])):
        label_map[mask] = int(fdi)
    return label_map


def render_instances(image_rgb, instance_result, roi_box=None, thickness=None):
    """원본 RGB 이미지 위에 치아별 외곽선 + teeth ROI 박스를 그린다."""
    from trainer import vtk_utils
    colors = vtk_utils.get_teeth_color_table(normalize=False)
    image = np.ascontiguousarray(image_rgb.copy())
    thickness = thickness or 2 * max(1, int(round(image.shape[1] / 1000)))
    for fdi, mask in zip(instance_result['fdi'], instance_result['masks']):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = tuple(int(c) for c in colors[int(fdi)])
        cv2.polylines(image, contours, True, color, thickness, lineType=cv2.LINE_AA)
    if roi_box is not None:
        cv2.rectangle(image, tuple(roi_box[:2]), tuple(roi_box[2:]), (255, 255, 0), max(1, thickness // 2))
    return image


def main(*,
         use_ema=True, with_source_concat=False,
         test_path='',
         save_dir = 'temp/resultsxray',
         drawing_mode='mask',
         polygon_thickness=None,
         overlap_report=None,
         max_images=None,
         convert=None,
         polygon_epsilon=2.0,
         input_width=640,
         segmentation_crop_size=(128, 64),
         pretrain_weights='outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth',
         face_crop=False,
         face_model=DEFAULT_FACE_MODEL,
         face_crop_margin=0.08,
         **kwargs):
    """face_crop: 얼굴 사진 -> MediaPipe 입 랜드마크로 teeth ROI crop -> RGB 추론 -> 원본 좌표 복원.
    natural 학습 데이터는 RGB 컬러(CocoDetection)로 읽으므로 face_crop 경로는 컬러 입력을 쓴다.
    """
    # FIXME: save_path
    from trainer import diskmanager, image_utils
    # save_dir = f'results/test/0828'
    if convert not in (None, 'labelme'):
        raise ValueError(f"Unsupported conversion format: {convert!r}")
    if polygon_epsilon < 0:
        raise ValueError("polygon_epsilon must be non-negative")
    if convert == 'labelme':
        drawing_mode = 'polygon'
        if overlap_report is None:
            overlap_report = False
    assert drawing_mode in ['mask', 'polygon'], "Invalid drawing mode specified."
    model = init_and_get_model(export=False, use_ema=use_ema,
                                segmentation_crop_size=tuple(segmentation_crop_size),
                                pretrain_weights=pretrain_weights)
    if overlap_report is None:
        overlap_report = drawing_mode == 'polygon'
    all_tooth_rows, all_pair_rows = [], []
    processed_count = 0
    # model = rf_detr.model.model
    
        # ''
        
        # next(rf_detr.model.model.parameters()).device
    # model = rf_detr.model.model
    # model.eval()
    from trainer import torch_utils
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle/Teeth Segmentation JSON/d2/img/'
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_teeth_seg_kaggle/Teeth Segmentation PNG/d2/img/'
    # path = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm'
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON'
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm/'
    path = test_path
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_0824/JPEGImages/'
    # path = 
    
    mask_save_dir = save_dir # 'temp/resultsxray'
    found = diskmanager.deep_search_files(path, exts=['.jpg', '.jpeg', 'JPG', 'JPEG', '.png', 'PNG'])
    cropper = load_face_cropper(face_model) if face_crop else None
    excluded = [file for file in found if is_inference_filename_excluded(file)]
    found = [file for file in found if not is_inference_filename_excluded(file)]
    for file in excluded:
        print(f'Skipping excluded filename: {file}')
    
    print('emtpy images')
    # found = glob.glob(f'{path}/*.jpg')
    i_break = 30
    for idx, file in enumerate(found):
        # if idx > i_break:
            # break
        if max_images is not None and idx >= max_images:
            break
        # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        roi_box = None
        if cropper is not None:
            img = image_utils.cv2_imread(file, flags=cv2.IMREAD_COLOR)
            try:
                landmarks, _ = cropper.detect_landmarks(img)
            except ValueError as e:
                print(f'[face-crop] skip {file}: {e}')
                continue
            roi_box = cropper.teeth_roi_box(landmarks, img.shape[:2], margin=face_crop_margin)
            x0, y0, x1, y1 = roi_box
            model_input = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            print(f'[face-crop] {os.path.basename(file)} roi={roi_box} crop_hw={model_input.shape[:2]}')
        else:
            img = image_utils.cv2_imread(file, flags=cv2.IMREAD_GRAYSCALE)
            model_input = img
        rsz_img, input_content_box = resize_image(model_input, stride=64, target_width=input_width)
        if rsz_img.ndim == 2:
            img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
        else:
            img2 = np.transpose(rsz_img, (2, 0, 1)) / 255.

        img_tensor = torch_utils.data_convert(img2[None])
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # print(outputs.keys())
            print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
            if 'pred_masks' in outputs:
                print(outputs['pred_masks'].shape)
        
            try:
                instance_results = [] if overlap_report or convert == 'labelme' or face_crop else None
                color_image, mask_iamge = draw_preditions_boxes(
                    img_tensor, outputs, save=False, origin_size=model_input.shape[:2],
                                        segmentation_mode='crop_and_resize',
                    mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                    segmentation_crop_box_scale=1.15,
                    with_source_concat=with_source_concat,
                    paste_masks_at_original_size=True,
                    input_content_box=input_content_box,
                    drawing_mode=drawing_mode,
                    polygon_thickness=polygon_thickness,
                    instance_results=instance_results,
                )


                assert len(mask_iamge) == 1, "Expected a single mask image."
                savename = os.path.join(mask_save_dir, os.path.relpath(file, path))
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                from trainer import image_utils
                stem, ext = os.path.splitext(savename)
                if roi_box is not None:
                    # crop 결과 -> 원본 좌표 복원. crop 단독 렌더는 디버깅용으로 따로 저장
                    image_utils.cv2_imwrite(f'{stem}_crop.png', color_image[0].astype(np.uint8)[..., ::-1])
                    instance_results[0] = restore_instance_result(instance_results[0], roi_box, img.shape[:2])
                    color_image = [render_instances(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB), instance_results[0], roi_box, polygon_thickness)]
                    mask_iamge = [instance_label_map(instance_results[0], img.shape[:2])]
                if convert == 'labelme':
                    from rfdetr.util.detr2labelme import save_labelme_annotation

                    source_path = Path(file).resolve()
                    output_image_path = Path(savename).resolve()
                    if source_path != output_image_path:
                        shutil.copy2(source_path, output_image_path)
                    labelme_path = save_labelme_annotation(
                        f'{stem}.json',
                        savename,
                        instance_results[0],
                        image_height=img.shape[0],
                        image_width=img.shape[1],
                        polygon_epsilon=polygon_epsilon,
                    )
                    print(f"Saved Labelme annotation to: {labelme_path}")
                else:
                    # 원본이 png면 렌더 이미지와 이름이 겹치므로 _mask 접미사
                    mask_path = f'{stem}_mask.png' if ext.lower() == '.png' else f'{stem}.png'
                    image_utils.cv2_imwrite(mask_path, mask_iamge[0].astype(np.uint8))
                    image_utils.cv2_imwrite(savename, color_image[0].astype(np.uint8)[..., ::-1])  # Convert RGB to BGR for saving with OpenCV
                    print(f"Saved mask image to: {mask_path}")
                processed_count += 1

                if overlap_report:
                    file_key = os.path.relpath(file, path)
                    tooth_rows, pair_rows, overlap_map = compute_tooth_overlaps(
                        instance_results[0], file_key=file_key
                    )
                    all_tooth_rows.extend(tooth_rows)
                    all_pair_rows.extend(pair_rows)
                    if overlap_map is not None:
                        highlight = draw_overlap_highlight(color_image[0], overlap_map)
                        stem, ext = os.path.splitext(savename)
                        image_utils.cv2_imwrite(f'{stem}_overlap{ext}', highlight[..., ::-1])

            except Exception as e:
                import traceback
                traceback.print_exc()
                print('draw_preditions_boxes failed')
    if overlap_report:
        write_overlap_report(all_tooth_rows, all_pair_rows, mask_save_dir, processed_count)
    # model = build_model(args)
    # checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
    # res = model.load_state_dict(checkpoint['model'], strict=False)
    
    print(model)
    

def init_and_get_model(config=None, device='cuda', export=False, use_ema=True,
                        segmentation_crop_size=(128, 64),
                        pretrain_weights='outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth'):
    global g_rfdetr_model
    config = config or dict()
    if g_rfdetr_model is None:
        rf_detr = RFDETRSmall(

            patch_size=16,
            num_windows=4,
            # num_queries=100,
            num_queries=100,
            group_detr=5,
            num_select=35,
            encoder='dinov2_windowed_tiny',
            # encoder='dinov2_windowed_base',

            # patch_size=24,
            # num_channels=1,
            # eval=True,
            # num_classes=32,
            num_classes=32,
            segmentation_head=True,
            segmentation_mode='crop_and_resize',
            segmentation_crop_size=segmentation_crop_size,
            coarse_hint_scale=0.35,
            segmentation_crop_box_scale=1.15,
            coarse_hint_dropout=0.30,

            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
            # pretrain_weights='e:/temp/checkpoint.pth',
            
            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg_0819_all/checkpoint.pth',
            # # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg_0819_crop_retrain/checkpoint.pth',
            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg_crop_regularization/checkpoint.pth',
            # pretrain_weights='output/bounding_bbox/checkpoint.pth',
            pretrain_weights=pretrain_weights,
            # 'output/xray_teeth33_dinov2tiny_small_seg_crop_retrain'
            # 'output/xray_teeth33_dinov2tiny_small_seg_0819_all/eval'
            **config,
            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
            
        )

        if use_ema:
            checkpoint_path = rf_detr.model.args.pretrain_weights
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False,
            )
            ema_state_dict = checkpoint.get('ema_model')
            if ema_state_dict is None:
                print(
                    f"EMA weights not found in {checkpoint_path}; "
                    "using regular model weights."
                )
            else:
                rf_detr.model.model.load_state_dict(
                    clean_state_dict(ema_state_dict),
                    strict=True,
                )
                print(f'Loaded EMA weights from: {checkpoint_path}')

        # g_rfdetr_model = rf_detr.model.model
        # g_rfdetr_model.eval()
        # g_rfdetr_model.cuda()
        
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                assert x.shape[0] == 1, "Only batch size of 1 is supported for this wrapper."
                res =  self.model(x)
                # bboxes / logits / masks
                # not export option
                if isinstance(res, dict):
                    bboxes = res['pred_boxes']
                    logits = res['pred_logits']
                    masks = res['pred_masks']
                else:
                    bboxes, logits, masks = res
                # # return bboxes, logits, masks
                # # not change ...drawing-bounding-box opiont
                # probs = logits.sigmoid()
                # bboxes, probs, masks = [torch.squeeze(v) for v in (bboxes, probs, masks)]
                # label = probs.argmax(dim=-1)
                # # posit = label > 0
                # res = [bboxes, probs, masks, label]
                # return res
                return bboxes, logits, masks
                # # res = bboxes[posit], probs[posit],  masks[posit].to(torch.float32), label[posit]
                # keys = [
                #     'pred_boxes',
                #     'pred_logits',
                #     'pred_masks',
                #     'pred_labels'
                # ]
                # return dict(zip(keys, res))
                

        if export:
            rf_detr.model.model.export()
            
            model = ModelWrapper(rf_detr.model.model)
            model.to(torch.device(device))
            print('device', next(model.parameters()).device)
            g_rfdetr_model = model
        else:
            g_rfdetr_model = rf_detr.model.model
            g_rfdetr_model.eval()
            g_rfdetr_model.to(torch.device(device))
            print('device', next(g_rfdetr_model.parameters()).device)
            
            
                
    return g_rfdetr_model
    

    
    # model.to()
def export_libtorch(output_path='e:/temp/model_libtorch.pt', shape=(384, 704), dtype=torch.float16):
    model = init_and_get_model(export=True)
    model.eval()
    model.requires_grad_(False)
    # dtype = torch.float16
    
    # model.half()
    model.to(dtype)
    # model
    # model.export()  # swap in forward_export + baked position embeddings (required before tracing)

    dummy = torch.randn(1, 3, *shape).cuda().to(dtype)
    # model(dummy)  # run once to ensure any lazy modules are initialized
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, check_trace=False)

    traced.save(output_path)
    print(f'saved libtorch model to {output_path}')
    
    
def coreml_export_main():
    from trainer import coreml_utils, torch_utils
    # device = torch.
    model = init_and_get_model(config=dict(
        antialias=False,
        ))
    model.eval()
    model.requires_grad_(False)
    
    # device = torch.device('cpu')
    # model.to()
    device_list = [
        torch.device(dev) for dev in ['mps']
    ]
    for device in device_list:
        x0 = torch.randn(1, 3, 384, 704)
        x0.to(device)
        
        y0 = model(x0)
        print(device, 'runing complete', x0.shape, torch_utils.get_shape(y0))

    # model.to(dtype)
    
    with torch.no_grad():
        traced = torch.jit.trace(model, (x0))


                
    output_names = [
        'pred_boxes', 'pred_logits', 'pred_masks',
        'pred_labels'
    ]
    coreml_model = coreml_utils.export_coreml(
        traced, [(1, 3, 384, 704)], output_names=output_names
        
    )
    coreml_model.save('temp.mlpackage')
    

def test_coreml_inference():
    x0 = np.random.randn(1, 3, 384, 704)
    from trainer import coreml_utils, torch_utils, timefn
    from rfdetr.datasets.xraypanoramic import get_size
    libtorch = 'outputs/temp.pt'
    
    @timefn
    def coreml_run(img):
        res = coreml_utils.coreml_predict(model, (img,))
        return res
        # print(torch_utils.get_shape(res))
        # dtype = torch.float16
        
    path = '/Users/meditai/Desktop/dataset/xray'
    from trainer import diskmanager, image_utils, torch_utils
    found = diskmanager.deep_search_files(path, exts=['.jpg', '.jpeg', '.JPG'])
    
    for file in found:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        size = get_size(img.shape[:2], refenrece_width=640, stride=64)
        rsz_img = cv2.resize(img, tuple(size[::-1]), interpolation=cv2.INTER_LINEAR)
        rsz_img = np.repeat(rsz_img[None], 3, axis=0).astype(np.float32) / 255.
        rsz_img = rsz_img[None]
        # res = coreml_utils.coreml_predict(model, (rsz_img,))
        res = coreml_run(rsz_img)
        input_tensor = torch_utils.data_convert(rsz_img)
        # tensor = torch_utils.data_convert(rsz_img[None], dtype=dtype)
        # with torch.no_grad():
            # res = model(tensor)
        # res = [torch.unsqueeze(v, 0) for v in res]
        res = [v[None] for v in res]
        res = torch_utils.data_convert(res, device='cpu')
        output_keys = [
                    'pred_boxes',
                    'pred_logits',
                    'pred_masks',
                    'pred_labels'
                ]
        res_dicts = dict(zip(output_keys, res))
        draw_preditions_boxes(
            # torch_utils.data_convert(rsz_img[None]),
            input_tensor,
            res_dicts,
            save=True,
            save_dir='outputs/resulst/export',
            segmentation_mode='crop_and_resize',
            mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
            segmentation_crop_box_scale=1.15,
        )
        
 
    # for _ in range(10):
    #     coreml_run()       
    '/Users/meditai/Desktop/dataset/xray'
def inference_libtorch_model_main():
    from rfdetr.datasets.xraypanoramic import get_size
    # libtorch = 'outputs/temp.pt'
    libtorch ='outputs/temp.pt'
    
    model = torch.jit.load(libtorch)
    model.cuda()
    model.eval()
    # model.requires_grad_(False)
    
    dtype = torch.float16
    
    filter_image_size = [
        (1000, 2000)
    ]
    
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle_2222/Radiographs/'
    path = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm'
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON'
    from trainer import diskmanager, image_utils, torch_utils
    found = diskmanager.deep_search_files(path, exts=['.jpg', '.jpeg', '.JPG'])
    save = True
    mask_save_dir = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm_masks'
    for file in found:
        # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = image_utils.cv2_imread(file, flags=cv2.IMREAD_GRAYSCALE)
        
        darwing, mask = predict_wrapper(img, model=model)
        
        # cv2.imwrite(f'temp/{os.path.basename(file)}', darwing.astype(np.uint8))
        image_utils.cv2_imwrite(f'temp/mask_{os.path.basename(file)}', darwing.astype(np.uint8)[..., ::-1])
        # shape = img.shape[:2]
        


def predict_wrapper(img, model=None, drawing_mode='mask',
                    return_instance_results=False):
    """Run one image while preserving the legacy two-value return by default.

    When ``return_instance_results`` is true, a third value is returned with
    the independent per-tooth masks produced before the single-winner FDI
    label image is composed.  This is the overlap-preserving input used by
    reverse registration; callers that only need ``(drawing, mask)`` are
    unaffected.
    """
    if model is None:
        model = init_and_get_model(export=False, use_ema=True)
    if img.ndim ==3:
        # simple gray image as pickking first channel
        img = img[..., 0]
    if isinstance(model, torch.jit.ScriptModule):
        # export model
        rsz_img, input_content_box = resize_image(img, stride=64, resize_hw=EXPORT_FIX_SIZE)
    else:
        rsz_img, input_content_box = resize_image(img, stride=64)
    img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.

    model_parameter = next(model.parameters(), None)
    convert_options = {}
    if model_parameter is not None:
        convert_options = {
            'device': model_parameter.device,
            'dtype': model_parameter.dtype,
        }
    img_tensor = torch_utils.data_convert(img2[None], **convert_options)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        # protocol export model or vanila modell, vanila model return dict, export model return tuple
        if isinstance(outputs, dict):
            pass
        else:
            bboxes, logits, masks = outputs
            outputs = {
                'pred_boxes': bboxes,
                'pred_logits': logits,
                'pred_masks': masks
            }
        # print(outputs.keys())
        print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
        if 'pred_masks' in outputs:
            print(outputs['pred_masks'].shape)
    
        instance_results = [] if return_instance_results else None
        # try:
        color_image, mask_iamge = draw_preditions_boxes(
            img_tensor, outputs, save=False, origin_size=img.shape[:2],
            segmentation_mode='crop_and_resize',
            mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
            segmentation_crop_box_scale=1.15,
            with_source_concat=False,
            paste_masks_at_original_size=True,
            input_content_box=input_content_box,
            drawing_mode=drawing_mode,
            instance_results=instance_results,
        )
            
    if return_instance_results:
        return color_image[0], mask_iamge[0], instance_results[0]
    return color_image[0], mask_iamge[0]


"""
    
# teeth-학습 환경..?    
nohup python rfdetr/cli/main.py \
  --data_mode natural \
  --xray_crop_mode teeth_roi --xray_crop_prob 1.0 \
  --eval_xray_crop \
  --input_width 256 \
  --pretrain_weights outputs/natural_teeth_roi/checkpoint_best_regular.pth \
  --segmentation_crop_size 64 48 \
  --epochs 400 \
  --checkpoint_interval 100 \
  --batch_size 1 \
  --device cuda:5 \
  --eval --eval_save \
  --output_dir outputs/natural_teeth_roi  > logs/natural_teeth_roi.log 2>&1 &


1024x1024 어깨라인 사진 기준으로 구강쪽 이미지는 

아래 크기는
(heightxwidth)
입벌리떄 200x330

가슴주변부 상반신샷 
105x240x

crop_size
48, 48 이면 충분


"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RF-DETR tooth segmentation inference')
    parser.add_argument(
        '--use-ema',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use ema_model weights from the checkpoint (default: enabled).',
    )
    parser.add_argument(
        '--with-source-concat',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Concatenate source image with output for visualization (default: disabled).',
    )
    
    parser.add_argument(
        '--test-path',
        type=str,
        default='',
        help='Path to the test images.',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='temp/resultsxray',
        help='Directory to save the results.',
    )
    parser.add_argument(
        '--drawing-mode', '--drawing_mode',
        dest='drawing_mode',
        type=str,
        default='mask',
        choices=['mask', 'polygon'],
        help="mask: 병합 label map 채우기 (겹친 영역은 FDI 큰 치아로 덮임). "
             "polygon: 치아별 외곽선만 그려서 인접 치아 중첩 영역 확인.",
    )
    parser.add_argument(
        '--polygon-thickness', '--polygon_thickness',
        dest='polygon_thickness',
        type=int,
        default=None,
        help='polygon 모드 선 두께(px). 미지정 시 기존 자동 두께의 2배 (최소 2px, 2041px -> 4px).',
    )
    parser.add_argument(
        '--overlap-report', '--overlap_report',
        dest='overlap_report',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='치아별 중첩 영역 산출 후 save-dir 에 CSV/markdown 리포트 저장 (polygon 모드 기본 활성).',
    )
    parser.add_argument(
        '--max-images', '--max_images',
        dest='max_images',
        type=int,
        default=None,
        help='처리할 최대 이미지 수 (디버깅용).',
    )
    parser.add_argument(
        '--convert',
        type=str,
        choices=['labelme'],
        default=None,
        help=(
            'Save predictions in the selected annotation format. '
            'labelme writes only the original image and polygon JSON together, '
            'and automatically uses --drawing-mode polygon.'
        ),
    )
    parser.add_argument(
        '--polygon-epsilon', '--polygon_epsilon',
        dest='polygon_epsilon',
        type=float,
        default=2.0,
        help=(
            'RDP polygon simplification error in pixels for Labelme output. '
            '0 disables simplification (default: 2.0).'
        ),
    )
    parser.add_argument(
        '--input-width', '--input_width',
        dest='input_width',
        type=int,
        default=640,
        help='리사이즈 기준 너비 (cli/main.py --input_width와 동일 기본값). '
             '학습 때 바꿨다면 추론도 같은 값을 넣어야 한다.',
    )
    parser.add_argument(
        '--segmentation-crop-size', '--segmentation_crop_size',
        dest='segmentation_crop_size',
        type=int, nargs=2, default=[128, 64], metavar=('H', 'W'),
        help='crop_and_resize refinement ROI 크기 (H W, cli/main.py --segmentation_crop_size와 동일 기본값). '
             '학습 때 바꿨다면 추론도 같은 값을 넣어야 한다.',
    )
    parser.add_argument(
        '--pretrain-weights', '--pretrain_weights',
        dest='pretrain_weights',
        type=str,
        default='outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth',
        help='추론에 사용할 checkpoint 경로.',
    )
    parser.add_argument(
        '--face-crop', '--face_crop',
        dest='face_crop',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='얼굴 사진 입력: MediaPipe 입 랜드마크로 teeth ROI crop 후 RGB 추론, 결과는 원본 좌표로 복원.',
    )
    parser.add_argument(
        '--face-model', '--face_model',
        dest='face_model',
        type=str,
        default=DEFAULT_FACE_MODEL,
        help='MediaPipe Face Landmarker .task 경로 (없으면 다운로드).',
    )
    parser.add_argument(
        '--face-crop-margin', '--face_crop_margin',
        dest='face_crop_margin',
        type=float,
        default=0.08,
        help='inner-lip bbox 대비 각 변 margin (학습 teeth_roi eval_margin=0.08과 동일 기본값).',
    )
    cli_args = parser.parse_args()
    main(
        **vars(cli_args),
    )
    # 
    # main(use_ema=cli_args.use_ema,
        #  with_source_concat=cli_args.with_source_concat, **)
    # export_libtorch('outputs/temp.pt', shape=(384, 704), dtype=torch.float16)
    # coreml_export_main()
    # test_coreml_inference()
    # inference_libtorch_model_main()

