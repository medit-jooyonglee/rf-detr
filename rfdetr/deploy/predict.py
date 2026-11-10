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
from rfdetr.datasets.xraypanoramic import apply_clahe
from rfdetr.datasets.xray_crop_aug import teeth_roi_box_center_aware
from trainer import torch_utils

# 가볍고 빠른 CLAHE 기본값 (단일 cv2 CLAHE 호출). 학습 측 augmentation 범위(1.5~3.5)의 중간값 --
# rfdetr/datasets/xraypanoramic.py DEFAULT_CONTRAST_AUG_CFG 참고. 배포 모델 호환성을 위해 기본은
# 비활성(--clahe로 켜기): 기존 배포 체크포인트는 CLAHE 분포를 학습하지 않았으므로, 끄는 쪽이 안전한 기본값이다.
DEFAULT_PREDICT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_PREDICT_CLAHE_TILE_GRID = 8

# two-pass 추론(two_pass_predict): 1차 전체 이미지 추론 -> teeth_roi_box_center_aware(11/21
# 정중선을 원점으로 고정한 대칭 crop)로 2차 crop 영역 산출. eval_margin은
# rfdetr/datasets/xray_crop_aug.py의 DEFAULT_TEETH_ROI['eval_margin']과 동일 규칙(치아 bbox
# 합집합 크기 대비 각 변 여유 비율).
DEFAULT_TWO_PASS_ROI_MARGIN = 0.08
# 2차 crop 면적이 원본 면적의 이 비율 이상이면("거의 안 줄어들면") 2차를 생략하고 1차 결과를
# 그대로 쓴다. periapical/peri류 입력은 이미 치아가 프레임을 거의 꽉 채워 crop이 거의 줄지
# 않으므로, 별도 peri 판별(파일명/해상도) 없이 이 기준 하나로 two-pass가 pano에만 적용된다.
DEFAULT_TWO_PASS_SHRINK_RATIO = 0.85
# 2차(crop 후 재추론) confidence threshold. 1차/one-pass와 같은 score 공간이라도 crop으로
# 전체 이미지 context(치아 전체 배치/위아래 기준점)가 사라지면서 confidence 캘리브레이션이
# 달라져 1차의 0.5를 그대로 쓰면 멀쩡히 맞춘 치아가 threshold에 걸려 사라진다 (image_4 사례:
# 정확히 맞춘 31/41번이 0.38/0.42로 탈락). cbct_pano+seoul_pano 82장 PR curve의 F1-최적점이
# 0.338, dice 기준 sweep(rfdetr/deploy/debug_two_pass_center.py)도 0.5->0.3으로 낮출수록
# 단조 개선 -- 두 독립 측정값 사이인 0.35를 기본값으로 쓴다.
DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD = 0.35

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
    # Treat both slash styles as separators so a Windows path is interpreted
    # correctly even when tests/inference run on Linux.
    filename = os.path.basename(os.fspath(file_path).replace('\\', '/')).casefold()
    return any(
        keyword.casefold() in filename
        for keyword in TEMP_EXCLUDED_FILENAME_KEYWORDS
    )

def get_target_image_size(image_shape, rererence_width: int = 640):
    """Scale the longer side to the requested size while preserving ratio."""
    ih, iw = image_shape[:2]
    if ih <= 0 or iw <= 0 or rererence_width <= 0:
        raise ValueError("image dimensions and input size must be positive")
    scale = rererence_width / max(ih, iw)
    return max(1, int(round(ih * scale))), max(1, int(round(iw * scale)))
    

def get_image_size(shape, stride=64):
    h0, w0 = shape
    size = (
        max(stride, int(np.ceil(h0 / stride) * stride)),
        max(stride, int(np.ceil(w0 / stride) * stride)),
    )
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


def load_export_model(model_path, device='cuda', export_shape=EXPORT_FIX_SIZE):
    """export_libtorch로 저장한 libtorch 모델과 고정 입력 크기 (H, W)를 반환.
    입력 크기는 같은 폴더의 *_info.json(export 시 저장)을 우선하고, 없으면 export_shape."""
    import json
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    info_path = os.path.splitext(model_path)[0] + '_info.json'
    if os.path.isfile(info_path):
        with open(info_path, encoding='utf-8') as f:
            export_shape = json.load(f)['input_shape_nchw'][-2:]
    else:
        print(f'[warning] {info_path} not found; using export shape {tuple(export_shape)}')
    return model, tuple(int(v) for v in export_shape)


def main(*,
         use_ema=True, with_source_concat=False,
         test_path='',
         save_dir = 'temp/resultsxray',
         drawing_mode='polygon',
         polygon_thickness=None,
         overlap_report=None,
         max_images=None,
         convert=None,
         polygon_epsilon=2.0,
         input_width=640,
         segmentation_crop_size=(128, 64),
         pretrain_weights='outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth',
         apply_clahe_enhance=False,
         clahe_clip_limit=DEFAULT_PREDICT_CLAHE_CLIP_LIMIT,
         clahe_tile_grid=DEFAULT_PREDICT_CLAHE_TILE_GRID,
         used_export=False,
         export_shape=EXPORT_FIX_SIZE,
         two_pass=False,
         roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN,
         roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO,
         pass2_confidence_threshold=DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD,
         **kwargs):
    """used_export: pretrain_weights를 export된 libtorch(.pt) 모델로 보고 추론 (고정 입력 크기 letterbox)."""
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
    export_resize_hw = None
    if used_export:
        model, export_resize_hw = load_export_model(pretrain_weights, export_shape=export_shape)
        model_parameter = next(model.parameters())
        convert_options = {'device': model_parameter.device, 'dtype': model_parameter.dtype}
        print(f'Loaded exported libtorch model: {pretrain_weights} '
              f'(input {export_resize_hw}, {model_parameter.dtype})')
    else:
        model = init_and_get_model(export=False, use_ema=use_ema,
                                    segmentation_crop_size=tuple(segmentation_crop_size),
                                    pretrain_weights=pretrain_weights)
        convert_options = {}
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
    found = diskmanager.deep_search_files(path, exts=['.jpg', '.jpeg', 'JPG', 'JPEG'])
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
        img = image_utils.cv2_imread(file, flags=cv2.IMREAD_GRAYSCALE)
        if apply_clahe_enhance:
            img = apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)

        try:
            if two_pass and not used_export:
                color_image0, mask_image0, instance_result0, used_two_pass, crop_box = two_pass_predict(
                    img, model, drawing_mode=drawing_mode,
                    mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                    segmentation_crop_box_scale=1.15, with_source_concat=with_source_concat,
                    roi_margin=roi_margin, roi_shrink_ratio=roi_shrink_ratio,
                    pass2_confidence_threshold=pass2_confidence_threshold,
                    convert_options=convert_options,
                )
                print(f'two_pass={used_two_pass} crop_box={crop_box}')
                color_image, mask_iamge = [color_image0], [mask_image0]
                instance_results = [instance_result0]
            else:
                rsz_img, input_content_box = resize_image(img, stride=64, target_width=input_width,
                                                          resize_hw=export_resize_hw)
                img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
                img_tensor = torch_utils.data_convert(img2[None], **convert_options)
                with torch.no_grad():
                    outputs = model(img_tensor)
                    if not isinstance(outputs, dict):
                        # export 모델은 (pred_boxes, pred_logits, pred_masks) tuple을 반환
                        outputs = dict(zip(('pred_boxes', 'pred_logits', 'pred_masks'), outputs))
                    # print(outputs.keys())
                    print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
                    if 'pred_masks' in outputs:
                        print(outputs['pred_masks'].shape)

                    instance_results = [] if overlap_report or convert == 'labelme' else None
                    color_image, mask_iamge = draw_preditions_boxes(
                        img_tensor, outputs, save=False, origin_size=img.shape[:2],
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
                mask_path = f'{stem}.png'
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
            print(e)
            print('draw_preditions_boxes failed')
    if overlap_report:
        write_overlap_report(all_tooth_rows, all_pair_rows, mask_save_dir, processed_count)
    

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
                    f"[warning] EMA weights (ema_model) not found in {checkpoint_path}; "
                    "continuing with regular model weights."
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
def export_libtorch(output_path='e:/temp/model_libtorch.pt', shape=(384, 704), dtype=torch.float16,
                    pretrain_weights='outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth',
                    use_ema=True, segmentation_crop_size=(128, 64)):
    model = init_and_get_model(export=True, use_ema=use_ema,
                               segmentation_crop_size=tuple(segmentation_crop_size),
                               pretrain_weights=pretrain_weights)
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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    traced.save(output_path)
    print(f'saved libtorch model to {output_path}')
    # 배포 쪽에서 입력 크기/정밀도/원본 checkpoint를 확인할 수 있도록 같이 저장
    import json
    checkpoint_keys = torch.load(pretrain_weights, map_location='cpu', weights_only=False).keys()
    info_path = os.path.splitext(output_path)[0] + '_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'pretrain_weights': os.path.abspath(pretrain_weights),
            # init_and_get_model과 동일 규칙: ema_model 키가 없으면 regular(model) 가중치
            'weights': 'ema_model' if use_ema and 'ema_model' in checkpoint_keys else 'model',
            'input_shape_nchw': [1, 3, *shape],
            'dtype': str(dtype).replace('torch.', ''),
            'segmentation_crop_size': list(segmentation_crop_size),
            'outputs': ['pred_boxes', 'pred_logits', 'pred_masks'],
        }, f, indent=2)
    print(f'saved export info to {info_path}')
    
    
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
    # libtorch ='outputs/temp.pt'
    libtorch = 'outputs/export/v1022_overlapfix/model_libtorch.pt'
    
    
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
        


def _predict_single_pass(img, model, convert_options, drawing_mode='polygon',
                          mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                          segmentation_crop_box_scale=1.15, with_source_concat=False,
                          fdi_refinement=True, resize_hw=None, confidence_threshold=0.5):
    """img(전체 또는 crop) 1회 추론 + draw_preditions_boxes. img 자신의 해상도를 기준으로
    (color_image, mask_image, instance_result)를 반환한다 (batch 차원은 이미 벗겨져 있음).
    two_pass_predict()의 1차/2차 호출과 predict_wrapper()의 one-pass 경로가 공유한다."""
    rsz_img, content_box = resize_image(img, stride=64, resize_hw=resize_hw)
    img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
    img_tensor = torch_utils.data_convert(img2[None], **convert_options)

    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, dict):
            pass
        else:
            bboxes, logits, masks = outputs
            outputs = {'pred_boxes': bboxes, 'pred_logits': logits, 'pred_masks': masks}
        print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
        if 'pred_masks' in outputs:
            print(outputs['pred_masks'].shape)

        instance_results = []
        color_image, mask_image = draw_preditions_boxes(
            img_tensor, outputs, save=False, origin_size=img.shape[:2],
            segmentation_mode='crop_and_resize',
            mask_probability_threshold=mask_probability_threshold,
            segmentation_crop_box_scale=segmentation_crop_box_scale,
            with_source_concat=with_source_concat,
            paste_masks_at_original_size=True,
            input_content_box=content_box,
            drawing_mode=drawing_mode,
            instance_results=instance_results,
            fdi_refinement=fdi_refinement,
            confidence_threshold=confidence_threshold,
        )
    return color_image[0], mask_image[0], instance_results[0]


def two_pass_predict(img, model, drawing_mode='polygon',
                      mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                      segmentation_crop_box_scale=1.15, with_source_concat=False,
                      fdi_refinement=True, roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN,
                      roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO, convert_options=None,
                      pass1_confidence_threshold=0.5,
                      pass2_confidence_threshold=DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD):
    """1차: 전체 이미지 추론. 1차에서 검출된 치아 bbox로
    (``rfdetr.datasets.xray_crop_aug.teeth_roi_box_center_aware``, 11/21 정중선을 원점으로 고정한
    대칭 crop -- 학습 때 teeth_roi eval crop의 추론 전용 변형) 2차 crop 영역을 구하고, 그 영역만
    다시 letterbox해 고해상도로 재추론한다 (넓은 panoramic 전체를 640폭으로 욱여넣는 대신 치아
    영역만 640폭으로 키우므로 치아당 유효 입력 해상도가 올라간다).

    **peri 이미지와의 구분(예외처리)**: 이미 periapical/peri류처럼 치아가 프레임 대부분을
    차지하는 입력은 1차 bbox 합집합이 원본과 거의 같아서 crop이 "거의 줄지 않는다". crop 면적이
    원본 면적의 ``roi_shrink_ratio``(기본 0.85) 이상이면 2차를 생략하고 1차 결과를 그대로
    반환한다 -- 파일명/해상도 기반 peri 판별 로직 없이, crop이 실제로 줄어드는지만으로 자연스럽게
    pano에만 2차가 적용된다.

    **pass1/pass2 confidence threshold를 분리한 이유**: crop으로 전체 이미지 context(치아 전체
    배치, 상/하악 판단 기준점)가 사라지면서 2차의 confidence 캘리브레이션이 1차와 달라진다. 같은
    0.5를 2차에도 쓰면 정확히 맞춘 치아가 threshold에 걸려 사라지는 사례가 실측됐다
    (rfdetr/deploy/debug_two_pass_roi.py의 image_4 사례: 31/41번이 0.38/0.42로 정답을 내고도
    탈락). cbct_pano+seoul_pano 82장 PR curve의 F1-최적 threshold도 1차 ~0.42, 2차 ~0.34로
    서로 다르게 나온다 (rfdetr/deploy/pr_curve_two_pass.py).

    Returns:
        (color_image, mask_image, instance_result, used_two_pass, crop_box). ``crop_box``는
        ``(x0, y0, x1, y1)`` 원본 좌표 -- 2차를 실행했을 때만 유효한 crop, 생략됐으면 참고용으로
        계산된(미사용) crop 또는 None.
    """
    h, w = img.shape[:2]
    if convert_options is None:
        model_parameter = next(model.parameters(), None)
        convert_options = ({'device': model_parameter.device, 'dtype': model_parameter.dtype}
                            if model_parameter is not None else {})

    color1, mask1, inst1 = _predict_single_pass(
        img, model, convert_options, drawing_mode=drawing_mode,
        mask_probability_threshold=mask_probability_threshold,
        segmentation_crop_box_scale=segmentation_crop_box_scale,
        with_source_concat=with_source_concat, fdi_refinement=fdi_refinement,
        confidence_threshold=pass1_confidence_threshold)

    if inst1['boxes'].shape[0] == 0:
        return color1, mask1, inst1, False, None

    crop_box = teeth_roi_box_center_aware(inst1['boxes'], inst1['fdi'], (h, w),
                                          cfg={'eval_margin': roi_margin})
    if crop_box is None:
        return color1, mask1, inst1, False, None

    x0, y0, x1, y1 = crop_box
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    crop_w, crop_h = x1i - x0i, y1i - y0i
    if crop_w <= 0 or crop_h <= 0 or crop_w * crop_h >= roi_shrink_ratio * w * h:
        # crop이 원본 대비 거의 줄지 않음 (peri류로 이미 꽉 차 있음) -> 2차 생략, 1-pass로 종료
        return color1, mask1, inst1, False, crop_box

    crop_img = img[y0i:y1i, x0i:x1i]
    _, _, inst2 = _predict_single_pass(
        crop_img, model, convert_options, drawing_mode=drawing_mode,
        mask_probability_threshold=mask_probability_threshold,
        segmentation_crop_box_scale=segmentation_crop_box_scale,
        with_source_concat=False, fdi_refinement=fdi_refinement,
        confidence_threshold=pass2_confidence_threshold)

    # crop-local 좌표 -> 원본 좌표로 평행이동 (각주: 2차 결과가 1차를 완전히 대체한다. 1차에서
    # 검출된 치아는 teeth_roi_box_center_aware 구성상 margin 포함 crop 안에 항상 들어있지만,
    # 2차 자체가 crop 경계에서 치아를 놓치거나 새로 찾는 경우는 보정하지 않는다).
    offset = np.array([x0i, y0i, x0i, y0i], dtype=inst2['boxes'].dtype)
    inst2['boxes'] = inst2['boxes'] + offset
    full_masks = np.zeros((inst2['masks'].shape[0], h, w), dtype=bool)
    full_masks[:, y0i:y1i, x0i:x1i] = inst2['masks']
    inst2['masks'] = full_masks

    from rfdetr.datasets.visualize import render_detection
    from trainer import image_utils, vtk_utils
    colors = vtk_utils.get_teeth_color_table(normalize=False)
    colors[0, :] = 0
    display_image = image_utils.to_magnitude_images(img)
    color2, mask2 = render_detection(
        display_image, inst2['boxes'], inst2['fdi'], colors, masks=inst2['masks'],
        drawing_mode=drawing_mode, with_source_concat=with_source_concat,
    )
    return color2, mask2, inst2, True, crop_box


def predict_wrapper(img, model=None, drawing_mode='polygon',
                    return_instance_results=False,
                    render=True,
                    apply_clahe_enhance=False,
                    clahe_clip_limit=DEFAULT_PREDICT_CLAHE_CLIP_LIMIT,
                    clahe_tile_grid=DEFAULT_PREDICT_CLAHE_TILE_GRID,
                    two_pass=False,
                    roi_margin=DEFAULT_TWO_PASS_ROI_MARGIN,
                    roi_shrink_ratio=DEFAULT_TWO_PASS_SHRINK_RATIO,
                    pass2_confidence_threshold=DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD,
                    return_two_pass_info=False):
    """Run one image while preserving the legacy two-value return by default.

    When ``return_instance_results`` is true, a third value is returned with
    the independent per-tooth masks produced before the single-winner FDI
    label image is composed.  This is the overlap-preserving input used by
    reverse registration; callers that only need ``(drawing, mask)`` are
    unaffected.

    render: when False the polygon/mask overlay is not drawn (saves ~100 ms per 2 MP image); the returned
    drawing is then the plain source image and the label map is None. Use with ``return_instance_results``.

    apply_clahe_enhance: optional lightweight CLAHE contrast enhancement before
    inference (default off -- existing deployed checkpoints were not trained
    with CLAHE'd inputs; see rfdetr/datasets/xraypanoramic.py DEFAULT_CONTRAST_AUG_CFG
    for the matching training-time augmentation).

    two_pass: see ``two_pass_predict`` -- 1차 전체 이미지 추론 후 1차 검출 bbox 기준
    teeth_roi 영역만 고해상도로 재추론한다. peri류처럼 crop이 거의 줄지 않으면 자동으로
    1-pass로 끝난다 (roi_shrink_ratio). export(torch.jit) 모델에는 적용하지 않는다
    (고정 입력 크기라 가변 crop을 다시 letterbox할 수 없음).
    """
    if model is None:
        model = init_and_get_model(export=False, use_ema=True)
    if img.ndim ==3:
        # simple gray image as pickking first channel
        img = img[..., 0]
    if apply_clahe_enhance:
        img = apply_clahe(img, clip_limit=clahe_clip_limit, tile_grid=clahe_tile_grid)

    model_parameter = next(model.parameters(), None)
    convert_options = {}
    if model_parameter is not None:
        convert_options = {
            'device': model_parameter.device,
            'dtype': model_parameter.dtype,
        }

    if two_pass:
        if isinstance(model, torch.jit.ScriptModule):
            print('[warning] two_pass is not supported for exported (torch.jit) models; '
                  'falling back to one-pass.')
        else:
            color_image, mask_image, instance_result, used_two_pass, crop_box = two_pass_predict(
                img, model, drawing_mode=drawing_mode, roi_margin=roi_margin,
                roi_shrink_ratio=roi_shrink_ratio, convert_options=convert_options,
                pass2_confidence_threshold=pass2_confidence_threshold)
            if return_instance_results:
                if return_two_pass_info:
                    return color_image, mask_image, instance_result, dict(
                        used_two_pass=used_two_pass, crop_box=crop_box)
                return color_image, mask_image, instance_result
            return color_image, mask_image

    if isinstance(model, torch.jit.ScriptModule):
        # export model
        rsz_img, input_content_box = resize_image(img, stride=64, resize_hw=EXPORT_FIX_SIZE)
    else:
        rsz_img, input_content_box = resize_image(img, stride=64)
    img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.
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
            render=render,
        )

    if return_instance_results:
        return color_image[0], mask_iamge[0], instance_results[0]
    return color_image[0], mask_iamge[0]



if __name__ == '__main__':

    # inference_libtorch_model_main()
    
    # exit(0)
    parser = argparse.ArgumentParser(description='RF-DETR tooth segmentation inference')
    parser.add_argument(
        '--use-ema',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use ema_model weights from the checkpoint (default: enabled).',
    )
    
    parser.add_argument(
        '--export',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='추론 대신 libtorch(torch.jit.trace) 모델로 export. '
             '--pretrain-weights checkpoint를 {save_dir}/{version}/model_libtorch.pt 로 저장.',
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='export 저장 하위 폴더명 (--export 시 필수). 저장 경로: {save_dir}/{version}/',
    )
    parser.add_argument(
        '--export-shape', '--export_shape',
        dest='export_shape',
        type=int, nargs=2, default=list(EXPORT_FIX_SIZE), metavar=('H', 'W'),
        help=f'export 고정 입력 크기 (H W, 기본 {EXPORT_FIX_SIZE[0]} {EXPORT_FIX_SIZE[1]}). '
             'predict_wrapper가 jit 모델에 쓰는 EXPORT_FIX_SIZE와 같아야 한다.',
    )
    parser.add_argument(
        '--used-export', '--used_export',
        dest='used_export',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='--pretrain-weights를 export된 libtorch 모델(model_libtorch.pt)로 보고 추론. '
             '입력 크기는 *_info.json(없으면 --export-shape), 정밀도는 모델 dtype을 따른다.',
    )
    parser.add_argument(
        '--export-dtype', '--export_dtype',
        dest='export_dtype',
        type=str, default='float16', choices=['float16', 'float32'],
        help='export 모델 정밀도 (기본 float16).',
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
        default='polygon',
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
        '--input-size', '--input_size', '--input-width', '--input_width',
        dest='input_width',
        type=int,
        default=640,
        help='이미지 긴 변을 해당 크기로 종횡비 유지 resize한 뒤 64 배수로 padding. '
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
        '--contrast-enhance', '--contrast_enhance',
        '--clahe', '--apply-clahe', '--apply_clahe',
        dest='apply_clahe_enhance',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='추론 전 가볍고 빠른 CLAHE 명암 강화 적용 (기본 비활성). 저해상도/고대비 치근단 방사선처럼 '
             '기존 pano 데이터와 명암 분포가 많이 다른 입력에 도움이 될 수 있다 -- 단, 기존 배포 '
             '체크포인트는 CLAHE 입력 분포를 학습하지 않았으므로 기본값은 꺼둔다 (배포 모델 호환성). '
             'rfdetr/datasets/xraypanoramic.py의 contrast augmentation으로 재학습한 체크포인트에서만 켜서 쓸 것.',
    )
    parser.add_argument(
        '--clahe-clip-limit', '--clahe_clip_limit',
        dest='clahe_clip_limit',
        type=float,
        default=DEFAULT_PREDICT_CLAHE_CLIP_LIMIT,
        help=f'--clahe 사용 시 CLAHE clipLimit (기본 {DEFAULT_PREDICT_CLAHE_CLIP_LIMIT}).',
    )
    parser.add_argument(
        '--clahe-tile-grid', '--clahe_tile_grid',
        dest='clahe_tile_grid',
        type=int,
        default=DEFAULT_PREDICT_CLAHE_TILE_GRID,
        help=f'--clahe 사용 시 CLAHE tileGridSize (정사각형, 기본 {DEFAULT_PREDICT_CLAHE_TILE_GRID}).',
    )
    parser.add_argument(
        '--two-pass', '--two_pass',
        dest='two_pass',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='1차 전체 이미지 추론 후, 1차 검출 치아 bbox 기준 teeth_roi 영역만 고해상도로 '
             '재추론 (two_pass_predict 참고). periapical/peri류처럼 치아가 이미 프레임 대부분을 '
             '차지하는 입력은 crop이 거의 줄지 않아 --roi-shrink-ratio 조건에 의해 자동으로 '
             '1-pass로 끝난다 (export 모델에는 적용되지 않음).',
    )
    parser.add_argument(
        '--roi-margin', '--roi_margin',
        dest='roi_margin',
        type=float,
        default=DEFAULT_TWO_PASS_ROI_MARGIN,
        help=f'--two-pass 2차 crop 여유 비율, 치아 bbox 합집합 크기 대비 (기본 {DEFAULT_TWO_PASS_ROI_MARGIN}).',
    )
    parser.add_argument(
        '--roi-shrink-ratio', '--roi_shrink_ratio',
        dest='roi_shrink_ratio',
        type=float,
        default=DEFAULT_TWO_PASS_SHRINK_RATIO,
        help='--two-pass 2차 crop 면적이 원본 면적 대비 이 비율 이상이면("거의 안 줄어들면") '
             f'2차를 생략하고 1-pass로 끝낸다 (기본 {DEFAULT_TWO_PASS_SHRINK_RATIO}).',
    )
    parser.add_argument(
        '--pass2-confidence-threshold', '--pass2_confidence_threshold',
        dest='pass2_confidence_threshold',
        type=float,
        default=DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD,
        help='--two-pass 2차(crop 후 재추론) confidence threshold. crop으로 context가 사라지며 '
             f'1차(0.5 고정)보다 캘리브레이션이 달라져 기본값을 낮게 둠 (기본 '
             f'{DEFAULT_TWO_PASS_PASS2_CONFIDENCE_THRESHOLD}, PR curve F1-최적점 참고).',
    )
    cli_args = parser.parse_args()
    if cli_args.export:
        if not cli_args.version:
            parser.error('--export requires --version (output: {save_dir}/{version}/)')
        export_libtorch(
            os.path.join(cli_args.save_dir, cli_args.version, 'model_libtorch.pt'),
            shape=tuple(cli_args.export_shape),
            dtype=getattr(torch, cli_args.export_dtype),
            pretrain_weights=cli_args.pretrain_weights,
            use_ema=cli_args.use_ema,
            segmentation_crop_size=cli_args.segmentation_crop_size,
        )
    else:
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

