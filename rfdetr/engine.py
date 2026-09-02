# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import random
import time
import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.coco import compute_multi_scale_scales

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from rfdetr.util.misc import NestedTensor
import numpy as np

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.inference_mode():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()
                if args.segmentation_head:
                    for t in targets:
                        if "masks" in t:
                            t["masks"] = F.interpolate(
                                t["masks"].unsqueeze(1).float(), size=scale, mode='nearest'
                            ).squeeze(1) > 0.5
        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )


            scaler.scale(losses).backward()
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    iou50_idx, area_idx, maxdet_idx = (
        int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j   = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall    = float(rec_thrs[best_j])
    macro_f1        = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid   = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
        ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
        rc = macro_recall

        #Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : pc,
            "recall"     : rc,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall
    }

def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    
    coco_evaluator = CocoEvaluator(base_ds, iou_types) # if getattr(args, 'coco_evaluate', True) else None
    # coco_evaluator = CocoEvaluator(base_ds, iou_types) if getattr(args, 'coco_evaluate', True) else None

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)
            
            if getattr(args, 'eval_save', False):
                draw_preditions_boxes(
                    samples,
                    outputs,
                    save=True,
                    segmentation_mode=getattr(
                        args, 'segmentation_mode', 'full_image'
                    ),
                    segmentation_crop_box_scale=getattr(
                        args, 'segmentation_crop_box_scale', 1.15
                    ),
                )

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results_all)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator

def non_max_suppression(boxes, scores, threshold):
    """
    NumPy를 이용한 NMS (Non-Maximum Suppression) 구현
    
    Parameters:
    - boxes: (N, 4) 형태의 바운딩 박스 배열 ([xmin, ymin, xmax, ymax])
    - scores: (N,) 형태의 각 박스별 신뢰도 점수
    - threshold: IoU 임계값 (이 값보다 크면 겹치는 박스로 판단하여 제거)
    
    Returns:
    - keep: NMS를 통과한 살아남은 박스들의 인덱스 배열
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # 좌표 추출
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 각 박스의 넓이 계산
    areas = (x2 - x1) * (y2 - y1)
    
    # 점수가 높은 순서대로 정렬 (내림차순)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 가장 점수가 높은 박스의 인덱스 선택
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 남은 박스들과 선택된 박스 간의 교차 영역(Intersection) 좌표 계산
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 교차 영역의 너비와 높이 계산 (음수일 경우 0으로 처리)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # IoU (Intersection over Union) 계산
        # IoU = 교차 영역 / (박스 i의 넓이 + 남은 박스들의 넓이 - 교차 영역)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # IoU가 임계값(threshold)보다 작은 박스들만 남김
        inds = np.where(iou <= threshold)[0]
        
        # order 배열 업데이트 (다음 반복을 위해)
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def draw_preditions_boxes(
    new_samples,
    outputs,
    save=False,
    save_dir='results',
    nms_refinement=True,
    fname='', origin_size=None,
    segmentation_mode='full_image',
    mask_probability_threshold=0.5,
    segmentation_crop_box_scale=1.15,
    with_source_concat=False,
    paste_masks_at_original_size=False,
    input_content_box=None,
):
    """Draw predictions, optionally restoring the rendered output to its original size.

    Args:
        origin_size: Original image size as ``(height, width)``. When provided,
            the input image, bounding boxes, and segmentation mask are restored
            from the model input size to this size before rendering.
        paste_masks_at_original_size: In ``crop_and_resize`` mode, paste ROI mask
            logits directly onto an ``origin_size`` canvas before thresholding.
            This avoids resizing an already-thresholded label map.
        input_content_box: Optional valid-image region ``(x0, y0, x1, y1)`` in
            model-input pixels. Use this when the input was letterboxed so boxes,
            masks, and the rendered image are restored without including padding.
    """
    from trainer import torch_utils
    from rfdetr.datasets.teeth import draw_bboxes
    from trainer import utils_numpy, image_utils, time_strftime, vtk_utils
    from rfdetr.datasets.xraypanoramic import label_to_fdi
    import cv2
    from rfdetr.util import box_ops
    def denorm_boxes_to_xyxy(boxes, img_w, img_h):
        shape = boxes.shape
        x_c, y_c, w, h = np.split(boxes.reshape([-1, 4]), 4, axis=1)
        x0 = (x_c - 0.5 * w) * img_w
        y0 = (y_c - 0.5 * h) * img_h
        x1 = (x_c + 0.5 * w) * img_w
        y1 = (y_c + 0.5 * h) * img_h
        return np.concatenate([x0, y0, x1, y1], axis=1).reshape(shape)
    
    
    if hasattr(new_samples, 'tensors'):
        inputs_arrays  = torch_utils.to_numpy(new_samples.tensors)
    elif isinstance(new_samples, torch.Tensor):
        inputs_arrays  = torch_utils.to_numpy(new_samples)
    normalized_boxes = outputs['pred_boxes']
    boxes = torch_utils.to_numpy(normalized_boxes)
    mask_boxes = boxes
    if segmentation_mode == 'crop_and_resize':
        from rfdetr.models.segmentation_head import expand_normalized_boxes
        mask_boxes = torch_utils.to_numpy(
            expand_normalized_boxes(
                normalized_boxes,
                segmentation_crop_box_scale,
            )
        )
    probs = outputs['pred_logits'].sigmoid()
    probs = torch_utils.to_numpy(probs.to(torch.float32))
    
    width, height = inputs_arrays.shape[-2:][::-1]
    render_height, render_width = height, width
    if origin_size is not None:
        if len(origin_size) != 2:
            raise ValueError("origin_size must be a (height, width) pair")
        render_height, render_width = (int(value) for value in origin_size)
        if render_height <= 0 or render_width <= 0:
            raise ValueError("origin_size values must be positive")
    if input_content_box is not None:
        if origin_size is None:
            raise ValueError("input_content_box requires origin_size")
        if len(input_content_box) != 4:
            raise ValueError("input_content_box must be an (x0, y0, x1, y1) tuple")
        content_x0, content_y0, content_x1, content_y1 = (
            float(value) for value in input_content_box
        )
        if not (
            0 <= content_x0 < content_x1 <= width
            and 0 <= content_y0 < content_y1 <= height
        ):
            raise ValueError("input_content_box must lie inside the model input")

    def restore_boxes(boxes_xyxy):
        restored = boxes_xyxy.copy()
        if input_content_box is None:
            restored[..., [0, 2]] *= render_width / width
            restored[..., [1, 3]] *= render_height / height
        else:
            restored[..., [0, 2]] = (
                restored[..., [0, 2]] - content_x0
            ) * (render_width / (content_x1 - content_x0))
            restored[..., [1, 3]] = (
                restored[..., [1, 3]] - content_y0
            ) * (render_height / (content_y1 - content_y0))
        return restored
    # posit = pred_scores[i] > threshold
    # (batch, num_queries, 4) 
    boxes = denorm_boxes_to_xyxy(boxes, width, height)
    mask_boxes = denorm_boxes_to_xyxy(mask_boxes, width, height)
    # logits = torch_utils.to_numpy(outputs['pred_logits'].to(torch.float32))
    pred_scores = np.max(probs, axis=-1)
    pred_label = np.argmax(probs, axis=-1)

    confidence_threshold = 0.5
    selected_query_indices = None
    if nms_refinement:
        selected_query_indices = []
        nms_bboxes = []
        nms_labels = []
        nms_scores = []
        nms_mask_bboxes = []

        for ib in range(boxes.shape[0]):
            posit = (pred_scores[ib] > confidence_threshold) & (pred_label[ib] > 0)
            keep_indices = non_max_suppression(
                boxes[ib][posit], pred_scores[ib][posit], threshold=0.4
            )
            keep_indices = np.where(posit)[0][keep_indices]
            selected_query_indices.append(keep_indices)
            print(f'batch {ib}: nms {posit.sum()}--->{len(keep_indices)}')
            nms_bboxes.append(boxes[ib][keep_indices])
            nms_mask_bboxes.append(mask_boxes[ib][keep_indices])
            nms_labels.append(pred_label[ib][keep_indices])
            nms_scores.append(pred_scores[ib][keep_indices])

        pred_label = nms_labels
        pred_scores = nms_scores
        boxes = nms_bboxes
        mask_boxes = nms_mask_bboxes
    
    pred_masks = outputs.get('pred_masks')
    confidence_threshold = 0.5
    
    if pred_masks is not None:
        if not 0.0 < mask_probability_threshold < 1.0:
            raise ValueError("mask_probability_threshold must be between 0 and 1.")
        mask_logit_threshold = math.log(
            mask_probability_threshold / (1.0 - mask_probability_threshold)
        )
        if segmentation_mode == 'crop_and_resize':
            from rfdetr.models.segmentation_head import paste_masks_in_image

            # ROI masks are expensive to reconstruct at full image resolution.
            # Select positive, NMS-surviving queries before pasting so discarded
            # background/duplicate queries never allocate a full-image canvas.
            pred_masks_label = []
            for batch_index in range(pred_masks.shape[0]):
                batch_masks = pred_masks[batch_index]
                if selected_query_indices is not None:
                    query_indices = torch.as_tensor(
                        selected_query_indices[batch_index],
                        device=pred_masks.device,
                        dtype=torch.long,
                    )
                    batch_masks = batch_masks.index_select(0, query_indices)

                paste_boxes = mask_boxes[batch_index]
                paste_size = (height, width)
                if paste_masks_at_original_size and origin_size is not None:
                    paste_boxes = restore_boxes(paste_boxes)
                    paste_size = (render_height, render_width)
                pasted_masks = paste_masks_in_image(
                    batch_masks,
                    torch.as_tensor(paste_boxes, device=pred_masks.device),
                    paste_size,
                )
                pred_masks_label.append(
                    torch_utils.to_numpy(pasted_masks > mask_logit_threshold)
                )
        else:
            pred_masks = F.interpolate(
                pred_masks,
                size=inputs_arrays.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            pred_masks_label = torch_utils.to_numpy(
                pred_masks > mask_logit_threshold
            )
            if selected_query_indices is not None:
                pred_masks_label = [
                    pred_masks_label[ib][indices]
                    for ib, indices in enumerate(selected_query_indices)
                ]
    else:
        pred_masks_label = None

    if isinstance(pred_label, list):
        label = []
        fdi = []
        for pl in pred_label:
            lb = fd = label_to_fdi(pl)
            label.append(lb)
            fdi.append(fd)
    else:

        label = fdi = label_to_fdi(pred_label)
    # sort_pred_args = np.argsort(pred_scores, axis=-1)[:, ::-1]
    
    
    # boxes = boxes[label > 0]
    num_batch = inputs_arrays.shape[0]
    # np.squeeze(np.argmax(logtis, axis=-1))
    
    images = np.transpose(inputs_arrays, [0, 2, 3, 1])
    
    os.makedirs('results', exist_ok=True)

    
    # 32 num_classes
    
    colors = vtk_utils.get_teeth_color_table(normalize=False)
    
    colors[0, :] = 0
    
    # debug_num_clolrs = 4
    debug_4num_clolrs = False
    if debug_4num_clolrs:
        # for debugging 10 / 20 / 30 40
        colors[10:20, :] = np.array([255, 0, 0])
        colors[20:30, :] = np.array([0, 255, 0])
        colors[30:40, :] = np.array([0, 0, 255])
        colors[40:50, :] = np.array([255, 255, 0])
    # colors = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [128, 128, 128], [128, 0, 0], [0, 128, 0]])
    # num_classes = 10
    threshold = 0.5
    res_images = []
    mask_images = []
    for i in range(num_batch):
        
        image = image_utils.to_magnitude_images(images[i])
        if (
            input_content_box is not None
            or (render_height, render_width) != (height, width)
        ):
            if input_content_box is not None:
                image = image[
                    int(content_y0):int(content_y1),
                    int(content_x0):int(content_x1),
                ]
            image = cv2.resize(
                image,
                (render_width, render_height),
                interpolation=cv2.INTER_LINEAR,
            )
        
        # posit = np.logical_and(label[i] > 0, label[i] < num_classes)
        # posit = pred_scores[i] > threshold
        
        # print("posit sum: ", np.sum(posit), '/', probs[i].shape[0])
        
        
        posit_boxes = boxes[i]
        posit_labels = label[i]
        boxes_xy = posit_boxes.copy()
        if (
            input_content_box is not None
            or (render_height, render_width) != (height, width)
        ):
            boxes_xy = restore_boxes(boxes_xy)
        # boxes_xy = 
        
        src_image = image.copy()
        
        # t_boxes_xy = torch_utils.data_convert(boxes_xy)
        # iou, _ = box_ops.box_iou(t_boxes_xy, t_boxes_xy)
        draw_bboxes(image, boxes_xy, colors=colors[posit_labels])
        if pred_masks_label is not None and len(pred_masks_label) > 0:
            posit_masks = pred_masks_label[i]
            if len(posit_masks) == 0:
                label_image = np.zeros(posit_masks.shape[-2:], dtype=np.int64)
            else:
                label_image = posit_labels[:, None, None] * posit_masks
                label_image = np.max(label_image, axis=0)
            mask_is_already_restored = (
                segmentation_mode == 'crop_and_resize'
                and paste_masks_at_original_size
                and origin_size is not None
            )
            if (
                label_image.shape != (render_height, render_width)
                or (input_content_box is not None and not mask_is_already_restored)
            ):
                if input_content_box is not None:
                    label_image = label_image[
                        int(content_y0):int(content_y1),
                        int(content_x0):int(content_x1),
                    ]
                restore_label_image = cv2.resize(
                    label_image,
                    (render_width, render_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                restore_label_image = label_image
            mask_images.append(restore_label_image)
            color_label_image = colors[restore_label_image]
            # image_utils.
            image = utils_numpy.apply_blending_mask(image, color_label_image)
            if with_source_concat:
                image = np.concatenate([src_image, image], axis=1)
                
        res_images.append(image)

        if save:
            # save_dir = 
            save_dir = os.path.join(save_dir, time.strftime('%Y%m%d'))
            if fname:
                save_name = os.path.join(save_dir, fname)
            else:
                save_name = f'{save_dir}/bounding_draw_{time_strftime()}.png'            
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            
            # cv2.imwrite(save_name, image.astype(np.uint8)[..., ::-1])
            image_utils.cv2_imwrite(save_name, image.astype(np.uint8)[..., ::-1])
            print("Saved image with bounding boxes to: ", save_name)
    return res_images, mask_images
