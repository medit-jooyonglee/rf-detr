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
                draw_preditions_boxes(samples, outputs)

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


def draw_preditions_boxes(new_samples, outputs, save=False, save_dir='results'):
    from trainer import torch_utils
    from rfdetr.datasets.teeth import draw_bboxes
    from trainer import utils_numpy, image_utils, time_strftime, vtk_utils
    from rfdetr.datasets.xraypanoramic import label_to_fdi
    import cv2
    from rfdetr.util import box_ops
    
    if hasattr(new_samples, 'tensors'):
        inputs_arrays  = torch_utils.to_numpy(new_samples.tensors)
    elif isinstance(new_samples, torch.Tensor):
        inputs_arrays  = torch_utils.to_numpy(new_samples)
    boxes = torch_utils.to_numpy(outputs['pred_boxes'])
    probs = outputs['pred_logits'].sigmoid()
    probs = torch_utils.to_numpy(probs.to(torch.float32))
    # logits = torch_utils.to_numpy(outputs['pred_logits'].to(torch.float32))
    pred_scores = np.max(probs, axis=-1)
    pred_label = np.argmax(probs, axis=-1)
    label = fdi = label_to_fdi(pred_label)
    sort_pred_args = np.argsort(pred_scores, axis=-1)[:, ::-1]
    # top_k = 40

    
    pred_masks = outputs.get('pred_masks')
    if pred_masks is not None:
        pred_masks = F.interpolate(pred_masks, size=inputs_arrays.shape[-2:], mode='bilinear', align_corners=False)
        pred_masks_label = torch_utils.to_numpy(pred_masks > 0)
    else:
        pred_masks_label = None
    
    # boxes = np.squeeze(boxes)
    # boxes = boxes[label > 0]
    num_batch = inputs_arrays.shape[0]
    # np.squeeze(np.argmax(logtis, axis=-1))
    width, height = inputs_arrays.shape[-2:][::-1]
    images = np.transpose(inputs_arrays, [0, 2, 3, 1])
    
    os.makedirs('results', exist_ok=True)
    def denorm_boxes_to_xyxy(boxes, img_w, img_h):
        x_c, y_c, w, h = np.split(boxes, 4, axis=1)
        x0 = (x_c - 0.5 * w) * img_w
        y0 = (y_c - 0.5 * h) * img_h
        x1 = (x_c + 0.5 * w) * img_w
        y1 = (y_c + 0.5 * h) * img_h
        return np.concatenate([x0, y0, x1, y1], axis=1)
    
    
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
    for i in range(num_batch):
        
        image = image_utils.to_magnitude_images(images[i])
        
        # posit = np.logical_and(label[i] > 0, label[i] < num_classes)
        posit = pred_scores[i] > threshold
        
        print("posit sum: ", np.sum(posit), '/', probs[i].shape[0])
        
        
        posit_boxes = boxes[i][posit]
        posit_labels = label[i][posit]
        boxes_xy = denorm_boxes_to_xyxy(posit_boxes, width, height)
        
        
        
        # t_boxes_xy = torch_utils.data_convert(boxes_xy)
        # iou, _ = box_ops.box_iou(t_boxes_xy, t_boxes_xy)
        draw_bboxes(image, boxes_xy, colors=colors[posit_labels])
        if pred_masks_label is not None:
            posit_masks = pred_masks_label[i][posit]
            label_image = posit_labels[:, None, None] * posit_masks
            label_image = np.max(label_image, axis=0)
            restore_label_image = cv2.resize(label_image, (width, height), interpolation=cv2.INTER_NEAREST)
            color_label_image = colors[restore_label_image]
            # image_utils.
            image = utils_numpy.apply_blending_mask(image, color_label_image)
        res_images.append(image)

        if save:
            os.makedirs(save_dir, exist_ok=True)
            save_name = f'{save_dir}/bounding_draw_{time_strftime()}.png'        
            cv2.imwrite(save_name, image.astype(np.uint8)[..., ::-1])
            print("Saved image with bounding boxes to: ", save_name)
    return res_images
        