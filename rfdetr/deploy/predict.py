import os
import glob
import cv2
import ast
import random
import argparse
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
# from rfdetr.deploy._onnx import OnnxOptimizer
import re
import sys
from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall

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

def main(use_ema=True, with_source_concat=False):
    # FIXME: save_path
    from trainer import diskmanager, image_utils
    save_dir = f'results/test/0828'
    model = init_and_get_model(export=False, use_ema=use_ema)
    
    # model = rf_detr.model.model
    
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
    
    def resize_image(image, stride=64, target_width=640):
        """Match ResizeCocoSample's keep-ratio letterbox used for CBCT training."""
        size = get_target_image_size(image.shape[:2], rererence_width=target_width)
        size = get_image_size(size, stride=stride)
        target_h, target_w = size
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

        # ''
        
        # next(rf_detr.model.model.parameters()).device
    # model = rf_detr.model.model
    # model.eval()
    from trainer import torch_utils
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle/Teeth Segmentation JSON/d2/img/'
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_teeth_seg_kaggle/Teeth Segmentation PNG/d2/img/'
    path = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm'
    # path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON'
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm/'
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_0824/JPEGImages/'
    # path = 
    
    mask_save_dir = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm_mask_results_0828_testestt222_merge222_prob'
    found = diskmanager.deep_search_files(path, exts=['.jpg', '.jpeg'])
    # found = glob.glob(f'{path}/*.jpg')
    i_break = 30
    for idx, file in enumerate(found):
        # if idx > i_break:
            # break
        # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = image_utils.cv2_imread(file, flags=cv2.IMREAD_GRAYSCALE)
        rsz_img, input_content_box = resize_image(img, stride=64)
        img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.

        img_tensor = torch_utils.data_convert(img2[None])
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # print(outputs.keys())
            print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
            if 'pred_masks' in outputs:
                print(outputs['pred_masks'].shape)
        
            try:
                color_image, mask_iamge = draw_preditions_boxes(
                    img_tensor, outputs, save=False, origin_size=img.shape[:2],
                                        segmentation_mode='crop_and_resize',
                    mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
                    segmentation_crop_box_scale=1.15,
                    with_source_concat=with_source_concat,
                    paste_masks_at_original_size=True,
                    input_content_box=input_content_box,
                )
                
                
                assert len(mask_iamge) == 1, "Expected a single mask image."
                savename = os.path.join(mask_save_dir, os.path.relpath(file, path))
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                # cv2.imwrite(savename, mask_iamge[0].astype(np.uint8))
                from trainer import image_utils
                image_utils.cv2_imwrite(savename.replace('.jpg', '.png'), mask_iamge[0].astype(np.uint8))
                image_utils.cv2_imwrite(savename, color_image[0].astype(np.uint8)[..., ::-1])  # Convert RGB to BGR for saving with OpenCV
                print(f"Saved mask image to: {savename.replace('.jpg', '.png')}")

            except Exception as e:
                print(e)
                print('draw_preditions_boxes failed')
    # model = build_model(args)
    # checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
    # res = model.load_state_dict(checkpoint['model'], strict=False)
    
    print(model)
    

def init_and_get_model(config=None, device='cuda', export=False, use_ema=True):
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
            segmentation_crop_size=(128, 64),
            coarse_hint_scale=0.35,
            segmentation_crop_box_scale=1.15,
            coarse_hint_dropout=0.30,

            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
            # pretrain_weights='e:/temp/checkpoint.pth',
            
            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg_0819_all/checkpoint.pth',
            # # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg_0819_crop_retrain/checkpoint.pth',
            # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg_crop_regularization/checkpoint.pth',
            # pretrain_weights='output/bounding_bbox/checkpoint.pth',
            pretrain_weights='output/bounding_bbox_regularization/checkpoint.pth',
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
                # return bboxes, logits, masks
                probs = logits.sigmoid()
                bboxes, probs, masks = [torch.squeeze(v) for v in (bboxes, probs, masks)]
                label = probs.argmax(dim=-1)
                # posit = label > 0
                res = [bboxes, probs, masks, label]
                return res
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
    
# def export_libtorch():
#     pass
#     model = init_and_get_model({})
#     x0 = torch.randn(1, 3, 64*5, 64*10).cuda()
#     res = torch.jit.trace(
#         model,)
#         # torch.randn(1, 3, 224, 224).cuda()
    
    
    
    # model.to()
def export_libtorch(output_path='e:/temp/model_libtorch.pt', shape=(384, 704)):
    model = init_and_get_model({})
    model.eval()
    model.requires_grad_(False)
    # dtype = torch.float16
    dtype = torch.float16
    # model.half()
    model.to(dtype)
    # model
    # model.export()  # swap in forward_export + baked position embeddings (required before tracing)

    dummy = torch.randn(1, 3, *shape).cuda().half() #.cuda()
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
        shape = img.shape[:2]
        if np.all([np.all(shape > np.array(size0)) for size0 in filter_image_size]):
            pass
        else:
            print(f"Skipping file {file} due to size {shape}.")
            continue
        # size = get_size(img.shape[:2], refenrece_width=640, stride=64)
        size = (384, 704)
        origin_size = img.shape[:2]
        rsz_img = cv2.resize(img, tuple(size[::-1]), interpolation=cv2.INTER_LINEAR)
        rsz_img = np.repeat(rsz_img[None], 3, axis=0).astype(np.float32) / 255.
        tensor = torch_utils.data_convert(rsz_img[None], dtype=dtype)
        try:
            with torch.no_grad():
                res = model(tensor)
        except Exception as e:
            print(e)
            print('model inference failed')
            continue
        res = [torch.unsqueeze(v, 0) for v in res]
        output_keys = [
                    'pred_boxes',
                    'pred_logits',
                    'pred_masks',
                    'pred_labels'
                ]
        res_dicts = dict(zip(output_keys, res))
        color_image, mask_iamge = draw_preditions_boxes(
            tensor,
            res_dicts,
            save=save,
            save_dir='outputs/export/',
            fname=os.path.relpath(file, path),
            origin_size=origin_size,
            segmentation_mode='crop_and_resize',
            mask_probability_threshold=MASK_PROBABILITY_THRESHOLD,
            segmentation_crop_box_scale=1.15,
        )
        assert len(mask_iamge) == 1, "Expected a single mask image."
        savename = os.path.join(mask_save_dir, os.path.relpath(file, path))
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        # cv2.imwrite(savename, mask_iamge[0].astype(np.uint8))
        from trainer import image_utils
        image_utils.cv2_imwrite(savename.replace('.jpg', '.png'), mask_iamge[0].astype(np.uint8))
        image_utils.cv2_imwrite(savename, color_image[0].astype(np.uint8)[..., ::-1])  # Convert RGB to BGR for saving with OpenCV
        print(f"Saved mask image to: {savename.replace('.jpg', '.png')}")

    
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
    cli_args = parser.parse_args()
    main(use_ema=cli_args.use_ema,
         with_source_concat=cli_args.with_source_concat)
    # export_libtorch('outputs/temp.pt')
    # coreml_export_main()
    # test_coreml_inference()
    # inference_libtorch_model_main()
#
