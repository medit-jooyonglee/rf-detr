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
import onnxsim
import numpy as np
from PIL import Image

import rfdetr.util.misc as utils
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

def main():
    from trainer import diskmanager, image_utils

    rf_detr = RFDETRSmall(
        
        patch_size=16,
        num_windows=4,
        # num_queries=100,
        num_queries=100,
        group_detr=5,
        num_select=30,
        encoder='dinov2_windowed_tiny',
        # encoder='dinov2_windowed_base',
        
        # patch_size=24,
        # num_channels=1,
        # eval=True,
        # num_classes=32,
        num_classes=32,
        segmentation_head=True,

        # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
        pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0099.pth'
        
    )
    
    model = rf_detr.model.model
    
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
        # size = get_image_size(image.shape[:2], stride=stride)
        size = get_target_image_size(image.shape[:2], rererence_width=target_width)
        size = get_image_size(size, stride=stride)
        # return image.resize(size, Image.BILINEAR)
        return cv2.resize(image, tuple(size[::-1]), interpolation=cv2.INTER_LINEAR)

        # ''
        
        # next(rf_detr.model.model.parameters()).device
    model = rf_detr.model.model
    model.eval()
        # "data/xray_teeth33/test/images/000000.png",]
    from trainer import torch_utils
    # path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle/Teeth Segmentation JSON/d2/img/'
    path = 'E:/dataset/reverse_tomosynthesis/cbct_ios_dcm'
    
    found = diskmanager.deep_search_files(path, exts=['.jpg', '.png', '.jpeg'])
    # found = glob.glob(f'{path}/*.jpg')
    i_break = 30
    for idx, file in enumerate(found):
        # if idx > i_break:
            # break
        # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = image_utils.cv2_imread(file, flags=cv2.IMREAD_GRAYSCALE)
        rsz_img = resize_image(img, stride=64)
        img2 = np.repeat(rsz_img[None], 3, axis=0) / 255.

        img_tensor = torch_utils.data_convert(img2[None])
        
        with torch.no_grad():
            outputs = model(img_tensor)
            print(outputs.keys())
            print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
            if 'pred_masks' in outputs:
                print(outputs['pred_masks'].shape)
        
            try:
                draw_preditions_boxes(
                    img_tensor, outputs, save=True, save_dir=f'results/test/',
                )
            except Exception as e:
                print(e)
                print('draw_preditions_boxes failed')
    # model = build_model(args)
    # checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
    # res = model.load_state_dict(checkpoint['model'], strict=False)
    
    print(model)
main()
