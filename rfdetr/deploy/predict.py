import os
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

from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRSegPreviewConfig,
    TrainConfig,
    SegmentationTrainConfig,
    ModelConfig
)

def main():
    # print("git:\n  {}\n".format(utils.get_sha()))
    # print(args)
    # # convert device to device_id
    # if args.device == 'cuda':
    #     device_id = "0"
    # elif args.device == 'cpu':
    #     device_id = ""
    # else:
    #     device_id = str(int(args.device))
    
    from rfdetr.main import populate_args
    #     args.device = f"cuda:{device_id}"
        
    # rf_detr = RFDETRSmall(
    
    #     patch_size=16,
    #     num_windows=4,
    #     # num_queries=100,
    #     num_queries=100,
    #     group_detr=5,
    #     num_select=30,
    #     encoder='dinov2_windowed_tiny',
    #     # encoder='dinov2_windowed_base',
        
    #     # patch_size=24,
    #     # num_channels=1,
    #     # eval=True,
    #     # num_classes=32,
    #     num_classes=num_classes,
    #     segmentation_head=args.segmentation_head,
    #     # pretrain_weights="output/checkpoint0099.pth",
    #     # pretrain_weights="output/xray_teeth/checkpoint0059.pth",
    #     # pretrain_weights='output/xray_teeth33/checkpoint0039.pth'
    #     # pretrain_weights='output/xray_teeth33/checkpoint_best_regular.pth'
    #     # pretrain_weights='output/xray_teeth33_nano/checkpoint_best_total.pth'
    #     # pretrain_weights='output/xray_teeth33_small/checkpoint_best_regular.pth'
    #     # pretrain_weights= args.pretrain_weights # 'output/xray_teeth33_small_seg/checkpoint0039.pth',
    #     # pretrain_weights='output/xray_teeth33_dinov2tiny_small/checkpoint0039.pth',
    #     pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
        
    # )
        
    config = RFDETRBaseConfig(
        patch_size=16,
        num_windows=4,
        # num_queries=100,
        num_queries=100,
        group_detr=5,
        num_select=30,
        encoder='dinov2_windowed_tiny',
        num_classes=32,
        segmentation_head=True,
        pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
    )
    
    args = populate_args(**dict(config))
    

    # # device for export onnx
    # # TODO: export onnx with cuda failed with onnx error
    # device = torch.device("cpu")
    # os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    
    model = build_model(args)
    checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
    res = model.load_state_dict(checkpoint['model'], strict=False)
    
    print(model)
main()
