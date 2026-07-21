# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
# from rf100vl import get_rf100vl_projects
import roboflow
from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall
import torch
import os

def download_dataset(rf_project: roboflow.Project, dataset_version: int):
    versions = rf_project.versions()
    if dataset_version is not None:
        versions = [v for v in versions if v.version == str(dataset_version)]
        if len(versions) == 0:
            raise ValueError(f"Dataset version {dataset_version} not found")
        version = versions[0]
    else:
        version = max(versions, key=lambda v: v.id)
    location = os.path.join("datasets/", rf_project.name + "_v" + version.version)
    if not os.path.exists(location):
        location = version.download(
            model_format="coco", location=location, overwrite=False
        ).location
    
    return location


def train_from_rf_project(rf_project: roboflow.Project, dataset_version: int):
    location = download_dataset(rf_project, dataset_version)
    print(location)
    rf_detr = RFDETRBase()
    device_supports_cuda = torch.cuda.is_available()
    rf_detr.train(
        dataset_dir=location,
        epochs=1,
        device="cuda" if device_supports_cuda else "cpu",
    )


def train_from_coco_dir(coco_dir: str):
    rf_detr = RFDETRBase()
    device_supports_cuda = torch.cuda.is_available()
    
    rf_detr.train(
        dataset_dir=coco_dir,
        epochs=300,
        device="cuda" if device_supports_cuda else "cpu",
        dataset_file='coco',
        coco_path=coco_dir,
        batch_size=1,
        num_workers=0,
        
        # eval=True,
        # resume="output/checkpoint_best_ema.pth"
    )


def train_from_teeth_dir(teeth_dir: str):
    rf_detr = RFDETRBase(
        # segmentation_head=True,
        patch_size=8,
        num_windows=4,
        num_queries=25,
        group_detr=5,
        num_select=20,
        pretrain_weights="output/checkpoint0099.pth",
        # ='output/xray_teeth33_nano'
    )
    device_supports_cuda = torch.cuda.is_available()
    
    rf_detr.train(
        dataset_dir=teeth_dir,
        epochs=100,
        device="cuda" if device_supports_cuda else "cpu",
        dataset_file='teeth',
        # coco_path=teeth_dir,
        batch_size=1,
        num_workers=0,
        square_resize=False,
        # segmentation_head=True,
        mask_ce_loss_coef=5.,
        mask_dice_loss_coef=5.,
        mask_point_sample_ratio=16,
        grad_accum_steps=1,
        coco_evaluate=False,
        multi_scale=False,
        num_queries=25,
        num_select=20,
    )
    
    


def get_my_arg_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--coco_dir", type=str, default='/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle/Teeth Segmentation JSON/d2/')
    parser.add_argument("--num_classes", type=int, required=False, default=32)
    # parser.add_argument("--project_name", type=str, required=False, default=None)
    parser.add_argument("--annot_file", type=str, required=False, default='../../xray_coco_33_seg.json')
    # parser.add_argument("--annot_file", type=str, required=False, default='../../xray_coco_33_seg.json')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--eval_save', action='store_true',
                        # default=True,
                        help='Run save results after training')
    parser.add_argument('--segmentation_head', action='store_true',
                        # default=True, 
                        help='Run save results after training')
    
    parser.add_argument('--pretrain_weights', type=str,
                        # default=True, 
                        default='',
                        help='Run save results after training')
    
    # parser.add_argument('--pretrain_weights', type=str,
    #                     # default=True, 
    #                     default='',
    #                 help='Run save results after training')
    args = parser.parse_args()
    return args


# cli/main.py -> rfdetr/main.py -> rfdetr/enggine.py::train_one_epocch // evaluate
# rfdetr/models/lwdetr // models & build-model
# rfdetr/detr.py ;; rfdetr main-class model & export utils
# rfdetr/main.py 
#   / trainer // train & evaluate & test trainer 
#  mODEL: trainer wrapper class


def train_from_xray_teeth_dir():
    
    args = get_my_arg_parse()
    dataset_dir = args.coco_dir
    num_classes = args.num_classes
    annot_file = args.annot_file
    
    
    args.segmentation_head = True
    args.eval = True
    args.eval_save = True
    
    # rf_detr = RFDETRSmall(
        
    #     patch_size=16,
    #     num_windows=4,
    #     num_queries=50,
    #     group_detr=5,
    #     num_select=30,
    #     encoder='dinov2_windowed_tiny',

    #     num_classes=32)
    
    # rf_detr = RFDETRNano(
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
        num_classes=num_classes,
        segmentation_head=args.segmentation_head,
        # pretrain_weights="output/checkpoint0099.pth",
        # pretrain_weights="output/xray_teeth/checkpoint0059.pth",
        # pretrain_weights='output/xray_teeth33/checkpoint0039.pth'
        # pretrain_weights='output/xray_teeth33/checkpoint_best_regular.pth'
        # pretrain_weights='output/xray_teeth33_nano/checkpoint_best_total.pth'
        # pretrain_weights='output/xray_teeth33_small/checkpoint_best_regular.pth'
        # pretrain_weights= args.pretrain_weights # 'output/xray_teeth33_small_seg/checkpoint0039.pth',
        # pretrain_weights='output/xray_teeth33_dinov2tiny_small/checkpoint0039.pth',
        pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
        
    )
    device_supports_cuda = torch.cuda.is_available()
    
    rf_detr.train(
        dataset_dir=dataset_dir,
        epochs=500,
        device="cuda" if device_supports_cuda else "cpu",
        dataset_file='xray_teeth',
        # coco_path=teeth_dir,
        batch_size=4,
        num_workers=0,
        square_resize=False,
        # segmentation_head=True,
        mask_ce_loss_coef=5.,
        mask_dice_loss_coef=5.,
        mask_point_sample_ratio=16,
        grad_accum_steps=1,
        coco_evaluate=False,
        multi_scale=False,
        num_queries=50,
        num_select=35,
        checkpoint_interval = 50,
        output_dir='output/xray_teeth33_dinov2tiny_small_seg',
        # annot_file='../../xray_coco_33.json',
        # annot_file='../../xray_coco_33.json',
        annot_file=annot_file,
        # resume='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth',
        # annot_file=''
        # annot_file='../../xray_coco.json',
        # pretrain_weights="output/checkpoint0099.pth",
        segmentation_head=args.segmentation_head,
        eval_save=args.eval_save,
        eval=args.eval,
        # **args.__dict__
    )


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", type=str, required=False)
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--workspace", type=str, required=False, default=None)
    parser.add_argument("--project_name", type=str, required=False, default=None)
    parser.add_argument("--dataset_version", type=int, required=False, default=None)
    args = parser.parse_args()
    return args


def trainer():
    args = get_arg_parse()
    
    if args.coco_dir is not None:
        train_from_coco_dir(args.coco_dir)
        return

    if (args.workspace is None and args.project_name is not None) or (
        args.workspace is not None and args.project_name is None
    ):
        raise ValueError(
            "Either both workspace and project_name must be provided or none of them"
        )

    if args.workspace is not None:
        rf = roboflow.Roboflow(api_key=args.api_key)
        project = rf.workspace(args.workspace).project(args.project_name)
    else:
        projects = get_rf100vl_projects(api_key=args.api_key)
        project = projects[0].rf_project

    train_from_rf_project(project, args.dataset_version)

# cli/main.py -> rfdetr/main.py -> rfdetr/enggine.py::train_one_epocch // evaluate
# model-build model-config - >rfdetr/models/lwdetr.py // build_model(...)
if __name__ == "__main__":
    # trainer()
    
    # coco_dir = 'E:/dataset/teeth_seg_3d/render_2dset2'
    # coco_dir = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2'
    # coco_dir = '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle/Teeth Segmentation JSON/d2/'
    # 'E:\dataset\reverse_tomosynthesis\kaggle_xrays\xray_teeth_seg_kaggle\Teeth Segmentation JSON\d2'
    # coco_dir = '/data1/jooyonglee/teeth_segmentation3d/render_set/teeth_seg_3d/'
    torch.cuda.set_device(torch.device('cuda:4'))
    # train_from_teeth_dir(coco_dir)
    train_from_xray_teeth_dir()

    # coco_dir = 'E:/dataset/coco/base'
    # train_from_coco_dir(coco_dir)