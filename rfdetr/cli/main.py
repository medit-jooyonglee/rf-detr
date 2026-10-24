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
import json

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
    parser.add_argument("--device", type=str, default='cuda:0', help='Device to use for training (e.g., "cuda:0" or "cpu")')
    # parser.add_argument("--project_name", type=str, required=False, default=None)
    parser.add_argument("--annot_file", type=str, required=False, default='../../xray_coco_33_seg.json')
    # parser.add_argument("--annot_file", type=str, required=False, default='../../xray_coco_33_seg.json')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only (skip training)')
    parser.add_argument('--eval_save', action='store_true',
                        # default=True,
                        help='Run save results after training')
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--segmentation_head', action='store_true',
                        # default=True, 
                        help='Run save results after training')
    
    parser.add_argument('--pretrain_weights', type=str,
                        # default=True,
                        default='outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth',
                        help='Run save results after training')
    parser.add_argument('--epochs', type=int, default=250, help='학습 epoch 수')
    parser.add_argument('--input_width', type=int, default=640,
                        help='리사이즈 기준 너비 (periapical 제외). 실제 입력 너비는 (input_width // 64 + 1) * 64 (640 -> 704, 256 -> 320)')
    parser.add_argument('--segmentation_crop_size', type=int, nargs=2, default=[128, 64], metavar=('H', 'W'),
                        help='crop_and_resize refinement ROI 크기 (H W, 4의 배수). 추론(predict)도 같은 값을 써야 한다')
    parser.add_argument('--checkpoint_interval', type=int, default=30, help='checkpoint{epoch}.pth 저장 주기(epoch)')
    parser.add_argument('--data_mode', type=str, default='xray', choices=['xray', 'natural'],
                        help='(Linux) 학습 데이터: xray(파노라마 ①②③) | natural(smile_design_picture 사진, coco만)')
    parser.add_argument('--output_dir', type=str,
                        default='output/xray_teeth33_dinov2tiny_small_seg_crop_regularization',
                        help='학습 결과(checkpoint, log) 저장 경로')
    parser.add_argument('--xray_crop_mode', type=str, nargs='+', default=None, choices=['periapical', 'teeth_roi'],
                        help='학습 샘플 crop (검증에도 적용하려면 --eval_xray_crop). 여러 개 지정 가능 (예: periapical teeth_roi). '
                             'periapical: 치근단 방사선 모사 (이웃 치아 3~5개, 상악/하악/상하악, 사선/둥근 모서리). '
                             'teeth_roi: two-pass 2차 입력 (전체 치아 영역만 crop). 상세: rfdetr/datasets/xray_crop_aug.py')
    parser.add_argument('--xray_crop_prob', type=float, nargs='+', default=[1.0],
                        help='mode별 선택 확률. 값 1개면 모든 mode에 같은 값, 여러 개면 --xray_crop_mode 순서대로. '
                             '샘플마다 mode 하나를 확률대로 고르고 나머지(1-합)는 전체 영상. 합이 1을 넘으면 비율로 정규화 '
                             '(예: periapical teeth_roi + 0.3 0.3 -> PA 30%%, ROI 30%%, 전체 영상 40%%)')
    parser.add_argument('--xray_crop_cfg', type=str, default=None,
                        help='crop 세부 파라미터 JSON (xray_crop_aug.DEFAULT_PERIAPICAL / DEFAULT_TEETH_ROI 덮어쓰기). '
                             'mode별 지정: \'{"periapical": {"num_teeth": [3, 4]}, "teeth_roi": {"margin": [0.05, 0.1]}}\'')
    parser.add_argument('--eval_xray_crop', action='store_true',
                        help='val/valid/test에도 xray crop 적용. 회전/색상 증강은 끄고 crop된 GT로 평가. '
                             '--eval --eval_save와 함께 사용하면 crop 영상 예측 저장')
    parser.add_argument('--eval_xray_crop_seed', type=int, default=42,
                        help='검증 crop 시드 (이미지별 고정, 기본 42)')
    parser.add_argument('--save_crop_samples', type=int, default=0,
                        help='N>0: --xray_crop_mode의 mode별로 학습 샘플 N장을 outputs/xray_crop_samples/{mode}/ 에 저장하고 종료')
    parser.add_argument('--build_tufts', action='store_true',
                        help='Tufts(kaggle_2222) teeth_polygon.pkl -> ann/*.json 변환(기존 ann 자동 백업) + 변환 점검 후 종료')
    parser.add_argument('--build_json', action='store_true',
                        help='학습 dataset_dir로 평가 GT coco json(원본 좌표, *_v2.json)을 재빌드/검증하고 종료 '
                             '(기존 _v2.json은 자동 백업). --build_tufts와 같이 주면 Tufts 변환 후 빌드')
    parser.add_argument('--mask_boundary_loss_coef', type=float, default=0.0,
                        help='GT 경계 근처 픽셀 BCE 가중치(coarse+crop_and_resize 둘 다). 0=비활성화(기본). '
                             '코드 내부 기본값은 2.0(lwdetr.py). 활성화하려면 예: --mask_boundary_loss_coef 2.0')
    parser.add_argument('--refine_outer_boundary_loss_coef', type=float, default=0.0,
                        help='crop_and_resize에서 GT 바깥쪽(false-positive) 경계 링 BCE 가중치. 0=비활성화(기본). '
                             '코드 내부 기본값은 4.0. 활성화하려면 예: --refine_outer_boundary_loss_coef 4.0')
    parser.add_argument('--overlap_loss_coef', type=float, default=0.0,
                        help='crop_and_resize에서 이웃 치아와 겹치는 GT 영역에 추가 BCE 가중치(신규, 중첩 영역 학습 강화용). '
                             '0=비활성화(기본). 활성화하려면 예: --overlap_loss_coef 4.0')
    parser.add_argument('--lr', type=float, default=1e-4, help='head/decoder 학습률 (코드 기본값 1e-4)')
    parser.add_argument('--lr_encoder', type=float, default=1.5e-4,
                        help='DINOv2 backbone 학습률 (코드 기본값 1.5e-4, head lr보다 큼). '
                             '작은 데이터셋으로 freeze_encoder=False 전체 파인튜닝 시 catastrophic forgetting 방지하려면 '
                             'head lr보다 훨씬 낮게(예: 1e-5) 줄 것')

    # parser.add_argument('--pretrain_weights', type=str,
    #                     # default=True, 
    #                     default='',
    #                 help='Run save results after training')
    args = parser.parse_args()
    if args.eval_xray_crop and not args.xray_crop_mode:
        parser.error('--eval_xray_crop requires --xray_crop_mode')
    return args


# cli/main.py -> rfdetr/main.py -> rfdetr/enggine.py::train_one_epocch // evaluate
# rfdetr/models/lwdetr // models & build-model
# rfdetr/detr.py ;; rfdetr main-class model & export utils
# rfdetr/main.py 
#   / trainer // train & evaluate & test trainer 
#  mODEL: trainer wrapper class


def train_from_xray_teeth_dir():
    # public xray dataset from build
    # 1. Teeth Segmentation on dental X-ray images
    # 2. The Tufts Dental Database 2022
    # medit-datset from label-me(https://github.com/medit-AI/labelme.git) and converting. 
    # 
    
    import platform
    if platform.system().lower() == 'windows':
        dataset_dir = [
            # kaggle set 
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2',
            'E:/dataset/reverse_tomosynthesis/kaggle_xrays/kaggle_2222/',
            
        ]
        coco_directories = [
                # ('E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm', 
                # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm/annotations.json')
                ('E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm_latest_0824', 
                'E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm_latest_0824/annotations.json')
            ]
        # annot_file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_coco_33_seg.json'
        annot_file = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/xray_coco_33_seg_v2.json'
        # json file_name 상대경로 기준 (dataset_dir들의 공통 상위 폴더)
        json_base_dirs = ['E:/dataset/reverse_tomosynthesis/kaggle_xrays']
        coco_directories_val = None  # 0824 데이터는 train/val 미분할
    elif platform.system().lower() == 'linux':
        
        mode = get_my_arg_parse().data_mode
        
        if mode == 'xray':
            
            dataset_dir = [
                '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_teeth_seg_kaggle/Teeth Segmentation JSON/d2/',
                '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle_2222/',
                # '/data1/jooyonglee/reverse_tomo/xray_panoramic/kaggle_2222/'
                
            ]
            coco_directories = [
                    # ('E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm', 
                    # 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm/annotations.json')
                    ('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022/train/', 
                    '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022/train/annotations.json')
                ]
            # ③ 자체 데이터 val split (평가 GT에 합쳐서 평가)
            coco_directories_val = [
                    ('/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022/val/',
                    '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm_latest_1022/val/annotations.json')
                ]
            
            # coco_directories_val = None
        
            annot_file = '/data1/jooyonglee/reverse_tomo/xray_panoramic/xray_coco_33_seg.json'
            json_base_dirs = ['/data1/jooyonglee/reverse_tomo/xray_panoramic']
        elif mode == 'natural':
            
            dataset_dir = [
                
            ]
            coco_directories = [
                    ('/data1/jooyonglee/smile_design_picture/samples/train/', 
                    '/data1/jooyonglee/smile_design_picture/samples/train/annotations.json'),
                    (
                        '/data1/jooyonglee/smile_design_picture/flux2/sub_samples/train/',
                        '/data1/jooyonglee/smile_design_picture/flux2/sub_samples/train/annotations.json',
                    )
                ]
            # ③ 자체 데이터 val split (평가 GT에 합쳐서 평가)
            coco_directories_val = [
                    ('/data1/jooyonglee/smile_design_picture/samples/train/', 
                    '/data1/jooyonglee/smile_design_picture/samples/train/annotations.json'),
                    (
                        '/data1/jooyonglee/smile_design_picture/flux2/sub_samples/val/',
                        '/data1/jooyonglee/smile_design_picture/flux2/sub_samples/val/annotations.json',
                    )
                ]
            
            # coco_directories_val = None
        
            annot_file = ''
            json_base_dirs = []
            
    args = get_my_arg_parse()

    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(torch.device(args.device))
    # 평가 GT json: 기존 json은 images[].width/height, area가 리사이즈 기준이라 segm 평가가 0이 된다.
    # 원본 좌표로 재빌드한 *_v2.json을 쓴다. (docs/troubleshoots/xray_coco_json_segm_eval.md)
    # Windows(xray_coco_33_seg_v2.json), Linux(xray_coco_33_seg.json) 모두 annot_file이 이미 원본 좌표(v2) json이다
    annot_file_v2 = annot_file
    if args.build_tufts:
        # ② Tufts 포맷 변환 (평가 json의 Tufts polygon도 ann/*.json에서 읽으므로 --build_json보다 먼저)
        from rfdetr.util.dataloader import tufts_to_coco
        from rfdetr.datasets.xray_overlap_stats import check_tufts_conversion
        tufts_dir = dataset_dir[1]
        tufts_to_coco(tufts_dir=tufts_dir)
        check_tufts_conversion(num_samples=100, tufts_dir=tufts_dir)  # IoU 평균 ~0.99 이면 정상
    if args.build_json:
        import time
        from rfdetr.datasets.xraypanoramic import build_xray_coco_json, verify_xray_coco_json
        if os.path.exists(annot_file_v2):
            backup = annot_file_v2.replace('.json', f"_backup_{time.strftime('%Y%m%d_%H%M%S')}.json")
            os.rename(annot_file_v2, backup)
            print(f'backup {annot_file_v2} -> {backup}')
        # 학습과 같은 dataset_dir(①②)로 빌드. ③ coco_directories는 train 전용이라 제외
        build_xray_coco_json(dataset_dir, json_base_dirs, annot_file_v2, num_classes=args.num_classes)
        verify_xray_coco_json(annot_file_v2, base_dir=json_base_dirs[0],
                              vis_dir='outputs/xray_coco_json_vis', num_vis=20)
    xray_crop_cfg = json.loads(args.xray_crop_cfg) if args.xray_crop_cfg else None
    if args.save_crop_samples > 0:
        assert args.xray_crop_mode, '--save_crop_samples needs --xray_crop_mode periapical|teeth_roi'
        from rfdetr.datasets.xray_crop_aug import save_crop_samples, split_crop_cfg
        mode_cfgs = split_crop_cfg(xray_crop_cfg, args.xray_crop_mode)
        for mode in args.xray_crop_mode:  # mode별로 따로 저장 (확률과 무관하게 해당 mode 100%)
            save_crop_samples(dataset_dir, coco_directories, mode, 'outputs/xray_crop_samples',
                              num=args.save_crop_samples, num_classes=args.num_classes, cfg=mode_cfgs[mode])
    if args.build_tufts or args.build_json or args.save_crop_samples > 0:
        return
    if not annot_file:
        pass  # 평가 json 없이 coco_directories의 GT로 평가 (natural)
    elif os.path.exists(annot_file_v2):
        annot_file = annot_file_v2
    else:
        print(f'[warning] {annot_file_v2} not found. run with --build_json first.')

    # dataset_dir = args.coco_dir
    num_classes = args.num_classes
    # annot_file = args.annot_file
    
    
    args.segmentation_head = True
    # args.eval = True
    # args.eval_save = True
    
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
        # tiny-version 따로 추가....
        encoder='dinov2_windowed_tiny',
        # encoder='dinov2_windowed_base',
        
        # patch_size=24,
        # num_channels=1,
        # eval=True,
        # num_classes=32,
        num_classes=num_classes,
        segmentation_head=args.segmentation_head,
        segmentation_mode='crop_and_resize',
        segmentation_crop_size=tuple(args.segmentation_crop_size),
        coarse_hint_scale=0.35,
        refine_outer_boundary_loss_coef=args.refine_outer_boundary_loss_coef,
        overlap_loss_coef=args.overlap_loss_coef,
        segmentation_crop_box_scale=1.15,
        coarse_hint_dropout=0.30,
        # pretrain_weights="output/xray_teeth33_dinov2tiny_small_seg_crop_retrain/checkpoint0059.pth",
        pretrain_weights=args.pretrain_weights,
        # "output/xray_teeth33_dinov2tiny_small_seg_crop_retrain/checkpoint0059.pth",
        # pre
        # pretrain_weights="output/xray_teeth/checkpoint0059.pth",
        # pretrain_weights='output/xray_teeth33/checkpoint0039.pth'
        # pretrain_weights='output/xray_teeth33/checkpoint_best_regular.pth'
        # pretrain_weights='output/xray_teeth33_nano/checkpoint_best_total.pth'
        # pretrain_weights='output/xray_teeth33_small/checkpoint_best_regular.pth'
        # pretrain_weights= args.pretrain_weights # 'output/xray_teeth33_small_seg/checkpoint0039.pth',
        # pretrain_weights='output/xray_teeth33_dinov2tiny_small/checkpoint0039.pth',
        # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0499.pth'
        # pretrain_weights='output/xray_teeth33_dinov2tiny_small_seg/checkpoint0199.pth',
        # pretrain_weights= 'output/xray_teeth33_dinov2tiny_small_seg/checkpoint.pth',

        # for dataset- field
        
        # 'output/xray_teeth33_dinov2tiny_small_seg/checkpoint0299.pth'
        
    )
    device_supports_cuda = torch.cuda.is_available()
    
    rf_detr.train(
        dataset_dir=dataset_dir,
        coco_directories=coco_directories,
        coco_directories_val=coco_directories_val,
        epochs=args.epochs,
        device=args.device,
        dataset_file='xray_teeth',
        # coco_path=teeth_dir,
        batch_size=args.batch_size,
        num_workers=0,
        square_resize=False,
        # segmentation_head=True,
        mask_ce_loss_coef=5.,
        mask_dice_loss_coef=5.,
        mask_point_sample_ratio=16,
        grad_accum_steps=1,
        coco_evaluate=False,
        multi_scale=False,
        num_queries=100,
        group_detr=5,
        num_select=35,
        segmentation_mode='crop_and_resize',
        segmentation_crop_size=tuple(args.segmentation_crop_size),
        coarse_hint_scale=0.35,
        refine_outer_boundary_loss_coef=args.refine_outer_boundary_loss_coef,
        overlap_loss_coef=args.overlap_loss_coef,
        mask_boundary_loss_coef=args.mask_boundary_loss_coef,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        segmentation_crop_box_scale=1.15,
        coarse_hint_dropout=0.30,
        checkpoint_interval=args.checkpoint_interval,
        input_width=args.input_width,
        output_dir=args.output_dir,
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
        # 기본은 학습에만 crop. eval_xray_crop으로 검증에도 명시적으로 활성화.
        xray_crop_mode=args.xray_crop_mode,
        xray_crop_prob=args.xray_crop_prob,
        xray_crop_cfg=xray_crop_cfg,
        eval_xray_crop=args.eval_xray_crop,
        eval_xray_crop_seed=args.eval_xray_crop_seed,
        # **args.__dict__
        
        # dataset -config
        # splits={
        #     'train': (0, 0.3),
        #     'val': (0.85, 0.9),
        #     'valid': (0.85, 0.9),
        #     'test': (0.9, 1.0),
        # },
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

    # train_from_teeth_dir(coco_dir)
    train_from_xray_teeth_dir()

    # coco_dir = 'E:/dataset/coco/base'
    # train_from_coco_dir(coco_dir)



# nohup python -m rfdetr.cli.main \
#   --data_mode natural \  
#   --xray_crop_mode teeth_roi --xray_crop_prob 1.0 \
#   --eval_xray_crop \
#   --input_width 256 \
#   --segmentation_crop_size 64 48 \
#   --epochs 400 \
#   --checkpoint_interval 100 \
#   --batch_size 1 \
#   --device cuda:5 \
#   --output_dir outputs/natural_teeth_roi > logs/natural_teeth_roi.log 2>&1 &
