#!/bin/bash
# fine-tune: periapical + teeth_roi crop augmentation (기존 augmentation 유지, train split만 적용)
# pretrained: outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth
cd "$(dirname "$0")"

OUTPUT_DIR=outputs/fine_tune_pa_teeth_roi
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=1 nohup python -m rfdetr.cli.main \
    --pretrain_weights outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth \
    --xray_crop_mode periapical teeth_roi \
    --xray_crop_prob 1.0 \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    > ${OUTPUT_DIR}/train.log 2>&1 &

echo "PID: $!  log: ${OUTPUT_DIR}/train.log"


nohup python -m rfdetr.cli.main \
    --pretrain_weights outputs/pretrained/segmentation/xray_teeth_seg/bounding_bbox_regularization/checkpoint.pth \
    --xray_crop_mode periapical teeth_roi \
    --xray_crop_prob 0.4 \
    --output_dir outputs/fine_tune_pa_teeth_roi \
    --batch_size 4 \
    --device cuda:5 \
    > logs/fine_tune_pa_teeth_roi.log 2>&1 &

