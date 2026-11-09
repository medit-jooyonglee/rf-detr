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



# cat << 'EOF'
nohup python -m rfdetr.cli.main \
  --pretrain_weights outputs/peri_data_fusion/checkpoint.pth \
  --device cuda:4 \
  --batch_size 4 \
  --epochs 500 \
  --checkpoint_interval 50 \
  --xray_crop_mode periapical teeth_roi \
  --xray_crop_prob 0.2 0.4 \
  --overlap_loss_coef 4.0 \
  --lr_encoder 1e-5 \
  --output_dir outputs/peri_data_fusion_1104_ce \
  > logs/peri_data_fusion_1104_ce.log 2>&1 &



nohup python -m rfdetr.cli.main \
  --pretrain_weights outputs/peri_data_fusion_1104_ce/checkpoint.pth \
  --device cuda:4 \
  --batch_size 4 \
  --epochs 300 \
  --checkpoint_interval 50 \
  --xray_crop_mode periapical teeth_roi \
  --xray_crop_prob 0.2 0.4 \
  --overlap_loss_coef 4.0 \
  --lr_encoder 1e-5 \
  --output_dir outputs/peri_data_fusion_1106_ce \
  > logs/peri_data_fusion_1106_ce.log 2>&1 &




# nohup python -m rfdetr.cli.main --pretrain_weights outputs/peri_data_fusion/checkpoint.pth --device cuda:4 --batch_size 4 --epochs 500 --checkpoint_interval 50 --xray_crop_mode periapical teeth_roi --xray_crop_prob 0.2 0.4 --overlap_loss_coef 4.0 --lr_encoder 1e-5 --output_dir outputs/peri_data_fusion_1104_ce > logs/peri_data_fusion_1104_ce.log 2>&1 &  
