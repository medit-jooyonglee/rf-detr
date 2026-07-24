import torchinfo
from rfdetr import RFDETRBase, RFDETRNano, RFDETRSmall

rf_detr = RFDETRSmall(
        
        patch_size=16,
        num_windows=4,
        num_queries=50,
        group_detr=5,
        num_select=30,
        encoder='dinov2_windowed_tiny',

        num_classes=32)
shape = (384, 640)
torchinfo.summary(rf_detr.model.model, input_size=[(1, 3, *shape)],device='cuda')
# torchinfo.summary(rf_detr.model.model.backbone.encoder, input_size=[(1, 3, *shape)],device='cuda')

torchinfo.summary(rf_detr.model.model.backbone[0].encoder, input_size=[(1, 3, *shape)],device='cuda')


import numpy as np
# image resolutsion
# \xray_teeth_seg_kaggle\Teeth Segmentation JSON\d2\img
shapes = [
        (2041, 1024),
        (1852, 1024),
        (1615, 850),
        (2640, 1256),
]

shapes_arrays = np.array(shapes)
scale = 640 / shapes_arrays[:, :1]
scale_shape = ( shapes_arrays * scale).astype(np.int32)
stride = 64
final_shape = (scale_shape // stride + 1) * stride

final_ratio = final_shape / shapes_arrays
ratio_keeps = final_ratio[:, 0] / final_ratio[:, 1]
#  = np.array() 
# np.array()


# 

# dr. baek
# 

#  periapical / 치근단
# (645, 515) (66, 614)


