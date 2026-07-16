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