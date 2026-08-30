# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

def scale_coarse_probability_hint(
    coarse_probability: torch.Tensor,
    scale: float = 0.35,
) -> torch.Tensor:
    """Center [0,1] probabilities at zero and limit their refinement strength."""
    if not 0.0 <= scale <= 1.0:
        raise ValueError("coarse_hint_scale must be between 0 and 1.")
    return (2.0 * coarse_probability - 1.0) * scale


def _crop_grid(boxes: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Build a grid_sample grid for normalized cxcywh boxes."""
    crop_h, crop_w = output_size
    boxes = boxes.clamp(0.0, 1.0)
    cx, cy, bw, bh = boxes.unbind(-1)
    bw = bw.clamp_min(1e-6)
    bh = bh.clamp_min(1e-6)

    ys = (torch.arange(crop_h, device=boxes.device, dtype=boxes.dtype) + 0.5) / crop_h
    xs = (torch.arange(crop_w, device=boxes.device, dtype=boxes.dtype) + 0.5) / crop_w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    x0 = cx - 0.5 * bw
    y0 = cy - 0.5 * bh
    sample_x = x0[..., None, None] + grid_x * bw[..., None, None]
    sample_y = y0[..., None, None] + grid_y * bh[..., None, None]
    grid = torch.stack((sample_x, sample_y), dim=-1)
    return grid.mul(2.0).sub(1.0)


def crop_tensor_by_boxes(
    tensor: torch.Tensor,
    boxes: torch.Tensor,
    output_size: tuple[int, int] = (128, 64),
) -> torch.Tensor:
    """Crop BxCxHxW tensors with BxNx4 normalized cxcywh boxes."""
    if tensor.ndim != 4 or boxes.ndim != 3:
        raise ValueError("Expected tensor [B,C,H,W] and boxes [B,N,4].")
    batch_size, channels = tensor.shape[:2]
    if boxes.shape[0] != batch_size:
        raise ValueError("Tensor and boxes batch dimensions must match.")

    num_boxes = boxes.shape[1]
    crop_h, crop_w = output_size
    if num_boxes == 0:
        return tensor.new_empty((batch_size, 0, channels, crop_h, crop_w))

    source = tensor[:, None].expand(-1, num_boxes, -1, -1, -1)
    source = source.reshape(batch_size * num_boxes, channels, *tensor.shape[-2:])
    grid = _crop_grid(boxes, output_size).reshape(
        batch_size * num_boxes, crop_h, crop_w, 2
    )
    crops = F.grid_sample(
        source,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    return crops.reshape(batch_size, num_boxes, channels, crop_h, crop_w)


def crop_instance_masks(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    output_size: tuple[int, int] = (128, 64),
) -> torch.Tensor:
    """Crop each NxHxW instance mask with its corresponding Nx4 box."""
    if masks.shape[0] == 0:
        return masks.new_empty((0, *output_size), dtype=torch.float32)
    crops = crop_tensor_by_boxes(
        masks[:, None].float(),
        boxes[:, None].detach(),
        output_size,
    )
    return crops[:, 0, 0]


def paste_masks_in_image(
    masks: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Paste K ROI mask logits into K full-image canvases."""
    image_h, image_w = image_size
    pasted = masks.new_zeros((masks.shape[0], image_h, image_w))
    for index, (mask, box) in enumerate(zip(masks, boxes_xyxy)):
        x0 = max(int(torch.floor(box[0]).item()), 0)
        y0 = max(int(torch.floor(box[1]).item()), 0)
        x1 = min(int(torch.ceil(box[2]).item()), image_w)
        y1 = min(int(torch.ceil(box[3]).item()), image_h)
        if x1 <= x0 or y1 <= y0:
            continue
        resized = F.interpolate(
            mask[None, None],
            size=(y1 - y0, x1 - x0),
            mode='bilinear',
            align_corners=False,
        )[0, 0]
        pasted[index, y0:y1, x0:x1] = resized
    return pasted


class _CropConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CropAndResizeSegmentationHead(nn.Module):
    """Binary U-Net refining image ROIs with coarse mask probabilities."""

    def __init__(self, output_size: tuple[int, int] = (128, 64), in_channels: int = 4):
        super().__init__()
        self.output_size = tuple(output_size)
        if min(self.output_size) < 4:
            raise ValueError(
                "Crop segmentation height and width must both be at least 4."
            )
        self.enc1 = _CropConvBlock(in_channels, 32)
        self.enc2 = _CropConvBlock(32, 64)
        self.bottleneck = _CropConvBlock(64, 128)
        self.dec2 = _CropConvBlock(128 + 64, 64)
        self.dec1 = _CropConvBlock(64 + 32, 32)
        self.mask_logits = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        if crops.shape[0] == 0:
            return crops.new_empty((0, *self.output_size))
        enc1 = self.enc1(crops)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))
        dec2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = F.interpolate(dec2, size=enc1.shape[-2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        return self.mask_logits(dec1)[:, 0]


class DepthwiseConvBlock(nn.Module):
    r""" Simplified ConvNeXt block without the MLP subnet
    """
    def __init__(self, dim, layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        return x + input


class MLPBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=0):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        ])
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x + input


class SegmentationHead(nn.Module):
    def __init__(self, in_dim, num_blocks: int, bottleneck_ratio: int=1, downsample_ratio: int=4):
        super().__init__()

        self.downsample_ratio = downsample_ratio
        self.interaction_dim = in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        self.blocks = nn.ModuleList([DepthwiseConvBlock(in_dim) for _ in range(num_blocks)])
        self.spatial_features_proj = nn.Identity() if bottleneck_ratio is None else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)

        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = nn.Identity() if bottleneck_ratio is None else nn.Linear(in_dim, self.interaction_dim)

        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()
    
    def forward(self, spatial_features: torch.Tensor, query_features: list[torch.Tensor], image_size: tuple[int, int], skip_blocks: bool=False) -> list[torch.Tensor]:
        # spatial features: (B, C, H, W)
        # query features: [(B, N, C)] for each decoder layer
        # output: (B, N, H*r, W*r)
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode='bilinear', align_corners=False)

        mask_logits = []
        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                spatial_features_proj = self.spatial_features_proj(spatial_features)
                qf = self.query_features_proj(self.query_features_block(qf))
                mask_logits.append(torch.einsum('bchw,bnc->bnhw', spatial_features_proj, qf) + self.bias)
        else:
            assert len(query_features) == 1, "skip_blocks is only supported for length 1 query features"
            qf = self.query_features_proj(self.query_features_block(query_features[0]))
            mask_logits.append(torch.einsum('bchw,bnc->bnhw', spatial_features, qf) + self.bias)

        return mask_logits
    
    def forward_export(self, spatial_features: torch.Tensor, query_features: list[torch.Tensor], image_size: tuple[int, int], skip_blocks: bool=False) -> list[torch.Tensor]:
        assert len(query_features) == 1, "at export time, segmentation head expects exactly one query feature"
        
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        spatial_features = F.interpolate(spatial_features, size=target_size, mode='bilinear', align_corners=False)

        if not skip_blocks:
            for block in self.blocks:
                spatial_features = block(spatial_features)
        
        spatial_features_proj = self.spatial_features_proj(spatial_features)

        qf = self.query_features_proj(self.query_features_block(query_features[0]))
        return [torch.einsum('bchw,bnc->bnhw', spatial_features_proj, qf) + self.bias]


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio=3, importance_sample_ratio=0.75
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords