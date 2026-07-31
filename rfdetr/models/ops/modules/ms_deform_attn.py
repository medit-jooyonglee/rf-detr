# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
Multi-Scale Deformable Attention Module
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import ms_deform_attn_core_pytorch, ms_deform_attn_core_pytorch_export


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module
    """
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the "
                          "dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()
        
        self._export = False

    def export(self):
        """export mode
        """
        self._export = True

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)
                     [0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def vanila_forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        attention_weights = F.softmax(attention_weights, -1)

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        if not self._export:
            return self.vanila_forward(
                query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask
            )

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # CoreML은 rank>5 텐서를 지원하지 않으므로, n_levels/n_points를 병합한
        # rank5/rank4 형태(sampling_locations/attention_weights)를 끝까지 유지한다.
        sampling_locations, attention_weights = refactored_intermediate_calc(
            self,
            query,
            reference_points,
            input_spatial_shapes,
            N,
            Len_q,
            merge_output=True,
        )

        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch_export(
            value, input_spatial_shapes, sampling_locations, attention_weights, self.n_points)
        output = self.output_proj(output)
        return output


# ------------------------------------------------------------------
# 기존 방식과 수정 방식을 비교하기 위한 헬퍼 연산 함수
# ------------------------------------------------------------------
def original_intermediate_calc(module: MSDeformAttn, query, reference_points, input_spatial_shapes, N, Len_q):
    """기존 Rank 6 연산 로직"""
    sampling_offsets = module.sampling_offsets(query).view(
        N, Len_q, module.n_heads, module.n_levels, module.n_points, 2
    )
    attention_weights = module.attention_weights(query).view(
        N, Len_q, module.n_heads, module.n_levels * module.n_points
    )

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets / module.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        )

    attention_weights = F.softmax(attention_weights, -1).view(
        N, Len_q, module.n_heads, module.n_levels, module.n_points
    )
    return sampling_locations, attention_weights


def refactored_intermediate_calc(module: MSDeformAttn, query, reference_points, input_spatial_shapes, N, Len_q,
                                  merge_output=False):
    """CoreML 호환 Rank 5 연산 로직

    :param merge_output   True면 n_levels/n_points를 병합한 rank5(locations)/rank4(weights) 형태를 그대로 반환한다.
                          False(기본값)면 기존 rank6/rank5 형태로 복원해서 반환한다 (original_intermediate_calc와 비교용).
    """
    n_total_points = module.n_levels * module.n_points
    sampling_offsets = module.sampling_offsets(query).view(
        N, Len_q, module.n_heads, n_total_points, 2
    )
    attention_weights = module.attention_weights(query).view(
        N, Len_q, module.n_heads, n_total_points
    )

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        offset_normalizer = offset_normalizer.repeat_interleave(
            module.n_points, dim=0
        )
        ref_pts = reference_points.repeat_interleave(module.n_points, dim=2)
        sampling_locations = (
            ref_pts[:, :, None, :, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, :]
        )

    elif reference_points.shape[-1] == 4:
        ref_pts_xy = reference_points[:, :, None, :, :2].repeat_interleave(
            module.n_points, dim=3
        )
        ref_pts_wh = reference_points[:, :, None, :, 2:].repeat_interleave(
            module.n_points, dim=3
        )
        sampling_locations = (
            ref_pts_xy
            + sampling_offsets / module.n_points * ref_pts_wh * 0.5
        )
    else:
        raise ValueError(f'not supported shape{reference_points.shape}')

    attention_weights = F.softmax(attention_weights, -1)

    if merge_output:
        return sampling_locations, attention_weights

    sampling_locations = sampling_locations.view(
        N, Len_q, module.n_heads, module.n_levels, module.n_points, 2
    )
    attention_weights = attention_weights.view(
        N, Len_q, module.n_heads, module.n_levels, module.n_points
    )
    return sampling_locations, attention_weights

