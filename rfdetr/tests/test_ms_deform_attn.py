import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn, original_intermediate_calc, refactored_intermediate_calc
# ------------------------------------------------------------------
# PyTest 검증 시나리오
# ------------------------------------------------------------------
@pytest.fixture
def setup_module_and_inputs():
    torch.manual_seed(42)
    N, Len_q, d_model = 2, 100, 256
    n_levels, n_heads, n_points = 4, 8, 4

    # 모듈 초기화 (MSDeformAttn 내 가중치와 구조 공유)
    # from __main__ import MSDeformAttn
    # from rfdetr.models.opts.modules.ms_deform_attn import MSDeformAttn
    attn_module = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
    attn_module.eval()

    query = torch.randn(N, Len_q, d_model)
    spatial_shapes = torch.tensor([[80, 80], [40, 40], [20, 20], [10, 10]], dtype=torch.float32)

    # 1. 각 레벨의 공간 크기 계산 (80*80 + 40*40 + 20*20 + 10*10 = 8500)
    Len_in = sum([h * w for h, w in spatial_shapes])
    input_flatten = torch.randn(N, int(Len_in), d_model)
    
    # 3. 레벨별 시작 인덱스 (Shape: [n_levels]) -> [0, 6400, 8000, 8400]
    level_start_index = torch.cat(
        [
            spatial_shapes.new_zeros((1,)),
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1],
        ]
    ).long()

    return attn_module, query, spatial_shapes, N, Len_q, n_levels, input_flatten, level_start_index


@pytest.mark.parametrize("ref_dim", [2, 4])
def test_deformable_attn_coreml_fix(setup_module_and_inputs, ref_dim):
    attn_module, query, spatial_shapes, N, Len_q, n_levels, input_flatten, level_start_index = setup_module_and_inputs
    
    # 2D 또는 4D Reference Points 생성
    ref_points = torch.rand(N, Len_q, n_levels, ref_dim)

    # 1. 기존 연산 수행
    orig_locs, orig_attn = original_intermediate_calc(
        attn_module, query, ref_points, spatial_shapes, N, Len_q
    )

    # 2. CoreML 호환 수정 연산 수행
    refact_locs, refact_attn = refactored_intermediate_calc(
        attn_module, query, ref_points, spatial_shapes, N, Len_q
    )

    # 3. 오차 검증 (Tol: 1e-6)
    assert torch.allclose(orig_locs, refact_locs, atol=1e-6), f"Sampling locations missmatch on ref_dim={ref_dim}"
    assert torch.allclose(orig_attn, refact_attn, atol=1e-6), f"Attention weights missmatch on ref_dim={ref_dim}"
    res1 = attn_module(query, ref_points, input_flatten, spatial_shapes, None)
    res2 = attn_module.vanila_forward(query, ref_points, input_flatten, spatial_shapes, None)
    assert res1.shape == res2.shape
    assert torch.allclose(res1, res2)
if __name__=='__main__':
    pytest.main([
        '--color=yes',
        '-rGA',
        __file__ + '::test_deformable_attn_coreml_fix'
    ])