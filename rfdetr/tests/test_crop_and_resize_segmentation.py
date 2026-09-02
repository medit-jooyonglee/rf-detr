import torch
import numpy as np
import torch.nn as nn

from rfdetr.models.lwdetr import LWDETR, SetCriterion
from rfdetr.models.segmentation_head import (
    CropAndResizeSegmentationHead,
    crop_instance_masks,
    crop_tensor_by_boxes,
    expand_normalized_boxes,
    scale_coarse_probability_hint,
    paste_masks_in_image,
)


def test_expand_normalized_boxes_adds_context_and_clips_edges():
    boxes = torch.tensor([
        [0.5, 0.5, 0.4, 0.6],
        [0.05, 0.05, 0.2, 0.2],
    ])

    expanded = expand_normalized_boxes(boxes, 1.15)

    assert torch.allclose(
        expanded[0], torch.tensor([0.5, 0.5, 0.46, 0.69])
    )
    expanded_xyxy = torch.stack((
        expanded[:, 0] - expanded[:, 2] / 2,
        expanded[:, 1] - expanded[:, 3] / 2,
        expanded[:, 0] + expanded[:, 2] / 2,
        expanded[:, 1] + expanded[:, 3] / 2,
    ), dim=1)
    assert expanded_xyxy.min() >= 0.0
    assert expanded_xyxy.max() <= 1.0


def _make_refinement_model(dropout=0.0):
    model = LWDETR.__new__(LWDETR)
    nn.Module.__init__(model)
    model.segmentation_crop_size = (128, 64)
    model.segmentation_crop_box_scale = 1.15
    model.coarse_hint_scale = 0.35
    model.coarse_hint_dropout = dropout
    model.segmentation_refinement_head = _SumChannelsHead()
    return model


class _SumChannelsHead(nn.Module):
    def forward(self, inputs):
        return inputs.sum(dim=1)


def test_refinement_detaches_coarse_mask_gradient():
    model = _make_refinement_model()
    model.train()
    images = torch.rand(1, 3, 32, 32, requires_grad=True)
    coarse_masks = torch.rand(1, 2, 8, 8, requires_grad=True)
    targets = [{
        'boxes': torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        'labels': torch.tensor([1]),
        'size': torch.tensor([32, 32]),
    }]

    refined = model._predict_target_crop_masks(
        images,
        targets,
        torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2]]]),
        torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        coarse_masks,
    )[0]
    refined.mean().backward()

    assert images.grad is not None
    assert images.grad.abs().sum() > 0
    assert coarse_masks.grad is None


def test_coarse_hint_dropout_drops_whole_instance_hints():
    model = _make_refinement_model(dropout=0.30)
    model.train()
    torch.manual_seed(0)

    hints = model._prepare_coarse_hint(torch.ones(64, 1, 4, 4))
    dropped = (hints == 0).flatten(1).all(dim=1)

    assert 0 < dropped.sum() < hints.shape[0]
    assert torch.all((hints == 0) | (hints == 0.35))


def test_crop_loss_includes_outer_false_positive_ring():
    criterion = SetCriterion.__new__(SetCriterion)
    nn.Module.__init__(criterion)
    criterion.segmentation_crop_size = (128, 64)
    criterion.segmentation_crop_box_scale = 1.15
    crop_logits = torch.zeros(1, 128, 64, requires_grad=True)
    target_mask = torch.zeros(1, 64, 64)
    target_mask[:, 20:44, 24:40] = 1.0

    losses = criterion._loss_crop_masks(
        {
            'pred_mask_crops': [crop_logits],
            'pred_logits': torch.zeros(1, 1, 2),
        },
        [{
            'boxes': torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            'masks': target_mask,
        }],
    )

    assert losses['loss_mask_refine_outer'] > 0
    sum(losses.values()).backward()
    assert crop_logits.grad is not None

def test_scale_coarse_probability_hint_centers_and_scales():
    probability = torch.tensor([0.0, 0.5, 1.0])
    hint = scale_coarse_probability_hint(probability, 0.35)
    assert torch.allclose(hint, torch.tensor([-0.35, 0.0, 0.35]))



def test_crop_tensor_by_boxes_shape_and_gradients():
    image = torch.rand(2, 3, 80, 160, requires_grad=True)
    boxes = torch.tensor([
        [[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.2, 0.3]],
        [[0.5, 0.5, 1.0, 1.0], [0.75, 0.5, 0.3, 0.6]],
    ])

    crops = crop_tensor_by_boxes(image, boxes, (128, 64))

    assert crops.shape == (2, 2, 3, 128, 64)
    crops.mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_crop_instance_masks_uses_corresponding_box():
    masks = torch.zeros(1, 40, 80)
    masks[:, 10:30, 20:60] = 1
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])

    crop = crop_instance_masks(masks, boxes, (128, 64))

    assert crop.shape == (1, 128, 64)
    assert crop.mean() > 0.95


def test_crop_head_output_shape_and_gradients():
    head = CropAndResizeSegmentationHead((128, 64))
    crops = torch.rand(3, 4, 128, 64, requires_grad=True)

    logits = head(crops)

    assert logits.shape == (3, 128, 64)
    logits.mean().backward()
    assert crops.grad is not None
    assert crops.grad[:, 3].abs().sum() > 0


def test_paste_masks_in_image_respects_box():
    masks = torch.ones(1, 128, 64)
    boxes = torch.tensor([[10.0, 5.0, 30.0, 25.0]])

    pasted = paste_masks_in_image(masks, boxes, (40, 50))

    assert pasted.shape == (1, 40, 50)
    assert torch.all(pasted[0, 5:25, 10:30] > 0)
    assert pasted[0, :5].count_nonzero() == 0
    assert pasted[0, :, :10].count_nonzero() == 0


def test_draw_predictions_pastes_only_positive_nms_queries(monkeypatch):
    import rfdetr.engine as engine
    import rfdetr.datasets.teeth as teeth
    import rfdetr.datasets.xraypanoramic as xraypanoramic
    import rfdetr.models.segmentation_head as segmentation_head
    from trainer import image_utils, utils_numpy, vtk_utils

    paste_query_counts = []
    paste_image_sizes = []
    pasted_boxes = []
    original_paste_masks = segmentation_head.paste_masks_in_image

    def recording_paste_masks(masks, boxes, image_size):
        paste_query_counts.append(masks.shape[0])
        paste_image_sizes.append(image_size)
        pasted_boxes.append(boxes.cpu())
        return original_paste_masks(masks, boxes, image_size)

    monkeypatch.setattr(
        segmentation_head, "paste_masks_in_image", recording_paste_masks
    )
    monkeypatch.setattr(teeth, "draw_bboxes", lambda *args, **kwargs: None)
    monkeypatch.setattr(xraypanoramic, "label_to_fdi", lambda labels: labels)
    monkeypatch.setattr(
        vtk_utils,
        "get_teeth_color_table",
        lambda normalize=False: np.zeros((50, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        image_utils,
        "to_magnitude_images",
        lambda image: np.zeros((*image.shape[:2], 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        utils_numpy, "apply_blending_mask", lambda image, color_mask: image
    )

    outputs = {
        "pred_boxes": torch.tensor(
            [[[0.2, 0.2, 0.1, 0.1],
              [0.5, 0.5, 0.4, 0.4],
              [0.5, 0.5, 0.4, 0.4],
              [0.8, 0.8, 0.1, 0.1]]]
        ),
        "pred_logits": torch.tensor(
            [[[10.0, -10.0, -10.0],
              [-10.0, 3.0, -10.0],
              [-10.0, 2.0, -10.0],
              [-10.0, -1.0, -10.0]]]
        ),
        "pred_masks": torch.ones(1, 4, 8, 4),
    }

    _, mask_images = engine.draw_preditions_boxes(
        torch.zeros(1, 3, 40, 50),
        outputs,
        origin_size=(80, 100),
        segmentation_mode="crop_and_resize",
        paste_masks_at_original_size=True,
        input_content_box=(5, 0, 45, 40),
    )

    # Query 0 is background, query 2 is removed by NMS, and query 3 is below
    # the confidence threshold. Only query 1 should be reconstructed.
    assert paste_query_counts == [1]
    assert paste_image_sizes == [(80, 100)]
    assert torch.allclose(
        pasted_boxes[0], torch.tensor([[21.25, 21.6, 78.75, 58.4]]), atol=1e-5
    )
    assert len(mask_images) == 1
    assert mask_images[0].shape == (80, 100)
