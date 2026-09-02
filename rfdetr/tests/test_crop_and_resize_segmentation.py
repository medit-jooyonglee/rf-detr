import torch
import numpy as np

from rfdetr.models.segmentation_head import (
    CropAndResizeSegmentationHead,
    crop_instance_masks,
    crop_tensor_by_boxes,
    scale_coarse_probability_hint,
    paste_masks_in_image,
)

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
        lambda image: np.zeros((*image.shape[-2:], 3), dtype=np.uint8),
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
    )

    # Query 0 is background, query 2 is removed by NMS, and query 3 is below
    # the confidence threshold. Only query 1 should be reconstructed.
    assert paste_query_counts == [1]
    assert paste_image_sizes == [(80, 100)]
    assert torch.allclose(
        pasted_boxes[0], torch.tensor([[30.0, 24.0, 70.0, 56.0]])
    )
    assert len(mask_images) == 1
    assert mask_images[0].shape == (80, 100)
