import torch

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
