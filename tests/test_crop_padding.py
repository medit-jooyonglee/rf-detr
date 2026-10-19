from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from rfdetr.engine import draw_preditions_boxes
from rfdetr.models.lwdetr import LWDETR, SetCriterion
from rfdetr.util.misc import nested_tensor_from_tensor_list


def test_render_uses_each_images_size():
    images = [torch.zeros(3, 32, 64), torch.zeros(3, 64, 32)]
    outputs = {
        'pred_boxes': torch.tensor([[[.5, .5, .4, .6]]] * 2),
        'pred_logits': torch.tensor([[[-10., 10.]]] * 2),
        'pred_masks': torch.full((2, 1, 16, 8), 10.),
    }
    options = dict(segmentation_mode='crop_and_resize', drawing_mode='polygon')
    rendered, masks = draw_preditions_boxes(nested_tensor_from_tensor_list(images), outputs, **options)
    for i, image in enumerate(images):
        standalone, standalone_masks = draw_preditions_boxes(
            image[None], {k: v[i:i + 1] for k, v in outputs.items()}, **options
        )
        np.testing.assert_array_equal(rendered[i], standalone[0])
        np.testing.assert_array_equal(masks[i], standalone_masks[0])


def test_refinement_hint_uses_same_region_with_padding():
    head = SimpleNamespace(
        segmentation_crop_box_scale=1.15, segmentation_crop_size=(16, 8),
        _prepare_coarse_hint=lambda x: x,
        segmentation_refinement_head=lambda x: x[:, 3],
    )
    image = torch.zeros(1, 3, 32, 64)
    coarse = torch.linspace(-4, 4, 32 * 64).reshape(1, 1, 32, 64)
    boxes = torch.tensor([[[.5, .5, .4, .6]]])
    padded_image = F.pad(image, (0, 32, 0, 32))
    padded_coarse = F.pad(coarse, (0, 32, 0, 32))
    padding = torch.ones(1, 64, 96, dtype=torch.bool)
    padding[:, :32, :64] = False
    reference = LWDETR._predict_crop_masks(head, image, boxes, coarse)
    padded = LWDETR._predict_crop_masks(head, padded_image, boxes, padded_coarse, padding)
    torch.testing.assert_close(padded, reference)
    target = {'boxes': boxes[0], 'size': torch.tensor([32, 64]), 'labels': torch.tensor([1])}
    logits = torch.tensor([[[-10., 10.]]])
    reference = LWDETR._predict_target_crop_masks(head, image, [target], boxes, logits, coarse)
    padded = LWDETR._predict_target_crop_masks(head, padded_image, [target], boxes, logits, padded_coarse)
    torch.testing.assert_close(padded[0], reference[0])


def test_refinement_loss_ignores_target_padding():
    criterion = SimpleNamespace(segmentation_crop_box_scale=1.15, segmentation_crop_size=(16, 8))
    mask = torch.zeros(1, 32, 64, dtype=torch.bool)
    mask[:, 8:24, 20:44] = True
    target = {'boxes': torch.tensor([[.5, .5, .4, .6]]),
              'size': torch.tensor([32, 64]), 'masks': mask}
    outputs = {'pred_mask_crops': [torch.linspace(-3, 3, 128).reshape(1, 16, 8)]}
    reference = SetCriterion._loss_crop_masks(criterion, outputs, [target])
    padded = SetCriterion._loss_crop_masks(
        criterion, outputs, [dict(target, masks=F.pad(mask, (0, 32, 0, 32)))]
    )
    for key in reference:
        torch.testing.assert_close(padded[key], reference[key])
