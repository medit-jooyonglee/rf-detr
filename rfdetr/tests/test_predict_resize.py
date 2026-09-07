import numpy as np

from rfdetr.deploy.predict import resize_image


def test_resize_image_uses_exact_export_size():
    image = np.zeros((1000, 2000), dtype=np.uint8)

    resized, content_box = resize_image(image, resize_hw=(384, 704))

    assert resized.shape == (384, 704)
    assert content_box == (0, 16, 704, 368)


def test_resize_image_rejects_invalid_export_size():
    image = np.zeros((10, 20), dtype=np.uint8)

    try:
        resize_image(image, resize_hw=(0, 704))
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid resize_hw to raise ValueError")
