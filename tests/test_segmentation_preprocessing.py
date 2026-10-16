import torch
from PIL import Image
from src.preprocessing.image_preprocessing import (
    preprocess_segmentation_image, align_segmentation_mask_to_classifier_view
)


def test_segmentation_input_no_imagenet_normalization_and_direct_resize():
    src = Image.new("RGB", (430, 240), (255, 0, 0))
    tensor = preprocess_segmentation_image(src)
    assert tensor.shape == (1, 3, 224, 224)
    assert torch.allclose(tensor[:, 0, :, :], torch.ones(1, 224, 224))
    assert torch.allclose(tensor[:, 1:, :, :], torch.zeros(1, 2, 224, 224))


def test_mask_letterboxing_landscape():
    m = torch.ones((1, 1, 224, 224))
    remapped = align_segmentation_mask_to_classifier_view(m, (430, 240))
    assert remapped.shape == (1, 1, 224, 224)
    assert torch.count_nonzero(remapped[:, :, :49]).item() == 0
    assert torch.all(remapped[:, :, 49:174] == 1)
    assert torch.count_nonzero(remapped[:, :, 174:]).item() == 0


def test_mask_letterboxing_portrait():
    m = torch.ones((1, 1, 224, 224))
    remapped = align_segmentation_mask_to_classifier_view(m, (240, 430))
    assert remapped.shape == (1, 1, 224, 224)
    assert torch.count_nonzero(remapped[:, :, :, :49]).item() == 0
    assert torch.all(remapped[:, :, :, 49:174] == 1)
    assert torch.count_nonzero(remapped[:, :, :, 174:]).item() == 0


def test_mask_letterboxing_square_unchanged():
    m = torch.zeros((1, 1, 224, 224))
    m[:, :, 60:85, 90:100] = 1.0
    remapped = align_segmentation_mask_to_classifier_view(m, (500, 500))
    assert torch.equal(remapped, m)
