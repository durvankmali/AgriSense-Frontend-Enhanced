from PIL import Image
import torch
from torchvision import transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def resize_and_pad(image: Image.Image, size: int = IMAGE_SIZE):
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a square of the requested size.
    """

    image = image.convert("RGB")

    width, height = image.size

    scale = min(
        size / width,
        size / height,
    )

    new_width = round(width * scale)
    new_height = round(height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    padded_image = Image.new(
        "RGB",
        (size, size),
        (0, 0, 0),
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    padded_image.paste(
        image,
        (left, top),
    )

    return padded_image


def preprocess_image(image: Image.Image):
    """
    Apply the preprocessing used by the AgriSense classifier.
    """

    image = resize_and_pad(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    tensor = transform(image)

    return tensor.unsqueeze(0)

def preprocess_segmentation_image(image: Image.Image) -> torch.Tensor:
    """Reproduce the U-Net's training input: RGB, direct 224x224 resize, [0,1].

    The classifier instead uses letterboxing and ImageNet normalization. Sharing
    its tensor with this independent U-Net produces unreliable masks.
    """
    resized = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    return transforms.ToTensor()(resized).unsqueeze(0)


def align_segmentation_mask_to_classifier_view(
    binary_mask: torch.Tensor, source_size: tuple[int, int], size: int = IMAGE_SIZE
) -> torch.Tensor:
    """Letterbox an independently predicted U-Net mask for the frontend canvas.

    Model mask coordinates were obtained from a stretched image (as in training).
    The viewer displays the original as centered letterbox, so resize the binary
    mask into the image footprint before compositing. Keep nearest-neighbor
    interpolation for discrete mask values; padding is always zero.
    """
    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError("Source image dimensions must be positive")
    scale = min(size / width, size / height)
    display_width = round(width * scale)
    display_height = round(height * scale)
    left = (size - display_width) // 2
    top = (size - display_height) // 2

    import torch.nn.functional as F
    resized = F.interpolate(
        binary_mask.detach().cpu().float(),
        size=(display_height, display_width),
        mode="nearest",
    )
    canvas = torch.zeros((1, 1, size, size), dtype=torch.float32)
    canvas[:, :, top:top + display_height, left:left + display_width] = resized
    return canvas
