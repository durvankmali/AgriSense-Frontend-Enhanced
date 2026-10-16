import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


def build_segmentation_model(
    device: torch.device | None = None,
):
    """
    Build the AgriSense U-Net segmentation model.
    """

    model = UNet()

    if device is not None:
        model = model.to(device)

    return model


def load_segmentation_model(
    checkpoint_path,
    device: torch.device | None = None,
):
    """
    Build the U-Net and load trained weights.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = build_segmentation_model(
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Keep only small evaluation metadata, not the checkpoint's optimizer tensors.
    # Holding both the model weights and the full checkpoint used >100 MB extra RAM.
    metadata = {
        key: value for key, value in checkpoint.items()
        if isinstance(value, (int, float, str, bool))
    }
    del checkpoint
    return model, metadata


def predict_segmentation(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.4,
):
    """
    Generate a binary disease-region mask.
    """

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)

    binary_mask = (
        probabilities >= threshold
    ).float()

    coverage = binary_mask.mean().item()

    return {
        "probability_map": probabilities,
        "mask": binary_mask,
        "coverage": coverage,
    }