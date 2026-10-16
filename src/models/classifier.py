import torch
import torch.nn as nn
from torchvision import models


def build_classifier(
    num_classes: int = 115,
    device: torch.device | None = None,
):
    
    model = models.resnet18(weights=None)

    # Freeze the complete backbone
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune Layer 4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the original ImageNet classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    if device is not None:
        model = model.to(device)

    return model


def load_classifier(
    checkpoint_path,
    num_classes: int = 115,
    device: torch.device | None = None,
):
    """
    Build the final classifier and load trained weights.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = build_classifier(
        num_classes=num_classes,
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


def predict_classifier(
    model,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
):
    """
    Generate a disease prediction from a preprocessed image tensor.
    """

    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    predicted_disease = class_names[predicted_index]

    CONFIDENCE_THRESHOLD = 0.65

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "class_index": predicted_index,
        "uncertain": confidence < CONFIDENCE_THRESHOLD,
    }