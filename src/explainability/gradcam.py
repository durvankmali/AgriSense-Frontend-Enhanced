"""Memory-conscious Grad-CAM for a shared, serially invoked inference model."""

import torch
import torch.nn.functional as F


class GradCAM:
    """Use one forward hook and autograd.grad, avoiding parameter gradient buffers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.forward_handle = target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, module, inputs, output):
        self.activations = output

    def generate(self, image_tensor: torch.Tensor, class_index: int):
        self.activations = None
        try:
            # Classifier layer4 is trainable; the output therefore has a grad graph.
            with torch.enable_grad():
                output = self.model(image_tensor)
                activations = self.activations
                if activations is None or not activations.requires_grad:
                    raise RuntimeError("Grad-CAM target layer did not retain gradients.")
                gradients = torch.autograd.grad(
                    output[:, class_index].sum(), activations,
                    retain_graph=False, create_graph=False,
                )[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations.detach()).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
                cam = cam.squeeze().detach()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                return cam.cpu().numpy()
        finally:
            # Most importantly, don't retain the previous request's autograd graph.
            self.activations = None

    def close(self):
        self.forward_handle.remove()
