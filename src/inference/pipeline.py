from PIL import Image

import time

import torch

from src.config import (
    CLASSIFIER_CHECKPOINT,
    DEVICE,
    SEGMENTATION_CHECKPOINT,
    SEGMENTATION_THRESHOLD,
    load_disease_classes,
)

from src.models.classifier import (
    load_classifier,
    predict_classifier,
)

from src.models.segmentation import (
    load_segmentation_model,
    predict_segmentation,
)

from src.preprocessing.image_preprocessing import (
    preprocess_image,
    preprocess_segmentation_image,
    align_segmentation_mask_to_classifier_view,
)

from src.explainability.gradcam import (
    GradCAM,
)


class AgriSensePipeline:
    """
    End-to-end AgriSense inference pipeline.

    Combines:
    - Disease classification
    - Disease-region segmentation
    - Grad-CAM explainability
    """

    def __init__(
        self,
        classifier_checkpoint=CLASSIFIER_CHECKPOINT,
        segmentation_checkpoint=SEGMENTATION_CHECKPOINT,
        device=DEVICE,
    ):
        self.device = device

        # Load disease classes
        self.class_names = load_disease_classes()

        # Load classifier
        self.classifier, self.classifier_checkpoint = (
            load_classifier(
                classifier_checkpoint,
                num_classes=len(self.class_names),
                device=device,
            )
        )

        # Load segmentation model
        self.segmenter, self.segmentation_checkpoint = (
            load_segmentation_model(
                segmentation_checkpoint,
                device=device,
            )
        )

        # Grad-CAM target layer
        self.gradcam = GradCAM(
            model=self.classifier,
            target_layer=self.classifier.layer4[-1],
        )

    def predict(self, image_path):
        """
        Run the complete AgriSense inference pipeline
        using an image file path.
        """

        image = Image.open(
            image_path
        ).convert("RGB")

        return self.predict_from_image(image)


    def _log_memory(self, stage):
        memory_mb = None

        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        memory_mb = int(line.split()[1]) / 1024
                        break
        except OSError:
            pass

        if memory_mb is not None:
            print(
                f"[PREDICT] {stage} | memory={memory_mb:.1f} MB",
                flush=True,
            )
        else:
            print(
                f"[PREDICT] {stage}",
                flush=True,
            )


    def predict_from_image(self, image):

        """
        Run the complete AgriSense inference pipeline
        using a PIL image.
        """

        start_time = time.perf_counter()

        print("[PREDICT] handler reached", flush=True)
        self._log_memory("start")

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        image = image.convert("RGB")
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        print(
            f"[PREDICT] preprocessing complete | "
            f"time={time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        self._log_memory("after preprocessing")

        # --------------------------------------------------
        # 1. Disease classification
        # --------------------------------------------------

        classification = predict_classifier(
            model=self.classifier,
            image_tensor=image_tensor,
            class_names=self.class_names,
            device=self.device,
        )

        print(
            f"[PREDICT] classification complete | "
            f"time={time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        self._log_memory("after classification")

        predicted_index = classification["class_index"]

        # --------------------------------------------------
        # 2. Disease segmentation
        # --------------------------------------------------

        # The U-Net's training preprocessing is NOT the same as the ResNet's.
        # Its input was directly resized and only scaled to [0, 1], whereas
        # the classifier requires letterboxing and ImageNet normalization.
        segmentation_tensor = preprocess_segmentation_image(image).to(self.device)
        segmentation = predict_segmentation(
            model=self.segmenter,
            image_tensor=segmentation_tensor,
            device=self.device,
            threshold=SEGMENTATION_THRESHOLD,
        )

        print(
            f"[PREDICT] segmentation complete | "
            f"time={time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        self._log_memory("after segmentation")

        # --------------------------------------------------
        # 3. Grad-CAM
        # --------------------------------------------------

        cam = self.gradcam.generate(
        image_tensor=image_tensor,
        class_index=predicted_index,
        )

        print(
            f"[PREDICT] gradcam complete | "
            f"time={time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        self._log_memory("after gradcam")

        # --------------------------------------------------
        # 4. Combined result
        # --------------------------------------------------

        # The frontend composites both masks over the classifier's padded
        # square view; map the U-Net mask into that same coordinate space.
        aligned_mask = align_segmentation_mask_to_classifier_view(
            segmentation["mask"], source_size=image.size
        )

        result = {
            "disease": classification["disease"],
            "confidence": classification["confidence"],
            "class_index": classification["class_index"],
            "uncertain": classification["uncertain"],
            "mask_coverage": segmentation["coverage"],
            "segmentation_mask": aligned_mask.numpy(),
            "gradcam": cam,
        }

        print(
            f"[PREDICT] result prepared | "
            f"time={time.perf_counter() - start_time:.2f}s",
            flush=True,
        )
        self._log_memory("final")

        return result


    def close(self):
        """
        Remove Grad-CAM hooks.
        """

        self.gradcam.close()