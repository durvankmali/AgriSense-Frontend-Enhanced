from pydantic import BaseModel


class PredictionResponse(BaseModel):
    disease: str
    confidence: float
    class_index: int
    uncertain: bool
    mask_coverage: float
    segmentation_mask: str
    gradcam: str