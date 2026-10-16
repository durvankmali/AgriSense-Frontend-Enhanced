# AgriSense segmentation correction

## Root cause

The U-Net was trained in `notebooks/07_segmentation_baseline.ipynb` with:
- `image.resize((224,224), BILINEAR)` (direct stretch)
- Tensor in `[0, 1]` (`image / 255.0`)
- **No** ImageNet mean/std normalization

The former production pipeline applied the **classifier** preprocessing to
U-Net as well: resize-and-pad and ImageNet normalization. This is a training /
inference mismatch, so masks can be small and misplaced.

## What's fixed

- Separate preprocessors for classifier and segmenter.
- ResNet preprocessing (padded, ImageNet normalization) remains unchanged.
- U-Net preprocessing exactly matches its training recipe.
- Mask is re-letterboxed to align with the original image in the browser.
- API format and Gemini Guide remain unchanged.
- Display text explicitly identifies masks as tentative estimates.

## Important limitations

This fix is **not** a model retrain. The trained segmentation weights remain
unchanged. Segmentation may still be wrong for out-of-distribution images,
including object/background confusion. Validate using labeled disease masks
before making strong accuracy claims.

## Run

Use `python -m uvicorn app.main:app --reload` in the extracted project root.
`GEMINI_API_KEY` is optional and used for text explanations only.
