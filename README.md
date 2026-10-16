# AgriSense — Crop Health Intelligence Platform

AgriSense is an end-to-end computer vision platform for crop disease identification and disease-region localization from plant images.

The project combines a fine-tuned ResNet-18 classifier, a lightweight U-Net segmentation model, Grad-CAM explainability, and a FastAPI inference backend with a web-based frontend.

## Project Overview

## 🎥 Application Demo

Watch AgriSense in action, including crop disease
classification, confidence scores, disease-region
segmentation, and explainability visualizations.

### Full Application Walkthrough

https://github.com/durvankmali/AgriSense-Frontend-Enhanced/blob/main/Demo/Agrisense_Demo_Video.mp4

## Application Screenshots

### Application Homepage 
![AgriSense Homepage](Demo/images/homepage.png)


## Disease Prediction | Grad-CAM | Disease Segmentation |   Visualization
![Disease Prediction](Demo/images/disease_prediction.png)


### Problem

Plant disease identification from images is challenging because real-world images can contain different resolutions, aspect ratios, backgrounds, lighting conditions, plant species, and disease appearances.

AgriSense addresses this by building a complete image-based pipeline that:

1. Identifies the disease from a plant image.
2. Localizes the predicted disease region.
3. Provides model confidence.
4. Generates a Grad-CAM explanation showing image regions influencing the classification.
5. Exposes the complete inference pipeline through an API and web interface.

## Key Features

- Multi-class plant disease classification across **115 disease classes**
- Fine-tuned **ResNet-18** classifier
- Lightweight **U-Net** for disease-region segmentation
- **Grad-CAM** visual explanations
- Confidence-based uncertainty indication
- Aspect-ratio-preserving image preprocessing
- Duplicate and conflicting-label detection
- Cross-split leakage control
- FastAPI backend
- Interactive browser frontend
- Dockerized deployment
- Git/GitHub version-controlled workflow

## System Architecture

```text
                         Uploaded Plant Image
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             Image Preprocessing        Image Preprocessing
                    |                           |
                    v                           v
             ResNet-18 Classifier       U-Net Segmentation
                    |                           |
                    v                           v
             Disease Prediction          Disease Region
             + Confidence                + Mask Coverage
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                         Grad-CAM Explanation
                                  |
                                  v
                    Explainable Prediction Result
```

The production inference pipeline returns:

- predicted disease
- confidence score
- class index
- uncertainty flag
- segmentation mask
- mask coverage
- Grad-CAM visualization

## Dataset

AgriSense uses the PlantSeg dataset.

### Original dataset

| Property | Value |
|---|---:|
| Images | 7,774 |
| Plants | 34 |
| Diseases | 115 |
| Missing metadata values | 0 |

The original dataset contained duplicate image contents and conflicting disease labels, so data-quality checks were performed before model development.

### Clean dataset

The final modeling dataset contains:

| Property | Value |
|---|---:|
| Records | 7,399 |
| Unique image contents | 7,399 |
| Training | 5,084 |
| Validation | 811 |
| Test | 1,504 |
| Duplicate image contents | 0 |
| Cross-split duplicate contents | 0 |

### Data quality processing

The cleaning workflow included:

- duplicate-content detection using image hashes
- conflicting-label detection
- removal of records associated with conflicting image contents
- cross-split leakage checks
- metadata and filesystem consistency checks
- image integrity checks
- image-size and aspect-ratio analysis
- plant-label validation

A total of **98 conflicting image contents**, corresponding to **205 records**, were removed.

The final dataset contains no duplicate image contents and no cross-split duplicate contents.

## Image Preprocessing

The final classifier and production inference pipeline use:

- RGB conversion
- resize while preserving aspect ratio
- padding to **224 × 224**
- black padding
- BILINEAR interpolation
- ImageNet normalization

Training additionally uses:

- random horizontal flipping
- random rotation up to 10 degrees

This approach avoids indiscriminate cropping of heterogeneous plant images.

## Disease Classification

### Model

**ResNet-18**, initialized from the torchvision architecture and fine-tuned for the 115 disease classes.

Training configuration:

- Trainable layers: `layer4` and final classification layer
- Optimizer: Adam
- Learning rate: `0.0001`
- Scheduler: ReduceLROnPlateau
- Early stopping
- Batch size: 16
- Training epochs: 10

The final checkpoint is:

```text
models/checkpoints/resnet18_extended_finetuned_latest.pth
```

### Final test performance

The authoritative test evaluation after aligning production preprocessing with the training/evaluation pipeline:

| Metric | Final Model |
|---|---:|
| Accuracy | **62.70%** |
| Macro F1 | **51.48%** |
| Weighted F1 | **61.24%** |
| Top-5 Accuracy | **88.50%** |

### Baseline comparison

| Metric | Baseline | Final Model | Improvement |
|---|---:|---:|---:|
| Accuracy | 44.02% | 62.70% | +18.68 pp |
| Macro F1 | 33.99% | 51.48% | +17.49 pp |
| Weighted F1 | 42.37% | 61.24% | +18.87 pp |

The final model substantially improved over the baseline while retaining the original multi-class problem formulation.

## Class Imbalance Experiment

A class-weighted training experiment was also evaluated.

Results:

- Accuracy: 42.09%
- Macro F1: 33.68%
- Weighted F1: 41.62%

Because this performed substantially worse than the selected model, class weighting was not used in the final classifier.

## Error Analysis

The final test set contained **561 incorrect predictions**.

Of these:

- Same-plant errors: 153 (27.27%)
- Cross-plant errors: 408 (72.73%)

This indicates that a large proportion of errors occur across different plant categories rather than only between diseases affecting the same plant.

The relationship between class support and F1 score was moderately positive:

- Pearson correlation: 0.468
- Spearman correlation: 0.488

This analysis was used to understand model behavior beyond overall accuracy.

## Disease-Region Segmentation

AgriSense uses a lightweight U-Net to estimate the disease region in an image.

### Model

- Encoder channels: 3 → 32 → 64 → 128
- Bottleneck: 256
- Decoder with skip connections
- Output: single-channel disease probability map
- Parameters: 1,928,417
- Loss: BCE + Dice
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Training epochs: 5

Checkpoint:

```text
models/checkpoints/unet_segmentation_best.pth
```

### Final test performance

At the calibrated threshold of **0.4**:

| Metric | Score |
|---|---:|
| Dice | **0.5233** |
| IoU | **0.3971** |
| Pixel Accuracy | **0.8198** |

The threshold was selected using validation-set Dice performance.

The segmentation model provides useful disease-region localization, although small and thin lesions can be missed and some predictions extend beyond the annotated region.

**Mask coverage should not be interpreted as disease severity.**

## Explainability with Grad-CAM

Grad-CAM is applied to the final convolutional block of ResNet-18.

The generated heatmap provides a coarse visual indication of image regions that influenced the predicted class.

A qualitative sample-level check showed that Grad-CAM can overlap meaningfully with annotated disease regions, but Grad-CAM is treated as an explanation mechanism rather than proof that the classifier has learned the true disease boundary.

## Confidence and Uncertainty

The production classifier uses a confidence threshold of **0.65**.

Predictions below this threshold are marked:

```text
uncertain = true
```

Validation analysis:

- Mean confidence: 66.13%
- Validation accuracy: 62.15%
- Predictions accepted at 0.65: 54.62%
- Accuracy among accepted validation predictions: 81.04%

The uncertainty flag is intended to communicate lower-confidence predictions. It is not a guarantee that a low-confidence prediction is incorrect.

## End-to-End Inference

```text
Upload Image
      |
      v
Validate File
      |
      v
Preprocess to 224x224
      |
      +--------------------+
      |                    |
      v                    v
ResNet-18              U-Net
      |                    |
      v                    v
Disease + Confidence   Disease Mask
      |                    |
      +----------+---------+
                 |
                 v
              Grad-CAM
                 |
                 v
          Prediction Result
```

## Backend

The application backend is implemented using **FastAPI**.

### Main endpoints

```text
GET  /health
POST /predict
GET  /docs
```

`/health` reports backend and model-loading status.

`/predict` accepts a plant image and returns the complete prediction result.

The FastAPI application also validates uploaded files, including supported image types, empty uploads, corrupted images, and file-size limits.

### Example response structure

```json
{
  "disease": "apple black rot",
  "confidence": 0.9739,
  "class_index": 0,
  "uncertain": false,
  "mask_coverage": 0.0141,
  "segmentation_mask": "...",
  "gradcam": "..."
}
```

## Frontend

The frontend is a responsive browser-based interface containing:

- image upload and drag-and-drop support
- image preview
- analysis/loading state
- prediction result
- confidence display
- disease-region visualization
- Grad-CAM visualization
- interpretation of the prediction
- uncertainty indication
- error handling

Frontend structure:

```text
frontend/
├── index.html
├── style.css
└── script.js
```

## Project Structure

```text
AgriSense/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   │   └── PlantSeg/
│   │       ├── images/
│   │       └── annotations/
│   └── processed/
│       └── agrisense_metadata_clean.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   └── checkpoints/
│       ├── resnet18_extended_finetuned_latest.pth
│       └── unet_segmentation_best.pth
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── ...
│   └── 11_*.ipynb
│
├── src/
│   ├── config.py
│   ├── data_analysis.py
│   ├── preprocessing/
│   │   └── image_preprocessing.py
│   ├── models/
│   │   ├── classifier.py
│   │   └── segmentation.py
│   ├── inference/
│   │   └── pipeline.py
│   └── explainability/
│       └── gradcam.py
│
├── .dockerignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/durvankmali/AgriSense.git
cd AgriSense
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The project is configured for CPU inference when CUDA is unavailable.

### 4. Start the FastAPI server

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the frontend

The frontend can be served locally using a simple static HTTP server or a VS Code Live Server setup.

The local frontend is configured to communicate with the FastAPI backend.

## Docker

The project includes a CPU-based Docker configuration.

Build:

```bash
docker build -t agrisense-api .
```

Run:

```bash
docker run --rm -p 8000:8000 agrisense-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Deployment

The project was prepared for deployment using Render:

- FastAPI backend deployed as a Docker Web Service
- frontend deployed as a Render Static Site
- model checkpoints managed with Git LFS
- health endpoint configured for service monitoring

The public deployment was useful for validating the complete deployment workflow.

### Deployment limitation

The free backend environment has limited memory. The full inference pipeline includes classification, segmentation, and Grad-CAM, which creates a significant memory footprint.

The public backend successfully returned predictions, but repeated inference can exceed the available free-service resources and cause the service process to restart.

For demonstrations, the **local deployment is the authoritative environment** because it provides stable execution of the complete pipeline.

## Limitations

1. Classification performance is not sufficient for autonomous agricultural diagnosis.
2. Some disease classes have limited samples.
3. Class imbalance remains present in the dataset.
4. Small and thin disease regions can be missed by segmentation.
5. Grad-CAM provides coarse visual explanations rather than precise disease boundaries.
6. Segmentation mask coverage is not a validated disease-severity measurement.
7. The PlantSeg dataset does not provide geolocation, so location-specific recommendations are not currently part of the implemented system.
8. Environmental/contextual information is not yet integrated into the prediction pipeline.
9. Public free-tier deployment is constrained by memory and service lifecycle limits.

## Future Scope

Potential extensions include:

- improved fine-tuning and class-specific training strategies
- stronger segmentation architectures
- richer disease localization metrics
- calibrated confidence estimates
- environmental and contextual data integration
- weather and location-aware risk analysis when appropriate data is available
- agricultural advisory integration using trusted external sources
- SQL-based prediction and analytics storage
- dashboard-based historical analysis
- improved cloud infrastructure for production inference
- model monitoring and performance tracking

These are future extensions and are not represented as currently implemented features.

## Development Workflow

The project was developed using a version-controlled workflow with Git and GitHub.

Major development stages included:

1. Dataset exploration
2. Data-quality analysis
3. Duplicate and leakage detection
4. Dataset cleaning
5. Preprocessing design
6. Baseline classification
7. Fine-tuning
8. Classification evaluation
9. Error analysis
10. Segmentation
11. Grad-CAM explainability
12. FastAPI integration
13. Frontend integration
14. Dockerization
15. Deployment testing
16. Resource optimization

Model checkpoints are stored separately from the source-code logic and tracked using Git LFS.

## Final Verification Summary

| Component | Status |
|---|---|
| Clean dataset | Complete |
| Duplicate/leakage checks | Complete |
| ResNet-18 classifier | Complete |
| U-Net segmentation | Complete |
| Grad-CAM | Complete |
| Confidence/uncertainty logic | Complete |
| FastAPI backend | Complete |
| Frontend | Complete |
| Docker | Complete |
| Local end-to-end inference | Verified |
| Public deployment experiment | Verified with resource limitations |

## License and Dataset

The project uses the PlantSeg dataset. Refer to the original dataset source and license information before redistribution of dataset images or annotations.

---

**AgriSense** is designed as an end-to-end applied computer vision project demonstrating the complete path from messy image data and quality control to model development, evaluation, explainability, API integration, and deployment.

## AI Guide — Client-friendly explanations (Exhibition Edition)

This edition adds an **AgriSense Guide** directly to the results screen:

- An immediate plain-language explanation of the detected class and its model confidence score.
- Explicit explanations of the red segmentation overlay, warm Grad-CAM attention map, and the limitations of each.
- Four visitor-friendly one-tap questions, plus a free-text question box.
- A new `POST /api/assistant` endpoint that explains a prediction **without running the vision models again**.
- A reliable **built-in guide** when the language-model API is not configured, is slow, has reached its free-tier quota or the exhibition call cap, or is offline.
- **Optional generative-AI answers** using the Google Gemini API, with the key held only by the FastAPI server.

### Using the guide without an API key

No configuration is needed. Run FastAPI normally, upload a JPG/PNG image, and read the explanation and suggested Q&A. The badge displays **BUILT-IN GUIDE** for the rule-based responses. This is intentionally not described as a live generative-AI model.

### Enabling generated answers

**1.** Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey). The free tier has usage limits and is subject to availability for your project/region. See `GEMINI_SETUP.md` for quick setup.

**2.** Set an environment variable **on the FastAPI server**, never in `frontend/script.js` or public source control:

PowerShell:

```powershell
$env:GEMINI_API_KEY = "your-secret-key"
$env:AGRISENSE_AI_MODEL = "gemini-3.8-flash"  # optional
$env:AGRISENSE_AI_MAX_CALLS_PER_HOUR = "30"  # optional shared cap
uvicorn app.main:app --reload
```

Ubuntu/WSL:

```bash
export GEMINI_API_KEY="your-secret-key"
export AGRISENSE_AI_MODEL="gemini-3.8-flash" # optional
export AGRISENSE_AI_MAX_CALLS_PER_HOUR="30" # optional
uvicorn app.main:app --reload
```

For Render: in the **backend** Web Service, set `GEMINI_API_KEY` under Environment, then redeploy. Do not add the key to a static frontend service. Check your active Gemini free-tier rate limits in AI Studio before a public demo. The in-process 30 calls/hour cap is a simple safeguard, **not** a distributed quota or guaranteed billing limit (it resets on process restart).

Docker:

```bash
docker run --rm -p 8000:8000 -e GEMINI_API_KEY="your-secret-key" agrisense-api
```

If the free quota is reached, the key is rejected, the selected model is unavailable, or the external provider is down, the interface continues to provide the built-in explanations; other AgriSense features continue working. Generated messages display **GEMINI AI EXPLANATION**. Generated language can be wrong, so validate important claims.

### Assistant API example

`POST /api/assistant` JSON:

```json
{
  "disease": "apple black rot",
  "confidence": 0.78,
  "uncertain": false,
  "mask_coverage": 0.08,
  "question": "What should I do next?"
}
```

The endpoint accepts only recognized trained disease labels and bounded question/score fields. Response includes `answer`, `mode` (`ai` or `guide`), and `note`. The language model receives **only** the submitted predicted disease label, confidence, uncertainty flag, highlighted-area ratio, and visitor's question; it does **not** receive the user's photo, masks, Grad-CAM image, or the model checkpoints. Uploaded photos still go to the AgriSense prediction backend as before.

The assistant is an **educational explanation layer**, not another disease classifier, a validated disease-treatment adviser, or a replacement for an agronomist. The model score is not a confirmed-diagnosis probability; highlighted-area ratio is not disease severity.


### Gemini free-tier details

- Default model: `gemini-3.8-flash` (change via `AGRISENSE_AI_MODEL` if your key has access to a different free-tier model).
- Gemini calls use only the diagnosis label, confidence, uncertainty flag, estimated mask coverage and visitor question; **not the photo**.
- The Gemini API key must be on the **backend**, not in client-side JavaScript or a shared repository.
- Gemini's free tier is limited and free-tier data may be used to improve Google's products. Do not type personal or confidential information in the public exhibition chat.
- If a particular model returns 404 or is unavailable for your new project, select an available free-tier text model in AI Studio, set `AGRISENSE_AI_MODEL`, and restart the backend.
- The hourly 30-call cap is per running backend process and **does not** override Google's API quotas.


## Segmentation input correction (exhibition reliability update)

This edition fixes a concrete inference-versus-training mismatch. ResNet-18
uses padded 224×224 inputs with ImageNet normalization. U-Net was trained on
**directly resized 224×224 RGB images scaled to [0,1], with no ImageNet
normalization**. The deployed version mistakenly fed U-Net the ResNet tensor.
This update constructs the correct U-Net input separately, then maps its mask
into the padded visualization coordinate frame to align with the original image.

The 0.4 U-Net threshold remains unchanged because it was selected using
validation masks. Mask coverage is a fraction of the *original image*, not an
estimate of disease severity; the displayed mask is a tentative region estimate
that can mark background or miss damage. **No weights were retrained**, and
there is no guarantee of disease-level segmentation correctness on unseen
photographic styles. For a trustworthy improvement beyond preprocessing,
evaluate on manually annotated examples and retrain if required.
