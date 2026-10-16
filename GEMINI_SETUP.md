# AgriSense — Gemini Free API Setup (Windows / VS Code)

This version uses Google Gemini to **explain** your existing ResNet-18 / U-Net image prediction in simple language. Gemini does not run the crop classifier and does not receive the uploaded image.

## 1. Create a key

Open https://aistudio.google.com/app/apikey and create a Gemini API key. Do not share it in screenshots, the frontend JavaScript, or GitHub.

## 2. Install and run locally (PowerShell in extracted project folder)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_KEY"
python -m uvicorn app.main:app --reload
```

You must set `GEMINI_API_KEY` in **the same terminal** before launching the backend. Open http://127.0.0.1:8000, upload a leaf photo, analyze, and ask the AgriSense Guide a question. When Gemini answers, the results display **GEMINI AI EXPLANATION**.

Optional model selection in PowerShell (before starting Uvicorn):

```powershell
$env:AGRISENSE_AI_MODEL = "gemini-3.8-flash"
$env:AGRISENSE_AI_MAX_CALLS_PER_HOUR = "30"
```

The default model is `gemini-3.8-flash`. Free-tier models and access change over time. In AI Studio check which models and quotas your key can use; you can set `AGRISENSE_AI_MODEL` to another accessible model, e.g. `gemini-2.5-flash` if enabled for your project.

**No key?** The app still works with rule-based educational explanations labeled **BUILT-IN GUIDE**. Prediction, heatmaps, and masks continue working normally.

## 3. Render deployment

1. Upload/commit this updated project to your backend's repository (excluding `.venv` and secrets).
2. Render dashboard → your **backend Web Service** → **Environment** → Add `GEMINI_API_KEY` with the secret value.
3. Optional: `AGRISENSE_AI_MODEL` = `gemini-3.8-flash`.
4. Redeploy the **backend**. If you host the frontend separately, deploy updated frontend files too.
5. Never set `GEMINI_API_KEY` in static hosting or client-side JS.

## 4. Troubleshooting

- **BUILT-IN GUIDE appears**: The key is missing, Gemini failed, the model is inaccessible, or a quota/call cap was reached. The backend `/api/assistant` response includes a `note` indicating the general reason.
- **429 / RESOURCE_EXHAUSTED**: Your Gemini rate/quota limit was reached. Wait until it resets, reduce requests, or review limits in AI Studio. The exhibition guide falls back automatically.
- **401 / 403**: Check that the API key is valid, unrestricted for the Generative Language API, and supported in your region.
- **404**: Check the model ID and account access in AI Studio; change `AGRISENSE_AI_MODEL` to one available to your project.
- **Gemini is slow**: Questions fall back to the built-in guide if a request fails or times out.

**Free tier limitations:** Free tier use is limited and Google may use submitted API content to improve its products. Don't type confidential information in the chat. AgriSense sends only the prediction label/scores and question — not the image — to Gemini.

Google references: https://ai.google.dev/gemini-api/docs/pricing and https://ai.google.dev/gemini-api/docs/rate-limits
