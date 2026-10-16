/* AgriSense Exhibition Edition — resilient upload, analysis, and evidence viewer. */

// FastAPI now serves the frontend and API from the same origin when possible.
// Legacy external frontend and VS Code Live Server are supported as fallbacks.
const configuredBase = document.querySelector('meta[name="agrisense-api-base"]')?.content.trim();
const hostname = window.location.hostname;
const isLocal = !hostname || ["localhost", "127.0.0.1"].includes(hostname);
const API_BASE = (configuredBase || (
    isLocal ? ((window.location.port === "8000") ? window.location.origin : "http://127.0.0.1:8000")
        : hostname === "agrisense-rcwy.onrender.com" ? "https://agrisense-api-9bzp.onrender.com"
            : window.location.origin
)).replace(/\/$/, "");
const API_URL = `${API_BASE}/predict`;
const MAX_FILE_SIZE = 10 * 1024 * 1024;

const $ = (id) => document.getElementById(id);
const imageInput = $("imageInput");
const browseButton = $("browseButton");
const dropZone = $("dropZone");
const uploadPlaceholder = $("uploadPlaceholder");
const previewContainer = $("previewContainer");
const imagePreview = $("imagePreview");
const removeImage = $("removeImage");
const fileName = $("fileName");
const analyzeButton = $("analyzeButton");
const analyzeText = $("analyzeText");
const buttonSpinner = $("buttonSpinner");
const resultsSection = $("resultsSection");
const errorMessage = $("errorMessage");
const errorText = $("errorText");
const closeError = $("closeError");
const newAnalysisButton = $("newAnalysisButton");
const diseaseName = $("diseaseName");
const resultLabel = $("resultLabel");
const confidenceValue = $("confidenceValue");
const confidenceFill = $("confidenceFill");
const classIndex = $("classIndex");
const maskCoverage = $("maskCoverage");
const imageDimensions = $("imageDimensions");
const originalResultImage = $("originalResultImage");
const segmentationImage = $("segmentationImage");
const gradcamImage = $("gradcamImage");
const statusButton = $("statusButton");
const statusText = $("statusText");
const analysisProgress = $("analysisProgress");
const progressText = $("progressText");
const compareCanvas = $("compareCanvas");
const showMask = $("showMask");
const showCam = $("showCam");
const overlayOpacity = $("overlayOpacity");
const opacityValue = $("opacityValue");
const overlayLegend = $("overlayLegend");

let selectedFile = null;
let selectionId = 0;
let activeController = null;
let busy = false;
let visualImages = null;
let visualMode = "mask";
let waitMessageTimer = null;
let currentPrediction = null;
let guideSummaryController = null;
let guideQuestionController = null;
let guideRequestCounter = 0;
let guideBusy = false;
const guideHeadline = $("guideHeadline");
const guideSummary = $("guideSummary");
const guideModeBadge = $("guideModeBadge");
const guideServiceNote = $("guideServiceNote");
const guideMessages = $("guideMessages");
const guideEmpty = $("guideEmpty");
const guideInput = $("guideInput");
const guideForm = $("guideForm");
const guideSend = $("guideSend");
const guideChips = [...document.querySelectorAll("[data-guide-question]")];

function setServerState(state, message) {
    statusButton.classList.remove("status-ready", "status-checking", "status-offline");
    statusButton.classList.add(`status-${state}`);
    statusText.textContent = message;
    statusButton.setAttribute("aria-label", `${message}. Click to check connection again.`);
}

async function checkHealth() {
    setServerState("checking", "Connecting to AI...");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 9000);
    try {
        const response = await fetch(`${API_BASE}/health`, { signal: controller.signal, cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const health = await response.json();
        if (health.status !== "healthy" || health.models_loaded !== true) throw new Error("Models not available");
        setServerState("ready", `AI online · ${String(health.device || "CPU").toUpperCase()}`);
    } catch (_) {
        // A cold host may take longer than this probe; don't claim the model is broken.
        setServerState("offline", "AI waking / unavailable");
    } finally {
        clearTimeout(timeout);
    }
}
statusButton.addEventListener("click", checkHealth);

browseButton.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", (event) => {
    if (event.target.files && event.target.files[0]) handleFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-over");
    });
});
["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
    });
});
dropZone.addEventListener("drop", (event) => {
    if (event.dataTransfer?.files?.length) handleFile(event.dataTransfer.files[0]);
});

function stopRequest() {
    if (activeController) {
        activeController.abort();
        activeController = null;
    }
    clearTimeout(waitMessageTimer);
    setLoading(false);
}

function handleFile(file) {
    hideError();
    if (!["image/jpeg", "image/png"].includes(file.type)) {
        showError("Please choose a JPG or PNG image.");
        return;
    }
    if (!file.size || file.size > MAX_FILE_SIZE) {
        showError("The image must be under 10 MB and cannot be empty.");
        return;
    }

    // Prevent the previous request or FileReader from changing this selection.
    ++selectionId;
    stopRequest();
    clearResults();
    selectedFile = file;
    fileName.textContent = file.name;
    uploadPlaceholder.classList.add("hidden");
    previewContainer.classList.remove("hidden");
    analyzeButton.disabled = false;

    const thisSelection = selectionId;
    const reader = new FileReader();
    reader.onload = (event) => {
        if (thisSelection !== selectionId) return;
        imagePreview.src = event.target.result;
        originalResultImage.src = event.target.result;
    };
    reader.onerror = () => { if (thisSelection === selectionId) showError("Unable to preview the image."); };
    reader.readAsDataURL(file);
}

imagePreview.addEventListener("load", () => {
    imageDimensions.textContent = `${imagePreview.naturalWidth} × ${imagePreview.naturalHeight}`;
});
removeImage.addEventListener("click", (event) => { event.stopPropagation(); resetUpload(); });

function clearResults() {
    resultsSection.classList.add("hidden");
    visualImages = null;
    segmentationImage.removeAttribute("src");
    gradcamImage.removeAttribute("src");
    resetGuide();
}
function resetUpload() {
    ++selectionId;
    stopRequest();
    selectedFile = null;
    imageInput.value = ""; // Also allows choosing the same file again.
    imagePreview.removeAttribute("src");
    originalResultImage.removeAttribute("src");
    fileName.textContent = "Image selected";
    imageDimensions.textContent = "—";
    uploadPlaceholder.classList.remove("hidden");
    previewContainer.classList.add("hidden");
    analyzeButton.disabled = true;
    clearResults();
    hideError();
}

function setLoading(loading) {
    busy = loading;
    analyzeButton.disabled = loading || !selectedFile;
    buttonSpinner.classList.toggle("hidden", !loading);
    analysisProgress.classList.toggle("hidden", !loading);
    analyzeText.textContent = loading ? "Analyzing plant..." : "Analyze Image";
    if (!loading) clearTimeout(waitMessageTimer);
}

analyzeButton.addEventListener("click", analyzeImage);
async function analyzeImage() {
    if (!selectedFile || busy) return;
    hideError();
    clearResults();
    setLoading(true);
    progressText.textContent = "Processing your image with the AgriSense models...";

    const mySelection = selectionId;
    const controller = new AbortController();
    activeController = controller;
    // Render's free service can take time to wake; allow a generous deadline.
    const deadline = setTimeout(() => controller.abort(), 120000);
    waitMessageTimer = setTimeout(() => {
        if (mySelection === selectionId && busy) {
            progressText.textContent = "Still working. A cloud-hosted AI server may need time to wake up...";
        }
    }, 7000);
    const form = new FormData();
    form.append("file", selectedFile, selectedFile.name);

    try {
        const response = await fetch(API_URL, { method: "POST", body: form, signal: controller.signal });
        if (!response.ok) {
            let message = `The server returned HTTP ${response.status}.`;
            try {
                const error = await response.json();
                if (typeof error.detail === "string") message = error.detail;
            } catch (_) { /* Server can return non-JSON during a restart. */ }
            if ([502, 503, 504].includes(response.status)) {
                message = "The cloud AI server is temporarily unavailable or restarting. Retry after it comes online.";
            }
            throw new Error(message);
        }
        const result = await response.json();
        if (mySelection !== selectionId) return;
        if (!result.disease || !result.segmentation_mask || !result.gradcam) {
            throw new Error("The API returned an incomplete result. Please retry.");
        }
        displayResults(result);
        setServerState("ready", "AI online · ready");
    } catch (error) {
        if (mySelection !== selectionId) return; // User replaced the image intentionally.
        console.error("AgriSense request error:", error);
        showError(getReadableError(error));
        if (error instanceof TypeError) checkHealth();
    } finally {
        clearTimeout(deadline);
        if (activeController === controller) activeController = null;
        if (mySelection === selectionId) setLoading(false);
    }
}

function displayResults(result) {
    const confidence = Math.max(0, Math.min(100, Number(result.confidence) * 100));
    const coverage = Math.max(0, Math.min(100, Number(result.mask_coverage) * 100));
    resultLabel.textContent = result.uncertain ? "LOW-CONFIDENCE MODEL PREDICTION" : "MODEL PREDICTION";
    diseaseName.textContent = formatDiseaseName(result.disease);
    confidenceValue.textContent = confidence.toFixed(1);
    classIndex.textContent = String(result.class_index);
    maskCoverage.textContent = `${coverage.toFixed(1)}%`;
    confidenceFill.style.width = `${confidence}%`;
    segmentationImage.src = `data:image/png;base64,${result.segmentation_mask}`;
    gradcamImage.src = `data:image/png;base64,${result.gradcam}`;
    resultsSection.classList.remove("hidden");
    showGuideForPrediction(result);
    renderInteractiveMap(selectionId);
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function formatDiseaseName(name) {
    return String(name || "Unknown").replaceAll("_", " ").replace(/\s+/g, " ").trim()
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });
}
async function renderInteractiveMap(mySelection) {
    try {
        const images = await Promise.all([
            loadImage(originalResultImage.src),
            loadImage(segmentationImage.src),
            loadImage(gradcamImage.src),
        ]);
        if (mySelection !== selectionId) return;
        visualImages = { original: images[0], mask: images[1], cam: images[2] };
        drawComposite();
    } catch (_) {
        if (mySelection === selectionId) {
            visualImages = null;
            console.warn("Unable to render overlay; raw model maps remain available below.");
        }
    }
}

function drawComposite() {
    if (!visualImages) return;
    const { original, mask, cam } = visualImages;
    const width = compareCanvas.width;
    const height = compareCanvas.height;
    const ctx = compareCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, width, height);

    // Match the API's resize_and_pad: a centered square, NOT a stretched crop.
    const scale = Math.min(width / original.naturalWidth, height / original.naturalHeight);
    const displayWidth = Math.round(original.naturalWidth * scale);
    const displayHeight = Math.round(original.naturalHeight * scale);
    const left = Math.floor((width - displayWidth) / 2);
    const top = Math.floor((height - displayHeight) / 2);
    ctx.drawImage(original, left, top, displayWidth, displayHeight);

    const source = visualMode === "mask" ? mask : cam;
    const temp = document.createElement("canvas");
    temp.width = width;
    temp.height = height;
    const tctx = temp.getContext("2d", { willReadFrequently: true });
    tctx.drawImage(source, 0, 0, width, height);
    const pixels = tctx.getImageData(0, 0, width, height);
    const opacity = Number(overlayOpacity.value) / 100;
    for (let i = 0; i < pixels.data.length; i += 4) {
        const v = pixels.data[i] / 255;
        if (visualMode === "mask") {
            pixels.data[i] = 255;
            pixels.data[i + 1] = 86;
            pixels.data[i + 2] = 65;
            pixels.data[i + 3] = v > .5 ? 255 * opacity : 0;
        } else {
            pixels.data[i] = 255;
            pixels.data[i + 1] = Math.round(175 - 100 * v);
            pixels.data[i + 2] = 40;
            pixels.data[i + 3] = Math.round(255 * opacity * v);
        }
    }
    tctx.putImageData(pixels, 0, 0);
    ctx.drawImage(temp, 0, 0);
}

function setVisualMode(nextMode) {
    visualMode = nextMode;
    showMask.classList.toggle("active", nextMode === "mask");
    showCam.classList.toggle("active", nextMode === "cam");
    showMask.setAttribute("aria-pressed", String(nextMode === "mask"));
    showCam.setAttribute("aria-pressed", String(nextMode === "cam"));
    overlayLegend.innerHTML = nextMode === "mask"
        ? '<i class="legend-dot legend-mask"></i> ESTIMATED REGION'
        : '<i class="legend-dot legend-cam"></i> AI ATTENTION';
    drawComposite();
}
showMask.addEventListener("click", () => setVisualMode("mask"));
showCam.addEventListener("click", () => setVisualMode("cam"));
overlayOpacity.addEventListener("input", () => {
    opacityValue.textContent = `${overlayOpacity.value}%`;
    drawComposite();
});

function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove("hidden");
}
function hideError() { errorMessage.classList.add("hidden"); }
closeError.addEventListener("click", hideError);
function getReadableError(error) {
    if (error.name === "AbortError") {
        return "The request was interrupted or took too long. Check your AI server status and try again.";
    }
    if (error instanceof TypeError) {
        return "Could not connect to the AI server. The host may be sleeping, restarting, or blocked by the network. Check the server status above, then retry.";
    }
    return error.message || "An unexpected error occurred. Please retry.";
}
newAnalysisButton.addEventListener("click", () => {
    resetUpload();
    document.querySelector(".workspace").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("printReportButton").addEventListener("click", () => window.print());

checkHealth();


/* ==================== AI EXPLANATION LAYER ====================
 * Classification, mask and CAM are generated by the existing vision models.
 * Assistant receives ONLY the returned label/scores and visitor text.
 * It cannot see the uploaded photo. Never place API keys in browser code.
 */
function resetGuide() {
    ++guideRequestCounter;
    guideSummaryController?.abort();
    guideQuestionController?.abort();
    guideSummaryController = null;
    guideQuestionController = null;
    currentPrediction = null;
    guideBusy = false;
    guideSummary.textContent = "An explanation of your result will appear here.";
    guideHeadline.textContent = "Here is what your scan suggests";
    guideMessages.replaceChildren();
    guideEmpty.classList.remove("hidden");
    guideMessages.appendChild(guideEmpty);
    guideInput.value = "";
    toggleGuideBusy(false);
    setGuideMode("instant");
}
function toggleGuideBusy(isBusy) {
    guideBusy = isBusy;
    guideInput.disabled = isBusy;
    guideSend.disabled = isBusy;
    guideSend.textContent = isBusy ? "Thinking..." : "Ask Guide ↗";
    guideChips.forEach(button => { button.disabled = isBusy; });
}
function setGuideMode(mode, note = "") {
    const type = mode === "ai" ? "GEMINI AI EXPLANATION" :
        mode === "guide" ? "BUILT-IN GUIDE" : "INSTANT EXPLANATION";
    guideModeBadge.textContent = type;
    guideModeBadge.classList.toggle("mode-guide", mode !== "ai");
    guideServiceNote.textContent = note;
}
function safePercent(value) {
    const v = Number(value);
    return Number.isFinite(v) ? Math.max(0, Math.min(100, v * 100)).toFixed(0) : "—";
}
function showGuideForPrediction(result) {
    // The instant explanation stays useful even with no connection or LLM key.
    currentPrediction = {
        disease: String(result.disease),
        confidence: Number(result.confidence),
        uncertain: Boolean(result.uncertain),
        mask_coverage: Number(result.mask_coverage),
    };
    const label = formatDiseaseName(result.disease);
    guideHeadline.textContent = `The closest match is ${label}`;
    guideSummary.textContent =
        `AgriSense matched your image to “${label}” with a ${safePercent(result.confidence)}% model score. ` +
        (result.uncertain ? "The match is uncertain, so take extra care interpreting it. " :
            "The model selected this as its leading class, but the condition is not confirmed. ") +
        "The red overlay estimates possibly affected image regions; the warm-colored map shows what influenced the choice. " +
        "The highlighted-area percentage is not a measure of severity. Verify symptoms before acting.";
    setGuideMode("instant");
    // Optional upgrade: generated, result-specific summary. Does not block UI.
    loadGuideSummary(currentPrediction, selectionId, guideRequestCounter);
}
function offlineGuideAnswer(question) {
    const q = question.toLowerCase();
    const label = formatDiseaseName(currentPrediction?.disease || "the suggested condition");
    if (/confiden|sure|certain|accur|trust|probab/.test(q)) {
        return `The model scored ${label} at ${safePercent(currentPrediction.confidence)}%. This is a model preference score, not the chance of a confirmed diagnosis. Check the plant in person.`;
    }
    if (/sever|serious|spread|danger|how bad/.test(q)) {
        return "We cannot measure disease severity from this photo. The highlighted area is a segmentation estimate, not the amount of actual plant damage. A local agricultural expert can help assess the plant.";
    }
    if (/color|red|orange|map|mask|cam|highlight|region|area/.test(q)) {
        return "Red marks a region the segmentation model predicts may be affected. The warm-colored attention map shows which parts influenced the classifier. Neither is a confirmed disease boundary.";
    }
    if (/next|what should|treat|spray|pesticide|care|cure|do now|action/.test(q)) {
        return `Inspect the plant in good light and check nearby plants. AgriSense suggested ${label}, but avoid applying a treatment based only on this photo. Consult an agricultural extension worker for a confirmed diagnosis and locally appropriate advice.`;
    }
    return `The trained model's closest match is ${label} (${safePercent(currentPrediction.confidence)}% model score). This is not a verified diagnosis. The maps show estimated affected regions and visual attention. Check the real plant before deciding what to do.`;
}
async function requestGuide(question, controller) {
    const response = await fetch(`${API_BASE}/api/assistant`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...currentPrediction, question }),
        signal: controller.signal,
    });
    if (!response.ok) throw new Error(`AI explainer unavailable (${response.status})`);
    const data = await response.json();
    if (typeof data.answer !== "string" || !data.answer.trim()) throw new Error("Empty answer");
    return data;
}
async function loadGuideSummary(prediction, mySelection, myCounter) {
    const controller = new AbortController();
    guideSummaryController = controller;
    const deadline = setTimeout(() => controller.abort(), 28000);
    try {
        const data = await requestGuide("Explain this scan in simple words to a visitor, including uncertainty and one sensible next step.", controller);
        if (mySelection !== selectionId || myCounter !== guideRequestCounter || currentPrediction !== prediction) return;
        guideSummary.textContent = data.answer;
        setGuideMode(data.mode, data.note || "");
    } catch (_) {
        // Keep the already rendered, honest explanation if the optional service fails.
    } finally {
        clearTimeout(deadline);
        if (guideSummaryController === controller) guideSummaryController = null;
    }
}
function appendGuideMessage(role, message, mode) {
    guideEmpty.classList.add("hidden");
    const bubble = document.createElement("div");
    bubble.className = `guide-message ${role}`;
    const author = document.createElement("span");
    author.className = "guide-message-role";
    author.textContent = role === "visitor" ? "YOU" : (mode === "ai" ? "AGRISENSE · GEMINI AI" : "AGRISENSE · BUILT-IN GUIDE");
    const body = document.createElement("span");
    body.textContent = message;
    bubble.append(author, body);
    guideMessages.appendChild(bubble);
    guideMessages.scrollTop = guideMessages.scrollHeight;
    return bubble;
}
async function askGuide(question) {
    question = String(question || "").trim().slice(0, 400);
    if (!question || !currentPrediction || guideBusy) return;
    guideInput.value = "";
    appendGuideMessage("visitor", question);
    toggleGuideBusy(true);
    const waiting = appendGuideMessage("assistant", "Explaining your result...", "guide");
    const mySelection = selectionId;
    const myCounter = guideRequestCounter;
    const controller = new AbortController();
    guideQuestionController = controller;
    const deadline = setTimeout(() => controller.abort(), 28000);
    try {
        const data = await requestGuide(question, controller);
        if (mySelection !== selectionId || myCounter !== guideRequestCounter) return;
        waiting.remove();
        appendGuideMessage("assistant", data.answer, data.mode);
        setGuideMode(data.mode, data.note || "");
    } catch (_) {
        if (mySelection !== selectionId || myCounter !== guideRequestCounter) return;
        waiting.remove();
        appendGuideMessage("assistant", offlineGuideAnswer(question), "guide");
        setGuideMode("guide", "Connection unavailable. Showing the built-in guide.");
    } finally {
        clearTimeout(deadline);
        if (guideQuestionController === controller) guideQuestionController = null;
        if (mySelection === selectionId && myCounter === guideRequestCounter) toggleGuideBusy(false);
    }
}
guideChips.forEach(button => {
    button.addEventListener("click", () => askGuide(button.dataset.guideQuestion));
});
guideForm.addEventListener("submit", event => {
    event.preventDefault();
    askGuide(guideInput.value);
});
