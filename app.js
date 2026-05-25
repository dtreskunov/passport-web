// Passport Photo (web) — uses MediaPipe FaceLandmarker (WASM) for landmarks,
// then computes a US-passport-spec square crop from the captured frame.
//
// UX: welcome screen → either camera (tap → 3-2-1 countdown → snap) or
// upload → result screen with crop overlay + download.
// Everything runs in the browser; the camera feed never leaves the page.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── DOM ────────────────────────────────────────────────────────────────────
const welcome       = document.getElementById("welcome");
const cameraScreen  = document.getElementById("camera");
const resultScreen  = document.getElementById("result");
const useCameraBtn  = document.getElementById("useCameraBtn");
const useUploadBtn  = document.getElementById("useUploadBtn");
const fileInput     = document.getElementById("fileInput");
const video         = document.getElementById("video");
const cameraTap     = document.getElementById("cameraTap");
const cameraHint    = document.getElementById("cameraHint");
const countdownEl   = document.getElementById("countdown");
const cameraBackBtn = document.getElementById("cameraBackBtn");
const resultBackBtn = document.getElementById("resultBackBtn");
const capturedCv    = document.getElementById("captured");
const overlayCv     = document.getElementById("overlay");
const redoBtn       = document.getElementById("redoBtn");
const downloadBtn   = document.getElementById("downloadBtn");
const statusEl      = document.getElementById("status");
const headFracIn    = document.getElementById("headFrac");
const eyeFracIn     = document.getElementById("eyeFrac");
const outSizeIn     = document.getElementById("outSize");

// ── State ──────────────────────────────────────────────────────────────────
let landmarker = null;
let stream = null;
let capturedBitmap = null;     // ImageBitmap of the frozen frame
let capturedMirrored = false;  // true when frame came from selfie video (mirrored)
let landmarks = null;          // 478 normalized landmarks for the captured frame
let plan = null;               // {x, y, size, headRatio, eyeFromBottom, warnings}
let cameFrom = null;           // 'camera' | 'upload'
let countdownTimer = null;

// Eye landmarks for the eye line (eyelid + corners on both eyes).
const EYE_LM = [159, 145, 33, 133, 386, 374, 263, 362];

// ── Screen helpers ─────────────────────────────────────────────────────────
function show(name) {
  welcome.hidden      = name !== "welcome";
  cameraScreen.hidden = name !== "camera";
  resultScreen.hidden = name !== "result";
}

function setStatus(msg, level = "") {
  statusEl.textContent = msg;
  statusEl.className = "status" + (level ? " " + level : "");
}

// ── MediaPipe ──────────────────────────────────────────────────────────────
async function ensureLandmarker() {
  if (landmarker) return landmarker;
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    numFaces: 1,
  });
  return landmarker;
}

// ── Camera ─────────────────────────────────────────────────────────────────
async function enterCamera() {
  show("camera");
  countdownEl.hidden = true;
  cameraHint.hidden = false;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width:  { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    ensureLandmarker().catch(() => {});  // warm the model
  } catch (err) {
    alert("Camera error: " + err.message);
    show("welcome");
  }
}

function stopCamera() {
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;
}

function cancelCountdown() {
  if (countdownTimer) {
    clearTimeout(countdownTimer);
    countdownTimer = null;
  }
  countdownEl.hidden = true;
  cameraHint.hidden = false;
}

function startCountdown() {
  if (countdownTimer || !stream) return;
  cameraHint.hidden = true;
  let n = 3;
  const tick = () => {
    if (n <= 0) {
      countdownTimer = null;
      countdownEl.hidden = true;
      captureFromVideo();
      return;
    }
    countdownEl.hidden = false;
    // re-trigger CSS animation
    countdownEl.style.animation = "none";
    void countdownEl.offsetWidth;
    countdownEl.style.animation = "";
    countdownEl.textContent = String(n);
    n--;
    countdownTimer = setTimeout(tick, 1000);
  };
  tick();
}

async function captureFromVideo() {
  if (!video.videoWidth) return;
  const w = video.videoWidth, h = video.videoHeight;
  const off = new OffscreenCanvas(w, h);
  const octx = off.getContext("2d");
  // Mirror to match what the user sees on screen.
  octx.translate(w, 0);
  octx.scale(-1, 1);
  octx.drawImage(video, 0, 0, w, h);
  capturedBitmap = await createImageBitmap(off);
  capturedMirrored = true;
  stopCamera();
  cameFrom = "camera";
  await goToResult();
}

// ── Upload ─────────────────────────────────────────────────────────────────
async function handleUpload(e) {
  const file = e.target.files && e.target.files[0];
  e.target.value = "";  // reset so re-picking same file fires change
  if (!file) return;
  try {
    capturedBitmap = await createImageBitmap(file);
    capturedMirrored = false;
    cameFrom = "upload";
    await goToResult();
  } catch (err) {
    alert("Could not read image: " + err.message);
  }
}

// ── Result ─────────────────────────────────────────────────────────────────
async function goToResult() {
  show("result");
  capturedCv.width  = capturedBitmap.width;
  capturedCv.height = capturedBitmap.height;
  capturedCv.getContext("2d").drawImage(capturedBitmap, 0, 0);
  fitOverlayToCanvas();
  setStatus("Detecting…");
  downloadBtn.disabled = true;
  plan = null; landmarks = null;
  clearOverlay();
  try {
    const lm = await ensureLandmarker();
    const result = lm.detect(capturedCv);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      setStatus("No face detected. Retake with better lighting / framing.", "warn");
    } else {
      landmarks = result.faceLandmarks[0];
      computePlan();
    }
  } catch (err) {
    setStatus("Detection failed: " + err.message, "err");
  }
  downloadBtn.disabled = !plan;
  draw();
}

function retake() {
  cancelCountdown();
  plan = null; landmarks = null; capturedBitmap = null;
  clearOverlay();
  if (cameFrom === "camera") {
    enterCamera();
  } else {
    show("welcome");
  }
}

function backToWelcome() {
  cancelCountdown();
  stopCamera();
  plan = null; landmarks = null; capturedBitmap = null;
  show("welcome");
}

// ── Plan: passport-spec square crop ────────────────────────────────────────
function computePlan() {
  if (!landmarks || !capturedBitmap) { plan = null; return; }
  const W = capturedBitmap.width, H = capturedBitmap.height;
  const headFrac = clampNum(headFracIn.valueAsNumber, 0.40, 0.80, 0.55);
  const eyeFrac  = clampNum(eyeFracIn.valueAsNumber,  0.40, 0.80, 0.60);

  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    const x = Math.min(1, Math.max(0, p.x));
    const y = Math.min(1, Math.max(0, p.y));
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const bboxX = minX * W, bboxY = minY * H;
  const bboxW = (maxX - minX) * W, bboxH = (maxY - minY) * H;

  let eyeSum = 0, eyeCount = 0;
  for (const i of EYE_LM) {
    const p = landmarks[i];
    if (p && Number.isFinite(p.y)) {
      eyeSum += Math.min(1, Math.max(0, p.y)) * H;
      eyeCount++;
    }
  }
  if (eyeCount === 0) { plan = null; setStatus("Could not locate eyes.", "warn"); return; }
  const eyeY  = eyeSum / eyeCount;
  const faceX = bboxX + bboxW / 2;
  const chinY = bboxY + bboxH;

  // Face mesh runs hairline → chin. Crown sits ~25-30% above the bbox top;
  // 1.30 × bbox height ≈ full crown-to-chin head height.
  const headHeight = bboxH * 1.30;
  const crownEstY  = chinY - headHeight;

  const cropSize = Math.round(headHeight / headFrac);
  const eyeFromTopOfCrop = (1 - eyeFrac) * cropSize;
  const cropY = Math.round(eyeY - eyeFromTopOfCrop);
  const cropX = Math.round(faceX - cropSize / 2);

  const warnings = [];
  let x = cropX, y = cropY, s = cropSize;
  if (s > Math.min(W, H)) {
    warnings.push(`Crop ${s}px exceeds source ${W}×${H}; shrinking.`);
    s = Math.min(W, H);
  }
  if (x < 0) { x = 0; warnings.push("Shifted right to fit."); }
  if (y < 0) { y = 0; warnings.push("Shifted down to fit."); }
  if (x + s > W) { x = W - s; warnings.push("Shifted left to fit."); }
  if (y + s > H) { y = H - s; warnings.push("Shifted up to fit."); }

  const headRatio     = headHeight / s;
  const eyeFromBottom = (s - (eyeY - y)) / s;
  if (headRatio < 0.50 || headRatio > 0.69)
    warnings.push(`Head ${(headRatio * 100).toFixed(0)}% outside 50–69%.`);
  if (eyeFromBottom < 0.56 || eyeFromBottom > 0.69)
    warnings.push(`Eye ${(eyeFromBottom * 100).toFixed(0)}% outside 56–69%.`);

  plan = {
    x, y, size: s,
    eyeY, faceX, chinY, crownEstY,
    bboxX, bboxY, bboxW, bboxH,
    headRatio, eyeFromBottom, warnings,
  };

  const ok = warnings.length === 0;
  setStatus(
    `${s}×${s} • head ${(headRatio*100).toFixed(0)}% • eye ${(eyeFromBottom*100).toFixed(0)}%` +
    (ok ? " • within spec" : " • " + warnings.join(" ")),
    ok ? "" : "warn"
  );
}

function clampNum(v, lo, hi, dflt) {
  if (!Number.isFinite(v)) return dflt;
  return Math.min(hi, Math.max(lo, v)) / 100;
}

// ── Overlay drawing ────────────────────────────────────────────────────────
function fitOverlayToCanvas() {
  const stage = capturedCv.parentElement;
  const stageRect = stage.getBoundingClientRect();
  const capRect = capturedCv.getBoundingClientRect();
  overlayCv.width  = capturedCv.width;
  overlayCv.height = capturedCv.height;
  overlayCv.style.width  = capRect.width  + "px";
  overlayCv.style.height = capRect.height + "px";
  overlayCv.style.left   = (capRect.left - stageRect.left) + "px";
  overlayCv.style.top    = (capRect.top  - stageRect.top)  + "px";
}

function clearOverlay() {
  const ctx = overlayCv.getContext("2d");
  ctx.clearRect(0, 0, overlayCv.width, overlayCv.height);
}

function draw() {
  const ctx = overlayCv.getContext("2d");
  ctx.clearRect(0, 0, overlayCv.width, overlayCv.height);
  if (!plan) return;

  // Crop rectangle.
  ctx.setLineDash([10, 6]);
  ctx.strokeStyle = "rgba(255, 215, 0, 0.95)";
  ctx.lineWidth = Math.max(2, capturedCv.width / 400);
  ctx.strokeRect(plan.x, plan.y, plan.size, plan.size);

  // Eye line (subtle).
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = Math.max(1, capturedCv.width / 800);
  ctx.strokeStyle = "rgba(166, 92, 220, 0.75)";
  ctx.beginPath();
  ctx.moveTo(plan.x, plan.eyeY);
  ctx.lineTo(plan.x + plan.size, plan.eyeY);
  ctx.stroke();

  ctx.setLineDash([]);
  ctx.lineWidth = 1;
}

// ── Download ───────────────────────────────────────────────────────────────
function download() {
  if (!plan || !capturedBitmap) return;
  const target = clampInt(outSizeIn.valueAsNumber, 600, 1200, 900);
  const out = document.createElement("canvas");
  out.width = target;
  out.height = target;
  const octx = out.getContext("2d");
  octx.imageSmoothingEnabled = true;
  octx.imageSmoothingQuality = "high";
  octx.drawImage(
    capturedBitmap,
    plan.x, plan.y, plan.size, plan.size,
    0, 0, target, target
  );
  out.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `passport_${target}x${target}.jpg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }, "image/jpeg", 0.95);
}

function clampInt(v, lo, hi, dflt) {
  if (!Number.isFinite(v)) return dflt;
  return Math.round(Math.min(hi, Math.max(lo, v)));
}

// ── Wire up ────────────────────────────────────────────────────────────────
useCameraBtn.addEventListener("click", enterCamera);
useUploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleUpload);

cameraBackBtn.addEventListener("click", backToWelcome);
resultBackBtn.addEventListener("click", backToWelcome);

// Tap target: use pointerdown for instant response. Ignore taps that hit
// the back button (it stops propagation via stopPropagation below).
cameraTap.addEventListener("pointerdown", startCountdown);
cameraBackBtn.addEventListener("pointerdown", e => e.stopPropagation());

redoBtn.addEventListener("click", retake);
downloadBtn.addEventListener("click", download);

for (const el of [headFracIn, eyeFracIn]) {
  el.addEventListener("change", () => {
    if (!landmarks) return;
    computePlan();
    draw();
    downloadBtn.disabled = !plan;
  });
}

window.addEventListener("resize", () => {
  if (!resultScreen.hidden && capturedBitmap) {
    fitOverlayToCanvas();
    draw();
  }
});

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  useCameraBtn.disabled = true;
  useCameraBtn.title = "This browser does not support camera access.";
}

show("welcome");
