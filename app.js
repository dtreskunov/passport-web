// Passport Photo (web) — uses MediaPipe FaceLandmarker (WASM) for landmarks,
// then computes a US-passport-spec square crop from the captured frame.
//
// UX: welcome screen → either camera (tap → 3-2-1 countdown → snap) or
// upload → result screen with crop overlay + download.
// Everything runs in the browser; the camera feed never leaves the page.

import {
  FaceLandmarker,
  FaceDetector,
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
const spinnerEl     = document.getElementById("spinner");
const redoBtn       = document.getElementById("redoBtn");
const downloadBtn   = document.getElementById("downloadBtn");
const statusEl      = document.getElementById("status");

// Crop spec defaults (within US passport spec; mid of allowed bands).
const HEAD_FRAC = 0.55;   // head height / image height
const EYE_FRAC  = 0.60;   // eye line from bottom of image
const OUT_SIZE  = 900;    // output side in px

// ── State ──────────────────────────────────────────────────────────────────
let landmarker = null;
let detector = null;            // FaceDetector (BlazeFace) for small-face fallback
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

async function ensureDetector() {
  if (detector) return detector;
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  detector = await FaceDetector.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
  });
  return detector;
}

// Run the landmarker on the captured canvas. If no face is found, use the
// BlazeFace detector (which handles smaller / further faces) to locate the
// face, crop a padded square around it, scale that region up, run the
// landmarker on the scaled region, and remap landmarks back to canvas coords.
async function detectLandmarks() {
  const lm = await ensureLandmarker();

  // Detection runs on a downsampled copy so the main thread isn't tied up
  // for seconds on a multi-MP frame. Landmarks come back as normalized
  // [0,1] coords, so they still apply to the full-res captured canvas.
  const DETECT_MAX = 1024;
  const W = capturedCv.width, H = capturedCv.height;
  const scale = Math.min(1, DETECT_MAX / Math.max(W, H));
  let detectSrc = capturedCv;
  if (scale < 1) {
    const dw = Math.round(W * scale), dh = Math.round(H * scale);
    detectSrc = new OffscreenCanvas(dw, dh);
    detectSrc.getContext("2d").drawImage(capturedCv, 0, 0, dw, dh);
  }

  let res = lm.detect(detectSrc);
  if (res.faceLandmarks && res.faceLandmarks.length) return res.faceLandmarks[0];

  const det = await ensureDetector();
  const dres = det.detect(detectSrc);
  const box = dres?.detections?.[0]?.boundingBox;
  if (!box) return null;

  // BlazeFace box is in detectSrc pixel space; map back to capturedCv coords.
  const cx = (box.originX + box.width  / 2) / scale;
  const cy = (box.originY + box.height / 2) / scale;
  // Pad to ~3.5× face height so the mesh has hair / chin / shoulders context.
  const pad = Math.max(box.width, box.height) / scale * 3.5;
  let sx = Math.max(0, Math.round(cx - pad / 2));
  let sy = Math.max(0, Math.round(cy - pad / 2));
  let sw = Math.min(W - sx, Math.round(pad));
  let sh = Math.min(H - sy, Math.round(pad));

  const TARGET = 1024;
  const cropScale = TARGET / Math.max(sw, sh);
  const dw = Math.round(sw * cropScale), dh = Math.round(sh * cropScale);
  const crop = new OffscreenCanvas(dw, dh);
  crop.getContext("2d").drawImage(capturedCv, sx, sy, sw, sh, 0, 0, dw, dh);

  res = lm.detect(crop);
  if (!res.faceLandmarks || !res.faceLandmarks.length) return null;

  // Remap normalized landmarks from the cropped region back to canvas-relative
  // normalized coordinates.
  return res.faceLandmarks[0].map(p => ({
    x: (sx + p.x * sw) / W,
    y: (sy + p.y * sh) / H,
    z: p.z,
  }));
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
        // 1920 px on the long edge is plenty for a 900 px output crop
        // and keeps the live preview smooth. Asking for the sensor max
        // (e.g. 4K) makes the video stream choppy on most webcams.
        width:  { ideal: 1920 },
        height: { ideal: 1920 },
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
  setStatus("");
  spinnerEl.hidden = false;
  downloadBtn.disabled = true;
  plan = null; landmarks = null;
  clearOverlay();
  // Yield so the browser actually paints the spinner before we hog the main
  // thread with the synchronous landmarker.detect() call.
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  try {
    const lm = await detectLandmarks();
    if (!lm) {
      setStatus("No face detected. Retake with better lighting / framing.", "warn");
    } else {
      landmarks = lm;
      computePlan();
    }
  } catch (err) {
    setStatus("Detection failed: " + err.message, "err");
  }
  spinnerEl.hidden = true;
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
  const headFrac = HEAD_FRAC;
  const eyeFrac  = EYE_FRAC;

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

  // Distinguish "crop had to be shrunk / shifted to fit the source" (the
  // source frame isn't tall/wide enough for a spec crop) from "crop fits
  // but ratios drift from spec" (recoverable by adjusting the source).
  const tooSmall = cropSize > Math.min(W, H);
  let s = tooSmall ? Math.min(W, H) : cropSize;

  // Recenter on face/eye line *after* size clamp so the box stays centered.
  const wantX = Math.round(faceX - s / 2);
  const wantY = Math.round(eyeY - (1 - eyeFrac) * s);
  let x = wantX, y = wantY;
  const lack = { above: false, below: false, left: false, right: false };
  if (x < 0)      { x = 0;     lack.left  = true; }
  if (y < 0)      { y = 0;     lack.above = true; }
  if (x + s > W)  { x = W - s; lack.right = true; }
  if (y + s > H)  { y = H - s; lack.below = true; }
  // tooSmall means the source can't even hold the square; treat as lacking
  // on whichever axis is short.
  if (tooSmall) {
    if (H < W) { lack.above = true; lack.below = true; }
    else       { lack.left = true;  lack.right = true; }
  }

  const headRatio     = headHeight / s;
  const eyeFromBottom = (s - (eyeY - y)) / s;
  const headBad = headRatio < 0.50 || headRatio > 0.69;
  const eyeBad  = eyeFromBottom < 0.56 || eyeFromBottom > 0.69;

  plan = {
    x, y, size: s,
    eyeY, faceX, chinY, crownEstY,
    bboxX, bboxY, bboxW, bboxH,
    headRatio, eyeFromBottom,
  };

  const dirs = [];
  if (lack.above) dirs.push("above");
  if (lack.below) dirs.push("below");
  if (lack.left || lack.right) dirs.push("beside");
  if (dirs.length || headBad || eyeBad) {
    const where = dirs.length ? dirs.join(" / ") : "around";
    setStatus(`Retake with more space ${where} your head`, "warn");
  } else {
    setStatus("");
  }
}


// ── Overlay drawing ────────────────────────────────────────────────────────
function fitOverlayToCanvas() {
  // Match the overlay's pixel grid to the captured canvas so we can draw the
  // crop mask in source-pixel coordinates. CSS handles the visible sizing
  // (object-fit: cover on both keeps them perfectly aligned).
  overlayCv.width  = capturedCv.width;
  overlayCv.height = capturedCv.height;
}

function clearOverlay() {
  const ctx = overlayCv.getContext("2d");
  ctx.clearRect(0, 0, overlayCv.width, overlayCv.height);
}

function draw() {
  const ctx = overlayCv.getContext("2d");
  const W = overlayCv.width, H = overlayCv.height;
  ctx.clearRect(0, 0, W, H);
  if (!plan) return;

  // Dim everything outside the crop rectangle.
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.rect(plan.x, plan.y, plan.size, plan.size);
  ctx.fill("evenodd");
}

// ── Download ───────────────────────────────────────────────────────────────
function download() {
  if (!plan || !capturedBitmap) return;
  const target = OUT_SIZE;
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
