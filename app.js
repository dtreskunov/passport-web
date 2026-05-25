// Passport Photo (web) — uses MediaPipe FaceLandmarker (WASM) for landmarks,
// then computes a US-passport-spec square crop from the captured frame.
//
// Everything runs in the browser; the camera feed never leaves the page.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── DOM ────────────────────────────────────────────────────────────────────
const video        = document.getElementById("video");
const liveWrap     = document.getElementById("liveWrap");
const liveOverlay  = document.getElementById("liveOverlay");
const capturedCv   = document.getElementById("captured");
const overlayCv    = document.getElementById("overlay");
const startBtn     = document.getElementById("startBtn");
const captureBtn   = document.getElementById("captureBtn");
const redoBtn      = document.getElementById("redoBtn");
const downloadBtn  = document.getElementById("downloadBtn");
const statusEl     = document.getElementById("status");
const headFracIn   = document.getElementById("headFrac");
const eyeFracIn    = document.getElementById("eyeFrac");
const outSizeIn    = document.getElementById("outSize");

// ── State ──────────────────────────────────────────────────────────────────
let landmarker = null;
let stream = null;
let capturedBitmap = null;   // ImageBitmap of the frozen frame
let landmarks = null;        // 478 normalized landmarks for the captured frame
let plan = null;             // {x, y, size, headRatio, eyeFromBottom, warnings}

// MediaPipe FaceLandmarker landmark indices. We don't rely on any single
// "magic" index for the chin or crown — the bounding box of all landmarks
// gives us a reliable face extent. We use multiple eye landmarks averaged
// together for the eye line so a single bad point can't skew the result.
const EYE_LM = [
  // Left eye: top / bottom of eyelid + outer / inner corners
  159, 145, 33, 133,
  // Right eye (viewer's right, subject's left)
  386, 374, 263, 362,
];

// ── Status helpers ─────────────────────────────────────────────────────────
function setStatus(msg, level = "") {
  statusEl.textContent = msg;
  statusEl.className = "status" + (level ? " " + level : "");
}

// ── Init MediaPipe ─────────────────────────────────────────────────────────
async function ensureLandmarker() {
  if (landmarker) return landmarker;
  setStatus("Loading face model…");
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
async function startCamera() {
  try {
    setStatus("Requesting camera…");
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
    startBtn.disabled = true;
    captureBtn.disabled = false;
    setStatus("Frame your face and click Capture.");
    // Kick off model load in the background.
    ensureLandmarker().catch(err => setStatus("Model load failed: " + err.message, "err"));
  } catch (err) {
    setStatus("Camera error: " + err.message, "err");
  }
}

function stopCamera() {
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;
}

// ── Capture ────────────────────────────────────────────────────────────────
async function capture() {
  if (!video.videoWidth) { setStatus("Camera not ready yet.", "warn"); return; }
  captureBtn.disabled = true;
  setStatus("Capturing & detecting…");

  // Snapshot the current frame into an ImageBitmap so we own the pixels.
  const w = video.videoWidth, h = video.videoHeight;
  const off = new OffscreenCanvas(w, h);
  const octx = off.getContext("2d");
  octx.drawImage(video, 0, 0, w, h);
  capturedBitmap = await createImageBitmap(off);

  // Show frozen frame, hide live preview.
  capturedCv.width = w;
  capturedCv.height = h;
  const cctx = capturedCv.getContext("2d");
  cctx.drawImage(capturedBitmap, 0, 0);
  liveWrap.hidden = true;
  capturedCv.hidden = false;
  overlayCv.hidden = false;
  fitOverlayToCanvas();

  // Stop the camera — we have the frame; no need to keep streaming.
  stopCamera();

  // Run detection.
  try {
    const lm = await ensureLandmarker();
    const result = lm.detect(capturedCv);
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      setStatus("No face detected. Retake with better lighting / framing.", "warn");
      landmarks = null;
    } else {
      landmarks = result.faceLandmarks[0];
      computePlan();
    }
  } catch (err) {
    setStatus("Detection failed: " + err.message, "err");
    landmarks = null;
  }

  redoBtn.disabled = false;
  downloadBtn.disabled = !plan;
  draw();
}

function retake() {
  capturedBitmap = null;
  landmarks = null;
  plan = null;
  capturedCv.hidden = true;
  overlayCv.hidden = true;
  liveWrap.hidden = false;
  redoBtn.disabled = true;
  downloadBtn.disabled = true;
  startBtn.disabled = false;
  setStatus("Click \"Start camera\" to begin.");
}

// ── Plan: compute the passport-spec square crop ────────────────────────────
function computePlan() {
  if (!landmarks || !capturedBitmap) { plan = null; return; }
  const W = capturedBitmap.width, H = capturedBitmap.height;
  const headFrac = clampNum(headFracIn.valueAsNumber, 0.40, 0.80, 0.55);
  const eyeFrac  = clampNum(eyeFracIn.valueAsNumber,  0.40, 0.80, 0.60);

  // Face bounding box from all landmarks. Bounded to [0,1] so a stray
  // out-of-frame landmark can't blow up the geometry.
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

  // Eye line: average a handful of eyelid + corner landmarks across both eyes.
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

  // The face mesh extends from near the hairline to the chin, so bbox
  // height ≈ face-length (forehead → chin). Crown sits ~25-30% higher than
  // bbox top. Use 1.30 × bbox height as full crown-to-chin head height —
  // this gives a credible head ratio for most subjects.
  const headHeight = bboxH * 1.30;
  const crownEstY  = chinY - headHeight;

  const cropSize = Math.round(headHeight / headFrac);

  const eyeFromTopOfCrop = (1 - eyeFrac) * cropSize;
  const cropY = Math.round(eyeY - eyeFromTopOfCrop);
  const cropX = Math.round(faceX - cropSize / 2);

  // Clamp to image bounds (warn if we had to shift or shrink).
  const warnings = [];
  let x = cropX, y = cropY, s = cropSize;
  if (s > Math.min(W, H)) {
    warnings.push(`Crop ${s}px exceeds source ${W}×${H}; shrinking.`);
    s = Math.min(W, H);
  }
  if (x < 0) { x = 0; warnings.push("Shifted right to fit image."); }
  if (y < 0) { y = 0; warnings.push("Shifted down to fit image."); }
  if (x + s > W) { x = W - s; warnings.push("Shifted left to fit image."); }
  if (y + s > H) { y = H - s; warnings.push("Shifted up to fit image."); }

  const headRatio     = headHeight / s;
  const eyeFromBottom = (s - (eyeY - y)) / s;

  if (headRatio < 0.50 || headRatio > 0.69)
    warnings.push(`Head ratio ${(headRatio * 100).toFixed(1)}% outside 50–69% spec.`);
  if (eyeFromBottom < 0.56 || eyeFromBottom > 0.69)
    warnings.push(`Eye line ${(eyeFromBottom * 100).toFixed(1)}% outside 56–69% spec.`);

  plan = {
    x, y, size: s,
    eyeY, faceX, chinY, crownEstY,
    bboxX, bboxY, bboxW, bboxH,
    headRatio, eyeFromBottom, warnings,
  };

  const ok = warnings.length === 0;
  setStatus(
    `Crop ${s}×${s}  •  head ${(headRatio*100).toFixed(1)}%  •  ` +
    `eye ${(eyeFromBottom*100).toFixed(1)}% from bottom` +
    (ok ? "  •  within spec" : "  •  " + warnings.join(" ")),
    ok ? "" : "warn"
  );
}

function clampNum(v, lo, hi, dflt) {
  if (!Number.isFinite(v)) return dflt;
  return Math.min(hi, Math.max(lo, v)) / 100;
}

// ── Overlay drawing ────────────────────────────────────────────────────────
function fitOverlayToCanvas() {
  // Match the overlay's pixel size + CSS size to the captured canvas as
  // rendered on screen.
  const rect = capturedCv.getBoundingClientRect();
  overlayCv.width  = capturedCv.width;
  overlayCv.height = capturedCv.height;
  overlayCv.style.width  = rect.width  + "px";
  overlayCv.style.height = rect.height + "px";
  // Position absolute relative to .stage; use the captured canvas's position.
  overlayCv.style.left = capturedCv.offsetLeft + "px";
  overlayCv.style.top  = capturedCv.offsetTop  + "px";
}

function draw() {
  const ctx = overlayCv.getContext("2d");
  ctx.clearRect(0, 0, overlayCv.width, overlayCv.height);
  if (!plan) return;

  // Face mesh point cloud (small dots) — useful for confirming detection.
  ctx.fillStyle = "rgba(46, 213, 115, 0.45)";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * capturedCv.width, p.y * capturedCv.height, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Face bbox (thin teal).
  ctx.strokeStyle = "rgba(56, 189, 248, 0.7)";
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(plan.bboxX, plan.bboxY, plan.bboxW, plan.bboxH);

  // Reference lines: eye (purple), chin (blue), estimated crown (red).
  ctx.lineWidth = 1;
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = "rgba(166, 92, 220, 0.9)"; hline(ctx, plan.eyeY);
  ctx.strokeStyle = "rgba(70, 130, 220, 0.9)"; hline(ctx, plan.chinY);
  ctx.strokeStyle = "rgba(220, 80, 80, 0.9)";  hline(ctx, plan.crownEstY);

  // Crop rectangle.
  ctx.setLineDash([8, 5]);
  ctx.strokeStyle = "rgba(255, 215, 0, 0.95)";
  ctx.lineWidth = 3;
  ctx.strokeRect(plan.x, plan.y, plan.size, plan.size);
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
}

function hline(ctx, y) {
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(overlayCv.width, y);
  ctx.stroke();
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
startBtn.addEventListener("click", startCamera);
captureBtn.addEventListener("click", capture);
redoBtn.addEventListener("click", retake);
downloadBtn.addEventListener("click", download);

for (const el of [headFracIn, eyeFracIn]) {
  el.addEventListener("change", () => {
    computePlan();
    draw();
    downloadBtn.disabled = !plan;
  });
}

window.addEventListener("resize", () => {
  if (!capturedCv.hidden) { fitOverlayToCanvas(); draw(); }
});

if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  setStatus("This browser does not support camera access.", "err");
  startBtn.disabled = true;
}
