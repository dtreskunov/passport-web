// Passport Photo (web) — uses MediaPipe FaceLandmarker (WASM) for landmarks,
// then computes a US-passport-spec square crop from the captured frame.
//
// UX: welcome screen → either camera (tap → 3-2-1 countdown → snap) or
// upload → result screen with crop overlay + download.
// Everything runs in the browser; the camera feed never leaves the page.

import {
  FaceLandmarker,
  FaceDetector,
  ImageSegmenter,
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
const fixBgBtn      = document.getElementById("fixBgBtn");
const downloadBtn   = document.getElementById("downloadBtn");
const statusEl      = document.getElementById("status");

// Crop spec defaults (within US passport spec; mid of allowed bands).
const HEAD_FRAC = 0.62;   // head height / image height (spec: 0.50 – 0.69)
const EYE_FRAC  = 0.60;   // eye line from bottom of image (spec: 0.56 – 0.69)
const OUT_SIZE  = 900;    // output side in px

// ── State ──────────────────────────────────────────────────────────────────
let landmarker = null;
let detector = null;            // FaceDetector (BlazeFace) for small-face fallback
let segmenter = null;           // ImageSegmenter for background replacement
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
  statusEl.hidden = !msg;
}

function defaultRetakeLabel() {
  return cameFrom === "upload" ? "Choose a different image" : "Retake";
}

// Warning is surfaced on the Retake button itself: the default label
// ("Retake" / "Choose a different image") by default, or the warning text
// in warn color when the crop is out of spec.
function setWarning(msg) {
  if (msg) {
    redoBtn.textContent = msg;
    redoBtn.classList.add("warn");
  } else {
    redoBtn.textContent = defaultRetakeLabel();
    redoBtn.classList.remove("warn");
  }
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
  // Warm up: first detect() pays GPU shader-compile cost (~hundreds of ms).
  // Run it now on a dummy frame so the user-facing detect is cheap.
  try {
    const warm = new OffscreenCanvas(64, 64);
    const wctx = warm.getContext("2d");
    wctx.fillStyle = "#888";
    wctx.fillRect(0, 0, 64, 64);
    landmarker.detect(warm);
  } catch { /* warm-up is best-effort */ }
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

async function ensureSegmenter() {
  if (segmenter) return segmenter;
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  segmenter = await ImageSegmenter.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  });
  return segmenter;
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
  // 640 is plenty for the landmarker; larger sizes don't improve accuracy.
  const DETECT_MAX = 640;
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
// Photo settings (imageWidth/imageHeight) chosen at camera-start time so
// ImageCapture.takePhoto() returns a still that matches the preview track's
// aspect ratio. Null = let the device pick (its sensor-native aspect, which
// usually differs from the preview).
let photoSettings = null;

async function pickPhotoSettings(track) {
  if (!("ImageCapture" in window)) return null;
  try {
    const ic = new ImageCapture(track);
    const caps = await ic.getPhotoCapabilities();
    const ts = track.getSettings();
    if (!caps?.imageHeight || !caps?.imageWidth || !ts.width || !ts.height) return null;
    const aspect = ts.width / ts.height;
    const snap = (v, step, lo, hi) => {
      const s = step || 1;
      let n = Math.round(v / s) * s;
      return Math.max(lo, Math.min(hi, n));
    };
    // Use the camera's max photo height, derive width from the preview
    // aspect, then clamp/snap into the photo width's allowed range.
    let h = caps.imageHeight.max;
    let w = snap(h * aspect, caps.imageWidth.step,
                  caps.imageWidth.min, caps.imageWidth.max);
    // If width got clamped, recompute height from width so the aspect holds.
    if (Math.abs(w / h - aspect) > 0.01) {
      h = snap(w / aspect, caps.imageHeight.step,
               caps.imageHeight.min, caps.imageHeight.max);
    }
    return { imageWidth: w, imageHeight: h };
  } catch {
    return null;
  }
}
// ── Camera ─────────────────────────────────────────────────────────────────
async function enterCamera() {
  show("camera");
  countdownEl.hidden = true;
  cameraHint.hidden = false;
  try {
    // Preview track only needs to look sharp at the current viewport; the
    // still photo is grabbed at the camera's native sensor resolution via
    // ImageCapture.takePhoto() below. This keeps the live stream cheap.
    const dpr = window.devicePixelRatio || 1;
    const previewIdeal = Math.min(
      640,
      Math.round(Math.max(window.innerWidth, window.innerHeight) * dpr),
    );
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width:  { ideal: previewIdeal },
        height: { ideal: previewIdeal },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    photoSettings = await pickPhotoSettings(stream.getVideoTracks()[0]);
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

async function mirrorBitmap(src) {
  const w = src.width, h = src.height;
  const off = new OffscreenCanvas(w, h);
  const octx = off.getContext("2d");
  octx.translate(w, 0);
  octx.scale(-1, 1);
  octx.drawImage(src, 0, 0, w, h);
  return await createImageBitmap(off);
}

let captureId = 0;  // invalidates in-flight hi-res swaps after retake/back

async function captureFromVideo() {
  if (!video.videoWidth) return;
  const myId = ++captureId;
  const myStream = stream;  // snapshot so retake doesn't kill the new stream

  // 1. Grab the current low-res preview frame and run detection on it
  //    immediately so the user sees a result quickly.
  const previewBmp = await createImageBitmap(video);
  capturedBitmap = await mirrorBitmap(previewBmp);
  capturedMirrored = true;
  cameFrom = "camera";

  // 2. Kick off the high-res photo capture in parallel; swap it in once
  //    ready. Landmarks are normalized [0,1] so they remain valid.
  const hiResPromise = (async () => {
    const track = myStream?.getVideoTracks?.()[0];
    if (!track || !("ImageCapture" in window)) return null;
    try {
      const ic = new ImageCapture(track);
      const blob = await ic.takePhoto(photoSettings || undefined);
      const bmp = await createImageBitmap(blob);
      return await mirrorBitmap(bmp);
    } catch {
      return null;
    }
  })();

  await goToResult();  // detection runs on the low-res frame

  const hi = await hiResPromise;
  if (myId === captureId) {
    if (hi) {
      // If photoSettings was applied, the still and the preview have the
      // same aspect ratio, so normalized landmarks from the preview are
      // already valid for the hi-res frame — no re-detect needed. Without
      // photoSettings the aspects can differ, so re-run detection.
      const needsRedetect = !photoSettings
        || Math.abs((hi.width / hi.height) - (capturedCv.width / capturedCv.height)) > 0.01;
      capturedBitmap = hi;
      capturedCv.width  = hi.width;
      capturedCv.height = hi.height;
      capturedCv.getContext("2d").drawImage(hi, 0, 0);
      fitOverlayToCanvas();
      if (needsRedetect) {
        try {
          const lm = await detectLandmarks();
          if (lm) landmarks = lm;
        } catch { /* keep preview landmarks */ }
      }
      if (landmarks) computePlan();
      updateBgUI();
      draw();
    }
    // Only stop the stream we owned; a retake may have started a new one.
    if (myStream) {
      for (const t of myStream.getTracks()) t.stop();
      if (stream === myStream) { stream = null; video.srcObject = null; }
    }
  }
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
  setWarning("");
  fixBgBtn.hidden = true;
  spinnerEl.hidden = false;
  downloadBtn.disabled = true;
  plan = null; landmarks = null;
  clearOverlay();  // Yield so the browser actually paints the spinner before we hog the main
  // thread with the synchronous landmarker.detect() call.
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  try {
    const lm = await detectLandmarks();
    if (!lm) {
      setStatus("No face detected. Retake with better lighting / framing.", "err");
    } else {
      landmarks = lm;
      computePlan();
    }
  } catch (err) {
    setStatus("Detection failed: " + err.message, "err");
  }
  spinnerEl.hidden = true;
  downloadBtn.disabled = !plan;
  updateBgUI();
  draw();
}

function retake() {
  cancelCountdown();
  captureId++;  // invalidate any in-flight hi-res swap from the prior shot
  plan = null; landmarks = null; capturedBitmap = null;
  clearOverlay();
  if (cameFrom === "camera") {
    enterCamera();
  } else if (cameFrom === "upload") {
    fileInput.click();
  } else {
    show("welcome");
  }
}

function backToWelcome() {
  cancelCountdown();
  captureId++;
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
  if (eyeCount === 0) { plan = null; setStatus("Could not locate eyes.", "err"); return; }
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

  // Warn when the final crop is out of US passport spec, or when the face
  // sits so close to a side edge that the crop had to be shifted
  // horizontally (head/eye ratios are vertical metrics and won't catch
  // sideways clamping on their own).
  const sideClamped = lack.left || lack.right;
  if (headBad || eyeBad || sideClamped) {
    const dirs = [];
    if (lack.above) dirs.push("above");
    if (lack.below) dirs.push("below");
    if (sideClamped) dirs.push("beside");
    const where = dirs.length ? dirs.join(" / ") : "around";
    setWarning(`Retake with more space ${where} your head`);
  } else {
    setWarning("");
  }
}


// ── Background detection / fix ─────────────────────────────────────────────

// Box (mean) filter over a (2r+1)x(2r+1) window using a summed-area table.
// `src` and `dst` are Float32Arrays of length W*H. `dst` may equal `src`.
// O(W*H), independent of r.
function boxFilter(src, dst, W, H, r) {
  const II = new Float64Array((W + 1) * (H + 1));
  for (let y = 1; y <= H; y++) {
    const rowOff = y * (W + 1);
    const prevRow = rowOff - (W + 1);
    let rowSum = 0;
    for (let x = 1; x <= W; x++) {
      rowSum += src[(y - 1) * W + (x - 1)];
      II[rowOff + x] = II[prevRow + x] + rowSum;
    }
  }
  for (let y = 0; y < H; y++) {
    const y0 = y - r < 0 ? 0 : y - r;
    const y1 = y + r + 1 > H ? H : y + r + 1;
    const r0 = y0 * (W + 1), r1 = y1 * (W + 1);
    for (let x = 0; x < W; x++) {
      const x0 = x - r < 0 ? 0 : x - r;
      const x1 = x + r + 1 > W ? W : x + r + 1;
      const area = (x1 - x0) * (y1 - y0);
      dst[y * W + x] =
        (II[r1 + x1] - II[r0 + x1] - II[r1 + x0] + II[r0 + x0]) / area;
    }
  }
}

// Edge-aware guided filter (He, Sun, Tang 2010). Refines a coarse value map
// `p` using a high-resolution guide `I`; output `q` snaps to edges in I.
// For matting, p is the coarse alpha and I is the photo's luminance.
function guidedFilter(I, p, W, H, r, eps) {
  const N = W * H;
  const tmp     = new Float32Array(N);
  const meanI   = new Float32Array(N);
  const meanP   = new Float32Array(N);
  const corrI   = new Float32Array(N);
  const corrIp  = new Float32Array(N);

  boxFilter(I, meanI, W, H, r);
  boxFilter(p, meanP, W, H, r);
  for (let i = 0; i < N; i++) tmp[i] = I[i] * I[i];
  boxFilter(tmp, corrI, W, H, r);
  for (let i = 0; i < N; i++) tmp[i] = I[i] * p[i];
  boxFilter(tmp, corrIp, W, H, r);

  const a = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const vI    = corrI[i]  - meanI[i] * meanI[i];
    const covIp = corrIp[i] - meanI[i] * meanP[i];
    a[i] = covIp / (vI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  // Reuse meanI/meanP buffers for mean_a / mean_b.
  boxFilter(a, meanI, W, H, r);
  boxFilter(b, meanP, W, H, r);

  const q = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const v = meanI[i] * I[i] + meanP[i];
    q[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return q;
}

// Refine a 256-pixel coarse confidence mask into a per-pixel alpha that
// snaps to real edges in the photo, via guided filtering at a working
// resolution capped to ~0.7 MP for memory/time. The final alpha is
// bilinearly upsampled to the full resolution of the photo.
function refineMaskGuided(coarseMaskCv, photoCv, W, H) {
  const TARGET = 700_000;
  const scale = Math.min(1, Math.sqrt(TARGET / (W * H)));
  const sW = Math.max(64, Math.round(W * scale));
  const sH = Math.max(64, Math.round(H * scale));

  // Downsample guide (photo) to working resolution and pull luma.
  const gCv = new OffscreenCanvas(sW, sH);
  const gctx = gCv.getContext("2d", { willReadFrequently: true });
  gctx.imageSmoothingEnabled = true;
  gctx.imageSmoothingQuality = "high";
  gctx.drawImage(photoCv, 0, 0, sW, sH);
  const gd = gctx.getImageData(0, 0, sW, sH).data;

  // Bilinearly upsample the 256-px coarse mask to the same working res.
  const mCv = new OffscreenCanvas(sW, sH);
  const mctx = mCv.getContext("2d", { willReadFrequently: true });
  mctx.imageSmoothingEnabled = true;
  mctx.imageSmoothingQuality = "high";
  mctx.drawImage(coarseMaskCv, 0, 0, sW, sH);
  const md = mctx.getImageData(0, 0, sW, sH).data;

  const sN = sW * sH;
  const I = new Float32Array(sN);
  const p = new Float32Array(sN);
  // Binarize the coarse mask at 0.5 before refinement. The segmenter's
  // soft confidence extends well past the true person silhouette, and
  // feeding that soft band into the guided filter lets the photo guide
  // drag alpha out along weak furniture edges in the background. With a
  // hard 0/1 input, the filter only has to soften the one true edge.
  for (let i = 0, j = 0; i < gd.length; i += 4, j++) {
    I[j] = (0.2126 * gd[i] + 0.7152 * gd[i + 1] + 0.0722 * gd[i + 2]) / 255;
    p[j] = md[i + 3] >= 128 ? 1 : 0;
  }

  // Small radius so the filter only does local edge alignment (a few
  // pixels around the binarized boundary), not regional smoothing.
  const r = Math.max(2, Math.round(Math.min(sW, sH) / 200));
  const eps = 1e-4;
  const q = guidedFilter(I, p, sW, sH, r, eps);

  // Remap the refined alpha: snap the lowest band to 0 and the highest
  // band to 1, with a linear ramp between. Without this, the guided
  // filter leaves a faint soft "skirt" extending into the true bg, and
  // the matting equation in that skirt evaluates to α·realBg + (1-α)·NEW,
  // which is darker than NEW whenever the real bg is darker than the new
  // gray — producing a visible shadow halo around the person. We keep
  // the ramp wide enough to preserve hair anti-aliasing.
  const LO = 0.25, HI = 0.85, SPAN = HI - LO;
  const qCv = new OffscreenCanvas(sW, sH);
  const qImg = new ImageData(sW, sH);
  for (let j = 0; j < sN; j++) {
    let v = q[j];
    v = v <= LO ? 0 : v >= HI ? 1 : (v - LO) / SPAN;
    qImg.data[j * 4 + 3] = (v * 255) | 0;
  }
  qCv.getContext("2d").putImageData(qImg, 0, 0);

  const upCv = new OffscreenCanvas(W, H);
  const uctx = upCv.getContext("2d", { willReadFrequently: true });
  uctx.imageSmoothingEnabled = true;
  uctx.imageSmoothingQuality = "high";
  uctx.drawImage(qCv, 0, 0, W, H);
  return uctx.getImageData(0, 0, W, H).data;   // RGBA, alpha = refined α
}

// US passport spec calls for a plain white or off-white background with no
// patterns or shadows. We sample everything in the crop except an expanded
// face/head/shoulders region (anything left should be pure background) and
// flag the photo when the area isn't uniform.
function checkBackground() {
  if (!plan || !capturedCv.width) return { ok: true, mean: 255, std: 0 };
  const SAMPLE = 128;
  const off = new OffscreenCanvas(SAMPLE, SAMPLE);
  const ctx = off.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(
    capturedCv,
    plan.x, plan.y, plan.size, plan.size,
    0, 0, SAMPLE, SAMPLE,
  );
  const data = ctx.getImageData(0, 0, SAMPLE, SAMPLE).data;

  // Build an exclusion box (in SAMPLE pixel coords) around the head/torso.
  // Expand the face bbox generously: enough to cover hair, ears, and the
  // shoulders that extend below the chin.
  const sx = (plan.bboxX - plan.x) / plan.size * SAMPLE;
  const sy = (plan.bboxY - plan.y) / plan.size * SAMPLE;
  const sw = plan.bboxW / plan.size * SAMPLE;
  const sh = plan.bboxH / plan.size * SAMPLE;
  const exL = sx - sw * 0.45;
  const exR = sx + sw * 1.45;
  const exT = sy - sh * 0.45;
  const exB = SAMPLE;   // everything below the forehead-line is body / not bg

  let sumY = 0, sumY2 = 0, n = 0;
  for (let y = 0; y < SAMPLE; y++) {
    for (let x = 0; x < SAMPLE; x++) {
      if (x >= exL && x <= exR && y >= exT && y <= exB) continue;
      const i = (y * SAMPLE + x) * 4;
      const Y = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
      sumY  += Y;
      sumY2 += Y * Y;
      n++;
    }
  }
  if (n < 32) return { ok: true, mean: 255, std: 0 };
  const mean = sumY / n;
  const std  = Math.sqrt(Math.max(0, sumY2 / n - mean * mean));
  // Spec really only cares about uniformity — a plain wall of any tone is
  // fine, and the fix step would just neutralize the brightness anyway.
  // Strict on std so patterns / objects / hard shadows still flag.
  const ok = std <= 30;
  return { ok, mean, std };
}

function updateBgUI() {
  if (!plan) { fixBgBtn.hidden = true; return; }
  fixBgBtn.hidden = checkBackground().ok;
}

// Replace the background of the captured frame with solid white using the
// MediaPipe selfie-multiclass segmenter (category 0 == background).
async function fixBackground() {
  if (!capturedBitmap) return;
  fixBgBtn.disabled = true;
  const origLabel = fixBgBtn.textContent;
  fixBgBtn.textContent = "Fixing background…";
  try {
    const seg = await ensureSegmenter();
    const W = capturedCv.width, H = capturedCv.height;
    // Segmenter's native input is 256x256; running larger doesn't help.
    const SEG = 256;
    const small = new OffscreenCanvas(SEG, SEG);
    small.getContext("2d").drawImage(capturedCv, 0, 0, SEG, SEG);
    const result = seg.segment(small);
    const bgMask = result.confidenceMasks[0];
    const bgArr  = bgMask.getAsFloat32Array();
    const mw = bgMask.width, mh = bgMask.height;

    // Stash mask alpha (= 1 - P(background)) as an RGBA canvas so the
    // browser can bilinearly upscale it to the photo's native resolution.
    const maskCv = new OffscreenCanvas(mw, mh);
    const mctx = maskCv.getContext("2d");
    const mImg = mctx.createImageData(mw, mh);
    for (let i = 0; i < bgArr.length; i++) {
      mImg.data[i * 4 + 3] = Math.round(255 * Math.max(0, Math.min(1, 1 - bgArr[i])));
    }
    mctx.putImageData(mImg, 0, 0);
    for (const m of result.confidenceMasks) m.close?.();
    result.close?.();

    // Refine the coarse 256-px mask with an edge-aware guided filter
    // (He et al. 2010), using the photo's luminance as the guide. The
    // resulting alpha snaps to actual hair / jaw / shoulder edges in the
    // photo instead of the soft bilinear-upsample blur that produced the
    // visible halo, and low-confidence ghost regions vanish into the
    // background mean without needing a separate sigmoid hardening pass.
    const upAlpha = refineMaskGuided(maskCv, capturedCv, W, H);

    // Pull the photo at full resolution.
    const srcCv = new OffscreenCanvas(W, H);
    const sctx = srcCv.getContext("2d", { willReadFrequently: true });
    sctx.drawImage(capturedBitmap, 0, 0, W, H);
    const sdata = sctx.getImageData(0, 0, W, H).data;
    const N = sdata.length;

    // Estimate the *old* background as a smooth spatially-varying color
    // field instead of one global average. Real backgrounds aren't a
    // single color (warmer near a wall, cooler near a window, etc.), and
    // subtracting one global mean from edge pixels in the matting
    // equation leaves a tinted halo. We sample the photo only where the
    // refined alpha says the pixel is definitely background, then
    // box-filter to extend the color into the foreground region.
    //
    // Operate at a downsampled resolution (~150 px short side) for speed;
    // the bg color field varies slowly so coarse is fine.
    const bgW = Math.max(32, Math.round(W / Math.max(W, H) * 200));
    const bgH = Math.max(32, Math.round(H / Math.max(W, H) * 200));
    const bgN = bgW * bgH;
    const bgCv = new OffscreenCanvas(bgW, bgH);
    const bgctx = bgCv.getContext("2d", { willReadFrequently: true });
    bgctx.imageSmoothingEnabled = true;
    bgctx.imageSmoothingQuality = "high";
    bgctx.drawImage(capturedCv, 0, 0, bgW, bgH);
    const bgSrc = bgctx.getImageData(0, 0, bgW, bgH).data;
    // Also downsample the alpha to bg grid for the mask.
    const aCv = new OffscreenCanvas(bgW, bgH);
    const actx = aCv.getContext("2d", { willReadFrequently: true });
    const aFull = new ImageData(W, H);
    for (let i = 3; i < upAlpha.length; i += 4) aFull.data[i] = upAlpha[i];
    const aFullCv = new OffscreenCanvas(W, H);
    aFullCv.getContext("2d").putImageData(aFull, 0, 0);
    actx.imageSmoothingEnabled = true;
    actx.imageSmoothingQuality = "high";
    actx.drawImage(aFullCv, 0, 0, bgW, bgH);
    const aLow = actx.getImageData(0, 0, bgW, bgH).data;

    const rBuf = new Float32Array(bgN);
    const gBuf = new Float32Array(bgN);
    const bBuf = new Float32Array(bgN);
    const wBuf = new Float32Array(bgN);
    for (let i = 0, j = 0; i < bgSrc.length; i += 4, j++) {
      // Weight = 1 where definitely background, 0 elsewhere.
      const w = aLow[i + 3] < 24 ? 1 : 0;
      wBuf[j] = w;
      rBuf[j] = bgSrc[i]     * w;
      gBuf[j] = bgSrc[i + 1] * w;
      bBuf[j] = bgSrc[i + 2] * w;
    }
    // Box-filter radius ≈ half the short side: gives every foreground
    // pixel access to bg samples even when the head fills the frame.
    const bgR = Math.max(8, Math.round(Math.min(bgW, bgH) / 2));
    boxFilter(rBuf, rBuf, bgW, bgH, bgR);
    boxFilter(gBuf, gBuf, bgW, bgH, bgR);
    boxFilter(bBuf, bBuf, bgW, bgH, bgR);
    boxFilter(wBuf, wBuf, bgW, bgH, bgR);
    // Pack interpolated local-bg color into an RGBA image and upsample to
    // full res so we can index it directly in the per-pixel matting loop.
    const lbImg = new ImageData(bgW, bgH);
    let globalR = 244, globalG = 244, globalB = 244, gn = 0;
    for (let j = 0; j < bgN; j++) {
      const w = wBuf[j];
      if (w > 1e-6) {
        const r = rBuf[j] / w, g = gBuf[j] / w, b = bBuf[j] / w;
        lbImg.data[j * 4]     = r | 0;
        lbImg.data[j * 4 + 1] = g | 0;
        lbImg.data[j * 4 + 2] = b | 0;
        lbImg.data[j * 4 + 3] = 255;
        globalR = (globalR * gn + r) / (gn + 1);
        globalG = (globalG * gn + g) / (gn + 1);
        globalB = (globalB * gn + b) / (gn + 1);
        gn++;
      } else {
        lbImg.data[j * 4]     = 244;
        lbImg.data[j * 4 + 1] = 244;
        lbImg.data[j * 4 + 2] = 244;
        lbImg.data[j * 4 + 3] = 255;
      }
    }
    const lbCv = new OffscreenCanvas(bgW, bgH);
    lbCv.getContext("2d").putImageData(lbImg, 0, 0);
    const lbUp = new OffscreenCanvas(W, H);
    const lbUctx = lbUp.getContext("2d", { willReadFrequently: true });
    lbUctx.imageSmoothingEnabled = true;
    lbUctx.imageSmoothingQuality = "high";
    lbUctx.drawImage(lbCv, 0, 0, W, H);
    const bgField = lbUctx.getImageData(0, 0, W, H).data;

    // Match the replacement background's brightness to the original so it
    // blends with whatever lighting the subject was shot in. Only cap at
    // the bright end (so we never produce a darker-than-original bg) and
    // apply a low floor to avoid pitch black on extreme cases. Neutral
    // gray (R=G=B) avoids introducing a colour cast.
    const oldLuma = 0.2126 * globalR + 0.7152 * globalG + 0.0722 * globalB;
    const newLuma = Math.max(160, Math.min(250, oldLuma));
    const NEW_R = newLuma | 0, NEW_G = newLuma | 0, NEW_B = newLuma | 0;

    // Per-pixel alpha matting:
    //   observed = α · fg + (1 - α) · oldBg   →   fg = (observed - (1-α)·oldBg) / α
    // We then composite the recovered fg onto the new background using α.
    // This pulls the dark/coloured halo *out* of the edge pixels instead of
    // blending it into the new white, eliminating the "obvious blur" look.
    // `oldBg` is sampled from the local bg color field, not one global
    // average, so warm walls and cool windows decontaminate correctly.
    const out = new ImageData(W, H);
    const od  = out.data;
    for (let i = 0; i < N; i += 4) {
      const a = upAlpha[i + 3] / 255;
      if (a <= 0) {
        od[i] = NEW_R; od[i + 1] = NEW_G; od[i + 2] = NEW_B; od[i + 3] = 255;
        continue;
      }
      if (a >= 0.997) {
        od[i] = sdata[i]; od[i + 1] = sdata[i + 1]; od[i + 2] = sdata[i + 2]; od[i + 3] = 255;
        continue;
      }
      const oldR = bgField[i], oldG = bgField[i + 1], oldB = bgField[i + 2];
      const ia = 1 - a;
      const fr = (sdata[i]     - ia * oldR) / a;
      const fg = (sdata[i + 1] - ia * oldG) / a;
      const fb = (sdata[i + 2] - ia * oldB) / a;
      const cr = fr < 0 ? 0 : fr > 255 ? 255 : fr;
      const cg = fg < 0 ? 0 : fg > 255 ? 255 : fg;
      const cb = fb < 0 ? 0 : fb > 255 ? 255 : fb;
      od[i]     = (a * cr + ia * NEW_R) | 0;
      od[i + 1] = (a * cg + ia * NEW_G) | 0;
      od[i + 2] = (a * cb + ia * NEW_B) | 0;
      od[i + 3] = 255;
    }

    const outCv = new OffscreenCanvas(W, H);
    outCv.getContext("2d").putImageData(out, 0, 0);
    const bmp = await createImageBitmap(outCv);
    capturedBitmap = bmp;
    capturedCv.getContext("2d").drawImage(bmp, 0, 0);
    draw();
    updateBgUI();
  } catch (err) {
    setStatus("Background fix failed: " + err.message, "err");
  } finally {
    fixBgBtn.disabled = false;
    fixBgBtn.textContent = origLabel;
  }
}


// ── Overlay drawing ────────────────────────────────────────────────────────
function fitOverlayToCanvas() {
  // Match the overlay's pixel grid to the captured canvas so we can draw the
  // crop mask in source-pixel coordinates. CSS handles the visible sizing
  // (object-fit: contain on both keeps them perfectly aligned).
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
fixBgBtn.addEventListener("click", fixBackground);
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
