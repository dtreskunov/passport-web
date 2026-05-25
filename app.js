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
const HEAD_FRAC = 0.55;   // head height / image height
const EYE_FRAC  = 0.60;   // eye line from bottom of image
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

// Warning is surfaced on the Retake button itself: "Retake" by default,
// or the warning text in warn color when the crop is out of spec.
function setWarning(msg) {
  if (msg) {
    redoBtn.textContent = msg;
    redoBtn.classList.add("warn");
  } else {
    redoBtn.textContent = "Retake";
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
// US passport spec calls for a plain white or off-white background with no
// patterns or shadows. We sample the upper corners of the crop (almost
// always above the head, so reliably background) and flag the photo when the
// area is too dark or too non-uniform.
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
  const CORNER = Math.floor(SAMPLE * 0.25);
  let sumY = 0, sumY2 = 0, n = 0;
  for (let y = 0; y < CORNER; y++) {
    for (let x = 0; x < CORNER; x++) {
      for (const xi of [x, SAMPLE - 1 - x]) {
        const i = (y * SAMPLE + xi) * 4;
        const Y = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        sumY  += Y;
        sumY2 += Y * Y;
        n++;
      }
    }
  }
  if (n === 0) return { ok: true, mean: 255, std: 0 };
  const mean = sumY / n;
  const std  = Math.sqrt(Math.max(0, sumY2 / n - mean * mean));
  // Permissive on brightness (subjects shot in dim rooms have darker but
  // still acceptable backgrounds); strict on uniformity (patterns /
  // shadows / objects raise the variance).
  const ok = mean >= 160 && std <= 22;
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

    // Upscale to full resolution with a tiny blur to round off the 256-px
    // grid, then sigmoid-sharpen the alpha. The blur smooths the grid
    // staircase; the sigmoid then collapses the wide α∈(0,1) band into a
    // 1-2 pixel anti-aliased edge. Low-confidence ghost regions (the
    // segmenter "seeing" a partial person at the frame boundary, for
    // example) get pushed to 0 and disappear entirely.
    const upMask = new OffscreenCanvas(W, H);
    const uctx = upMask.getContext("2d", { willReadFrequently: true });
    uctx.imageSmoothingEnabled = true;
    uctx.imageSmoothingQuality = "high";
    uctx.filter = `blur(${Math.max(1, Math.round(W / 800))}px)`;
    uctx.drawImage(maskCv, 0, 0, W, H);
    uctx.filter = "none";
    const upAlpha = uctx.getImageData(0, 0, W, H).data;
    // α' = clamp((α − 0.5) · GAIN + 0.5). GAIN 6 keeps a ~16% wide soft
    // band around the threshold, which is ~1-2 px after upsampling.
    const GAIN = 6;
    for (let i = 3; i < upAlpha.length; i += 4) {
      const a = upAlpha[i] / 255;
      const s = (a - 0.5) * GAIN + 0.5;
      upAlpha[i] = s <= 0 ? 0 : s >= 1 ? 255 : (s * 255) | 0;
    }

    // Pull the photo at full resolution.
    const srcCv = new OffscreenCanvas(W, H);
    const sctx = srcCv.getContext("2d", { willReadFrequently: true });
    sctx.drawImage(capturedBitmap, 0, 0, W, H);
    const sdata = sctx.getImageData(0, 0, W, H).data;
    const N = sdata.length;

    // Estimate the *old* background color by averaging pixels the mask
    // says are definitely background (alpha < ~3%). That color is what's
    // currently contaminating the edge pixels of the person.
    let br = 0, bg_ = 0, bb = 0, bn = 0;
    for (let i = 0; i < N; i += 16) {        // every 4th pixel is plenty
      if (upAlpha[i + 3] < 8) {
        br += sdata[i]; bg_ += sdata[i + 1]; bb += sdata[i + 2]; bn++;
      }
    }
    const oldR = bn ? br  / bn : 244;
    const oldG = bn ? bg_ / bn : 244;
    const oldB = bn ? bb  / bn : 244;

    // Match the replacement background's brightness to the original so it
    // blends with whatever lighting the subject was shot in. Only cap at
    // the bright end (so we never produce a darker-than-original bg) and
    // apply a low floor to avoid pitch black on extreme cases. Neutral
    // gray (R=G=B) avoids introducing a colour cast.
    const oldLuma = 0.2126 * oldR + 0.7152 * oldG + 0.0722 * oldB;
    const newLuma = Math.max(160, Math.min(250, oldLuma));
    const NEW_R = newLuma | 0, NEW_G = newLuma | 0, NEW_B = newLuma | 0;

    // Per-pixel alpha matting:
    //   observed = α · fg + (1 - α) · oldBg   →   fg = (observed - (1-α)·oldBg) / α
    // We then composite the recovered fg onto the new background using α.
    // This pulls the dark/coloured halo *out* of the edge pixels instead of
    // blending it into the new white, eliminating the "obvious blur" look.
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
