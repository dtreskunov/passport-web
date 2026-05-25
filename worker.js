// MediaPipe worker — hosts FaceLandmarker, FaceDetector and ImageSegmenter
// off the main thread so that wasm instantiation, model load, GPU shader
// compile and per-detect inference no longer freeze the UI.
//
// Protocol: main thread posts { id, op, ... } and the worker replies with
// { id, ok: true, result } or { id, ok: false, error }. Bitmaps are
// transferred (zero-copy) on the way in; mask Float32Arrays are
// transferred on the way out.

// Classic worker (not a module) because MediaPipe's tasks-vision package
// internally calls importScripts() to load its wasm loader, and that call
// is disallowed inside module workers. We dynamic-import() the ESM bundle
// from the classic worker — supported in all evergreen browsers.
let FaceLandmarker, FaceDetector, ImageSegmenter, FilesetResolver;
const visionReady = import(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"
).then((m) => {
  ({ FaceLandmarker, FaceDetector, ImageSegmenter, FilesetResolver } = m);
});

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

let filesetPromise = null;
let landmarkerPromise = null;
let detectorPromise = null;
let segmenterPromise = null;

function getFileset() {
  if (!filesetPromise) {
    filesetPromise = visionReady.then(() =>
      FilesetResolver.forVisionTasks(WASM_BASE)
    );
  }
  return filesetPromise;
}

function getLandmarker() {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    const fs = await getFileset();
    return FaceLandmarker.createFromOptions(fs, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
    });
  })();
  return landmarkerPromise;
}

function getDetector() {
  if (detectorPromise) return detectorPromise;
  detectorPromise = (async () => {
    const fs = await getFileset();
    return FaceDetector.createFromOptions(fs, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
    });
  })();
  return detectorPromise;
}

function getSegmenter() {
  if (segmenterPromise) return segmenterPromise;
  segmenterPromise = (async () => {
    const fs = await getFileset();
    return ImageSegmenter.createFromOptions(fs, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
  })();
  return segmenterPromise;
}

// Warm: load landmarker and run a dummy detect so the GPU shader-compile
// cost is paid up-front rather than on the first real detect.
async function warm() {
  const lm = await getLandmarker();
  const c = new OffscreenCanvas(64, 64);
  const cctx = c.getContext("2d");
  cctx.fillStyle = "#888";
  cctx.fillRect(0, 0, 64, 64);
  try { lm.detect(c); } catch { /* best-effort */ }
}

async function detectLandmarks(bitmap) {
  const lm = await getLandmarker();
  const res = lm.detect(bitmap);
  if (res.faceLandmarks && res.faceLandmarks.length) {
    // Return as a flat Float32Array for cheap transfer: [x0,y0,z0, x1,y1,z1, …].
    const pts = res.faceLandmarks[0];
    const out = new Float32Array(pts.length * 3);
    for (let i = 0; i < pts.length; i++) {
      out[i * 3]     = pts[i].x;
      out[i * 3 + 1] = pts[i].y;
      out[i * 3 + 2] = pts[i].z;
    }
    return { landmarks: out };
  }
  return { landmarks: null };
}

async function detectFace(bitmap) {
  const det = await getDetector();
  const res = det.detect(bitmap);
  const box = res?.detections?.[0]?.boundingBox;
  if (!box) return { box: null };
  return { box: { x: box.originX, y: box.originY, w: box.width, h: box.height } };
}

async function segment(bitmap) {
  const seg = await getSegmenter();
  const res = seg.segment(bitmap);
  const mask = res.confidenceMasks[0];
  // Float32Array of background-confidence values, mw*mh.
  const arr = mask.getAsFloat32Array();
  // Copy because MediaPipe owns the underlying buffer and will free it.
  const out = new Float32Array(arr.length);
  out.set(arr);
  const mw = mask.width, mh = mask.height;
  for (const m of res.confidenceMasks) m.close?.();
  res.close?.();
  return { mask: out, mw, mh };
}

self.addEventListener("message", async (ev) => {
  const { id, op } = ev.data;
  try {
    let result, transfer = [];
    switch (op) {
      case "warm":
        await warm();
        result = null;
        break;
      case "landmarks": {
        const r = await detectLandmarks(ev.data.bitmap);
        ev.data.bitmap.close?.();
        if (r.landmarks) transfer.push(r.landmarks.buffer);
        result = r;
        break;
      }
      case "face": {
        const r = await detectFace(ev.data.bitmap);
        ev.data.bitmap.close?.();
        result = r;
        break;
      }
      case "segment": {
        const r = await segment(ev.data.bitmap);
        ev.data.bitmap.close?.();
        transfer.push(r.mask.buffer);
        result = r;
        break;
      }
      default:
        throw new Error("unknown op: " + op);
    }
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err) });
  }
});
