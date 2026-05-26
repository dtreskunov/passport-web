# Passport Photo (web)

Static, client-side companion to the desktop `PassportPhoto` app. Takes a
US-passport-spec square crop from your webcam using
[MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
(WebAssembly) for landmark detection.

No build step, no server, no upload — the camera feed and detection run
entirely in the browser. The model file (~3 MB) is fetched from Google's CDN
on first use.

## One-time setup (per clone)

Enable the cache-busting pre-commit hook so deployed asset URLs get a fresh
`?v=` stamp on each commit (avoids the 10-min GitHub Pages browser cache):

```sh
git config core.hooksPath .githooks
```

## Run locally

`getUserMedia` requires a secure context, so you need `http://localhost` or
HTTPS — `file://` will not work.

Pick any static server:

```powershell
cd $HOME\dev\passport-web

# Node (recommended — installed via winget: OpenJS.NodeJS.LTS)
npx --yes serve -l 8080 .

# Python 3
python -m http.server 8000
```

Open <http://localhost:8080/> (or whichever port you chose) and click
**Start camera**.

## Workflow

1. **Start camera** → grants webcam access.
2. **Capture** → freezes the current frame and runs face detection.
3. The yellow dashed rectangle is the proposed passport crop. Green dots
   mark the detected iris centers and chin; the purple line is the eye
   axis, blue is chin, red is the estimated crown.
4. Tweak **Head %** and **Eye % from bottom** to recompose within spec.
5. **Download crop** → saves a JPEG at the chosen output size.

## US passport spec

| Property                          | Value       |
| --------------------------------- | ----------- |
| Aspect ratio                      | 1:1         |
| Output size (per side)            | 600–1200 px |
| Head height / image height        | 50–69%      |
| Eye line from bottom of image     | 56–69%      |

The status line shows the measured ratios and warns if either falls outside
the spec band.

## Caveats

- The face mesh does not include hair, so the **crown is estimated** from
  the chin→eye distance using a standard 2:1 anthropometric ratio. With
  voluminous hair you may want to manually lower the head-frac.
- US State Department rules forbid digital alteration beyond cropping. This
  tool only crops; it does not retouch, smooth, or relight.
- Browser support: any recent Chromium-based browser, Firefox, or Safari
  with WebAssembly + WebGL. iOS Safari needs a user gesture to start the
  camera (the **Start camera** button satisfies this).
