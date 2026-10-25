# Face-blur experiments

This benchmark evaluates the current `video_privacy.py` implementation against **WIDER FACE** still-image boxes and attributes. WIDER images are transformed into short, deterministic synthetic clips (12-fps pan/zoom, low-light and motion-blur variants, plus hard cuts); these are a **proxy, not user footage or real video**. Splits are assigned by source-image identifier before synthetic frames/clips are built, so no source still crosses tuning and held-out splits. WIDER is a research benchmark, not a substitute for consented, representative deployment footage. Check dataset and InsightFace model terms before any non-research/commercial use.

## Proxy inputs (external to repository)

The annotated archive is available without login from the public Hugging Face mirror `CUHK-CSE/wider_face`, file `data/wider_face_split.zip` (official WIDER FACE bounding-box/attribute labels). For this small run, only individually selected train stills are fetched from the public mirror `yuuMQ/Retinaface_WiderFaceDataset`, under `widerface/train/images/<category>/<file>`. Neither annotations nor downloaded images belong in this repository. The checked-in results contain dataset revision and aggregate SHA-256 fingerprints, not image filenames, IDs, images, or model weights. The external preparation manifest contains the selected-image hashes.

## Reproduce

The small proxy uses 8 randomly seeded, labeled positive images and all 4 WIDER train images annotated with no faces. The annotation mirror is pinned to `db171f1b7fedf4d3453e81297ff02f9915356d19`; the individual-image mirror is pinned to `04cab2dc3e18be03e2935b731c1dfa9eb29a31b5`. The external manifest records the selected-image SHA-256 values. The harness results include annotation and aggregate source-manifest SHA-256 fingerprints, split seed, source counts, and clip-builder version; filenames/image IDs and image pixels are not committed.

Use Python 3.12 in `tools/Dockerfile.face-blur-benchmark`; benchmark-only packages stay in `tools/requirements-benchmark.txt`. `libgl1`, `libglib2.0-0`, and `libgomp1` are required by OpenCV/ONNX Runtime. From the repository root, build a clean source copy (build and run from the copied source; datasets, model weights, MLflow store, and failure frames are on external volumes):

```powershell
$repo = (Get-Location).Path
$context = Join-Path $repo '.faceblur-context'
$archive = Join-Path $repo '.faceblur-context.tar'
tar.exe --exclude=.venv --exclude=data --exclude=logs --exclude=.git --exclude=__pycache__ --exclude=.pytest_cache --exclude=.copilot-azure --exclude=.faceblur-context -cf $archive .
New-Item -ItemType Directory -Force $context | Out-Null
tar.exe -xf $archive -C $context
docker build -f (Join-Path $context 'tools\Dockerfile.face-blur-benchmark') -t robotics-face-blur-benchmark:local $context
Remove-Item -Recurse -Force $context, $archive
```

Prepare the proxy into a directory outside the repo, then run each ordered step from PowerShell. This example mounts only the source tools, while dataset, weights, MLflow file store, and failure frames live in the host temp directory. Results are the only experiment output mounted back into the repo. Each step is a separate bounded Docker run:

```powershell
$repo = (Get-Location).Path
$commit = (git rev-parse HEAD).Trim()
$external = Join-Path $env:TEMP 'face-blur-wider-proxy'
$mlflow = Join-Path $env:TEMP 'face-blur-mlflow'
$failures = Join-Path $env:TEMP 'face-blur-failures'
New-Item -ItemType Directory -Force $external, $mlflow, $failures | Out-Null
$mounts = @('-v', "${repo}\tools:/app/tools", '-v', "${external}:/external",
  '-v', "${mlflow}:/mlflow", '-v', "${failures}:/failures",
  '-e', 'MLFLOW_TRACKING_URI=file:///mlflow', '-e', 'MLFLOW_ALLOW_FILE_STORE=true',
  '-e', "GIT_COMMIT=$commit")
docker run --rm @mounts robotics-face-blur-benchmark:local python -m tools.prepare_wider_proxy `
  --output-dir /external/wider --seed 20261025 --positive-count 8
foreach ($step in @('baseline','tune','tiles','track','conservative','final')) {
  docker run --rm @mounts robotics-face-blur-benchmark:local python -m tools.face_blur_benchmark `
    --annotations /external/wider/wider_face_train_bbx_gt.txt `
    --images-root /external/wider/images --weights-dir /external/weights `
    --failure-dir /failures --results-dir /app/tools/results `
    --step $step --image-limit 0 --frames 2 --bootstraps 1000 --git-commit $commit
}
```

For this completed experiment, only the public WIDER annotation archive and 12 selected JPEGs were downloaded; no user footage was available. YuNet weights are fetched from the pinned OpenCV Zoo URL and SHA-256 verified by production code. SCRFD uses InsightFace `buffalo_sc` from the official InsightFace release; each result records the actual detector ONNX SHA-256. InsightFace model weights are licensed for non-commercial research only, unlike the MIT-licensed package, so the SCRFD result is a research comparison and is not a production recommendation. Results are written to `tools/results/face_blur_*.json`, with one MLflow run per evaluated config/split in experiment `face-blur-privacy`. Do not point this benchmark at Azure MLflow. To replay saved JSON runs to an authorized tracking URI, use `python -m tools.replay_to_mlflow --results-dir tools/results`; the replayer requires only `mlflow-skinny` and the JSON files.

## Measurement definitions

`R_exposed` is the ratio of annotated visible face-frames with less than the specified coverage threshold (50%, 90%, or 100%) from the actual pixelation output ROI, including the production 20% horizontal and 25% top/bottom margins and the configured tracker output, with the production 0.25-second hold represented at 12 fps (3 frames). Per-category rates cover face height `<32`, `32–64`, and `>64` pixels, WIDER occlusion/pose attributes, synthetic low-light/motion-blur clips, and hard-cut frames. Report 95% bootstrap intervals by clip. Over-masking area is masked pixels outside all GT face boxes divided by all evaluated frame pixels. False-mask rate is the share of frames with no annotated faces that nevertheless contain a mask. Runtime is CPU detector/tracker milliseconds per frame and excludes clip construction and model initialization. The 90% cutoff is a metric definition, **not an acceptance threshold**; an acceptance threshold must come from the applicable privacy requirement.

The experiment grid is deliberately small and should not be read as a claim of statistical power. Failure images are saved only to the external failure directory and are never committed. Step 7 must not fine-tune unless held-out failures demonstrate a repeated, learnable pattern; transformed copies of a small public proxy are not appropriate training evidence.

## Results

No production defaults in `video_privacy.py` were changed. The proxy contains 12 source stills (8 tune, 4 held out), transformed into 28 tune clips / 64 tune frames and 14 held-out clips / 32 held-out frames. All frames, including synthetic hard-cut frames, were passed through the output-mask evaluator. CIs below are 95% clip-bootstrap intervals; with only four held-out source images, zero-width intervals do not demonstrate a zero population risk.

### Held-out comparison at 90% GT coverage

| Category | YuNet baseline | SCRFD baseline | Tuning-selected final (SCRFD, no tracker) |
|---|---:|---:|---:|
| Overall (n=72 face-frames) | 25.00% [11.76, 35.90] | 23.61% [17.57, 31.43] | 2.78% [0, 8.57] |
| Face height <32 px | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| Face height 32–64 px (n=48) | 20.83% [7.14, 38.89] | 20.83% [5.56, 30.77] | 4.17% [0, 11.54] |
| Face height >64 px (n=24) | 33.33% [33.33, 33.33] | 29.17% [20.83, 33.33] | 0% [0, 0] |
| Occlusion (n=48) | 37.50% [20.00, 46.75] | 35.42% [23.33, 54.56] | 4.17% [0, 13.64] |
| Pose/profile | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| Low light (n=18) | 33.33% [25.00, 40.00] | 33.33% [20.00, 50.00] | 11.11% [0, 25.00] |
| Motion blur (n=48) | 25.00% [9.09, 50.00] | 22.92% [10.86, 34.48] | 4.17% [0, 12.50] |
| Scene-transition faces | N/A (n=0) | N/A (n=0) | N/A (n=0) |

Overall `R_exposed` at 50% / 90% / 100% coverage (each with its own 95% clip-bootstrap CI) was YuNet **20.83% [8.82, 30.26] / 25.00% [11.76, 35.90] / 37.50% [27.14, 47.22]**, SCRFD baseline **23.61% [17.95, 32.35] / 23.61% [17.57, 31.43] / 23.61% [17.95, 32.37]**, and the tuning-selected final **2.78% [0, 8.82] / 2.78% [0, 8.57] / 2.78% [0, 9.09]**. N/A means no annotated held-out face-frames in that category, not zero exposure. The final's two exposed face-frames came from the same low-light crowd clip; they were small/distant, low-contrast faces, and were missed by detection-only output. This is one source-level failure, not independent evidence across the dataset.

| Held-out configuration | Over-mask area fraction | False-mask rate | CPU ms/frame |
|---|---:|---:|---:|
| YuNet baseline (0.35/0.35, production tiles/tracker) | 9.55% | 100% | 729.11 |
| SCRFD baseline (0.35/0.35, production tiles/tracker) | 36.26% | 100% | 319.05 |
| Final selected by tuning-only score (0.25/0.35, 0.7-scale/3×3 tiles, no tracker) | 41.91% | 100% | 533.50 |

False-mask rate is 100% because every held-out no-face frame had at least one output mask. The reported exposure reduction therefore trades against substantial masking of non-face image area and false masks; no privacy or usability acceptance threshold is inferred. Runtime is a small CPU-container measurement and is noisy, especially at this sample size.

### Ordered experiment decisions

1. **Detector baseline:** Public SCRFD weights were obtained and SHA-256 recorded (`5e4447f5…a8b4ea3a`). On held-out WIDER proxy, baseline SCRFD did not beat YuNet at the 90% metric (23.61% vs 25.00%; overlapping clip-bootstrap CIs). Its lower measured latency came with much higher over-mask area. InsightFace's package is MIT-licensed, but these pretrained weights are restricted to non-commercial research; this is not a production recommendation.
2. **Threshold/NMS:** All nine SCRFD combinations were measured on tune; the three best tuning points were evaluated held out. Tune-only selection chose score threshold **0.25**, NMS **0.35** (tune R90 0.28%, CI [0, 1.09%], over-mask area 60.9%). Held-out R90 was 0%, CI [0, 0], with 48.6% over-mask area and 100% false-mask rate. This small sample is not evidence of zero risk.
3. **Overlapping tiles:** Among the four tuning candidates, selected scale **0.7**, overlap **0.25**, grid **3×3**; tune R90 remained 0% while over-mask area decreased slightly from 60.9% to 60.1%. Held-out R90 was 0%, over-mask area 47.0%. A 0.5-scale candidate had held-out R90 6.94% [1.52, 11.84], so it was not selected. Current production tiling remains unchanged.
4. **Tracking:** Compared current velocity tracker, detection-only, IoU-only and Kalman-box tracking. Tune R90 was 0% for all; detection-only had the lowest tune over-mask area (55.11%). Held out, detection-only exposed 2.78% [0, 8.57], whereas the three trackers each had 0% observed exposure but about 47% over-mask area. The final config follows the declared tune-only lexicographic selection (R90, then over-mask area, then runtime), not held-out outcomes; the four-source held-out split is too small to resolve this trade-off.
5. **Conservative masking:** On the velocity-tracker base, lower confidence, 3× hold and growing box inflation produced no observed R90 improvement on either split. Tune over-mask area increased from 60.09% (unchanged 0.25 threshold / 3-frame hold) to 60.51% (9-frame hold, +5% per miss), 67.19% (0.15 threshold), or 67.90% (0.15 threshold, 9-frame hold, +10% per miss). No conservative variant was adopted.
6. **Every-frame failures:** Final selected detection-only configuration evaluated all 32 held-out output frames, including synthetic transitions; two R90 failures were saved outside the repository. Viewed failure frames show a low-contrast, distant face in a crowd under low light. No held-out GT faces occurred on the marked transition frames, so transition-specific R_exposed is N/A. No user footage was available.
7. **Fine-tuning:** Not performed. The two selected-final failures are consecutive frames of one source scene, while the held-out split has only four source images and lacks examples in several requested categories. This is insufficient independent, representative evidence for a learnable pattern, and the held-out split must not be used for training.

The saved result JSONs contain 44 configuration/split trials across the six executed steps (baseline, tune, tiles, track, conservative, final); the nine-point threshold grid and top-three held-out checks are separately logged. The selected final is **not** a recommendation: the measured proxy cannot establish a real-world privacy guarantee. A deployment decision requires a privacy-owner-defined acceptance threshold and consented, representative labeled clips, including distant/small faces, profiles, occlusion, low light, motion blur, crowded and no-face scenes, and scene cuts. Split these by source video/camera/session (never by adjacent frame), and evaluate every output frame.
