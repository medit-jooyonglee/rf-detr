# Mouth segmentation dataset builder

`build_dataset.py` turns the FLUX2 dental portraits into fixed-size mouth crops and FDI-valued tooth instance masks.

## Output contract

Each sample uses the same six-digit ID in these directories:

- `images/000000.png`: MediaPipe mouth crop (default `640x320`)
- `masks/000000.png`: lossless `uint8` mask; `0` is background, FDI numbers are foreground, and `255` is ignore
- `overlays/000000.png`: color-blended review image with FDI labels
- `metadata/000000.json`: source/crop transform, model hashes, instance statistics, assignment warnings, and timings
- `manifest.json` and `manifest.jsonl`: records gathered from the per-sample metadata
- `work/`: resumable intermediate output of the legacy segmentation model

The mask representation can be converted to COCO without recovering information from a rendered image: one connected FDI-valued region is one instance/category annotation.

FDI assignment is a positional heuristic because the bundled segmentation model predicts instances, not tooth identities. Predictions outside MediaPipe's inner-mouth ROI are removed. Review samples with status `review` (or `segmentation.review_required: true`) before treating labels as ground truth; `failed` samples are not usable.

## Environment

The existing `filterseg` Conda environment has the CUDA/scientific packages needed by the tooth model. Clone it so MediaPipe cannot disturb packages used by other projects, then install versions compatible with NumPy 1.x:

```powershell
conda create -n mouthseg --clone filterseg
conda run -n mouthseg python -m pip install "numpy==1.26.4" "opencv-contrib-python==4.11.0.86" "mediapipe==0.10.21"
```

The builder downloads Google's Face Landmarker model to `thirds/face_detector/models/face_landmarker.task` on first use. Supply `--no-download-model` for an offline run.

## Commands

Run a three-image smoke test:

```powershell
conda run -n mouthseg python thirds/face_detector/build_dataset.py --limit 3 --output artifacts/mouth_seg_smoke
```

Inspect `artifacts/mouth_seg_smoke/overlays`, then run all images:

```powershell
conda run -n mouthseg python thirds/face_detector/build_dataset.py --output artifacts/mouth_seg_dataset
```

Stages are resumable. For example, crop everything first and segment later:

```powershell
conda run -n mouthseg python thirds/face_detector/build_dataset.py --stage crop --output artifacts/mouth_seg_dataset
conda run -n mouthseg python thirds/face_detector/build_dataset.py --stage segment --output artifacts/mouth_seg_dataset
```

Use `--overwrite` to regenerate completed artifacts. Use `--prune-work` only when intermediate checkpoint files are no longer needed for resuming.

## Small binary-label seed set

For a teeth-vs-background proof of concept, create tighter `512x256` crops and then select 50 diverse samples in which the lower teeth are visible:

```powershell
conda run -n mouthseg python thirds/face_detector/build_dataset.py --stage crop --output artifacts/teeth_crop_dataset --crop-width 512 --crop-height 256 --horizontal-margin 0.15 --vertical-margin 0.25
conda run -n mouthseg python thirds/face_detector/select_label_seed.py --dataset artifacts/teeth_crop_dataset --output artifacts/teeth_label_seed_50 --count 50 --overwrite
```

The seed directory contains copied images, empty `masks/` for annotations, one metadata JSON per image, two contact sheets for quick review, and a combined manifest. Binary masks should use `0` for background and `255` for all visible teeth.
