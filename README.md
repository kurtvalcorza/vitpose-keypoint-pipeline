# ViTPose-base two-stage human pose estimation pipeline

DIMER inference wrapper for **ViTPose-base** (`usyd-community/vitpose-base`), the plain-ViT top-down pose estimator from the University of Sydney's ViTAE group, paired with the pinned **RT-DETR R50-VD** person detector (`PekingU/rtdetr_r50vd`, the sibling `rtdetr-detection-pipeline`'s checkpoint) as stage 1. Both snapshots are pinned to immutable Hugging Face revisions and loaded only from digest-verified local directories. The pipeline takes one image and optional person boxes; without boxes it detects `person` first, then returns 17 COCO keypoints per person with heatmap scores under two caller-owned thresholds, and carries a bounded fine-tuning contract for the ViTPose heatmap head and last encoder blocks on labelled persons. It provides no bottom-up association, 3-D pose, tracking or hand/face keypoints.

## Upstream alignment

- Model: `usyd-community/vitpose-base` @ `95be2991424e646950d656bb7fc15ec9be119700` (Apache-2.0)
- Person detector: `PekingU/rtdetr_r50vd` @ `df939e661d8c52e80608d1ec566561aabd25a4e7` (Apache-2.0)
- Upstream task: top-down 2-D human keypoint detection (17 COCO joints) on person crops
- Repository adaptation: a bounded fine-tuning contract (`evaluate`, `adapt`, `save_artifact`, `from_artifact`) over the ViTPose heatmap head and the last encoder blocks with the heatmap mean-squared error; the detector, the patch embedding and the earlier blocks stay frozen and the base weights are never modified on disk

## Quick start

```python
from PIL import Image
from vitpose_keypoint_pipeline import VitPoseKeypointPipeline

pipe = VitPoseKeypointPipeline.from_pretrained()          # stages + verifies BOTH snapshots first
result = pipe.estimate(Image.open("people.jpg"))          # stage 1 finds `person` boxes, stage 2 the keypoints
for pose in result["poses"]:
    print(pose["box"], pose["person_score"], pose["n_keypoints"])
    for kp in pose["keypoints"]:                          # joints at or above keypoint_threshold
        print(f"  {kp['name']:12} x={kp['x']:.1f} y={kp['y']:.1f} score={kp['score']:.2f}")

# bring your own boxes (skips stage 1) and tune the thresholds per call
result = pipe.estimate(Image.open("sketch.png"), person_boxes=[[180, 80, 460, 560]], keypoint_threshold=0.5)
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the test suite (`tests/test_model_backed.py` runs only where both snapshots are staged and skips otherwise). On a fresh clone both manifests are committed but the weights are not: `VitPoseKeypointPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files of both snapshots at their pinned revisions, then verifies them.

## Weights layout

```
weights/vitpose-base/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (4 files)
  config.json                # VitPoseForPoseEstimation: 17 COCO keypoints, skeleton edges
  preprocessor_config.json   # 192x256 affine crops, ImageNet mean/std
  model.safetensors          # git-ignored, 360,007,012 bytes
  README.md
weights/rtdetr-r50vd/
  dimer-base-manifest.json   # the person detector (4 files)
  config.json  preprocessor_config.json  README.md
  model.safetensors          # git-ignored, 172,175,856 bytes
```

## Adaptation contract

```python
from vitpose_keypoint_pipeline import (
    VitPoseKeypointPipeline, build_sample_dataset, fetch_corpus, load_byod_dataset, mean_pose_baseline, read_corpus, split_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 498 COCO persons, small + JPEG, 290 / 72 / 136
# or: splits = split_dataset(load_byod_dataset("my_people.zip"), seed=42)  # your photographs + keypoints.csv

pipe = VitPoseKeypointPipeline.from_pretrained()                      # cuda:0 if available, else cpu
prior = mean_pose_baseline(splits["train"], splits["test"])           # pck, oks, per_category
frozen = pipe.evaluate(splits["test"])
result = pipe.adapt(splits["train"], splits["validation"], epochs=8, lr=5e-5, trainable_blocks=2)
adapted = pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")                                 # adapter.safetensors + manifest.json
again = VitPoseKeypointPipeline.from_artifact("outputs/adapter")
```

- Records are `{id, image, box, keypoints, category?}`: `image` a PIL image within the ceilings, `box` one `[x0, y0, x1, y1]` person box inside it, `keypoints` a mapping from COCO joint name (`KEYPOINT_NAMES`) to `[x, y]` for every labelled joint (at least `MIN_LABELLED` = 12 of the 17); `validate_dataset` checks the shape (8..5,000 records, unique ids, box and joints inside the image) before any model import, and `split_dataset` keeps every image (by decoded pixels) in one split and every person with its image; `check_split_disjoint` asserts no image is shared.
- The default sample (`samples.py`) is 300 COCO val2017 photographs whose Flickr licence is CC BY 2.0 or CC BY-SA 2.0 (COCO licence ids 4 and 5; the `SAMPLE_IMAGES` pins carry each photograph's byte size, SHA-256, Flickr page, licence and its persons' CC BY 4.0 keypoint labels), fetched from `images.cocodataset.org` at run time and cached git-ignored under `weights/coco-val-persons/`. `degrade` is the stated input shift — the whole photograph downscaled so the person box is `TARGET_HEIGHT` (40) px tall, re-encoded as JPEG at `JPEG_QUALITY` (30), box and joints scaled to match — and `read_corpus(..., target_height=None)` gives the clean counterparts; `category` is the labelling density (`full` 16–17 joints, `partial` 14–15, `sparse` 12–13). `build_sample_dataset` draws a seeded 180 / 45 / 75 image-level split (`SAMPLE_DIGEST` pins the draw).
- `evaluate(records)` runs `predict_keypoints` on every record — `estimate` with the record's box as a caller box, all 17 joints — and returns `pose_metrics` (`metrics.py`): PCK (joints within `PCK_FRACTION` of the box's longest side) and mean OKS (the COCO kernel with COCO's per-joint sigmas and the box area), overall and per category, plus `per_record`, `verdict` (`measured` / `measured-small-sample`) and `adapted`. `box_centre_baseline` and `mean_pose_baseline` are the two box-only references the tutorial scores beside the model.
- `adapt(train, val=None, *, epochs=8, lr=5e-5, batch_size=8, trainable_blocks=2, seed=0, progress=None)` trains only the heatmap head and the last `trainable_blocks` encoder blocks with the final backbone LayerNorm (18,376,977 of 89,994,513 parameters by default): every box is affine-warped to the processor's 192×256 crop exactly as at inference, the labelled joints become Gaussian targets (sigma `HEATMAP_SIGMA` = 2) on the 64×48 heatmap grid, and the head's heatmaps are trained with the joint-weighted mean-squared error ViTPose was trained with; AdamW without weight decay, gradient clipping at 1.0, seeded shuffling, BatchNorm statistics frozen; epoch 0 records the frozen model and the epoch with the highest mean of validation PCK and OKS is kept. The update is transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.vitpose-base.adapter.v1`: base id, revision and weight digest, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies both base snapshots, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the head and the recorded blocks, and overlays the tensors onto a freshly loaded base.

## Input ceilings and thresholds

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PERSONS = 50`; `DETECTION_THRESHOLD = 0.3` on the detector's `person` sigmoid and `KEYPOINT_THRESHOLD = 0.3` on each joint's heatmap maximum (both the pinned ViTPose README example's values); `KEYPOINT_NAMES` (17, `config.json` order) and `SKELETON_EDGES` (19). Person boxes are pixel xyxy inside the image. See `MODEL_CARD.md` for who owns tuning the thresholds, the score semantics, the measured timings and the adaptation build record.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/tutorials/vitpose_keypoint_colab.ipynb)

`tutorials/vitpose_keypoint_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's three modules (`pipeline.py`, `metrics.py`, `samples.py`), both model identities, both manifest digest lists and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`; see `tutorials/README.md`). It stages and digest-verifies both snapshots, fetches the 300 digest-pinned COCO photographs and turns their 498 labelled persons into the stated small-person input (box 40 px tall, JPEG 30) with the clean counterparts kept for comparison, splits them by photograph without leakage, estimates a synthetic drawing through the two-stage inference contract with an input manifest and a rejection probe (the detector finds no `person` on the drawing; stage 2 receives the drawn box), scores the frozen model's PCK and mean OKS on the 136 held-out small persons beside the box-centre and mean-pose baselines and on the clean counterparts (the frozen model is near the mean-pose prior on small persons and far above it on clean ones), runs a bounded fine-tuning of the heatmap head and the last two encoder blocks with the heatmap MSE and epoch selection on the mean validation PCK and OKS, re-scores the held-out split per category on both inputs, re-estimates the drawing and four persons as side-by-side skeleton panels, exports the adapter and reloads it with verified joint parity, and writes `vitpose_keypoint_train.csv`, `vitpose_keypoint_input_manifest.json`, `vitpose_keypoint_scene_frozen.png`, `vitpose_keypoint_scene_adapted.png`, `vitpose_keypoint_evaluation_report.json`, `vitpose_keypoint_examples/`, `vitpose_keypoint_adapter/` and `vitpose_keypoint_result.json` under `outputs/`.

The default path runs on CPU and uses CUDA automatically when present (about two and a half minutes on an RTX 5070 Ti after the downloads; a 2-vCPU hosted runtime will take much longer for the eight ViT-B epochs). The metrics it prints are one seeded split of one 498-person sample under one degradation — evidence that the adaptation contract works, not a pose benchmark or production-fitness evidence.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. The clean-runtime run of the `E2E` notebook is pending; complete `docs/release-verification.md` against the exact release revision before calling the notebook release-grade. See `STATUS.md`.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests for both snapshots, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes for both snapshots.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). Both upstream checkpoints are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
