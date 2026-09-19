# Weight provenance and DIMER hosting

Two pinned snapshots, each with its own manifest, verifier and stager; the pipeline refuses to load unless both verify.

## Stage 2 — ViTPose-base (the profile's model)

- Upstream: `usyd-community/vitpose-base`
- Immutable revision: `95be2991424e646950d656bb7fc15ec9be119700` (the Hub's `main` resolved to this commit on 2026-09-14)
- Weight format: SafeTensors (`model.safetensors`, 360,007,012 bytes, float32); the upstream repository hosts no pickle checkpoint at this revision.
- Manifest: `weights/vitpose-base/dimer-base-manifest.json` (4 files: `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`; 360,020,483 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter and the Hub's licence tag)
- Loader: Transformers `VitPoseForPoseEstimation` / `AutoImageProcessor` (`VitPoseImageProcessor`, 192×256 affine crops) with `trust_remote_code=False`, `local_files_only=True` from the verified directory; the config's `use_pretrained_backbone` and `use_timm_backbone` are both false, so nothing is fetched at construction.

## Stage 1 — RT-DETR R50-VD person detector

- Upstream: `PekingU/rtdetr_r50vd` — the same checkpoint the sibling `rtdetr-detection-pipeline` wraps; only its `person` class is used here. (The pinned ViTPose README's example uses `PekingU/rtdetr_r50vd_coco_o365`; this repository pins the COCO-only sibling so the fleet carries one detector identity.)
- Immutable revision: `df939e661d8c52e80608d1ec566561aabd25a4e7`
- Weight format: SafeTensors (`model.safetensors`, 172,175,856 bytes, float32); no pickle checkpoint at this revision.
- Manifest: `weights/rtdetr-r50vd/dimer-base-manifest.json` (4 files: `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`; 172,190,863 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0
- Loader: Transformers `RTDetrForObjectDetection` / `AutoImageProcessor` with `trust_remote_code=False`, `local_files_only=True`, `use_pretrained_backbone=False` (in-library `RTDetrResNet` backbone).

## Hosting and staging

- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use for both checkpoints subject to preservation of the licence and notices. The Git repository does not vendor either checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror both pinned snapshots in its model store under the upstream licences. A DIMER profile for this pipeline needs **both** snapshots.
- Fresh clone: `stage_missing_files(allow_download=True)` and `stage_missing_detector_files(allow_download=True)` fetch only the manifest-listed files absent on disk, at the pinned revisions, into their snapshot directories; `verify_snapshot()` and `verify_detector_snapshot()` then check every file before any load (`from_pretrained` does all four steps). `weights/**` is marked `-text` in `.gitattributes` so Windows `core.autocrlf` cannot rewrite the committed small files and break their digests.
- The smoke run loaded and ran both stages with `HF_HUB_OFFLINE=1`; nothing is fetched at inference time.
- Adaptation sample cache: `weights/coco-val-persons/` (git-ignored) holds the 300 pinned COCO val2017 photographs (`<image_id>.jpg`, about 52 MB) fetched by `samples.fetch_corpus` from `http://images.cocodataset.org/val2017/`; every cached file is re-hashed against `SAMPLE_IMAGES` on read. Adapters written by `save_artifact` (`adapter.safetensors` + `manifest.json`; the heatmap head, the last encoder blocks and the final LayerNorm only) are outputs, not part of either snapshot, and `from_artifact` re-verifies both snapshots before overlaying them.
