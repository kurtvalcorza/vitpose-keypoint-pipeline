"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package. This package pins TWO snapshots (ViTPose and the RT-DETR
person detector): ``rewrites`` makes the shared weights root working-directory-relative and ``extra_weights`` carries
the detector manifest inline with its own stage/verify helpers.

This template configures an E2E keypoint fine-tuning workflow: both pinned snapshots are digest-verified and loaded,
300 permissively licensed COCO val2017 photographs (498 labelled persons) are fetched with per-file digests and
turned into small, JPEG-degraded persons (box 40 px tall, quality 30), the records are validated and split by image,
a synthetic drawing is estimated through the two-stage inference contract, the frozen model's PCK and mean OKS over
the held-out persons are measured beside two box-only baselines and against the clean counterparts, a bounded
fine-tuning of the heatmap head and the last two encoder blocks runs with the heatmap MSE, the held-out split is
scored again per category, the drawing and four held-out persons are re-estimated with the adapted model, and the
adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "vitpose_keypoint_pipeline",
    "repo_name": "vitpose-keypoint-pipeline",
    "stem": "vitpose_keypoint",
    "notebook_name": "vitpose_keypoint_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    # VTP-S2: the default path also reaches images.cocodataset.org (plain HTTP, every file SHA-256-pinned), so the
    # generated access bullet names both hosts instead of "the Hugging Face Hub only".
    "external_access": (
        "the Hugging Face Hub, to fetch the pinned `{MODEL_ID}` snapshot and the pinned detector snapshot (~{total_mb:.0f} MB "
        "together) at their pinned revisions (`{MODEL_REVISION:.12}…` for the pose model), and "
        "`images.cocodataset.org`, for the 365 pinned photographs named in the data bullet above — over plain HTTP, so "
        "integrity rests on the per-file byte size and SHA-256, which every file is checked against. No credentials are "
        "required; nothing is installed from this repository."
    ),
    "pipeline_class": "VitPoseKeypointPipeline",
    "weights_key": "vitpose-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "rtdetr-r50vd",
            "var": "DETECTOR_MANIFEST",
            "dir": "DETECTOR_WEIGHTS_DIR",
            "identity": ["DETECTOR_MODEL_ID", "DETECTOR_REVISION"],
            "stage": "stage_missing_detector_files",
            "verify": "verify_detector_snapshot",
        }
    ],
    "model_load": "VitPoseKeypointPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers"],
    "title": "ViTPose-base + RT-DETR — DIMER E2E keypoint fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/vitpose-keypoint-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/tutorials/vitpose_keypoint_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-usyd--community%2Fvitpose--base-ffcc4d?style=flat",
            "https://huggingface.co/usyd-community/vitpose-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-ViTAE--Transformer%2FViTPose-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/ViTAE-Transformer/ViTPose",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2204.12484-b31b1b.svg", "https://arxiv.org/abs/2204.12484"),
    ],
    "capability": "two-stage human pose estimation (`person` boxes from the pinned `PekingU/rtdetr_r50vd` detector or boxes you supply, then 17 COCO keypoints per person from the pinned `usyd-community/vitpose-base` weights) and bounded supervised fine-tuning of the ViTPose heatmap head and last encoder blocks on labelled persons",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `usyd-community/vitpose-base` and `PekingU/rtdetr_r50vd` snapshots (a 360 MB and a 172 MB `model.safetensors`; "
        "no pickle is opened anywhere), fetches the 365 pinned COCO val2017 photographs from the COCO image host (about 63 MB, "
        "each refused on any byte-size or SHA-256 mismatch), turns their 594 labelled persons into the stated small-person "
        "input (the whole image downscaled so the person box is 40 px tall, then JPEG at quality 30) with the boxes and joints "
        "scaled to match, splits 300 of them by image into 180 / 45 / 75 photographs (train, validation and the development test, whose scores earlier revisions read) and keeps the other 65, never scored before this revision, as the fresh test (290 / 72 / 136 / 96 persons), estimates a synthetic "
        "drawing through the two-stage inference contract with an input manifest and a rejection probe, measures the frozen "
        "model's PCK and mean OKS over the 96 fresh-test persons beside the box-centre and mean-pose baselines and against the "
        "same persons at full resolution, runs the pre-registered recipe grid (the heatmap head alone, with the last two and with the last four encoder blocks) with "
        "the heatmap mean-squared error and epoch selection on the mean validation PCK and OKS, chooses the configuration on validation by a rule fixed in advance, scores the fresh test for it only under the pre-registered settings (and, labelled, the development test) "
        "again per category, re-estimates the drawing and four held-out persons with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify keypoint parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.5 §3.1, §5). On a T4 the grid is estimated at about 4 minutes (an estimate from the Colab T4 run of the previous revision (blob 49fc2c05), not a measurement); a CUDA runtime "
        "is used automatically when present, and on a 2-vCPU hosted runtime expect the three fine-tunes to take well over an hour."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell (it first puts the model back to the pinned base) to supply photographs plus a `keypoints.csv` (`file`, `person`, `x0`, `y0`, `x1`, `y1`, optional `category`, and one "
        "`<joint>_x` / `<joint>_y` pair per labelled COCO joint, blank when unlabelled — at least 12 of the 17 per person, at "
        "at least **12 persons on 12 single-person images**, or more persons per image with at least 8 on the training images — "
        "`min_byod_records()`). Your persons are used as labelled — the notebook does not degrade them — and pass through the "
        "same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity "
        "cells as the COCO sample. Uploaded files stay inside this runtime. BYOD is optional and never part of the default path; a BYOD run is exploratory, so its held-out rows are your validation persons."
    ),
    "intro": (
        "ViTPose [1] is a **top-down** pose estimator: it needs a person box first. Stage 1 runs the RT-DETR R50-VD detector [2] "
        "(the same checkpoint the sibling `rtdetr-detection-pipeline` wraps) over the whole image and keeps the boxes "
        "labelled `person` above a threshold; stage 2 affine-warps each box to a 192×256 crop, runs a plain ViT-B "
        "backbone with a heatmap decoder (89,994,513 parameters, of which the head is 4,199,697; published under the "
        "**Apache-2.0** licence), and reads 17 COCO keypoints — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, "
        "ankles — back in input-pixel coordinates, each with the maximum of its 64×48 heatmap as its score (a ranking "
        "signal, not a calibrated probability).\n\n"
        "What this notebook adds to inference is **adaptation with labelled persons** under a stated input shift. "
        "ViTPose was trained on COCO persons at the crop resolution a 256-px box gives it; on the same photographs the "
        "frozen model reaches a PCK of **0.977** on the 96 fresh-test persons (the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook)), so there is "
        "nothing honest to gain there. The input the adaptation is about is the **distant, low-quality person** a "
        "surveillance camera or a wide shot hands a pose estimator: each photograph is downscaled so the person box is "
        "**40 px tall** and re-encoded as JPEG at quality 30, and on that input the frozen model drops to **0.613** — "
        "not far above the 0.419 that placing every joint at its mean position in the box achieves with no model at all. "
        "So the honest question is narrow: does a bounded fine-tuning — of the heatmap head alone or with the last two or four encoder blocks, chosen on validation by a rule fixed in advance — on "
        "290 such persons move the held-out **PCK** and **mean OKS** on an image-disjoint test split, per category, past the "
        "two **box-only baselines** — while the same model still scores the clean, full-resolution persons as before? "
        "Nothing here is a claim about your cameras: it is one seeded split of one sample under one stated degradation.\n\n**Pre-registered recipe (fixed before any grid score was read).** The task (each person downscaled to a 40-px box and re-encoded as JPEG at quality 30) and the training settings (learning rate 5e-5, batch 8, eight epochs, split seed 42, fine-tuning seed 0) were set during earlier development, when the scores of what is now the **development test** had been read; they are not tuned here. This revision chooses only among three configurations, on validation: **G1** the heatmap head only, **G2** the head and the last two encoder blocks, **G3** the head and the last four, each keeping its best epoch by the mean of validation PCK and OKS. The rule: a configuration counts only if its best validation score beats the frozen model's (epoch 0) by more than 0.01, and if none does, the frozen model is the result; among those that count, a configuration with more trainable blocks replaces a smaller one only if it beats it by more than 0.01. The margin is a preference for the smaller model, not a significance test; bootstrap intervals are printed beside it. The **fresh test**, 65 qualifying val2017 photographs (96 persons) that no revision scored before, is scored only under the pre-registered settings, for the chosen configuration, after the choice (see `docs/release-verification.md` for the recorded evaluation and its reproductions); the development test (75 photographs, 136 persons) is scored beside it for continuity and is labelled as such. **Exploratory runs.** Every setting named here is part of the pre-registration, together with the data source (no BYOD), the split seed and the grid's margin; they are all set in Section 4. Change any of them (a form field, BYOD, Section 10's experiment, or a smaller batch for memory) and the run is **exploratory**: it says so when it starts, compares on validation only, never scores the fresh test, and writes its reports to separate `*_exploratory_<n>.json` files. A later hosted run with identical settings is a **reproduction** of the pre-registered evaluation; every report carries `test_look` (`pre-registered evaluation`, `reproduction` or `not scored (exploratory)`) and the list of settings that differ. Every score uses the labelled COCO person boxes, not the detector's, so it is an upper bound on the two-stage system. (See **What did not work** for what an earlier revision did.)\n\n**Data overlap.** The photographs come from COCO val2017 [3]. ViTPose-base was trained on COCO train2017, not val2017, but the upstream training configuration [7] used val2017 to select its released checkpoint (`save_best='AP'`) and to report results. None of these photographs was trained on, yet the frozen model's clean scores are mildly optimistic as an estimate on unseen photographs; we expect the degraded-input scores and the adaptation gain to be much less affected, but this is not measured here.\n\n"
        "**Snapshot note:** both pinned revisions [5], [6] ship `model.safetensors` (4-file manifests) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies both before either processor or model is constructed."
    ),
    "guided": {
        "opening": [
            "> **⚠️ Notice.** For research, training and evaluation only. Not validated for production,\n> clinical, diagnostic or safety-critical use. Running this notebook downloads third-party\n> model weights and datasets under their own licenses (see the model card) and uses compute\n> on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are\n> responsible for having the right to use it. Results here are tutorial evidence, not\n> benchmarks. Everything is provided \"as is\", without warranty of any kind. To the maximum\n> extent permitted by law, the maintainers and the DIMER platform accept no liability for\n> its use. Pose estimation of people is not validated for, and must not be used for,\n> surveillance, tracking, identification or profiling of people, or workplace-safety decisions\n> (see the model card's prohibited uses).",
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a pretrained pose estimator behaves under an input shift — small, low-quality persons — and what a bounded fine-tune recovers, read honestly against box-only baselines and the clean input. No prior experience with pose estimation or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. A GPU is strongly recommended: on CPU the three fine-tunes of the recipe grid take well over an hour.\n\n**Input → Model → Output.**\n\n| | Two-stage pose estimation | Bounded fine-tuning |\n|---|---|---|\n| Input | one photograph (16..4,096 px); person boxes from the detector or from you | labelled persons: 290 training and 72 validation in the sample (plus a 96-person fresh test and a 136-person development test), each person downscaled to a 40-px box and JPEG-30, split by photograph |\n| Model | RT-DETR finds `person` boxes; ViTPose warps each box to 192 × 256 and reads 17 COCO joints from 64 × 48 heatmaps | the heatmap head alone, or with the last two or four encoder blocks, trained on the joint-weighted heatmap MSE; validation PCK + OKS chooses the epoch and, by a pre-registered rule, the configuration |\n| Output | 17 joints per box with heatmap-maximum scores — a ranking signal, not a probability | a safetensors adapter, and held-out PCK and mean OKS on the small persons and on their clean counterparts |\n\n**How to use this notebook.** Choose a runtime (a GPU is strongly recommended), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the two model snapshots — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run; changing any of them makes the run exploratory (validation only; the fresh test is not scored). Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer that names the run it quotes — the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook); quotes are refreshed from the first hosted run of this revision. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 COCO persons, the small-person degradation and an image-level split *(evaluation practice)* → 5 the two-stage inference contract on a drawing *(core concept: top-down pose)* → 6 box-only baselines and the frozen model, small and clean *(evaluation practice)* → 7 the pre-registered recipe grid *(core concept)* → 8 held-out evaluation → 9 skeletons, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify both immutable "
        "upstream snapshots; fetch a digest-pinned photograph set with its keypoint labels, degrade it under a stated recipe, "
        "validate it and split it by image without leakage; estimate a synthetic drawing through the public two-stage API and "
        "read the output contract correctly (17 joints for any box, heatmap maxima as scores, PCK against a self-drawn figure "
        "is not a benchmark); measure the frozen model's PCK and mean OKS beside two box-only baselines and against the clean "
        "counterparts, and read the per-category breakdown; run a bounded fine-tuning with the heatmap MSE, explicit "
        "hyperparameters and validation-based epoch selection; evaluate on an image-disjoint test split; look at the adapted "
        "skeletons next to the frozen ones and the labels; and export a safetensors adapter that reloads against the pinned "
        "base with verified parity."
    ),
    "exclusions": (
        "bottom-up or multi-person association without boxes (every person needs a box, from the detector or from you), "
        "3-D pose, tracking across frames, hand, face or whole-body keypoints beyond the 17 COCO joints, fine-tuning of the "
        "detector or of the patch embedding and earlier encoder blocks, COCO OKS-based average precision proper (only PCK and "
        "the per-joint OKS kernel averaged over labelled joints are computed here — no AP over thresholds), evaluation on the "
        "full COCO validation set or any benchmark (only one seeded 594-person sample from 365 of its images is scored), and "
        "any claim that a 40-px JPEG person stands in for your camera's degradation. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with pose estimation or fine-tuning. The notebook explains top-down pose, heatmaps, PCK, OKS, the box-only baselines, the clean counterparts and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter); a GPU is strongly recommended for Section 7. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU (float32) and uses CUDA automatically when available. In the Colab T4 run of the previous revision (blob 49fc2c05) scoring the 136 test persons took 3.0 s and one eight-epoch fine-tune with per-epoch validation took 62.8 s, and the whole run 320 s; this revision trains three configurations and scores validation after every epoch of each, estimated at about 3 minutes more on a T4 (not yet measured). In the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook) the grid took 20 minutes on a CPU. A 2-vCPU hosted runtime will take much longer (ViT-B forward and backward on 290 crops per epoch). The pinned `torch==2.14.0` install and the two checkpoints (360 MB ViTPose, 172 MB RT-DETR) are the large downloads of the run; the photographs are about 63 MB.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a bounding box in xyxy pixel coordinates is; what a keypoint heatmap is and why its maximum is not a probability; what PCK and OKS measure and why a self-drawn figure is a plumbing check while a held-out split under a stated degradation is a measurement of that degradation only.",
        "- **Data contract:** records are `{id, image, box, keypoints, category?}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4096 px, `box` one `[x0, y0, x1, y1]` person box inside it, `keypoints` a mapping from COCO joint name to `[x, y]` for every labelled joint (at least 12 of the 17), an optional `category` of at most 32 characters. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a training split needs 8..5,000 records and validation and test at least one each, so with the default 15 % + 20 % image hold-out the effective BYOD minimum is **12 persons on 12 single-person images** (`min_byod_records()`); splitting keeps every image (by decoded pixels) in one split and every person with its image. BYOD accepts one zip (or directory) of photographs plus a `keypoints.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image is decoded, every box and joint checked to lie inside it, but nothing checks that a joint is where its name says — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data (photographs of identifiable people you have no consent to process) to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the two model snapshots, the default path fetches 365 JPEG files from `http://images.cocodataset.org/val2017/<id>.jpg` (about 63 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's Flickr page and licence (CC BY 2.0 or CC BY-SA 2.0 — COCO licence ids 4 and 5 only) are kept in its record, and the 2017 keypoint annotations it carries are CC BY 4.0 (COCO Consortium). Nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. COCO persons, the small-person degradation and the split\n\n"
                "`fetch_corpus` returns the 365 pinned photographs from the cache under `weights/coco-val-persons/` or the "
                "COCO image host — every cached file is re-hashed and every fetched file refused on any byte-size or SHA-256 "
                "mismatch — and `read_corpus` turns each labelled person into a record: `degrade` downscales the whole "
                "photograph (bicubic) so the person's box is `TARGET_HEIGHT` px tall, re-encodes it as JPEG at `JPEG_QUALITY`, "
                "and scales the box (continuous) by the same factor and every labelled joint as the pixel index it is, `(x + 0.5)·s − 0.5`, so a label stays on the image content it marks; the record's `category` says how many of the "
                "17 joints COCO labelled (`full` 16–17, `partial` 14–15, `sparse` 12–13 — a proxy for occlusion and "
                "truncation). `build_sample_dataset` draws a seeded image-level split of 300 photographs (180 / 45 / 75; every person "
                "follows its photograph: train, validation and the development test) and keeps the 65 other qualifying photographs, which no revision scored before, as the fresh test, and `read_corpus(..., target_height=None)` builds the **clean counterparts** of the "
                "test persons — the same photographs unchanged — for Section 6. `validate_dataset` then checks every record "
                "against the contract, `check_image_ids_disjoint` asserts that no COCO photograph is in two splits (each person's degraded copy has its own pixels, so the decoded-pixel check `check_split_disjoint` cannot see that here; for BYOD it catches an identical decoded image, not a resized or re-encoded copy), and "
                "the training split's summary table is written to `outputs/{stem}_train.csv`.\n\n*Data context.* The photographs are COCO val2017 [3] under their Flickr licences; ViTPose-base was trained on train2017, while val2017 picked its released checkpoint upstream [7] — so no photograph here was trained on, but the frozen clean scores are mildly optimistic.\n\n"
                "Look for: 365 photographs and 594 persons in four splits (300 in the draw plus the 65 of the fresh test), boxes about 40 px tall after degradation, three categories, three "
                "digests, and four refusal probes — a duplicate id, a joint outside its image, a person with too few labelled "
                "joints, and a dataset too small to use — each rejected before the model does anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder holding `keypoints.csv` and the photographs, as a path in this runtime — this works on Colab, Kaggle and "
                "Jupyter) or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** "
                "from this cell. This cell first puts the model back to the pinned base, so Section 6 and Section 7's epoch 0 read "
                "the frozen model. Your persons are not degraded. The effective minimum is 12 persons on 12 single-person "
                "images.\n\n"
                "**Predict before running:** a photograph can hold several persons. Why must all of them land in the same split?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "RUN_RECIPE_GRID = True  # @param {{type:\"boolean\"}}\n"
                "TRAINABLE_BLOCKS = 2  # @param {{type:\"integer\"}}\n"
                "# The pre-registered settings (data, split, degradation, training, grid). Changing any of them, a form field above, BYOD\n"
                "# or a smaller batch for memory, makes the run EXPLORATORY: validation only, and the fresh test is not scored.\n"
                "TARGET_HEIGHT, JPEG_QUALITY = 40, 30\n"
                "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 8, 0\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Section 7 (BYOD, or a new split): Section 6 and Section 7's epoch 0 must read the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_tensors = pipe.restore_base()\n"
                "if had_adapter or restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'the fine-tuned head and blocks were put back to the checkpoint; Sections 5-7 start from it again'}})\n"
                "class ByodRefusal(ValueError):\n"
                "    \"\"\"A BYOD input this notebook refuses: shown as one message, without a traceback.\"\"\"\n"
                "    dimer_refusal = True\n"
                "\n"
                "    def _render_traceback_(self):\n"
                "        return [str(self)]\n"
                "\n"
                "\n"
                "# Every run of this cell starts a new run: the earlier split, scores and choice are dropped (also when a BYOD input is\n"
                "# refused), so later cells cannot read them as this run's; nothing already written to outputs/ is deleted.\n"
                "for stale in ('splits', 'train_records', 'val_records', 'test_records', 'dev_test_records', 'clean_test', 'clean_dev', 'held_out_records', 'held_out_clean', 'RUN_SETTINGS', 'preregistration', 'frozen_test', 'frozen_clean', 'frozen_dev', 'frozen_dev_clean', 'frozen_scored_with', 'adapt_result', 'recipe_selection', 'adapted_test', 'test_look'):\n"
                "    globals().pop(stale, None)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_zip.exists():\n"
                "            raise ByodRefusal(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding keypoints.csv and the photographs, then run this cell again.')\n"
                "        file_name = byod_zip.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise ByodRefusal('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ByodRefusal(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ByodRefusal(f'{{file_name}}: upload one .zip holding keypoints.csv and the photographs, then run this cell again.')\n"
                "        byod_zip = Path('work') / 'byod.zip'\n"
                "        byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_zip.write_bytes(payload)\n"
                "    try:\n"
                "        records = load_byod_dataset(byod_zip)\n"
                "        splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    except ValueError as exc:  # the loader's and the splitter's refusals name the file, row and field; anything else keeps its traceback\n"
                "        raise ByodRefusal(f'BYOD refused ({{file_name}}): {{exc}}. Fix the file and run this cell again; earlier files in outputs/ are kept, but the earlier split is no longer active.') from None\n"
                "    clean_test = clean_dev = None\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 20:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test persons: PCK and OKS move in large steps and carry no dispersion estimate; add persons before reading them\"}})\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/coco-val-persons')\n"
                "    corpus = read_corpus(corpus_files, target_height=TARGET_HEIGHT, quality=JPEG_QUALITY)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)  # train / validation / test (fresh) / development_test\n"
                "    test_images = {{r['image_id'] for r in splits['test']}}\n"
                "    dev_images = {{r['image_id'] for r in splits['development_test']}}\n"
                "    clean_test = read_corpus({{k: v for k, v in corpus_files.items() if k in test_images}}, target_height=None)\n"
                "    clean_dev = read_corpus({{k: v for k, v in corpus_files.items() if k in dev_images}}, target_height=None)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}}'\n"
                "    raw_rows = {{'photographs': len(corpus_files), 'bytes': sum(len(v) for v in corpus_files.values()), 'persons': len(corpus), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "# The training split must hold MIN_RECORDS; the other splits only need one person each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "photographs_per_split = check_image_ids_disjoint(splits)  # the sample's photo-disjointness, by COCO image id\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dev_test_records = splits.get('development_test')\n"
                "RUN_SETTINGS = {{'data_source': 'BYOD' if USE_BYOD else PREREGISTERED_DATA_SOURCE, 'byod': bool(USE_BYOD), 'split_seed': SPLIT_SEED, 'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY, 'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'adapt_seed': ADAPT_SEED, 'run_recipe_grid': bool(RUN_RECIPE_GRID), 'recipe_grid': [[name, blocks] for name, blocks in RECIPE_GRID] if RUN_RECIPE_GRID else [[f'blocks={{TRAINABLE_BLOCKS}}', TRAINABLE_BLOCKS]], 'recipe_margin': RECIPE_MARGIN}}\n"
                "preregistration = check_preregistration(RUN_SETTINGS)  # True only if EVERY pre-registered setting matches\n"
                "if preregistration['preregistered']:\n"
                "    held_out_name, held_out_records, held_out_clean = 'fresh test', test_records, clean_test\n"
                "else:\n"
                "    # Exploratory: compared on validation only; the fresh test is never scored in this run (the labelled development test may be shown).\n"
                "    held_out_name, held_out_records, held_out_clean = 'validation (exploratory run: the fresh test is not scored)', val_records, None\n"
                "    print({{'exploratory_run': True, 'settings_differing': preregistration['settings_differing'], 'note': 'a setting differs from the pre-registration, so Sections 6-10 score validation only and never the fresh test; reports go to separate *_exploratory_<n>.json files'}})\n"
                "if len(val_records) < 20:\n"
                "    print({{'caution': f'only {{len(val_records)}} validation persons: epoch and recipe selection on so few persons is close to a coin toss; add persons before trusting the choice'}})\n"
                "for clean, persons, label in ((clean_test, test_records, 'test'), (clean_dev, dev_test_records, 'development test')):\n"
                "    if clean is not None:\n"
                "        clean[:] = validate_dataset(clean)['records']\n"
                "        if [r['id'] for r in clean] != [r['id'] for r in persons]:  # a contract check: clean counterparts align with their persons\n"
                "            raise RuntimeError(f'the clean counterparts do not line up with the {{label}} persons')\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'photographs': photographs_per_split, 'degradation': {{'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY}}, 'licence': CORPUS_LICENSE}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'images': manifest['n_images'], 'category_counts': manifest['category_counts'], 'box_height': manifest['box_height'], 'labelled_joints': manifest['labelled_joints'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'category': example.get('category'), 'image': list(example['image'].size), 'box': [round(v, 1) for v in example['box']], 'labelled': len(example['keypoints']), 'scale': example.get('scale'), 'flickr': example.get('flickr_url'), 'licence': example.get('license')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'joint outside its image': [{{**train_records[0], 'keypoints': {{**train_records[0]['keypoints'], 'Nose': [-5.0, 0.0]}}}}, *train_records[1:8]],\n"
                "    'too few labelled joints': [{{**train_records[0], 'keypoints': dict(list(train_records[0]['keypoints'].items())[:5])}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '**What to notice:** 365 photographs and 594 persons, boxes about 40 px tall, the three categories, the three digests and the four refusals.\n\n<details><summary>Check your reasoning</summary>Persons in one photograph share its lighting, background and compression, and some overlap; split by person, the test set would contain the same photograph as training and the score would measure memory of that image. Keeping every person with its photograph makes the test a question about unseen photographs.</details>'
            ),
        },
        {
            "md": (
                "## 5. Estimate a synthetic drawing through the two-stage inference contract\n\n"
                "*Core concept.* A top-down pose estimator needs a person box first and always returns 17 joints for it, "
                "whether or not the box contains a person. **Predict before running:** will the photograph-trained detector find "
                "the cartoon person?\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a 640×640 cartoon person "
                "drawn with Pillow from a dictionary of 17 joint coordinates in COCO order (`REFERENCE_JOINTS`), with the drawn "
                "figure's box `DRAWN_BOX` — an image family the adaptation never sees, and a figure the adapted model will "
                "estimate again in Section 9. `validate_inputs` applies exactly the checks `estimate` applies (image sides, "
                "1..`MAX_PERSONS` boxes inside the image, both thresholds in `[0, 1]`) and returns an input manifest; a box that "
                "leaves the image is validated too and its rejection recorded as a finding. `detect_people` runs stage 1 alone: "
                "on the cartoon the photograph-trained RT-DETR finds **no** `person` at the default 0.3 (recorded as a finding), "
                "so `estimate` receives the drawn box as a caller box (`box_source: caller`) and returns 17 joints with heatmap "
                "maxima as scores — a box that contains no person would still yield 17 joints, with low scores as the only "
                "signal. The per-image `evaluation_report` against the self-drawn joints is `sample-sanity` (PCK within "
                "`PCK_FRACTION` of the box's longest side) — plumbing evidence, not a measurement; whether the model is *good "
                "at small photographed persons* is what Section 6 measures."
            ),
            "code": (
                "CX = 320\n"
                "REFERENCE_JOINTS = {{\n"
                "    'Nose': (CX, 130), 'L_Eye': (CX + 12, 118), 'R_Eye': (CX - 12, 118), 'L_Ear': (CX + 30, 125), 'R_Ear': (CX - 30, 125),\n"
                "    'L_Shoulder': (CX + 60, 200), 'R_Shoulder': (CX - 60, 200), 'L_Elbow': (CX + 95, 290), 'R_Elbow': (CX - 95, 290),\n"
                "    'L_Wrist': (CX + 120, 380), 'R_Wrist': (CX - 120, 380), 'L_Hip': (CX + 40, 350), 'R_Hip': (CX - 40, 350),\n"
                "    'L_Knee': (CX + 50, 450), 'R_Knee': (CX - 50, 450), 'L_Ankle': (CX + 55, 545), 'R_Ankle': (CX - 55, 545),\n"
                "}}\n"
                "DRAWN_BOX = [180.0, 80.0, 460.0, 560.0]\n\n\n"
                "def cartoon_person(width=640, height=640):\n"
                "    \"\"\"A flat cartoon person drawn from REFERENCE_JOINTS: head, torso, arms, legs on a plain background.\"\"\"\n"
                "    img = Image.new('RGB', (width, height), (225, 232, 240))\n"
                "    d = ImageDraw.Draw(img)\n"
                "    d.rectangle([0, 520, width, height], fill=(150, 160, 140))\n"
                "    skin, shirt, pants = (222, 180, 140), (200, 60, 60), (40, 60, 140)\n"
                "    j = REFERENCE_JOINTS\n"
                "    d.ellipse([CX - 42, 88, CX + 42, 172], fill=skin, outline=(120, 80, 60), width=3)\n"
                "    for eye in ('L_Eye', 'R_Eye'):\n"
                "        d.ellipse([j[eye][0] - 5, j[eye][1] - 5, j[eye][0] + 5, j[eye][1] + 5], fill=(30, 30, 30))\n"
                "    d.arc([CX - 18, 135, CX + 18, 160], 10, 170, fill=(120, 40, 40), width=3)\n"
                "    d.polygon([j['R_Shoulder'], j['L_Shoulder'], (j['L_Hip'][0] + 10, j['L_Hip'][1]), (j['R_Hip'][0] - 10, j['R_Hip'][1])], fill=shirt)\n"
                "    d.line([(CX, 172), (CX, 200)], fill=skin, width=18)\n"
                "    for side in ('L', 'R'):\n"
                "        d.line([j[f'{{side}}_Shoulder'], j[f'{{side}}_Elbow'], j[f'{{side}}_Wrist']], fill=shirt, width=26, joint='curve')\n"
                "        d.line([j[f'{{side}}_Elbow'], j[f'{{side}}_Wrist']], fill=skin, width=22, joint='curve')\n"
                "        d.ellipse([j[f'{{side}}_Wrist'][0] - 14, j[f'{{side}}_Wrist'][1] - 14, j[f'{{side}}_Wrist'][0] + 14, j[f'{{side}}_Wrist'][1] + 14], fill=skin)\n"
                "        d.line([j[f'{{side}}_Hip'], j[f'{{side}}_Knee'], j[f'{{side}}_Ankle']], fill=pants, width=34, joint='curve')\n"
                "        d.ellipse([j[f'{{side}}_Ankle'][0] - 22, j[f'{{side}}_Ankle'][1] - 10, j[f'{{side}}_Ankle'][0] + 22, j[f'{{side}}_Ankle'][1] + 14], fill=(40, 40, 40))\n"
                "    return img\n\n\n"
                "scene = cartoon_person()\n"
                "scene_reference = {{k: (float(x), float(y)) for k, (x, y) in REFERENCE_JOINTS.items()}}\n"
                "scene_name = 'synthetic_person_640x640.png'\n"
                "scene_sha256 = hashlib.sha256(np.asarray(scene).tobytes()).hexdigest()\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PERSONS': MAX_PERSONS, 'DETECTION_THRESHOLD': DETECTION_THRESHOLD, 'KEYPOINT_THRESHOLD': KEYPOINT_THRESHOLD, 'PCK_FRACTION': PCK_FRACTION, 'MIN_LABELLED': MIN_LABELLED, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(scene, person_boxes=[DRAWN_BOX], detection_threshold=DETECTION_THRESHOLD, keypoint_threshold=KEYPOINT_THRESHOLD, names=[scene_name])\n"
                "try:\n"
                "    validate_inputs(scene, person_boxes=[[0, 0, scene.width + 1, 10]])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'box-outside-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "stage1 = pipe.detect_people(scene, threshold=DETECTION_THRESHOLD)\n"
                "scene_findings = []\n"
                "if stage1['n_persons'] == 0:\n"
                "    scene_findings.append('stage 1 found no `person` on the drawing at the default threshold; stage 2 receives the drawn box as a caller box')\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': scene_name, 'sha256': scene_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings']), 'stage1_persons': stage1['n_persons'], 'scene_findings': scene_findings}})\n\n\n"
                "def estimate_scene(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.estimate(scene, person_boxes=[DRAWN_BOX], detection_threshold=DETECTION_THRESHOLD, keypoint_threshold=KEYPOINT_THRESHOLD)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    pose = result['poses'][0]\n"
                "    checks = {{\n"
                "        'one_person': result['n_persons'] == 1 and result['box_source'] == 'caller',\n"
                "        'seventeen_joints_in_order': [kp['name'] for kp in pose['all_keypoints']] == list(KEYPOINT_NAMES),\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION and result['detector_revision'] == DETECTOR_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'estimate output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, [scene_reference], sample_kind='synthetic (authored in this notebook)')\n"
                "    annotated = scene.copy()\n"
                "    draw = ImageDraw.Draw(annotated)\n"
                "    joints = {{kp['name']: (kp['x'], kp['y']) for kp in pose['all_keypoints']}}\n"
                "    for a, b in SKELETON_EDGES:\n"
                "        draw.line([joints[KEYPOINT_NAMES[a]], joints[KEYPOINT_NAMES[b]]], fill=(255, 200, 0), width=3)\n"
                "    annotated.save(f'outputs/{stem}_scene_{{label}}.png')\n"
                "    print({{label: {{'seconds': round(elapsed, 3), 'checks': checks, 'mean_score': round(pose['mean_keypoint_score'], 3), 'pck': round(report['metrics'][0]['value'], 3), 'mean_error_px': round(report['metrics'][0]['mean_error_px'], 1), 'verdict': report['verdict']}}}})\n"
                "    return result, report\n\n\n"
                "frozen_scene_result, frozen_scene = estimate_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "**What to notice:** `stage1_persons`, `box_source`, the joints' mean score and the self-drawn PCK.\n\n<details><summary>Check your reasoning</summary>No: in the Colab T4 run of the previous revision (blob 49fc2c05) RT-DETR found no `person` on the cartoon at the default 0.3, so `estimate` used the drawn box as a caller box — and ViTPose still returned 17 joints, matching the self-drawn ones (PCK 1.0). A box with no person in it would also get 17 joints; only the low heatmap scores would say so. This is plumbing evidence, not a measurement.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines, the frozen model on the small persons — and on the clean ones\n\n"
                "Two box-only baselines frame the adaptation, each read two ways by `pose_metrics` (carried in `metrics.py`): "
                "**PCK** — the share of labelled joints within `PCK_FRACTION` (0.1) of the box's longest side of their label, "
                "the package's own `keypoint_pck` convention — and **mean OKS** — the COCO object-keypoint-similarity kernel per "
                "joint (COCO's per-joint sigmas, the box area as the object scale) averaged over the labelled joints, the "
                "primitive of the COCO keypoint AP without its thresholds; it is scaled here by the person box area, not COCO's segmentation area, so these OKS values run higher than COCO's and are not comparable with published numbers — averaged over the persons and per category. The "
                "**box-centre** baseline puts every joint at the box centre: the floor. The **mean-pose** baseline puts every "
                "joint at its mean position relative to the box over the training split — the layout prior a box alone gives "
                "away, and the number a pose estimator must beat to be doing anything. The **frozen model** is scored by "
                "`pipe.evaluate`, which runs every record's box through `estimate` and scores all 17 joints. Every score in this notebook uses the labelled COCO box as a caller box (the detector is not in the loop), so it is an upper bound on the two-stage system, where RT-DETR would first have to find a 40-px person. It is scored "
                "twice: on the **small persons** the adaptation is about, and on their **clean counterparts** — the same "
                "persons in the unchanged photographs — which is where the checkpoint was trained to work. **What to look for:** "
                "the frozen model against the mean-pose prior on the small persons and on the clean ones, and the per-category "
                "rows.\n\n"
                "*Evaluation practice.* The cell records a **verdict** — whether the frozen model beats the box-centre floor — "
                "instead of asserting it.\n\n"
                "**Predict before running:** how much PCK will a 40-px JPEG person cost the frozen model, compared with the same "
                "person at full resolution?"
            ),
            "code": (
                "METRICS = ('pck', 'oks')\n"
                "baseline_centre = box_centre_baseline(held_out_records)  # the fresh test only under the pre-registered settings (Section 4)\n"
                "baseline_prior = mean_pose_baseline(train_records, held_out_records)\n"
                "print({{'box_centre_baseline': {{k: round(baseline_centre[k], 3) for k in METRICS}}, 'n': baseline_centre['n'], 'note': baseline_centre['baseline']}})\n"
                "print({{'mean_pose_baseline': {{k: round(baseline_prior[k], 3) for k in METRICS}}, 'note': baseline_prior['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(held_out_records)\n"
                "frozen_scored_with = dict(RUN_SETTINGS)\n"
                "print({{'frozen_small_persons': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'split': held_out_name, 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "frozen_clean = pipe.evaluate(held_out_clean) if held_out_clean is not None else None\n"
                "if frozen_clean is not None:\n"
                "    print({{'frozen_clean_counterparts': {{k: round(frozen_clean[k], 3) for k in METRICS}}}})\n"
                "frozen_dev = pipe.evaluate(dev_test_records) if dev_test_records else None\n"
                "frozen_dev_clean = pipe.evaluate(clean_dev) if clean_dev is not None else None\n"
                "if frozen_dev is not None:\n"
                "    print({{'development_test (read during earlier development; continuity only)': {{'frozen_small': {{k: round(frozen_dev[k], 3) for k in METRICS}}, 'frozen_clean': {{k: round(frozen_dev_clean[k], 3) for k in METRICS}}, 'n': frozen_dev['n']}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'pck': round(v['pck'], 3), 'oks': round(v['oks'], 3)}} for c, v in frozen_test['per_category'].items()}}\n"
                "print({{'by_category_frozen': frozen_fields}})\n"
                "frozen_verdict = 'frozen above the box-centre floor' if frozen_test['pck'] > baseline_centre['pck'] else 'frozen NOT above the box-centre floor'\n"
                "print({{'frozen_vs_floor': frozen_verdict}})"
            ),
        },
        {
            "md": (
                "**What to notice:** PCK and OKS for the floor, the mean-pose prior and the frozen model, small and clean, and the per-category rows.\n\n<details><summary>Check your reasoning</summary>About a third. In the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook) the frozen model scored PCK 0.977 on the clean fresh-test persons and 0.613 on the same persons at 40 px (OKS 0.962 and 0.585), against 0.419 for the mean-pose prior and 0.091 for the box centre; on the development test (the earlier test split, read during earlier development; continuity only) it scored 0.957 clean and 0.597 small. A good part of the small-person score comes from the box layout, not from the pixels. The clean score is mildly optimistic, because val2017 picked the upstream checkpoint.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning: the pre-registered recipe grid\n"
                "\n"
                "`pipe.adapt` trains only the heatmap head (two deconvolutions and the final convolution) and the last `trainable_blocks` encoder blocks with the final backbone LayerNorm, while the patch embedding, the earlier blocks and the whole detector stay frozen. Every record's box is affine-warped to the processor's 192×256 crop exactly as at inference, and its labelled joints become Gaussian targets (sigma `HEATMAP_SIGMA` = 2) on the 64×48 heatmap grid in the same frame, encoded with the crop-to-heatmap stride the processor decodes with (`(crop − 1) / (heatmap − 1)`, the unbiased convention ViTPose was trained with [4]), so a target decodes back to its joint exactly. The head's heatmaps are trained with the **joint-weighted mean-squared error** ViTPose was trained with (unlabelled joints carry no weight). AdamW without weight decay at a fixed learning rate, gradient clipping at 1.0, seeded shuffling, no scheduler, no augmentation; the head's BatchNorm statistics stay frozen so the adapter is parameters-only. Epoch 0 records the frozen model's validation metrics; every epoch is scored on the 72 validation persons, and each run keeps the epoch with the highest **mean of validation PCK and OKS**.\n"
                "\n"
                "This cell runs the **pre-registered grid** stated at the top of the notebook — G1 (head only, 4,199,697 trainable parameters), G2 (head and two blocks, 18,376,977) and G3 (head and four blocks, 32,552,721) — saves each configuration's best epoch under `work/recipe_grid/`, prints the validation score of every epoch of every configuration (epoch 0 is the frozen model), applies the rule with `select_recipe`, and loads the chosen configuration back from its saved artifact. It then prints photo-clustered bootstrap intervals for each configuration minus the frozen model and for the best minus the runner-up: sampling uncertainty over these 72 persons at one seed, printed beside the rule and not used by it. Test is not read here. `RUN_RECIPE_GRID = False` in Section 4 trains one configuration of your choice (`TRAINABLE_BLOCKS`) instead: an exploratory run that never scores the fresh test.\n"
                "\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call (or an artifact) changed are put back first, so epoch 0 is always the frozen model and the three configurations start from the same checkpoint. To compare a further change side by side without replacing the default exports, use Section 10.\n"
                "\n"
                "**Predict before running:** which configuration will the rule choose, and will it be the one with the highest validation score?"
            ),
            "code": (
                "import shutil\n"
                "\n"
                "# EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED, RUN_RECIPE_GRID and TRAINABLE_BLOCKS are set in Section 4 with the other pre-registered settings.\n"
                "\n"
                "\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 5)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in (*METRICS, 'score')}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n"
                "\n"
                "\n"
                "configs = RECIPE_GRID if RUN_RECIPE_GRID else ((f'blocks={{TRAINABLE_BLOCKS}}', TRAINABLE_BLOCKS),)\n"
                "if not RUN_RECIPE_GRID:\n"
                "    print({{'note': 'RUN_RECIPE_GRID = False: one configuration you chose and no selection, so this run is exploratory: validation only, and the fresh test is not scored'}})\n"
                "grid_dir = Path('work/recipe_grid')\n"
                "shutil.rmtree(grid_dir, ignore_errors=True)\n"
                "grid_results, grid_rows = {{}}, []\n"
                "t0 = time.perf_counter()\n"
                "for name, blocks in configs:\n"
                "    print({{'config': name, 'trainable_blocks': blocks}})\n"
                "    result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=blocks, seed=ADAPT_SEED, progress=report)\n"
                "    pipe.save_artifact(grid_dir / name, metadata={{'tutorial': '{stem}', 'recipe_config': name}})\n"
                "    grid_results[name] = result\n"
                "    grid_rows.append({{'name': name, 'trainable_blocks': blocks, 'trainable_parameters': result['n_trainable'], 'best_epoch': result['best_epoch'], 'best_score': result['history'][result['best_epoch']]['val']['score'], 'seconds': result['seconds']}})\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "frozen_val_score = grid_results[configs[0][0]]['history'][0]['val']['score']\n"
                "validation_table = {{name: [round(h['val']['score'], 4) for h in result['history']] for name, result in grid_results.items()}}\n"
                "print({{'validation_score_by_epoch (epoch 0 = frozen)': validation_table}})\n"
                "for row in grid_rows:\n"
                "    print({{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}})\n"
                "if RUN_RECIPE_GRID:\n"
                "    recipe_selection = select_recipe(grid_rows, frozen_val_score)\n"
                "else:\n"
                "    recipe_selection = {{'chosen': configs[0][0], 'trainable_blocks': TRAINABLE_BLOCKS, 'frozen_score': frozen_val_score, 'rule': 'no selection (RUN_RECIPE_GRID = False)'}}\n"
                "recipe_selection.update({{'validation_table': validation_table, 'grid': grid_rows, 'grid_seconds': adapt_seconds}})\n"
                "trained_with = {{(r['epochs'], r['batch_size'], r['lr'], r['seed']) for r in grid_results.values()}}\n"
                "if len(trained_with) != 1:\n"
                "    raise RuntimeError(f'the configurations trained with different settings: {{sorted(trained_with)}}')\n"
                "((trained_epochs, trained_batch, trained_lr, trained_seed),) = trained_with\n"
                "# The settings this grid actually trained with: Section 8 scores the fresh test only if they are all the pre-registered ones.\n"
                "recipe_selection['settings'] = {{**RUN_SETTINGS, 'epochs': trained_epochs, 'batch_size': trained_batch, 'learning_rate': trained_lr, 'adapt_seed': trained_seed, 'recipe_grid': [[name, blocks] for name, blocks in configs]}}\n"
                "recipe_selection['preregistration'] = check_preregistration(recipe_selection['settings'])\n"
                "chosen = recipe_selection['chosen']\n"
                "# Sampling uncertainty on validation: a photo-clustered paired bootstrap at one seed. Reported, not used by the rule.\n"
                "groups_val = [(r['image_id'] if 'image_id' in r else image_digest(r['image'])) for r in val_records]\n"
                "val_rows = {{}}\n"
                "for name in grid_results:\n"
                "    pipe.load_artifact(grid_dir / name)\n"
                "    val_rows[name] = pipe.evaluate(val_records)['per_record']\n"
                "pipe.restore_base()\n"
                "val_rows['frozen'] = pipe.evaluate(val_records)['per_record']\n"
                "ranked = sorted(grid_rows, key=lambda r: -r['best_score'])\n"
                "val_intervals = {{f\"{{r['name']}} - frozen\": paired_bootstrap(val_rows[r['name']], val_rows['frozen'], groups_val)['score'] for r in grid_rows}}\n"
                "if len(ranked) > 1:\n"
                "    val_intervals[f\"{{ranked[0]['name']}} - {{ranked[1]['name']}}\"] = paired_bootstrap(val_rows[ranked[0]['name']], val_rows[ranked[1]['name']], groups_val)['score']\n"
                "recipe_selection['validation_intervals_95'] = val_intervals\n"
                "print({{'validation 95% intervals (mean of PCK and OKS; photo-clustered paired bootstrap; not used by the rule)': {{k: [round(v['delta'], 3), round(v['low'], 3), round(v['high'], 3)] for k, v in val_intervals.items()}}}})\n"
                "if chosen == 'frozen':\n"
                "    pipe.restore_base()\n"
                "    adapt_result = None\n"
                "    print({{'recipe_selection': chosen, 'note': 'no configuration beat the frozen epoch-0 validation score by more than the margin: the frozen model is the result, and Section 8 records it without scoring the held-out split again'}})\n"
                "else:\n"
                "    pipe.load_artifact(grid_dir / chosen)  # the chosen configuration at its best epoch, overlaid like any verified artifact\n"
                "    adapt_result = grid_results[chosen]\n"
                "    print({{'recipe_selection': chosen, 'eligible': recipe_selection.get('eligible'), 'best_epoch': adapt_result['best_epoch'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'grid_seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the 3 × 9 table of validation scores, each configuration's `best_epoch`, and `recipe_selection`.\n\n<details><summary>Check your reasoning</summary>In the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook) the frozen model scored 0.648 on validation (mean of PCK and OKS); the best epochs scored 0.668 for G1 (epoch 6), 0.724 for G2 (epoch 8) and 0.747 for G3 (epoch 6), and the rule chose **G3, head and four blocks**: it beats the frozen model by 0.098 (G1 trails by 0.079, G2 trails by 0.022; eligible: G1, G2, G3). The 95% intervals: G1 - frozen +0.019 [+0.008, +0.031]; G2 - frozen +0.076 [+0.043, +0.113]; G3 - frozen +0.098 [+0.062, +0.140]; G3 - G2 +0.022 [+0.004, +0.041]. The best epochs were 6 for G1, 8 for G2 and 6 for G3: the comparison holds for this learning rate and eight-epoch budget only, and G3, the largest configuration tried, is the edge of the grid. The rule need not pick the highest score, and its 0.01 margin is a preference for the smaller model, not a test: on 72 validation persons the intervals above are several hundredths wide. If nothing beats the frozen model by more than 0.01, the frozen model is the result, and Section 8 records it.</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The **fresh-test** persons were not read by any revision before this one's choices: not for training, epoch selection, the recipe, the task or the hyperparameters. They are scored only under the pre-registered settings, for the configuration the pre-registered rule chose; in an exploratory run (Section 4 says so) this section reads validation instead, and `test_look` in the report says which it was: `pre-registered evaluation`, `reproduction` or `not scored (exploratory)`. The **development test** is the earlier revisions' test split; its scores were read while the task and hyperparameters were set, so it is scored beside the fresh test for continuity and is not confirmatory. No photograph appears in two splits. The verdicts use a stated tolerance of 0.01 (`no measurable change` within it), and a photo-clustered bootstrap interval for adapted minus frozen on the fresh test carries the sampling uncertainty. "
                "The adapted model is scored exactly as the frozen model was in Section 6 — on the small persons and on their "
                "clean counterparts — the systems are put side by side on both measures, and the per-category breakdown is "
                "repeated. Read it in this order: **PCK on the small persons** first, then **mean OKS**, then the **clean "
                "counterparts** — PCK *and* OKS, because they need not move together — then the per-category rows. The cell "
                "records a verdict — `improved`, `no gain` or `worse` — instead of asserting a gain: when the rule keeps the frozen model, or validation keeps epoch "
                "0 the adapted model is the frozen model, and on undegraded BYOD persons the frozen model may already be near "
                "the ceiling; the export, reload and result still run. 96 fresh-test persons from one seeded split under one degradation give a sampling interval but **no seed-to-seed "
                "dispersion estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a "
                "benchmark, and a gain at 40-px JPEG persons says nothing about motion blur, unusual viewpoints or your camera's "
                "own artefacts until you measure them.\n\n"
                "**Predict before running:** will the fine-tune cost the clean persons anything?"
            ),
            "code": (
                "run_check = recipe_selection['preregistration']\n"
                "if frozen_scored_with != RUN_SETTINGS or run_check['preregistered'] != preregistration['preregistered']:\n"
                "    raise RuntimeError('the settings changed after Section 4 (Section 6 scored, or Section 7 trained, with other settings): run again from Section 4')\n"
                "test_look = plan_test_look(run_check, 'outputs', '{stem}')  # pre-registered evaluation | reproduction | not scored (exploratory)\n"
                "print({{'test_look': test_look['test_look'], 'held_out': held_out_name, 'settings_differing': test_look['settings_differing']}})\n"
                "# The fresh test is scored only under the pre-registered settings; docs/release-verification.md records the evaluation and its reproductions.\n"
                "adapted_test = frozen_test if adapt_result is None else pipe.evaluate(held_out_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "if adapt_result is None:\n"
                "    print({{'note': 'the pre-registered rule kept the frozen model: the adapted rows are the frozen rows, and the held-out split is not scored again'}})\n"
                "    adapted_clean, adapted_dev, adapted_dev_clean = frozen_clean, frozen_dev, frozen_dev_clean\n"
                "else:\n"
                "    adapted_clean = pipe.evaluate(held_out_clean) if held_out_clean is not None else None\n"
                "    adapted_dev = pipe.evaluate(dev_test_records) if dev_test_records else None  # the labelled development test\n"
                "    adapted_dev_clean = pipe.evaluate(clean_dev) if clean_dev is not None else None\n"
                ""
                "adapted_fields = {{c: {{'n': v['n'], 'pck': round(v['pck'], 3), 'oks': round(v['oks'], 3)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'box_centre': round(baseline_centre[metric], 3), 'mean_pose': round(baseline_prior[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "if frozen_clean is not None:\n"
                "    for metric in METRICS:\n"
                "        comparison[metric].update({{'frozen_clean': round(frozen_clean[metric], 3), 'adapted_clean': round(adapted_clean[metric], 3)}})\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "TOLERANCE = RECIPE_MARGIN  # a stated tolerance for the verdicts (0.01), not a significance test; the intervals below carry the uncertainty\n"
                "\n"
                "\n"
                "def direction(new, old):\n"
                "    return 'improved' if new - old > TOLERANCE else ('worse' if old - new > TOLERANCE else 'no measurable change')\n"
                "# Reported verdicts, not assertions: a fine-tune that does not help is a result to record, and export and reload still run.\n"
                "comparison['verdicts'] = {{'frozen_vs_floor': frozen_verdict, 'adapted_vs_frozen_pck': direction(adapted_test['pck'], frozen_test['pck']), 'adapted_vs_frozen_oks': direction(adapted_test['oks'], frozen_test['oks']), 'adapted_above_mean_pose_pck': bool(adapted_test['pck'] > baseline_prior['pck'])}}\n"
                "if adapted_clean is not None:\n"
                "    comparison['verdicts'].update({{'clean_pck': direction(adapted_clean['pck'], frozen_clean['pck']), 'clean_oks': direction(adapted_clean['oks'], frozen_clean['oks'])}})\n"
                "comparison['delta_vs_mean_pose'] = {{metric: round(adapted_test[metric] - baseline_prior[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_pck': frozen_fields[c]['pck'], 'adapted_pck': adapted_fields[c]['pck'], 'frozen_oks': frozen_fields[c]['oks'], 'adapted_oks': adapted_fields[c]['oks']}} for c in frozen_fields}}\n"
                "if frozen_dev is not None:\n"
                "    comparison['development_test (read during earlier development; continuity only)'] = {{metric: {{'frozen': round(frozen_dev[metric], 3), 'adapted': round(adapted_dev[metric], 3), 'frozen_clean': round(frozen_dev_clean[metric], 3), 'adapted_clean': round(adapted_dev_clean[metric], 3)}} for metric in METRICS}}\n"
                "groups_test = [(r['image_id'] if 'image_id' in r else image_digest(r['image'])) for r in held_out_records]\n"
                "test_interval = paired_bootstrap(adapted_test['per_record'], frozen_test['per_record'], groups_test)\n"
                "comparison['test_interval_95 (adapted - frozen; photo-clustered paired bootstrap, one seed)'] = {{k: [round(v['delta'], 3), round(v['low'], 3), round(v['high'], 3)] for k, v in test_interval.items() if k != 'n'}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'test_look': test_look['test_look'],\n"
                "    'preregistered': test_look['preregistered'],\n"
                "    'settings_differing': test_look['settings_differing'],\n"
                "    'settings': recipe_selection['settings'],\n"
                "    'held_out': held_out_name,\n"
                "    'degradation': {{'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY}},\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'box_centre': {{k: v for k, v in baseline_centre.items() if k != 'per_record'}}, 'mean_pose': {{k: v for k, v in baseline_prior.items() if k != 'per_record'}}}},\n"
                "    'frozen_held_out': frozen_test,\n"
                "    'frozen_test': frozen_test if test_look['fresh_test_scored'] else None,\n"
                "    'frozen_clean': frozen_clean,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'held_out_metrics': adapted_test,\n"
                "    'test_metrics': adapted_test if test_look['fresh_test_scored'] else None,\n"
                "    'clean_metrics': adapted_clean,\n"
                "    'development_test': {{'note': 'read during earlier development; continuity only', 'frozen': frozen_dev, 'adapted': adapted_dev, 'frozen_clean': frozen_dev_clean, 'adapted_clean': adapted_dev_clean}},\n"
                "    'verdict_tolerance': TOLERANCE,\n"
                "    'comparison': comparison,\n"
                "    'recipe_selection': recipe_selection,\n"
                "    'adaptation': None if adapt_result is None else {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': None if adapt_result is None else adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "evaluation_report_path = Path('outputs/{stem}_evaluation_report.json')\n"
                "evaluation_report_path = evaluation_report_path.with_stem(evaluation_report_path.stem + test_look['suffix'])  # an exploratory run never overwrites the pre-registered report\n"
                "with open(evaluation_report_path, 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'verdicts': comparison['verdicts']}})\n"
                "print({{'report': str(evaluation_report_path), 'test_look': test_look['test_look'], 'adapted_beats_mean_pose': adapted_test['pck'] > baseline_prior['pck']}})"
            ),
        },
        {
            "md": (
                "**What to notice:** `delta_vs_frozen`, the clean rows (PCK and OKS), the per-category rows and the `verdicts`.\n\n<details><summary>Check your reasoning</summary>Not measurably. In the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook) the chosen configuration (G3, head and four blocks, epoch 6) raised the fresh-test small persons from PCK 0.613 to 0.773 and OKS 0.585 to 0.718 (adapted minus frozen, mean of PCK and OKS, 95% interval +0.146 [+0.102, +0.187]), past the mean-pose prior (0.419 / 0.418), while their clean counterparts moved from PCK 0.977 to 0.978 and OKS 0.962 to 0.961. On the development test (the earlier test split, read during earlier development; continuity only) the same model went from PCK 0.597 to 0.743 and OKS 0.574 to 0.698 (clean OKS 0.940 to 0.938). An earlier revision taught a clean-OKS cost of fine-tuning (0.94 → 0.93). In a controlled CPU comparison (two blocks, the same 72-person validation, the same CPU) that drop appears with the old coordinate conventions (0.940 → 0.930) and not with the fixed ones (0.940 → 0.940), but a 0.01 OKS difference on 136 persons is near what one split can resolve: read it as consistent with a geometry cause, not proof of one (see **What did not work**).</details>"
            ),
        },
        {
            "md": (
                "## 9. Look at the skeletons, export the adapter and reload it\n\n"
                "The drawing from Section 5 is estimated again by the adapted model — an image family the adaptation never "
                "saw, so this is a small look, not a measurement, at what the adaptation did *outside* its corpus "
                "self-drawn PCK at 1.0 — and four held-out small persons are written as side-by-side panels "
                "(`outputs/{stem}_examples/`: the degraded crop enlarged four times with the frozen skeleton, the adapted "
                "skeleton, and the labelled joints) so the numbers can be checked by eye: the adapted skeletons should sit on "
                "the blurred limbs where the frozen ones drift.\n\n"
                "`pipe.save_artifact` writes the chosen configuration's trained tensors (for G3, the head with four blocks, about 130 MB) — for G2 the head, the last two blocks and the final LayerNorm, about "
                "74 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `VitPoseKeypointPipeline.from_artifact` re-verifies both base "
                "snapshots, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses "
                "any tensor outside the head and the recorded blocks, and overlays the tensors onto a freshly loaded base — a "
                "new object from files, not the in-memory model (VER2). The cell asserts identical joints on eight test persons "
                "(VER4) — a contract check, so it stays a hard check.\n\n"
                "**Predict before running:** in the panels, where will the adapted skeleton differ most from the frozen one?"
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene_result, adapted_scene = estimate_scene(pipe, 'adapted')\n"
                "examples_dir = Path('outputs/{stem}_examples' + test_look['suffix'])\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "frozen_base = VitPoseKeypointPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR, device=pipe.device)\n\n\n"
                "def skeleton_panel(record, joints, colour, zoom=4):\n"
                "    x0, y0, x1, y1 = record['box']\n"
                "    pad = 0.15 * (y1 - y0)\n"
                "    crop_box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(record['image'].width, int(x1 + pad) + 1), min(record['image'].height, int(y1 + pad) + 1))\n"
                "    panel = record['image'].crop(crop_box)\n"
                "    panel = panel.resize((panel.width * zoom, panel.height * zoom), Image.Resampling.NEAREST)\n"
                "    draw = ImageDraw.Draw(panel)\n"
                "    at = {{k: ((v[0] - crop_box[0]) * zoom, (v[1] - crop_box[1]) * zoom) for k, v in joints.items()}}\n"
                "    for a, b in SKELETON_EDGES:\n"
                "        na, nb = KEYPOINT_NAMES[a], KEYPOINT_NAMES[b]\n"
                "        if na in at and nb in at:\n"
                "            draw.line([at[na], at[nb]], fill=colour, width=2)\n"
                "    for x, y in at.values():\n"
                "        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)\n"
                "    return panel\n\n\n"
                "for record in held_out_records[:4]:\n"
                "    panels = [skeleton_panel(record, frozen_base.predict_keypoints(record), (0, 120, 255)), skeleton_panel(record, pipe.predict_keypoints(record), (0, 200, 80)), skeleton_panel(record, record['keypoints'], (255, 255, 255))]\n"
                "    sheet = Image.new('RGB', (sum(p.width for p in panels) + 20, max(p.height for p in panels)), (255, 255, 255))\n"
                "    x = 0\n"
                "    for panel in panels:\n"
                "        sheet.paste(panel, (x, 0))\n"
                "        x += panel.width + 10\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}_{{record.get('category', 'person')}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panel_order': ['frozen skeleton', 'adapted skeleton', 'labelled joints'], 'zoom': 4}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter' + test_look['suffix'])\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "if pipe.adapter is None:\n"
                "    artifact_manifest, parity = None, None\n"
                "    print({{'artifact': 'none: the pre-registered rule kept the frozen model, so there is no adapter to export or reload'}})\n"
                "else:\n"
                "    pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source, 'recipe': recipe_selection['chosen']}})\n"
                "    artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "    print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n"
                "    reloaded = VitPoseKeypointPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR, device=pipe.device)\n"
                "    before = [pipe.predict_keypoints(r) for r in held_out_records[:8]]\n"
                "    after = [reloaded.predict_keypoints(r) for r in held_out_records[:8]]\n"
                "    parity = {{'identical_persons': sum(all(np.allclose(a[k], b[k]) for k in KEYPOINT_NAMES) for a, b in zip(before, after, strict=True)), 'of': len(before), 'max_abs_difference_px': float(max(abs(np.array(a[k]) - np.array(b[k])).max() for a, b in zip(before, after, strict=True) for k in KEYPOINT_NAMES))}}\n"
                "    print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "    assert parity['identical_persons'] == parity['of']\n"
                "\n"
                ""
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'detector_model_id': DETECTOR_MODEL_ID,\n"
                "    'detector_revision': DETECTOR_REVISION,\n"
                "    'detector_license': DETECTOR_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'test_look': test_look['test_look'],\n"
                "    'preregistered': test_look['preregistered'],\n"
                "    'settings_differing': test_look['settings_differing'],\n"
                "    'settings': recipe_selection['settings'],\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': CORPUS_IMAGES, 'pinned_persons': CORPUS_PERSONS, 'degradation': {{'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256, 'findings': scene_findings}}, 'stage1': stage1, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene, 'output_files': ['outputs/{stem}_scene_frozen.png', 'outputs/{stem}_scene_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': str(examples_dir),\n"
                "    'artifact': None if artifact_manifest is None else {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'recipe_selection': recipe_selection,\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "result_path = Path('outputs/{stem}_result.json')\n"
                "result_path = result_path.with_stem(result_path.stem + test_look['suffix'])\n"
                "with open(result_path, 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the four panels in `outputs/{stem}_examples/`, the drawing's PCK after adaptation, and the reload parity line.\n\n<details><summary>Check your reasoning</summary>On the limbs: wrists, elbows and ankles are the joints a 40-px JPEG blurs most, and that is where the frozen skeleton drifts. Reload parity is a contract check: in the Colab T4 run of the previous revision (blob 49fc2c05) eight of eight persons were identical, with a largest difference of 7.6e-06 px.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default one encoder block, a scope the grid did not try, so "
                "the head and the last block train — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshots, so it starts from the checkpoint and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/` and `outputs/{stem}_experiment_exploratory_<n>.json`, prints the default and the changed run side by side (epoch 0 must match), saves and "
                "reloads its own adapter to check parity, and checks that the default exports are byte-identical afterwards.\n\n"
                "It is an **exploratory** run of a configuration the rule did not choose: it compares on validation only and never scores the fresh test; never pick a recipe from it.\n\n**Predict:** will one block land between G1 and G2 on validation?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_BLOCKS = 1  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "elif adapt_result is None:\n"
                "    print({{'experiment': 'skipped: the pre-registered rule kept the frozen model, so there is no default adapter to compare against'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': evaluation_report_path, 'result': result_path}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshots: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = VitPoseKeypointPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=EXPERIMENT_TRAINABLE_BLOCKS, progress=report)\n"
                "    experiment_val = experiment_pipe.evaluate(val_records)  # exploratory: validation only; the fresh test is never scored here\n"
                "    experiment_check = check_preregistration({{**RUN_SETTINGS, 'epochs': experiment_result['epochs'], 'learning_rate': experiment_result['lr'], 'batch_size': experiment_result['batch_size'], 'adapt_seed': experiment_result['seed'], 'run_recipe_grid': False, 'recipe_grid': [[f'blocks={{EXPERIMENT_TRAINABLE_BLOCKS}}', EXPERIMENT_TRAINABLE_BLOCKS]]}})\n"
                "    experiment_pipe.save_artifact(experiment_dir / 'adapter', metadata={{'tutorial': '{stem}', 'experiment': True}})\n"
                "    experiment_reloaded = VitPoseKeypointPipeline.from_artifact(experiment_dir / 'adapter', weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_parity = all(np.allclose(experiment_pipe.predict_keypoints(r)[k], experiment_reloaded.predict_keypoints(r)[k], atol=1e-3) for r in val_records[:4] for k in KEYPOINT_NAMES)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_blocks': adapt_result['trainable_blocks'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_blocks': EXPERIMENT_TRAINABLE_BLOCKS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'epoch_0_val_score': {{'default': (adapt_result['history'][0]['val'] or {{}}).get('score'), 'experiment': (experiment_result['history'][0]['val'] or {{}}).get('score')}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'validation': {{metric: {{'frozen': round(experiment_result['history'][0]['val'][metric], 3), 'default': round(adapted_val[metric], 3), 'experiment': round(experiment_val[metric], 3)}} for metric in METRICS}},\n"
                "        'test_look': 'not scored (exploratory)',\n"
                "        'settings_differing': experiment_check['settings_differing'],\n"
                "        'trainable_parameters': {{'default': adapt_result['n_trainable'], 'experiment': experiment_result['n_trainable']}},\n"
                "        'experiment_reload_parity': experiment_parity,\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    n = 1\n"
                "    while Path(f'outputs/{stem}_experiment_exploratory_{{n}}.json').exists():\n"
                "        n += 1\n"
                "    experiment_report_path = Path(f'outputs/{stem}_experiment_exploratory_{{n}}.json')\n"
                "    with open(experiment_report_path, 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': [str(experiment_dir), str(experiment_report_path)]}})\n"
                "    del experiment_pipe, experiment_reloaded"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare `epoch_0_val_score` (it must be equal: both runs start from the checkpoint), the "
                "`test` rows and `experiment_reload_parity`.\n\n"
                "<details><summary>Check your reasoning</summary>In the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook) the grid's head-only configuration (G1) reached a best validation score of 0.668, against 0.724 for two blocks and 0.648 frozen, all at this fixed learning rate and budget: the head can only re-read the features it is given, and at 40 px the features themselves are what is wrong, so the encoder blocks have to learn to read blurred, block-edged limbs. No experiment run is recorded; on 136 persons, read a test difference of a point or two as noise.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A pose estimator trained on crops of full-resolution persons loses about a third of its PCK when the person is 40 px tall and JPEG-blocked (0.977 → 0.613 on the fresh test in the second CPU development run of this revision's package code (fix branch on base 8934a7d9, 17 October 2026; not yet a recorded run of this notebook), against a box-only prior of 0.419), and a bounded fine-tuning chosen on validation by a pre-registered rule (G3, head and four blocks) recovers part of that loss on the fresh test, which no revision had read before (PCK 0.773, mean OKS 0.718, scored only under the pre-registered settings; on the development test 0.743 / 0.698), while the clean persons stay within the 0.01 tolerance (PCK 0.977 → 0.978, OKS 0.962 → 0.961), with an adapter that reloads joint-for-joint. All scores use the labelled boxes, so they are an upper bound on the two-stage system. That is "
        "the claim: the adaptation contract works end to end on a real labelled set under a stated degradation, and the numbers "
        "it produces are read on two measures, per category, against two box-only baselines, the frozen model and the clean "
        "counterparts rather than in isolation.\n\n"
        "The fresh test is 96 persons on 65 photographs and the development test 136 persons, from one sample under one degradation; the validation split that picks "
        "the recipe and the epoch is 72 persons on 45 photographs, the comparison of configurations holds for this learning rate and eight-epoch budget only (with G3 at the edge of the grid), and both measures score joints against COCO's labels at a stated tolerance — PCK at a tenth of the "
        "box, OKS with COCO's sigmas — so a gain here says the contract works under 40-px JPEG-30 downscaling, not that the "
        "adapted model handles motion blur, low light, unusual viewpoints, crowds or your camera's own artefacts, and not that "
        "its heatmap scores are calibrated (they are not, before or after). The categories are COCO's labelling density, a proxy "
        "for occlusion, not a property of the person. Fine-tuning on one degradation also shapes what the model expects: the "
        "clean counterparts held here, but a different degradation would need its own measurement. The photographs come from COCO val2017: ViTPose-base was trained on train2017, not val2017, but upstream training used val2017 to pick its released checkpoint (`save_best='AP'`), so the frozen clean scores are mildly optimistic.\n\n"
        "Three things to carry to real data. **Baselines first:** the box-centre floor and the mean-pose prior on *your* labels "
        "are the numbers to read before any model number, per subset. **Degradation:** the persons the model learns from define "
        "what it learns to read; make your training input the way your deployment makes it (the contract's `degrade` is one "
        "stated recipe; BYOD uses your labels as they are). **Leakage:** keep every photograph in one split (the contract does "
        "this by decoded pixels and keeps every person with its photograph) and split by camera or session when your images "
        "come from few sources.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify both pinned model snapshots, fetch and digest-verify a real labelled photograph set and "
        "degrade it under a stated recipe, validate the demonstrated dataset contract without leakage, execute the two-stage "
        "inference contract and a bounded fine-tuning, evaluate against two box-only baselines, the frozen model and the clean "
        "counterparts on an image-disjoint split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, pose quality under any other degradation, calibration of "
        "the heatmap scores, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 trains one changed configuration in "
        "its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_BLOCKS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` (for example `1e-4`) there and run "
        "that cell again. Setting `RUN_RECIPE_GRID = False` with another `TRAINABLE_BLOCKS` in Section 4 and choosing **Run after** from Section 4 "
        "also starts from the pinned base — every `adapt` puts it back first — but it is an exploratory run: validation only, the fresh test is not scored, and its reports go to separate `*_exploratory_<n>` files. "
        "BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, and read the two baselines before "
        "the adapted number.\n\n"
        "## What did not work\n\n- **Choosing the recipe on test (superseded).** An earlier revision fixed two blocks at 40 px after development runs whose test scores had been read (head only at 64 px, two blocks at 32 px, four blocks at 32 px; recorded in the model card as superseded development runs, not evidence from this notebook). This revision keeps that task and those settings, chooses only the configuration, on validation, by the rule at the top, and scores the fresh test only under the pre-registered settings.\n- **A clean-OKS cost we could not confirm.** The earlier revision reported that fine-tuning cost the clean persons OKS (0.94 → 0.93 in the Kaggle T4 run of blob 050796f8). Two coordinate-convention errors were found and fixed: the degraded labels were scaled as `x·s`, 0.33–0.47 px off the image content COCO's pixel-index labels mark, and the heatmap targets were encoded with a `crop / heatmap` stride whose bias grows across the crop (mean 0.47 px, max 0.93 px on 40-px boxes) while the processor decodes with `(crop − 1) / (heatmap − 1)`. In a controlled CPU comparison (two blocks, the same 72-person validation, the same CPU) the clean OKS after fine-tuning was 0.930 with the old conventions and 0.940 with the fixed ones; in this revision it is 0.961 against 0.962 frozen on the fresh test. A 0.01 OKS difference on 136 persons is near what one split resolves, so the geometry is a likely cause of the earlier drop, not an established one.\n- **Reusing a test split.** The first pass of this fix still scored the earlier test split, whose scores had been read while the task and hyperparameters were set, and spent the 65 never-scored photographs on validation. They are now the fresh test, and the earlier test split is the labelled development test.\n\n## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 cannot fetch a photograph, or one fails its digest** — `fetch_corpus` names it; the default path needs `images.cocodataset.org` (plain HTTP; every file is checked against its pinned SHA-256). Run Section 4 again (cached photographs are re-hashed); delete `weights/coco-val-persons/` if a cached file is corrupt.\n- **Section 7 takes an hour or more** — that is the CPU cost of three eight-epoch fine-tunes; choose a GPU runtime and **Run all** again.\n- **Out of memory** — restart the session on a larger runtime (`BATCH_SIZE` is part of the pre-registered recipe; a smaller one makes the run exploratory: validation only, and the fresh test is not scored) and choose **Run all**; leave Section 10 off on a small runtime.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "keypoints.csv line N (file …): names an image that is not among the uploaded files"** — fix the `file` column of that row, or add the image to the zip.\n- **BYOD: "… must be numbers" / "… must be present and numeric"** — a box or joint coordinate on that line is not a number.\n- **BYOD: "the train split holds … persons" or "… too few to split"** — add images or persons; the message names the minimum.\n'
        "## Glossary\n\n"
        "- **Top-down pose** — find each person's box first, then estimate that person's joints inside it.\n"
        "- **Heatmap** — a 64 × 48 map per joint whose maximum is the predicted location; the maximum's value is a "
        "ranking score, not a probability.\n"
        "- **PCK** — the share of labelled joints within a tenth of the box's longest side of their label.\n"
        "- **OKS / sigma** — COCO's object-keypoint similarity: a per-joint Gaussian kernel whose width (sigma) is set per "
        "joint and scaled by the box area, averaged over the labelled joints.\n"
        "- **Box-centre / mean-pose baseline** — every joint at the box centre; every joint at its mean position in the box "
        "over the training split.\n"
        "- **Clean counterpart** — the same person in the unchanged, full-resolution photograph.\n"
        "- **Degradation** — the stated recipe: downscale so the box is 40 px tall, then JPEG at quality 30.\n"
        "- **Epoch selection** — keeping the epoch with the highest mean of validation PCK and OKS; epoch 0 is the frozen "
        "model.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline gives identical joints.\n"
        "- **BYOD** — bring your own data: your labelled persons through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The data was ___ persons on ___ photographs; test split ___ persons.\n"
        "- Mean-pose prior PCK ___; frozen ___ on the small persons and ___ on the clean ones.\n"
        "- After fine-tuning (epoch ___ kept): PCK ___, OKS ___; clean PCK ___, clean OKS ___ — a cost of ___.\n"
        "- What I would need before claiming the fine-tune helps on my camera: ___ (for example my own degradation, more test persons, several seeds).\n\n"
        "## References\n"
        "\n"
        "[1] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, 'ViTPose: Simple vision transformer baselines for human pose estimation,' in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022. [Online]. Available: https://arxiv.org/abs/2204.12484\n"
        "\n"
        "[2] Y. Zhao *et al.*, 'DETRs beat YOLOs on real-time object detection,' in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024. [Online]. Available: https://arxiv.org/abs/2304.08069\n"
        "\n"
        "[3] T.-Y. Lin *et al.*, 'Microsoft COCO: Common objects in context,' in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014, doi: 10.1007/978-3-319-10602-1_48. Keypoint annotations CC BY 4.0; photographs under their Flickr licences: https://cocodataset.org/#termsofuse\n"
        "\n"
        "[4] J. Huang, Z. Zhu, F. Guo, and G. Huang, 'The devil is in the details: Delving into unbiased data processing for human pose estimation,' in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020. [Online]. Available: https://arxiv.org/abs/1911.07524\n"
        "\n"
        "[5] usyd-community, 'vitpose-base,' Hugging Face model card, revision 95be2991424e. [Online]. Available: https://huggingface.co/usyd-community/vitpose-base/tree/95be2991424e646950d656bb7fc15ec9be119700\n"
        "\n"
        "[6] PekingU, 'rtdetr_r50vd,' Hugging Face model card, revision df939e661d8c. [Online]. Available: https://huggingface.co/PekingU/rtdetr_r50vd/tree/df939e661d8c52e80608d1ec566561aabd25a4e7\n"
        "\n"
        "[7] ViTAE-Transformer, 'ViTPose,' GitHub repository (configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py). [Online]. Available: https://github.com/ViTAE-Transformer/ViTPose\n"
        "\n"
        "Repository documents (not cited above): README https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/README.md · model card https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/MODEL_CARD.md · weight provenance https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/docs/WEIGHTS.md · DIMER Notebook Specification 2.5 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
