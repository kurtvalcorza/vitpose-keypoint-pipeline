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
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `usyd-community/vitpose-base` and `PekingU/rtdetr_r50vd` snapshots (a 360 MB and a 172 MB `model.safetensors`; "
        "no pickle is opened anywhere), fetches the 300 pinned COCO val2017 photographs from the COCO image host (about 52 MB, "
        "each refused on any byte-size or SHA-256 mismatch), turns their 498 labelled persons into the stated small-person "
        "input (the whole image downscaled so the person box is 40 px tall, then JPEG at quality 30) with the boxes and joints "
        "scaled to match, splits them by image into 180 / 45 / 75 photographs (290 / 72 / 136 persons), estimates a synthetic "
        "drawing through the two-stage inference contract with an input manifest and a rejection probe, measures the frozen "
        "model's PCK and mean OKS over the 136 held-out persons beside the box-centre and mean-pose baselines and against the "
        "same persons at full resolution, runs a bounded fine-tuning of the heatmap head and the last two encoder blocks with "
        "the heatmap mean-squared error and epoch selection on the mean validation PCK and OKS, scores the held-out persons "
        "again per category, re-estimates the drawing and four held-out persons with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify keypoint parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On an RTX 5070 Ti the whole path took under three minutes after the downloads; a CUDA runtime "
        "is used automatically when present, and on a 2-vCPU hosted runtime expect the eight epochs to take half an hour or more."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of photographs plus a `keypoints.csv` (`file`, `person`, `x0`, `y0`, `x1`, `y1`, optional `category`, and one "
        "`<joint>_x` / `<joint>_y` pair per labelled COCO joint, blank when unlabelled — at least 12 of the 17 per person, at "
        "least eight persons). Your persons are used as labelled — the notebook does not degrade them — and pass through the "
        "same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity "
        "cells as the COCO sample. Uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "ViTPose is a **top-down** pose estimator: it needs a person box first. Stage 1 runs the RT-DETR R50-VD detector "
        "(the same checkpoint the sibling `rtdetr-detection-pipeline` wraps) over the whole image and keeps the boxes "
        "labelled `person` above a threshold; stage 2 affine-warps each box to a 192×256 crop, runs a plain ViT-B "
        "backbone with a heatmap decoder (89,994,513 parameters, of which the head is 4,199,697; published under the "
        "**Apache-2.0** licence), and reads 17 COCO keypoints — nose, eyes, ears, shoulders, elbows, wrists, hips, knees, "
        "ankles — back in input-pixel coordinates, each with the maximum of its 64×48 heatmap as its score (a ranking "
        "signal, not a calibrated probability).\n\n"
        "What this notebook adds to inference is **adaptation with labelled persons** under a stated input shift. "
        "ViTPose was trained on COCO persons at the crop resolution a 256-px box gives it; on the same photographs the "
        "frozen model reaches a PCK of **0.957** (the build record's figure on the 136 held-out persons), so there is "
        "nothing honest to gain there. The input the adaptation is about is the **distant, low-quality person** a "
        "surveillance camera or a wide shot hands a pose estimator: each photograph is downscaled so the person box is "
        "**40 px tall** and re-encoded as JPEG at quality 30, and on that input the frozen model drops to **0.598** — "
        "not far above the 0.472 that placing every joint at its mean position in the box achieves with no model at all. "
        "So the honest question is narrow: does a bounded fine-tuning of the heatmap head and the last two encoder blocks on "
        "290 such persons move the held-out **PCK** and **mean OKS** on an image-disjoint test split, per category, past the "
        "two **box-only baselines** — while the same model still scores the clean, full-resolution persons as before? "
        "Nothing here is a claim about your cameras: it is one seeded split of one sample under one stated degradation.\n\n"
        "**Snapshot note:** both pinned revisions ship `model.safetensors` (4-file manifests) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies both before either processor or model is constructed."
    ),
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
        "full COCO validation set or any benchmark (only one seeded 498-person sample from 300 of its images is scored), and "
        "any claim that a 40-px JPEG person stands in for your camera's degradation. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. Stage 2 costs about 0.15 s per person on the build workstation's CPU and a few milliseconds on a GPU; the build record measured 1.1 s to prepare the 290 training crops and 31.5 s for the eight epochs with per-epoch validation scoring on an RTX 5070 Ti, and the whole default path took about two and a half minutes there with both snapshots and the photographs already cached. A 2-vCPU hosted runtime will take much longer (ViT-B forward and backward on 290 crops per epoch). The pinned `torch==2.14.0` install and the two checkpoints (360 MB ViTPose, 172 MB RT-DETR) are the large downloads of the run; the photographs are about 52 MB.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a bounding box in xyxy pixel coordinates is; what a keypoint heatmap is and why its maximum is not a probability; what PCK and OKS measure and why a self-drawn figure is a plumbing check while a held-out split under a stated degradation is a measurement of that degradation only.",
        "- **Data contract:** records are `{id, image, box, keypoints, category?}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4096 px, `box` one `[x0, y0, x1, y1]` person box inside it, `keypoints` a mapping from COCO joint name to `[x, y]` for every labelled joint (at least 12 of the 17), an optional `category` of at most 32 characters. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting keeps every image (by decoded pixels) in one split and every person with its image. BYOD accepts one zip (or directory) of photographs plus a `keypoints.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image is decoded, every box and joint checked to lie inside it, but nothing checks that a joint is where its name says — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data (photographs of identifiable people you have no consent to process) to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the two model snapshots, the default path fetches 300 JPEG files from `http://images.cocodataset.org/val2017/<id>.jpg` (about 52 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's Flickr page and licence (CC BY 2.0 or CC BY-SA 2.0 — COCO licence ids 4 and 5 only) are kept in its record, and the 2017 keypoint annotations it carries are CC BY 4.0 (COCO Consortium). Nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. COCO persons, the small-person degradation and the split\n\n"
                "`fetch_corpus` returns the 300 pinned photographs from the cache under `weights/coco-val-persons/` or the "
                "COCO image host — every cached file is re-hashed and every fetched file refused on any byte-size or SHA-256 "
                "mismatch — and `read_corpus` turns each labelled person into a record: `degrade` downscales the whole "
                "photograph (bicubic) so the person's box is `TARGET_HEIGHT` px tall, re-encodes it as JPEG at `JPEG_QUALITY`, "
                "and scales the box and the labelled joints by the same factor; the record's `category` says how many of the "
                "17 joints COCO labelled (`full` 16–17, `partial` 14–15, `sparse` 12–13 — a proxy for occlusion and "
                "truncation). `build_sample_dataset` draws a seeded image-level split (180 / 45 / 75 photographs; every person "
                "follows its photograph), and `read_corpus(..., target_height=None)` builds the **clean counterparts** of the "
                "test persons — the same photographs unchanged — for Section 6. `validate_dataset` then checks every record "
                "against the contract, `check_split_disjoint` asserts no photograph (by decoded-pixel digest) is shared, and "
                "the training split's summary table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 300 photographs and 498 persons, boxes about 40 px tall after degradation, three categories, three "
                "digests, and four refusal probes — a duplicate id, a joint outside its image, a person with too few labelled "
                "joints, and a dataset too small to use — each rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    clean_test = None\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/coco-val-persons')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    test_images = {{r['image_id'] for r in splits['test']}}\n"
                "    clean_test = read_corpus({{k: v for k, v in corpus_files.items() if k in test_images}}, target_height=None)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}}'\n"
                "    raw_rows = {{'photographs': len(corpus_files), 'bytes': sum(len(v) for v in corpus_files.values()), 'persons': len(corpus), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "if clean_test is not None:\n"
                "    clean_test = validate_dataset(clean_test)['records']\n"
                "    assert [r['id'] for r in clean_test] == [r['id'] for r in test_records]\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'degradation': {{'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY}}, 'licence': CORPUS_LICENSE}})\n"
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
                "## 5. Estimate a synthetic drawing through the two-stage inference contract\n\n"
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
                "## 6. Baselines, the frozen model on the small persons — and on the clean ones\n\n"
                "Two box-only baselines frame the adaptation, each read two ways by `pose_metrics` (carried in `metrics.py`): "
                "**PCK** — the share of labelled joints within `PCK_FRACTION` (0.1) of the box's longest side of their label, "
                "the package's own `keypoint_pck` convention — and **mean OKS** — the COCO object-keypoint-similarity kernel per "
                "joint (COCO's per-joint sigmas, the box area as the object scale) averaged over the labelled joints, the "
                "primitive of the COCO keypoint AP without its thresholds — averaged over the persons and per category. The "
                "**box-centre** baseline puts every joint at the box centre: the floor. The **mean-pose** baseline puts every "
                "joint at its mean position relative to the box over the training split — the layout prior a box alone gives "
                "away, and the number a pose estimator must beat to be doing anything. The **frozen model** is scored by "
                "`pipe.evaluate`, which runs every record's box through `estimate` and scores all 17 joints. It is scored "
                "twice: on the **small persons** the adaptation is about, and on their **clean counterparts** — the same "
                "persons in the unchanged photographs — which is where the checkpoint was trained to work. Expect the frozen "
                "model near the mean-pose prior on the small persons (0.598 against 0.472 in the build record) and far above "
                "it on the clean ones (0.957); read the per-category rows to see that sparse labels do not explain the gap."
            ),
            "code": (
                "METRICS = ('pck', 'oks')\n"
                "baseline_centre = box_centre_baseline(test_records)\n"
                "baseline_prior = mean_pose_baseline(train_records, test_records)\n"
                "print({{'box_centre_baseline': {{k: round(baseline_centre[k], 3) for k in METRICS}}, 'n': baseline_centre['n'], 'note': baseline_centre['baseline']}})\n"
                "print({{'mean_pose_baseline': {{k: round(baseline_prior[k], 3) for k in METRICS}}, 'note': baseline_prior['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_small_persons': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "frozen_clean = pipe.evaluate(clean_test) if clean_test is not None else None\n"
                "if frozen_clean is not None:\n"
                "    print({{'frozen_clean_counterparts': {{k: round(frozen_clean[k], 3) for k in METRICS}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'pck': round(v['pck'], 3), 'oks': round(v['oks'], 3)}} for c, v in frozen_test['per_category'].items()}}\n"
                "print({{'by_category_frozen': frozen_fields}})\n"
                "assert frozen_test['pck'] > baseline_centre['pck']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the heatmap head and the last encoder blocks\n\n"
                "`pipe.adapt` trains only the heatmap head (two deconvolutions and the final convolution) and the last "
                "`TRAINABLE_BLOCKS` encoder blocks with the final backbone LayerNorm — 18,376,977 of 89,994,513 parameters for "
                "two blocks — while the patch embedding, the earlier blocks and the whole detector stay frozen. Every record's "
                "box is affine-warped to the processor's 192×256 crop exactly as at inference, its labelled joints become "
                "Gaussian targets (sigma `HEATMAP_SIGMA` = 2) on the 64×48 heatmap grid in the same frame, and the head's "
                "heatmaps are trained with the **joint-weighted mean-squared error** ViTPose was trained with (unlabelled joints "
                "carry no weight). AdamW without weight decay at a fixed learning rate, gradient clipping at 1.0, seeded "
                "shuffling, no scheduler, no augmentation; the head's BatchNorm statistics stay frozen so the adapter is "
                "parameters-only. Epoch 0 records the frozen model's validation metrics; every epoch is scored on the 72 "
                "validation persons, and the epoch with the highest **mean of validation PCK and OKS** is kept.\n\n"
                "Watch the training loss fall from about 0.0028 towards 0.0013 while the validation PCK climbs from about 0.66 "
                "to 0.74 over eight epochs: the head is learning to read blurred, block-edged limbs it never saw at training "
                "resolution, which 290 persons are enough to teach. The build record's counter-examples — the head alone, which "
                "moved PCK by a hundredth, and four blocks at 32-px persons, which peaked and then overfit — are why the "
                "default is two blocks at 40 px."
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 5)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in (*METRICS, 'score')}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=TRAINABLE_BLOCKS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'preparation_seconds': adapt_result['preparation_seconds'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test persons were never used for training or epoch selection, and no photograph appears in two splits. "
                "The adapted model is scored exactly as the frozen model was in Section 6 — on the small persons and on their "
                "clean counterparts — the systems are put side by side on both measures, and the per-category breakdown is "
                "repeated. Read it in this order: **PCK on the small persons** first (the build record measured 0.598 → 0.721, "
                "past the mean-pose prior's 0.472), then **mean OKS** (0.565 → 0.684), then the **clean counterparts** (0.957 → "
                "0.958: the adaptation did not cost the resolution the checkpoint was built for), then the per-category rows, "
                "where fully labelled persons gain the most. The cell asserts the adapted PCK on the small persons is above the "
                "frozen one and reports the rest. A hundred-odd persons from one seeded split under one degradation give **no "
                "dispersion estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a "
                "benchmark, and a gain at 40-px JPEG persons says nothing about motion blur, unusual viewpoints or your camera's "
                "own artefacts until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "adapted_clean = pipe.evaluate(clean_test) if clean_test is not None else None\n"
                "adapted_fields = {{c: {{'n': v['n'], 'pck': round(v['pck'], 3), 'oks': round(v['oks'], 3)}} for c, v in adapted_test['per_category'].items()}}\n"
                "comparison = {{metric: {{'box_centre': round(baseline_centre[metric], 3), 'mean_pose': round(baseline_prior[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "if frozen_clean is not None:\n"
                "    for metric in METRICS:\n"
                "        comparison[metric].update({{'frozen_clean': round(frozen_clean[metric], 3), 'adapted_clean': round(adapted_clean[metric], 3)}})\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['delta_vs_mean_pose'] = {{metric: round(adapted_test[metric] - baseline_prior[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_pck': frozen_fields[c]['pck'], 'adapted_pck': adapted_fields[c]['pck'], 'frozen_oks': frozen_fields[c]['oks'], 'adapted_oks': adapted_fields[c]['oks']}} for c in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'degradation': {{'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY}},\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'box_centre': {{k: v for k, v in baseline_centre.items() if k != 'per_record'}}, 'mean_pose': {{k: v for k, v in baseline_prior.items() if k != 'per_record'}}}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'frozen_clean': frozen_clean,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'clean_metrics': adapted_clean,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['pck'] > frozen_test['pck']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_mean_pose': adapted_test['pck'] > baseline_prior['pck']}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the skeletons, export the adapter and reload it\n\n"
                "The drawing from Section 5 is estimated again by the adapted model — an image family the adaptation never "
                "saw, so this is a small look at what the adaptation did *outside* its corpus: the build record kept the "
                "self-drawn PCK at 1.0 — and four held-out small persons are written as side-by-side panels "
                "(`outputs/{stem}_examples/`: the degraded crop enlarged four times with the frozen skeleton, the adapted "
                "skeleton, and the labelled joints) so the numbers can be checked by eye: the adapted skeletons should sit on "
                "the blurred limbs where the frozen ones drift.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the head, the last two blocks and the final LayerNorm, about "
                "74 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `VitPoseKeypointPipeline.from_artifact` re-verifies both base "
                "snapshots, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses "
                "any tensor outside the head and the recorded blocks, and overlays the tensors onto a freshly loaded base — a "
                "new object from files, not the in-memory model (VER2). The cell asserts identical joints on eight test persons "
                "(VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene_result, adapted_scene = estimate_scene(pipe, 'adapted')\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
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
                "for record in test_records[:4]:\n"
                "    panels = [skeleton_panel(record, frozen_base.predict_keypoints(record), (0, 120, 255)), skeleton_panel(record, pipe.predict_keypoints(record), (0, 200, 80)), skeleton_panel(record, record['keypoints'], (255, 255, 255))]\n"
                "    sheet = Image.new('RGB', (sum(p.width for p in panels) + 20, max(p.height for p in panels)), (255, 255, 255))\n"
                "    x = 0\n"
                "    for panel in panels:\n"
                "        sheet.paste(panel, (x, 0))\n"
                "        x += panel.width + 10\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}_{{record.get('category', 'person')}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panel_order': ['frozen skeleton', 'adapted skeleton', 'labelled joints'], 'zoom': 4}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = VitPoseKeypointPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.predict_keypoints(r) for r in test_records[:8]]\n"
                "after = [reloaded.predict_keypoints(r) for r in test_records[:8]]\n"
                "parity = {{'identical_persons': sum(all(np.allclose(a[k], b[k]) for k in KEYPOINT_NAMES) for a, b in zip(before, after, strict=True)), 'of': len(before), 'max_abs_difference_px': float(max(abs(np.array(a[k]) - np.array(b[k])).max() for a, b in zip(before, after, strict=True) for k in KEYPOINT_NAMES))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_persons'] == parity['of']\n\n"
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
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': CORPUS_IMAGES, 'pinned_persons': CORPUS_PERSONS, 'degradation': {{'target_height_px': TARGET_HEIGHT, 'jpeg_quality': JPEG_QUALITY}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256, 'findings': scene_findings}}, 'stage1': stage1, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene, 'output_files': ['outputs/{stem}_scene_frozen.png', 'outputs/{stem}_scene_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A pose estimator trained on crops of full-resolution persons loses a third of its PCK when the person is 40 px tall "
        "and JPEG-blocked (0.957 → 0.598 in the build record, against a box-only prior of 0.472), and a bounded fine-tuning of "
        "the heatmap head and the last two encoder blocks on 290 such persons recovers a third of that loss (0.721 PCK, 0.684 "
        "mean OKS) without costing the clean input (0.957 → 0.958), with a 74 MB adapter that reloads joint-for-joint. That is "
        "the claim: the adaptation contract works end to end on a real labelled set under a stated degradation, and the numbers "
        "it produces are read on two measures, per category, against two box-only baselines, the frozen model and the clean "
        "counterparts rather than in isolation.\n\n"
        "The test split is 136 persons from one seeded draw of one sample under one degradation, the validation split that picks "
        "the epoch is 72, and both measures score joints against COCO's labels at a stated tolerance — PCK at a tenth of the "
        "box, OKS with COCO's sigmas — so a gain here says the contract works under 40-px JPEG-30 downscaling, not that the "
        "adapted model handles motion blur, low light, unusual viewpoints, crowds or your camera's own artefacts, and not that "
        "its heatmap scores are calibrated (they are not, before or after). The categories are COCO's labelling density, a proxy "
        "for occlusion, not a property of the person. Fine-tuning on one degradation also shapes what the model expects: the "
        "clean counterparts held here, but a different degradation would need its own measurement.\n\n"
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
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_BLOCKS = 0` and read how little the head "
        "alone recovers; set `TRAINABLE_BLOCKS = 4` and watch the validation score peak early and fall; raise `EPOCHS` and read "
        "whether the selection rule holds the best epoch; change `LEARNING_RATE` to `1e-4` and compare the climb; or bring your "
        "own labelled persons through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/ViTAE-Transformer/ViTPose\n"
        "- ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation (Xu, Zhang, Zhang, Tao, NeurIPS 2022): https://arxiv.org/abs/2204.12484\n"
        "- RT-DETR (Zhao et al., CVPR 2024): https://arxiv.org/abs/2304.08069\n"
        "- Microsoft COCO: Common Objects in Context (Lin et al., ECCV 2014; keypoint annotations CC BY 4.0, photographs under their Flickr licences): https://cocodataset.org/#termsofuse\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
