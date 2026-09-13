"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package. This package pins TWO snapshots (ViTPose and the
RT-DETR person detector): ``rewrites`` makes the shared weights root working-directory-relative and
``extra_weights`` carries the detector manifest inline with its own stage/verify helpers.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "vitpose_keypoint_pipeline",
    "repo_name": "vitpose-keypoint-pipeline",
    "stem": "vitpose_keypoint",
    "notebook_name": "vitpose_keypoint_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "VitPoseKeypointPipeline",
    "weights_key": "vitpose-base",
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
    "title": "ViTPose-base + RT-DETR — DIMER two-stage human pose estimation tutorial (standalone)",
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
    "capability": "two-stage human pose estimation — `person` boxes from the pinned `PekingU/rtdetr_r50vd` detector (or boxes you supply), then 17 COCO keypoints per person from the pinned `usyd-community/vitpose-base` weights",
    "intro": (
        "ViTPose is a **top-down** pose estimator: it needs a person box first. Stage 1 runs the RT-DETR R50-VD detector "
        "(the same checkpoint the sibling `rtdetr-detection-pipeline` wraps) over the whole image and keeps the boxes "
        "labelled `person` above a threshold; stage 2 affine-warps each box to a 192×256 crop, runs a plain ViT-B "
        "backbone (86M parameters) with a heatmap decoder, and reads 17 COCO keypoints — nose, eyes, ears, shoulders, "
        "elbows, wrists, hips, knees, ankles — back in input-pixel coordinates, each with the maximum of its heatmap as "
        "its score. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting "
        "happens in this notebook — the two upstream checkpoints supply the weights and processor configurations, and the "
        "carried module adds snapshot verification for both, the input contract (image side ceilings, optional caller "
        "boxes, two thresholds), a fixed output contract and the `keypoint_pck`, `validate_inputs` and "
        "`evaluation_report` helpers. The default sample is a cartoon person drawn in code whose joint positions are "
        "known exactly — and on which the photograph-trained detector finds **no** `person` at the default threshold, so "
        "the notebook records that finding and hands the drawn box to stage 2 itself; the resulting PCK is demonstration "
        "(plumbing) evidence for one drawing, not a pose benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify both "
        "immutable upstream model revisions, draw a synthetic person with known joints (or upload your own photograph) and "
        "validate it into an input manifest, run stage 1 and read what a closed-set detector does with a drawing, run "
        "stage 2 with a caller-supplied box, read heatmap scores and the two caller-owned thresholds correctly, exercise an "
        "optional BYOD path, produce an evaluation report that is `sample-sanity` with per-person `keypoint_pck` only when "
        "reference joints exist and `not-measurable` otherwise, and export machine-readable keypoints plus a skeleton "
        "overlay and provenance."
    ),
    "exclusions": (
        "bottom-up or multi-person association without boxes (every person needs a box, from the detector or from you), "
        "3-D pose, tracking across frames, hand, face or whole-body keypoints beyond the 17 COCO joints (ViTPose+ and "
        "other checkpoints cover those), COCO OKS-based average precision (which needs a labelled keypoint set; only PCK "
        "against joints you drew is computed here), the upstream MMPose evaluation stack, or any training. A box that "
        "contains no person still yields 17 keypoints, with low heatmap scores as the only signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 0.3 s for stage 1 on the 640×640 drawing and 0.15 s for stage 2 on one person in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the two checkpoints (360 MB ViTPose, 172 MB RT-DETR) are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what a keypoint heatmap is and why its maximum is not a probability; what percentage-of-correct-keypoints measures.",
        "- **Data:** the default sample is a deterministic 640×640 cartoon person drawn in code with Pillow (head, torso, limbs in flat colours on a plain background) from 17 joint coordinates you can read in the code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) — a photograph with one or more people fully visible works best — any colour mode, sides between 16 and 4096 px; on BYOD the detector supplies the boxes. Do not upload confidential or restricted data (photographs of identifiable people you have no consent to process) to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic person or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference: a 640×640 cartoon person — a skin-tone "
                "head with eyes and a mouth, a red shirt torso, red sleeves with skin forearms and hands, blue trousers and "
                "dark shoes — is drawn with Pillow from a dictionary of 17 joint coordinates in COCO order (`REFERENCE_JOINTS`), "
                "the same drawing the repository's smoke run used. Those coordinates are the reference for the `keypoint_pck` "
                "sanity check later, and the drawn figure's bounding box `DRAWN_BOX` is the person box stage 2 will receive; "
                "they are not a labelled dataset, so nothing here is a pose benchmark, and a flat cartoon is not a photograph. "
                "The image digest is printed for the record. BYOD is optional and disabled by default; when enabled, upload one "
                "image — stage 1 supplies the boxes and no reference joints exist, so the evaluation report will be "
                "`not-measurable`.\n\n"
                "Both thresholds are **caller-owned request parameters**, not pipeline constants: `detection_threshold` keeps a "
                "`person` box whose RT-DETR sigmoid score reaches it, `keypoint_threshold` keeps a joint whose heatmap maximum "
                "reaches it (joints below it are still listed under `all_keypoints`). Their package defaults (`DETECTION_THRESHOLD "
                "= 0.3`, `KEYPOINT_THRESHOLD = 0.3`) follow the pinned ViTPose README's two-stage example, not a calibration; they "
                "are exposed here as form parameters and passed explicitly on every call. Nothing is validated in this cell — the "
                "next section hands the image, the boxes and both thresholds to the pipeline's own validation stage, which is "
                "the only checker. Look for a dictionary naming the sample kind, the image size and digest, the thresholds and "
                "the reference joints."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "detection_threshold = 0.3  # @param {{type:\"number\"}}\n"
                "keypoint_threshold = 0.3  # @param {{type:\"number\"}}\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    reference_joints, drawn_box = None, None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic drawing: no randomness and no text rendering, so the digest is stable across Pillow builds.\n"
                "    image = cartoon_person()\n"
                "    reference_joints, drawn_box = {{k: (float(x), float(y)) for k, (x, y) in REFERENCE_JOINTS.items()}}, DRAWN_BOX\n"
                "    image_name = 'synthetic_person_640x640.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'detection_threshold': detection_threshold, 'keypoint_threshold': keypoint_threshold, 'drawn_box': drawn_box, 'n_reference_joints': None if reference_joints is None else len(reference_joints)}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `estimate` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, optional person boxes (1..`MAX_PERSONS`, each a "
                "non-empty xyxy box inside the image) and both thresholds in `[0, 1]` — and returns an **input manifest** naming "
                "the schema (including the 17 keypoint names and both models' preprocessing), the input's observed mode and size, "
                "the boxes and their source (`caller` or `detector`), the thresholds, the verdict and **both** model identities. "
                "The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a box that leaves the image and records the pipeline's own error message as a finding. Inside the "
                "pipeline the image is converted to RGB; stage 1 resizes it to 640×640 for the detector and stage 2 warps each "
                "box to 192×256 for ViTPose; keypoints are mapped back to input pixels, and nothing else is dropped or altered. "
                "On the synthetic path the manifest names the drawn box as a caller box, which is what stage 2 will receive."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PERSONS': MAX_PERSONS, 'DETECTION_THRESHOLD': DETECTION_THRESHOLD, 'KEYPOINT_THRESHOLD': KEYPOINT_THRESHOLD, 'KEYPOINT_NAMES': list(KEYPOINT_NAMES), 'PCK_FRACTION': PCK_FRACTION}}}})\n"
                "input_manifest = validate_inputs(image, person_boxes=None if drawn_box is None else [drawn_box], detection_threshold=detection_threshold, keypoint_threshold=keypoint_threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, person_boxes=[[0, 0, image.width + 1, 10]])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'box-outside-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in input_manifest.items() if k != 'schema'}}, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Stage 1 (detect people), stage 2 (keypoints) — and read the scores correctly\n\n"
                "`detect_people` runs stage 1 alone and returns the `person` boxes above `detection_threshold`, sorted by score. "
                "On the cartoon the photograph-trained RT-DETR finds **no** person at the default 0.3 — as recorded in the model "
                "card, the smoke run got zero boxes at 0.3 and seven low-confidence `person` proposals at 0.05 — so the notebook "
                "records that finding and calls `estimate` with the drawn box as a caller box (`box_source: caller`); on BYOD "
                "it calls `estimate` without boxes and stage 1 supplies them (`box_source: detector`). `estimate` returns one "
                "entry per person in `poses`: the `box`, the detector's `person_score` (or `None` for a caller box), "
                "`all_keypoints` — all 17 joints in `KEYPOINT_NAMES` order with pixel `x`, `y` and a `score` — and `keypoints`, "
                "the subset at or above `keypoint_threshold`. Each keypoint `score` is the **maximum of that joint's heatmap "
                "after the crop's affine warp — a ranking signal per joint, not a calibrated probability**, and a box that "
                "contains no person still yields 17 joints (the smoke run's blank box scored every joint below 0.04, which is "
                "the only signal you get). Inference is deterministic on a fixed device and dtype (no sampling, "
                "`torch.inference_mode`); CUDA kernel selection can move coordinates by a pixel and scores in the third decimal. "
                "The smoke run's stage 2 on this drawing returned all 17 joints at scores 0.85–0.97 with a mean error of about "
                "13 px against the drawn joints; that is one observation on a cartoon, not a calibration point."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "stage1 = pipe.detect_people(image, threshold=detection_threshold)\n"
                "t_stage1 = time.time() - t0\n"
                "print({{'stage1_persons': stage1['n_persons'], 'stage1_seconds': round(t_stage1, 2), 'top_person_scores': [round(p['score'], 3) for p in stage1['persons'][:3]]}})\n"
                "findings = []\n"
                "if drawn_box is not None:\n"
                "    if stage1['n_persons'] == 0:\n"
                "        findings.append('stage 1 found no `person` on the drawing at the default threshold; stage 2 receives the drawn box as a caller box')\n"
                "    t0 = time.time()\n"
                "    result = pipe.estimate(image, person_boxes=[drawn_box], detection_threshold=detection_threshold, keypoint_threshold=keypoint_threshold)\n"
                "else:\n"
                "    t0 = time.time()\n"
                "    result = pipe.estimate(image, detection_threshold=detection_threshold, keypoint_threshold=keypoint_threshold)\n"
                "t_stage2 = time.time() - t0\n"
                "print({{'box_source': result['box_source'], 'n_persons': result['n_persons'], 'stage2_seconds': round(t_stage2, 2), 'device': pipe.device, 'findings': findings}})\n"
                "for index, pose in enumerate(result['poses']):\n"
                "    print(f\"person {{index}}: box {{[round(v, 1) for v in pose['box']]}}  person_score {{pose['person_score']}}  kept {{pose['n_keypoints']}}/17  mean score {{pose['mean_keypoint_score']:.3f}}\")\n"
                "    for kp in pose['all_keypoints']:\n"
                "        flag = '' if kp['score'] >= keypoint_threshold else '  (below keypoint_threshold)'\n"
                "        print(f\"    {{kp['name']:12}} x {{kp['x']:7.1f}}  y {{kp['y']:7.1f}}  score {{kp['score']:.3f}}{{flag}}\")\n"
                "if result['n_persons'] == 0:\n"
                "    print('No person box was available, so stage 2 did not run; on BYOD, lower detection_threshold or supply boxes.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No pose metric is "
                "reported by default: COCO's OKS-based average precision needs a keypoint-labelled image set, and this "
                "repository ships none. The repository's metric helper is `keypoint_pck` — percentage of correct keypoints, a "
                "joint counting as correct when the predicted joint of the same name lies within `PCK_FRACTION` (0.1) of the "
                "person box's longest side of its reference (a stated convention, not OKS) — with the mean pixel error; when "
                "reference joints are supplied (one mapping per person, matched to `poses` in order) the report carries one "
                "`keypoint_pck` entry per person with the verdict `sample-sanity`, scoring **all 17 predicted joints** "
                "regardless of `keypoint_threshold`. On the synthetic path those references are joints **you drew yourself** "
                "and the box came from you, so a high PCK proves only that the input contract, the affine crop, the forward "
                "pass and the coordinate mapping round-trip. On BYOD no reference exists, the verdict is `not-measurable`, "
                "and the report states what would make the task measurable. The report is written to "
                "`outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "references = None if reference_joints is None else [reference_joints] * result['n_persons']\n"
                "report = evaluation_report(result, references or None, sample_kind=sample_kind)\n"
                "report['findings'] = findings\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'metrics'}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"person {{metric['person']}}: keypoint_pck {{metric['value']:.3f}}  ({{metric['correct']}}/{{metric['total']}} within {{metric['radius_px']:.0f}} px; mean error {{metric['mean_error_px']:.1f}} px)\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference joints exist for this input, so keypoint_pck is not computed; inspect the skeleton overlay instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (per person: box, source, all 17 keypoints with scores, the "
                "kept subset, both thresholds), the stage-1 detections, the evaluation report, the input manifest, the sample "
                "identity, digest, drawn box and reference joints, the notebook's source (repository, revision, embedded module "
                "digest, generator), **both** model identifiers and immutable revisions, the model licences, and the runtime "
                "identity (Python, `torch`, `transformers`, device). The keypoints are also written as CSV with explicit `image`, "
                "`person`, `keypoint`, `x`, `y`, `score`, `kept` columns, and a skeleton overlay PNG draws each person's box, the "
                "kept joints and the COCO skeleton edges between kept joints (a supplement to, not a replacement for, the "
                "machine-readable files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for pose in result['poses']:\n"
                "    draw.rectangle(pose['box'], outline=(0, 160, 0), width=2)\n"
                "    kept = {{kp['name']: (kp['x'], kp['y']) for kp in pose['keypoints']}}\n"
                "    for a, b in SKELETON_EDGES:\n"
                "        na, nb = KEYPOINT_NAMES[a], KEYPOINT_NAMES[b]\n"
                "        if na in kept and nb in kept:\n"
                "            draw.line([kept[na], kept[nb]], fill=(255, 200, 0), width=3)\n"
                "    for x, y in kept.values():\n"
                "        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 30, 30))\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'stage1': stage1,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'drawn_box': drawn_box, 'reference_joints': reference_joints}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'detector_model_id': DETECTOR_MODEL_ID,\n"
                "    'detector_revision': DETECTOR_REVISION,\n"
                "    'detector_license': DETECTOR_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_keypoints.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'person', 'keypoint', 'x', 'y', 'score', 'kept'])\n"
                "    for index, pose in enumerate(result['poses']):\n"
                "        for kp in pose['all_keypoints']:\n"
                "            writer.writerow([image_name, index, kp['name'], f\"{{kp['x']:.2f}}\", f\"{{kp['y']:.2f}}\", f\"{{kp['score']:.6f}}\", kp['score'] >= keypoint_threshold])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The keypoints are where ViTPose's heatmaps peak inside a box it was told contains a person; the score is a heatmap "
        "maximum, not a calibrated probability, and both thresholds are request parameters you own (the defaults are the "
        "upstream README example's values, not tuned operating points). On the synthetic drawing the `keypoint_pck` value in "
        "the evaluation report compares joints to a figure you drew yourself, inside a box you supplied, and the verdict is "
        "`sample-sanity`, which proves only that the input contract, the crop, the forward pass and the coordinate mapping "
        "work; it says nothing about photographs, occlusion, crowds, unusual poses, children, clothing, or camera viewpoints, "
        "and a BYOD result is a single-image observation with the verdict `not-measurable`. **Stage 1 is a closed-set "
        "photograph detector**: it found no `person` on this cartoon at the default threshold, so on drawings, sketches or "
        "stylised imagery you must supply boxes yourself, and on photographs a missed or merged person silently loses a pose. "
        "**Stage 2 returns 17 joints for any box** — an empty box in the smoke run scored every joint below 0.04 — so a box "
        "without a person produces a low-scoring skeleton rather than nothing, and only the scores tell you. The pipeline "
        "provides no bottom-up association, no 3-D pose, no tracking, no hand or face keypoints, no OKS evaluation and no "
        "training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify both pinned models, validate the demonstrated request, execute the public pipeline path, "
        "and emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does "
        "**not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production "
        "fitness on an unseen domain.\n\n"
        "**Next experiments:** lower `detection_threshold` to 0.05 and count the low-confidence `person` proposals stage 1 now "
        "returns on the drawing (the smoke run found seven); pass a box that contains only background to `estimate` and read "
        "the scores; move an elbow in `REFERENCE_JOINTS`, redraw, and watch the PCK error for that joint; enable `USE_BYOD` "
        "with a photograph of a person, let stage 1 supply the box, then hand-annotate a few joints and pass them as reference "
        "joints to `evaluation_report` to see the verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/vitpose-keypoint-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream pose model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream person detector: https://huggingface.co/PekingU/rtdetr_r50vd\n"
        "- Upstream code: https://github.com/ViTAE-Transformer/ViTPose\n"
        "- ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation (Xu et al., 2022): https://arxiv.org/abs/2204.12484\n"
        "- DETRs Beat YOLOs on Real-time Object Detection (Zhao et al., 2023): https://arxiv.org/abs/2304.08069\n"
        "- Microsoft COCO: Common Objects in Context — keypoint annotations (Lin et al., 2014): https://arxiv.org/abs/1405.0312"
    ),
}
