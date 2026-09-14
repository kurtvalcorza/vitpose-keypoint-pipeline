"""Two-stage human pose estimation with the pinned ``usyd-community/vitpose-base`` checkpoint (ViTPose)
and the pinned ``PekingU/rtdetr_r50vd`` person detector (RT-DETR).

Both models load only from digest-verified local snapshots (``weights/<key>/``) or, when explicitly
allowed, from the Hugging Face Hub at their pinned revisions — always with ``trust_remote_code=False``:
the architectures come from the pinned ``transformers`` release, the weights are SafeTensors, and no
model-repository code is executed. ViTPose is top-down: it needs a person box first, either from the
carried detector stage or from the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "usyd-community/vitpose-base"
MODEL_REVISION = "95be2991424e646950d656bb7fc15ec9be119700"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "vitpose-base"
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = _WEIGHTS_ROOT / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Stage 1 (person detection) is a SECOND pinned snapshot with its own manifest: the RT-DETR R50-VD
# checkpoint the sibling rtdetr-detection-pipeline wraps. Only its `person` class is used here.
DETECTOR_MODEL_ID = "PekingU/rtdetr_r50vd"
DETECTOR_REVISION = "df939e661d8c52e80608d1ec566561aabd25a4e7"
DETECTOR_LICENSE = "apache-2.0"
DETECTOR_KEY = "rtdetr-r50vd"
DETECTOR_WEIGHTS_DIR = _WEIGHTS_ROOT / DETECTOR_KEY
DETECTOR_PERSON_LABEL = "person"
# Detection threshold on the detector's per-class sigmoid for `person`: the value the pinned ViTPose
# README's two-stage example passes (threshold=0.3). Uncalibrated; the deployment owns tuning it.
DETECTION_THRESHOLD = 0.3
# Keypoint threshold on ViTPose's per-keypoint heatmap maximum: the README example's value (0.3).
# Keypoints below it are dropped from `keypoints` but still listed under `all_keypoints`.
KEYPOINT_THRESHOLD = 0.3
# The 17 COCO keypoints in the checkpoint's id2label order (config.json) and its skeleton edges.
KEYPOINT_NAMES = (
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
)
SKELETON_EDGES = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)
# Input ceilings. Each person crop is affine-warped to 192x256 (preprocessor_config.json) so the
# pose stage's cost is per person, not per pixel; the detector resizes the whole image to 640x640.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PERSONS = 50
# PCK normaliser for the sanity metric: a keypoint is "correct" when within this fraction of the
# person box's longest side of its reference (a stated convention, not the COCO OKS metric).
PCK_FRACTION = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the ViTPose snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _verify_manifest(root, MODEL_ID, MODEL_REVISION)


def verify_detector_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the RT-DETR person-detector snapshot against its DIMER manifest."""
    root = Path(path) if path is not None else DETECTOR_WEIGHTS_DIR
    return _verify_manifest(root, DETECTOR_MODEL_ID, DETECTOR_REVISION)


def _hub_download(relative_path: str, root: Path, model_id: str, revision: str) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(model_id, relative_path, revision=revision, local_dir=str(root))


def _stage_missing(
    root: Path,
    model_id: str,
    revision: str,
    allow_download: bool,
    downloader: Callable[[str, Path], None] | None,
) -> list[str]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {model_id}@{revision}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {revision}"
        )
    fetch = downloader or (lambda rel, dst: _hub_download(rel, dst, model_id, revision))
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch ViTPose manifest entries that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _stage_missing(root, MODEL_ID, MODEL_REVISION, allow_download, downloader)


def stage_missing_detector_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Same as `stage_missing_files` for the RT-DETR person-detector snapshot at DETECTOR_REVISION."""
    root = Path(path) if path is not None else DETECTOR_WEIGHTS_DIR
    return _stage_missing(root, DETECTOR_MODEL_ID, DETECTOR_REVISION, allow_download, downloader)


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def _check_boxes(boxes: Any, width: int, height: int) -> list[list[float]]:
    if isinstance(boxes, str) or not isinstance(boxes, Sequence):
        raise TypeError("person_boxes must be a sequence of [x0, y0, x1, y1] boxes")
    if not 1 <= len(boxes) <= MAX_PERSONS:
        raise ValueError(f"person_boxes has {len(boxes)} entries; expected 1..MAX_PERSONS={MAX_PERSONS}")
    checked: list[list[float]] = []
    for index, box in enumerate(boxes):
        if isinstance(box, str) or not isinstance(box, Sequence) or len(box) != 4:
            raise ValueError(f"person_boxes[{index}] must be [x0, y0, x1, y1]")
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"person_boxes[{index}] must hold numbers") from exc
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"person_boxes[{index}] {list(box)} is not a non-empty box inside the {width}x{height} image"
            )
        checked.append([x0, y0, x1, y1])
    return checked


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one image as PIL.Image.Image (any mode, converted to RGB) and optionally the person boxes as "
        "pixel xyxy; without boxes the carried RT-DETR stage detects `person` first"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "person_boxes": [1, MAX_PERSONS],
    "detection_threshold": [0.0, 1.0],
    "keypoint_threshold": [0.0, 1.0],
    "keypoints": list(KEYPOINT_NAMES),
    "preprocessing": (
        "stage 1 (when no boxes are given): the whole image resized to 640x640 for RT-DETR, `person` boxes "
        "kept above detection_threshold; stage 2: each person box affine-warped to a 192x256 crop "
        "(ImageNet mean/std) for ViTPose, heatmap maxima mapped back to input pixels"
    ),
    "output": "per person: the box, its source, 17 named keypoints with pixel coordinates and heatmap scores",
}


def _check_inputs(
    image: Any, person_boxes: Any, detection_threshold: Any, keypoint_threshold: Any
) -> tuple[Image.Image, list[list[float]] | None, float, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``estimate`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    boxes = None if person_boxes is None else _check_boxes(person_boxes, rgb.width, rgb.height)
    return (
        rgb,
        boxes,
        _check_threshold("detection_threshold", detection_threshold),
        _check_threshold("keypoint_threshold", keypoint_threshold),
    )


def validate_inputs(
    image: Image.Image,
    *,
    person_boxes: Sequence[Sequence[float]] | None = None,
    detection_threshold: float = DETECTION_THRESHOLD,
    keypoint_threshold: float = KEYPOINT_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``estimate`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, boxes, det_t, kp_t = _check_inputs(image, person_boxes, detection_threshold, keypoint_threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (estimate takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "person_boxes": boxes,
        "box_source": "caller" if boxes is not None else "detector",
        "detection_threshold": det_t,
        "keypoint_threshold": kp_t,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "detector_model_id": DETECTOR_MODEL_ID,
        "detector_revision": DETECTOR_REVISION,
    }


def keypoint_pck(
    predicted: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]],
    box: Sequence[float],
    *,
    fraction: float = PCK_FRACTION,
) -> dict[str, Any]:
    """Percentage of correct keypoints: a named reference joint counts as correct when the predicted
    joint of the same name lies within ``fraction`` of the box's longest side; missing joints are misses."""
    if not reference:
        raise ValueError("reference must contain at least one keypoint")
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box must be a non-empty [x0, y0, x1, y1]")
    radius = fraction * max(box[2] - box[0], box[3] - box[1])
    errors: dict[str, float | None] = {}
    correct = 0
    for name, (rx, ry) in reference.items():
        if name in predicted:
            px, py = predicted[name][0], predicted[name][1]
            err = math.hypot(px - rx, py - ry)
            errors[name] = err
            correct += err <= radius
        else:
            errors[name] = None
    measured = [e for e in errors.values() if e is not None]
    return {
        "value": correct / len(reference),
        "correct": correct,
        "total": len(reference),
        "radius_px": radius,
        "fraction": fraction,
        "mean_error_px": sum(measured) / len(measured) if measured else None,
        "errors_px": errors,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_keypoints: Sequence[Mapping[str, Sequence[float]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference_keypoints`` (one name -> (x, y) mapping per person, in the order of
    ``result['poses']``) the report carries one ``keypoint_pck`` entry per person as sample-sanity
    geometry evidence; without them the verdict is ``not-measurable`` and the report says what
    labelled data would make the task measurable.
    """
    poses = list(result["poses"])
    base = {
        "task": "two-stage human pose estimation: person boxes -> 17 COCO keypoints per person",
        "score_semantics": (
            "each keypoint score is the maximum of its ViTPose heatmap after the crop's affine warp — a "
            "ranking signal per joint, not a calibrated probability that the joint is where it says; the "
            "person score, when the detector stage ran, is RT-DETR's per-class sigmoid for `person`"
        ),
        "box_source": result.get("box_source"),
        "detection_threshold": result.get("detection_threshold", DETECTION_THRESHOLD),
        "keypoint_threshold": result.get("keypoint_threshold", KEYPOINT_THRESHOLD),
        "sample_kind": sample_kind,
        "n_persons": len(poses),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "detector_model_id": DETECTOR_MODEL_ID,
        "detector_revision": DETECTOR_REVISION,
    }
    if not reference_keypoints:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference keypoints were supplied for the evaluated image",
            "needs": (
                "images with COCO-style keypoint annotations from the deployment domain scored with "
                "OKS-based average precision (the COCO keypoint metric) or PCK at a stated normaliser; "
                "no such labelled set ships with this repository"
            ),
        }
    if len(reference_keypoints) != len(poses):
        raise ValueError(f"reference_keypoints has {len(reference_keypoints)} entries for {len(poses)} poses")
    metrics = []
    for index, (pose, reference) in enumerate(zip(poses, reference_keypoints, strict=True)):
        unknown = set(reference) - set(KEYPOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown keypoint names in reference {index}: {sorted(unknown)}")
        predicted = {kp["name"]: (kp["x"], kp["y"]) for kp in pose["all_keypoints"]}
        pck = keypoint_pck(predicted, reference, pose["box"])
        metrics.append(
            {
                "id": "keypoint_pck",
                "person": index,
                "value": pck["value"],
                "correct": pck["correct"],
                "total": pck["total"],
                "radius_px": pck["radius_px"],
                "mean_error_px": pck["mean_error_px"],
                "estimation": "one drawn person on a single image, no dispersion estimate",
            }
        )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} person(s) on one tutorial image whose joints you drew yourself; geometry "
            "sanity evidence, not a pose benchmark"
        ),
        "needs": (
            "a keypoint-labelled image set from the deployment domain (cameras, poses, occlusion, clothing) "
            "for any OKS-AP or PCK claim"
        ),
    }


@dataclass
class VitPoseKeypointPipeline:
    """``_detect(image, threshold)`` -> [{"box", "score"}] of persons; ``_pose(image, boxes)`` ->
    per box a list of 17 {"name", "x", "y", "score"} in KEYPOINT_NAMES order."""

    _detect: Callable[[Image.Image, float], list[dict[str, Any]]]
    _pose: Callable[[Image.Image, list[list[float]]], list[list[dict[str, Any]]]]
    device: str = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        detector_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> VitPoseKeypointPipeline:
        roots = {
            "pose": (
                Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR,
                MODEL_ID,
                MODEL_REVISION,
            ),
            "detector": (
                Path(detector_dir) if detector_dir is not None else DETECTOR_WEIGHTS_DIR,
                DETECTOR_MODEL_ID,
                DETECTOR_REVISION,
            ),
        }
        sources: dict[str, tuple[str, dict[str, Any]]] = {}
        for name, (root, model_id, revision) in roots.items():
            if (root / MANIFEST_NAME).is_file():
                _stage_missing(root, model_id, revision, allow_download, None)
                _verify_manifest(root, model_id, revision)
                sources[name] = (str(root), {"local_files_only": True})
            elif allow_download:
                sources[name] = (model_id, {"revision": revision})
            else:
                raise FileNotFoundError(
                    f"no verified {name} snapshot at {root} and allow_download=False; "
                    f"stage {model_id}@{revision} under {root}"
                )
        # Refuse invalid snapshots before importing model libraries.
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        det_src, det_kwargs = sources["detector"]
        det_processor = AutoImageProcessor.from_pretrained(det_src, trust_remote_code=False, **det_kwargs)
        detector = (
            RTDetrForObjectDetection.from_pretrained(
                det_src, trust_remote_code=False, use_pretrained_backbone=False, **det_kwargs
            )
            .to(resolved_device)
            .eval()
        )
        det_id2label = {int(k): v for k, v in detector.config.id2label.items()}
        pose_src, pose_kwargs = sources["pose"]
        pose_processor = AutoImageProcessor.from_pretrained(pose_src, trust_remote_code=False, **pose_kwargs)
        pose_model = (
            VitPoseForPoseEstimation.from_pretrained(pose_src, trust_remote_code=False, **pose_kwargs)
            .to(resolved_device)
            .eval()
        )
        pose_id2label = {int(k): v for k, v in pose_model.config.id2label.items()}
        if tuple(pose_id2label[i] for i in range(len(pose_id2label))) != KEYPOINT_NAMES:
            raise RuntimeError("snapshot id2label does not match KEYPOINT_NAMES")

        def detect(image: Image.Image, threshold: float) -> list[dict[str, Any]]:
            inputs = det_processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = detector(**inputs)
            result = det_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=[image.size[::-1]]
            )[0]
            return [
                {"box": [float(v) for v in box.tolist()], "score": float(score)}
                for box, label, score in zip(result["boxes"], result["labels"], result["scores"], strict=True)
                if det_id2label[int(label)] == DETECTOR_PERSON_LABEL
            ]

        def pose(image: Image.Image, boxes: list[list[float]]) -> list[list[dict[str, Any]]]:
            # The ViTPose processor takes boxes as xywh (the README converts xyxy -> xywh the same way).
            xywh = np.array([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], dtype=np.float32)
            inputs = pose_processor(image, boxes=[xywh], return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = pose_model(**inputs)
            # threshold=0.0 keeps every joint; the pipeline applies keypoint_threshold itself so both
            # the kept and the dropped joints can be reported.
            persons = pose_processor.post_process_pose_estimation(outputs, boxes=[xywh], threshold=0.0)[0]
            out: list[list[dict[str, Any]]] = []
            for person in persons:
                joints = [
                    {
                        "name": pose_id2label[int(label)],
                        "x": float(kp[0]),
                        "y": float(kp[1]),
                        "score": float(score),
                    }
                    for kp, label, score in zip(
                        person["keypoints"], person["labels"], person["scores"], strict=True
                    )
                ]
                out.append(joints)
            return out

        return cls(detect, pose, resolved_device)

    def detect_people(self, image: Image.Image, *, threshold: float = DETECTION_THRESHOLD) -> dict[str, Any]:
        """Stage 1 alone: `person` boxes from the carried RT-DETR, sorted by score."""
        rgb, _boxes, det_t, _kp_t = _check_inputs(image, None, threshold, KEYPOINT_THRESHOLD)
        found = sorted(self._detect(rgb, det_t), key=lambda d: -d["score"])
        for det in found:
            if set(det) != {"box", "score"} or len(det["box"]) != 4:
                raise RuntimeError(f"detector returned a malformed detection: {det!r}")
        return {
            "persons": found[:MAX_PERSONS],
            "n_persons": min(len(found), MAX_PERSONS),
            "threshold": det_t,
            "width": rgb.width,
            "height": rgb.height,
            "detector_model_id": DETECTOR_MODEL_ID,
            "detector_revision": DETECTOR_REVISION,
        }

    def estimate(
        self,
        image: Image.Image,
        *,
        person_boxes: Sequence[Sequence[float]] | None = None,
        detection_threshold: float = DETECTION_THRESHOLD,
        keypoint_threshold: float = KEYPOINT_THRESHOLD,
    ) -> dict[str, Any]:
        """Keypoints for every person: boxes from the caller, or from stage 1 when none are given."""
        rgb, boxes, det_t, kp_t = _check_inputs(image, person_boxes, detection_threshold, keypoint_threshold)
        if boxes is None:
            detected = self.detect_people(rgb, threshold=det_t)["persons"]
            boxes = [d["box"] for d in detected]
            person_scores: list[float | None] = [d["score"] for d in detected]
            source = "detector"
        else:
            person_scores = [None] * len(boxes)
            source = "caller"
        poses: list[dict[str, Any]] = []
        if boxes:
            per_person = self._pose(rgb, boxes)
            if len(per_person) != len(boxes):
                raise RuntimeError(f"pose stage returned {len(per_person)} results for {len(boxes)} boxes")
            for box, score, joints in zip(boxes, person_scores, per_person, strict=True):
                if [j["name"] for j in joints] != list(KEYPOINT_NAMES):
                    raise RuntimeError("pose stage returned keypoints out of KEYPOINT_NAMES order")
                kept = [j for j in joints if j["score"] >= kp_t]
                poses.append(
                    {
                        "box": [float(v) for v in box],
                        "person_score": score,
                        "keypoints": kept,
                        "all_keypoints": joints,
                        "n_keypoints": len(kept),
                        "mean_keypoint_score": sum(j["score"] for j in joints) / len(joints),
                    }
                )
        return {
            "poses": poses,
            "n_persons": len(poses),
            "box_source": source,
            "detection_threshold": det_t,
            "keypoint_threshold": kp_t,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "detector_model_id": DETECTOR_MODEL_ID,
            "detector_revision": DETECTOR_REVISION,
        }
