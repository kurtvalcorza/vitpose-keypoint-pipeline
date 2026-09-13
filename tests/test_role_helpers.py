"""Offline tests for the public validation, metric and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from vitpose_keypoint_pipeline import (
    DETECTION_THRESHOLD,
    DETECTOR_MODEL_ID,
    DETECTOR_REVISION,
    INPUT_SCHEMA,
    KEYPOINT_NAMES,
    KEYPOINT_THRESHOLD,
    MAX_IMAGE_SIDE,
    MAX_PERSONS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    PCK_FRACTION,
    evaluation_report,
    keypoint_pck,
    validate_inputs,
)

BOX = [180.0, 80.0, 460.0, 560.0]
REFERENCE = {name: (320.0 + 5.0 * i, 100.0 + 25.0 * i) for i, name in enumerate(KEYPOINT_NAMES)}


def _image(width: int = 640, height: int = 640) -> Image.Image:
    return Image.new("RGB", (width, height), (225, 232, 240))


def _pose(offset: float = 0.0, missing_score: float = 0.9) -> dict:
    joints = [
        {"name": name, "x": x + offset, "y": y, "score": missing_score if name == "L_Ear" else 0.9}
        for name, (x, y) in REFERENCE.items()
    ]
    return {
        "box": BOX,
        "person_score": None,
        "keypoints": [j for j in joints if j["score"] >= KEYPOINT_THRESHOLD],
        "all_keypoints": joints,
        "n_keypoints": sum(j["score"] >= KEYPOINT_THRESHOLD for j in joints),
        "mean_keypoint_score": 0.9,
    }


def _result(poses: list[dict], source: str = "caller") -> dict:
    return {
        "poses": poses,
        "n_persons": len(poses),
        "box_source": source,
        "detection_threshold": DETECTION_THRESHOLD,
        "keypoint_threshold": KEYPOINT_THRESHOLD,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), person_boxes=[BOX], names=["person.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["person_boxes"] == [1, MAX_PERSONS]
    assert manifest["schema"]["keypoints"] == list(KEYPOINT_NAMES)
    assert manifest["inputs"] == [{"id": "person.png", "mode": "RGB", "size": [640, 640]}]
    assert manifest["person_boxes"] == [BOX] and manifest["box_source"] == "caller"
    assert (manifest["detection_threshold"], manifest["keypoint_threshold"]) == (
        DETECTION_THRESHOLD,
        KEYPOINT_THRESHOLD,
    )
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (manifest["detector_model_id"], manifest["detector_revision"]) == (
        DETECTOR_MODEL_ID,
        DETECTOR_REVISION,
    )


def test_validate_inputs_detector_path_and_explicit_thresholds() -> None:
    manifest = validate_inputs(_image(), detection_threshold=0.2, keypoint_threshold=0.5)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["person_boxes"] is None and manifest["box_source"] == "detector"
    assert (manifest["detection_threshold"], manifest["keypoint_threshold"]) == (0.2, 0.5)


def test_validate_inputs_rejects_like_estimate() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="not a non-empty box"):
        validate_inputs(_image(), person_boxes=[[0, 0, 700, 10]])
    with pytest.raises(ValueError, match="detection_threshold"):
        validate_inputs(_image(), detection_threshold=-0.1)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_keypoint_pck_counts_joints_within_radius() -> None:
    predicted = {name: (x + 10.0, y) for name, (x, y) in REFERENCE.items()}
    pck = keypoint_pck(predicted, REFERENCE, BOX)
    assert pck["value"] == 1.0 and pck["correct"] == 17 and pck["total"] == 17
    assert pck["radius_px"] == pytest.approx(PCK_FRACTION * 480) and pck["mean_error_px"] == pytest.approx(
        10.0
    )
    far = {name: (x + 100.0, y) for name, (x, y) in REFERENCE.items()}
    assert keypoint_pck(far, REFERENCE, BOX)["value"] == 0.0
    partial = {name: xy for name, xy in REFERENCE.items() if name != "Nose"}
    pck = keypoint_pck(partial, REFERENCE, BOX)
    assert pck["correct"] == 16 and pck["errors_px"]["Nose"] is None
    with pytest.raises(ValueError, match="at least one keypoint"):
        keypoint_pck({}, {}, BOX)
    with pytest.raises(ValueError, match="non-empty"):
        keypoint_pck(predicted, REFERENCE, [0, 0, 0, 0])


def test_evaluation_report_not_measurable_without_references() -> None:
    report = evaluation_report(_result([_pose()]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_persons"] == 1 and report["box_source"] == "caller"
    assert "OKS" in report["needs"]
    assert report["baselines"] == []
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (report["detector_model_id"], report["detector_revision"]) == (
        DETECTOR_MODEL_ID,
        DETECTOR_REVISION,
    )
    assert "not a calibrated probability" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_reference_keypoints() -> None:
    report = evaluation_report(
        _result([_pose(offset=12.0, missing_score=0.1)]), [REFERENCE], sample_kind="synthetic"
    )
    assert report["verdict"] == "sample-sanity" and report["sample_kind"] == "synthetic"
    metric = report["metrics"][0]
    assert metric["id"] == "keypoint_pck" and metric["person"] == 0
    assert (
        metric["value"] == 1.0 and metric["correct"] == 17
    )  # low-score joints still count via all_keypoints
    assert metric["mean_error_px"] == pytest.approx(12.0)
    subset = {"Nose": REFERENCE["Nose"], "L_Ankle": (0.0, 0.0)}
    report = evaluation_report(_result([_pose()]), [subset])
    assert (report["metrics"][0]["correct"], report["metrics"][0]["total"]) == (1, 2)


def test_evaluation_report_rejects_mismatches_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="reference_keypoints has"):
        evaluation_report(_result([_pose()]), [REFERENCE, REFERENCE])
    with pytest.raises(ValueError, match="unknown keypoint names"):
        evaluation_report(_result([_pose()]), [{"Tail": (0.0, 0.0)}])
    empty = evaluation_report(_result([], source="detector"))
    assert empty["verdict"] == "not-measurable" and empty["n_persons"] == 0
