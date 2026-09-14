import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vitpose_keypoint_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DETECTION_THRESHOLD,
    DETECTOR_KEY,
    DETECTOR_MODEL_ID,
    DETECTOR_REVISION,
    DETECTOR_WEIGHTS_DIR,
    KEYPOINT_NAMES,
    KEYPOINT_THRESHOLD,
    MAX_IMAGE_SIDE,
    MAX_PERSONS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    SKELETON_EDGES,
    VitPoseKeypointPipeline,
    stage_missing_detector_files,
    stage_missing_files,
    verify_detector_snapshot,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION) and HEX40.match(DETECTOR_REVISION)
    assert MODEL_ID == "usyd-community/vitpose-base" and DETECTOR_MODEL_ID == "PekingU/rtdetr_r50vd"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert DETECTOR_WEIGHTS_DIR == REPO / "weights" / DETECTOR_KEY
    assert 0 < DETECTION_THRESHOLD < 1 and 0 < KEYPOINT_THRESHOLD < 1 and MAX_PERSONS == 50
    assert len(KEYPOINT_NAMES) == 17 and KEYPOINT_NAMES[0] == "Nose" and KEYPOINT_NAMES[-1] == "R_Ankle"
    assert len(SKELETON_EDGES) == 19 and all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON_EDGES)
    for key, model_id, revision in (
        (MODEL_KEY, MODEL_ID, MODEL_REVISION),
        (DETECTOR_KEY, DETECTOR_MODEL_ID, DETECTOR_REVISION),
    ):
        manifest = REPO / "weights" / key / "dimer-base-manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            assert (data["modelId"], data["revision"]) == (model_id, revision)


def test_keypoint_names_and_edges_match_snapshot_config():
    config = REPO / "weights" / MODEL_KEY / "config.json"
    if not config.is_file():
        pytest.skip("snapshot config.json not present")
    data = json.loads(config.read_text(encoding="utf-8"))
    assert tuple(data["id2label"][str(i)] for i in range(17)) == KEYPOINT_NAMES
    assert tuple(tuple(edge) for edge in data["edges"]) == SKELETON_EDGES


def _write_snapshot(
    root: Path, content: bytes, model_id: str, revision: str, sha: str | None = None, size: int | None = None
):
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": model_id,
        "revision": revision,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest_for_both_snapshots(tmp_path):
    pose, det = tmp_path / "pose", tmp_path / "det"
    pose.mkdir()
    det.mkdir()
    _write_snapshot(pose, b'{"model_type": "vitpose"}', MODEL_ID, MODEL_REVISION)
    _write_snapshot(det, b'{"model_type": "rt_detr"}', DETECTOR_MODEL_ID, DETECTOR_REVISION)
    assert verify_snapshot(pose)["files"] == 1 and verify_snapshot(pose)["model_id"] == MODEL_ID
    assert verify_detector_snapshot(det)["model_id"] == DETECTOR_MODEL_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(det)  # the detector manifest is not the pose manifest


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "vitpose"}'
    _write_snapshot(tmp_path, content, MODEL_ID, MODEL_REVISION, sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    content = b'{"model_type": "vitpose"}'
    _write_snapshot(tmp_path, content, MODEL_ID, MODEL_REVISION, size=len(content) + 1)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, content, MODEL_ID, MODEL_REVISION)
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, content, MODEL_ID, "0" * 40)
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        verify_snapshot(tmp_path / "nowhere")


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    content = b'{"model_type": "vitpose"}'
    _write_snapshot(tmp_path, content, MODEL_ID, MODEL_REVISION)
    (tmp_path / "config.json").unlink()
    fetched: list = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(content)

    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == ["config.json"]
    assert fetched == ["config.json"]
    assert verify_snapshot(tmp_path)["files"] == 1
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []


def test_stage_missing_detector_files_uses_detector_identity(tmp_path):
    content = b'{"model_type": "rt_detr"}'
    _write_snapshot(tmp_path, content, DETECTOR_MODEL_ID, DETECTOR_REVISION)
    (tmp_path / "config.json").unlink()
    fetched: list = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(content)

    assert stage_missing_detector_files(tmp_path, allow_download=True, downloader=downloader) == [
        "config.json"
    ]
    assert verify_detector_snapshot(tmp_path)["revision"] == DETECTOR_REVISION
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path)  # a detector manifest is refused by the pose stager


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    _write_snapshot(tmp_path, b"x", "someone/else", MODEL_REVISION)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True)


BOX = [180.0, 80.0, 460.0, 560.0]


def _joints(scale: float = 1.0, low: set[str] | None = None) -> list[dict]:
    low = low or set()
    return [
        {
            "name": name,
            "x": 320.0 + 5.0 * i * scale,
            "y": 100.0 + 25.0 * i,
            "score": 0.1 if name in low else 0.9,
        }
        for i, name in enumerate(KEYPOINT_NAMES)
    ]


def _fake_pipeline(persons: list | None = None, calls: list | None = None) -> VitPoseKeypointPipeline:
    persons = [] if persons is None else persons

    def detect(image: Image.Image, threshold: float) -> list[dict]:
        if calls is not None:
            calls.append(("detect", image.mode, threshold))
        return [p for p in persons if p["score"] >= threshold]

    def pose(image: Image.Image, boxes: list[list[float]]) -> list[list[dict]]:
        if calls is not None:
            calls.append(("pose", image.mode, [list(b) for b in boxes]))
        return [_joints(low={"L_Ear"} if i else set()) for i, _ in enumerate(boxes)]

    return VitPoseKeypointPipeline(detect, pose, "cpu")


def test_detect_people_filters_and_sorts():
    pipe = _fake_pipeline(
        [
            {"box": [1, 1, 5, 9], "score": 0.4},
            {"box": [2, 2, 6, 8], "score": 0.9},
            {"box": [0, 0, 1, 1], "score": 0.2},
        ]
    )
    result = pipe.detect_people(Image.new("L", (64, 64)))
    assert [p["score"] for p in result["persons"]] == [0.9, 0.4] and result["n_persons"] == 2
    assert result["threshold"] == DETECTION_THRESHOLD and (result["width"], result["height"]) == (64, 64)
    assert result["detector_model_id"] == DETECTOR_MODEL_ID
    assert pipe.detect_people(Image.new("RGB", (64, 64)), threshold=0.1)["n_persons"] == 3


def test_estimate_with_caller_boxes():
    calls: list = []
    pipe = _fake_pipeline(calls=calls)
    result = pipe.estimate(Image.new("L", (640, 640)), person_boxes=[BOX])
    assert result["box_source"] == "caller" and result["n_persons"] == 1
    pose = result["poses"][0]
    assert pose["box"] == BOX and pose["person_score"] is None
    assert [k["name"] for k in pose["all_keypoints"]] == list(KEYPOINT_NAMES)
    assert pose["n_keypoints"] == 17 and pose["mean_keypoint_score"] == pytest.approx(0.9)
    assert (result["keypoint_threshold"], result["detection_threshold"]) == (
        KEYPOINT_THRESHOLD,
        DETECTION_THRESHOLD,
    )
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert calls == [("pose", "RGB", [BOX])]  # no detector call when boxes are given


def test_estimate_runs_detector_when_no_boxes_and_applies_keypoint_threshold():
    calls: list = []
    pipe = _fake_pipeline(
        [{"box": [10.0, 10.0, 50.0, 90.0], "score": 0.8}, {"box": [60.0, 10.0, 100.0, 90.0], "score": 0.5}],
        calls,
    )
    result = pipe.estimate(Image.new("RGB", (120, 100)))
    assert result["box_source"] == "detector" and result["n_persons"] == 2
    assert [p["person_score"] for p in result["poses"]] == [0.8, 0.5]
    assert (
        result["poses"][1]["n_keypoints"] == 16
    )  # L_Ear at 0.1 dropped from `keypoints`, kept in `all_keypoints`
    assert len(result["poses"][1]["all_keypoints"]) == 17
    assert calls[0][0] == "detect" and calls[1][0] == "pose"
    empty = _fake_pipeline([]).estimate(Image.new("RGB", (120, 100)))
    assert empty["n_persons"] == 0 and empty["poses"] == [] and empty["box_source"] == "detector"


def test_estimate_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.estimate(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.estimate(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.estimate(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(TypeError, match="person_boxes"):
        pipe.estimate(Image.new("RGB", (64, 64)), person_boxes="box")
    with pytest.raises(ValueError, match="MAX_PERSONS"):
        pipe.estimate(Image.new("RGB", (64, 64)), person_boxes=[[0, 0, 1, 1]] * (MAX_PERSONS + 1))
    with pytest.raises(ValueError, match="MAX_PERSONS"):
        pipe.estimate(Image.new("RGB", (64, 64)), person_boxes=[])
    with pytest.raises(ValueError, match=r"person_boxes\[0\] must be"):
        pipe.estimate(Image.new("RGB", (64, 64)), person_boxes=[[0, 0, 1]])
    with pytest.raises(ValueError, match="not a non-empty box"):
        pipe.estimate(Image.new("RGB", (64, 64)), person_boxes=[[10, 10, 10, 20]])
    with pytest.raises(ValueError, match="not a non-empty box"):
        pipe.estimate(Image.new("RGB", (64, 64)), person_boxes=[[0, 0, 65, 10]])
    with pytest.raises(ValueError, match="detection_threshold"):
        pipe.estimate(Image.new("RGB", (64, 64)), detection_threshold=1.5)
    with pytest.raises(ValueError, match="keypoint_threshold"):
        pipe.estimate(Image.new("RGB", (64, 64)), keypoint_threshold=True)


def test_estimate_rejects_malformed_stage_outputs():
    bad_det = VitPoseKeypointPipeline(lambda *_: [{"box": [0, 0, 1], "score": 0.9}], lambda *_: [], "cpu")
    with pytest.raises(RuntimeError, match="malformed"):
        bad_det.detect_people(Image.new("RGB", (64, 64)))
    short = VitPoseKeypointPipeline(lambda *_: [], lambda *_: [], "cpu")
    with pytest.raises(RuntimeError, match="pose stage returned"):
        short.estimate(Image.new("RGB", (64, 64)), person_boxes=[[0, 0, 10, 10]])
    shuffled = VitPoseKeypointPipeline(lambda *_: [], lambda *_: [list(reversed(_joints()))], "cpu")
    with pytest.raises(RuntimeError, match="KEYPOINT_NAMES order"):
        shuffled.estimate(Image.new("RGB", (64, 64)), person_boxes=[[0, 0, 10, 10]])
