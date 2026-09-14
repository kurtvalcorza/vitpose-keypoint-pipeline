"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; valid snapshots still reach them.
"""

import hashlib
import json

import pytest

from vitpose_keypoint_pipeline.pipeline import (
    DETECTOR_MODEL_ID,
    DETECTOR_REVISION,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    VitPoseKeypointPipeline,
)

_CONFIG = json.dumps({"model_type": "vitpose", "scale_factor": 4}).encode()


def _snapshot(root, model_id, revision, tamper=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_bytes(_CONFIG)
    digest = "0" * 64 if tamper else hashlib.sha256(_CONFIG).hexdigest()
    manifest = {
        "modelId": model_id,
        "revision": revision,
        "files": [{"path": "config.json", "bytes": len(_CONFIG), "sha256": digest}],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        VitPoseKeypointPipeline.from_pretrained(
            device="cpu", weights_dir=tmp_path / "pose", detector_dir=tmp_path / "det", allow_download=False
        )


def test_from_pretrained_refuses_tampered_pose_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path / "pose", MODEL_ID, MODEL_REVISION, tamper=True)
    _snapshot(tmp_path / "det", DETECTOR_MODEL_ID, DETECTOR_REVISION)
    with pytest.raises(ValueError, match="sha256"):
        VitPoseKeypointPipeline.from_pretrained(
            device="cpu", weights_dir=tmp_path / "pose", detector_dir=tmp_path / "det", allow_download=False
        )


def test_from_pretrained_refuses_tampered_detector_snapshot_before_model_imports(
    tmp_path, forbid_model_imports
):
    _snapshot(tmp_path / "pose", MODEL_ID, MODEL_REVISION)
    _snapshot(tmp_path / "det", DETECTOR_MODEL_ID, DETECTOR_REVISION, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        VitPoseKeypointPipeline.from_pretrained(
            device="cpu", weights_dir=tmp_path / "pose", detector_dir=tmp_path / "det", allow_download=False
        )


def test_from_pretrained_valid_snapshots_reach_model_import(tmp_path, forbid_model_imports):
    _snapshot(tmp_path / "pose", MODEL_ID, MODEL_REVISION)
    _snapshot(tmp_path / "det", DETECTOR_MODEL_ID, DETECTOR_REVISION)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        VitPoseKeypointPipeline.from_pretrained(
            device="cpu", weights_dir=tmp_path / "pose", detector_dir=tmp_path / "det", allow_download=False
        )
