"""Model-backed checks that run only where the pinned snapshots are staged (local pre-flight): evaluation with the
per-category breakdown, a one-epoch adaptation of the head and the last block on a dozen drawn figures, the artifact
round trip, the loader's scope check, the transactional guarantee and — where CUDA is visible — the same path on the
accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vitpose_keypoint_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    DETECTOR_WEIGHTS_DIR,
    HEAD_PARAMETERS,
    KEYPOINT_NAMES,
    PARAMETER_COUNT,
    WEIGHT_FILE,
    VitPoseKeypointPipeline,
)

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]
CX, CY = 160, 60


def _joints():
    base = {
        "Nose": (0, 0), "L_Eye": (6, -6), "R_Eye": (-6, -6), "L_Ear": (15, -3), "R_Ear": (-15, -3),
        "L_Shoulder": (30, 35), "R_Shoulder": (-30, 35), "L_Elbow": (48, 80), "R_Elbow": (-48, 80),
        "L_Wrist": (60, 125), "R_Wrist": (-60, 125), "L_Hip": (20, 110), "R_Hip": (-20, 110),
        "L_Knee": (25, 160), "R_Knee": (-25, 160), "L_Ankle": (28, 208), "R_Ankle": (-28, 208),
    }
    return {name: [CX + dx, CY + dy] for name, (dx, dy) in base.items()}


def _record(i, size=(320, 300)):
    image = Image.new("RGB", size, (225, 232, 240))
    draw = ImageDraw.Draw(image)
    joints = _joints()
    draw.ellipse([CX - 20, CY - 20, CX + 20, CY + 20], fill=COLOURS[i % 6])
    for a, b in (("L_Shoulder", "L_Wrist"), ("R_Shoulder", "R_Wrist"), ("L_Hip", "L_Ankle"), ("R_Hip", "R_Ankle"), ("L_Shoulder", "R_Hip")):
        draw.line([tuple(joints[a]), tuple(joints[b])], fill=(40, 60, 140), width=12)
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    return {"id": f"r{i:03d}", "image": image, "box": [CX - 70.0, CY - 25.0, CX + 70.0, CY + 220.0], "keypoints": joints, "category": COLOURS[i % 6]}


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file() or not (DETECTOR_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshots not staged", allow_module_level=True)

KW = {"weights_dir": DEFAULT_WEIGHTS_DIR, "detector_dir": DETECTOR_WEIGHTS_DIR}


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(16)]


@pytest.fixture(scope="module")
def pipe():
    return VitPoseKeypointPipeline.from_pretrained(device="cpu", **KW)


def _same(a, b):
    return all(np.allclose(a[k], b[k]) for k in KEYPOINT_NAMES)


def test_identity_and_frozen_evaluation_with_the_breakdown(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith("head.")) == HEAD_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    metrics = pipe.evaluate(records[:8])
    assert metrics["n"] == 8 and 0.0 <= metrics["pck"] <= 1.0 and 0.0 <= metrics["oks"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics["per_category"]) == set(COLOURS) and metrics["verdict"] == "measured-small-sample"
    joints = pipe.predict_keypoints(records[0])
    assert set(joints) == set(KEYPOINT_NAMES) and all(len(v) == 2 for v in joints.values())


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], epochs=1, trainable_blocks=1, batch_size=4)
    assert result["n_trainable"] > HEAD_PARAMETERS and result["n_total"] == PARAMETER_COUNT and result["trainable_blocks"] == 1
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"pck", "oks", "score", "n"} and result["preparation_seconds"] > 0.0
    assert all(n.startswith(("head.", "backbone.encoder.layer.11.", "backbone.layernorm")) for n in result["trainable_names"])
    assert not any(n.startswith(("backbone.embeddings", "backbone.encoder.layer.0.")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["adapter"]["trainable_blocks"] == 1 and manifest["metadata"] == {"note": "test"}
    reloaded = VitPoseKeypointPipeline.from_artifact(artifact, device="cpu", **KW)
    assert all(_same(pipe.predict_keypoints(r), reloaded.predict_keypoints(r)) for r in records[:4])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[:4])["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, trainable_blocks=0, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    assert result["n_trainable"] == HEAD_PARAMETERS and all(n.startswith("head.") for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = VitPoseKeypointPipeline.from_artifact(artifact, device="cpu", **KW)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["trainable_blocks"] == 0


def test_adapt_refuses_bad_hyperparameters(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:12], None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:12], None, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.adapt(records[:12], None, epochs=1, trainable_blocks=13)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records[:12], None, epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="8..5000"):
        pipe.adapt(records[:4], None, epochs=1)
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, trainable_blocks=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        VitPoseKeypointPipeline.from_artifact(fewer, device="cpu", **KW)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["head.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        VitPoseKeypointPipeline.from_artifact(extra, device="cpu", **KW)
    other = tmp_path / "other_blocks"
    shutil.copytree(artifact, other)
    (other / "manifest.json").write_text(json.dumps({**manifest, "adapter": {**manifest["adapter"], "trainable_blocks": 2}}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        VitPoseKeypointPipeline.from_artifact(other, device="cpu", **KW)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, trainable_blocks=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # parameters and the frozen BatchNorm statistics
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_evaluate_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    cuda = VitPoseKeypointPipeline.from_pretrained(device="cuda:0", **KW)
    assert cuda.device == "cuda:0"
    metrics = cuda.evaluate(records[:8])
    assert 0.0 <= metrics["pck"] <= 1.0
    result = cuda.adapt(records[:12], records[12:], epochs=1, trainable_blocks=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = VitPoseKeypointPipeline.from_artifact(artifact, device="cuda:0", **KW)
    assert all(_same(cuda.predict_keypoints(r), reloaded.predict_keypoints(r)) for r in records[:4])
