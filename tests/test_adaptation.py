"""Offline checks of the adaptation contract: the keypoint record contract and its refusals, the stated degradation,
the pinned-corpus refusals and the draw, splitting, the BYOD loader, the metrics and baselines, `evaluate` through the
injected runner, the artifact-manifest checks, and the model-free refusals of `adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vitpose_keypoint_pipeline import (
    ARTIFACT_FORMAT,
    KEYPOINT_NAMES,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_IMAGES,
    SAMPLE_SPLIT,
    VitPoseKeypointPipeline,
    box_centre_baseline,
    build_sample_dataset,
    category_for,
    check_split_disjoint,
    dataset_digest,
    degrade,
    fetch_corpus,
    image_digest,
    load_byod_dataset,
    mean_pose_baseline,
    pose_metrics,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from vitpose_keypoint_pipeline import pipeline as pl
from vitpose_keypoint_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]
CX, CY = 160, 60


def _joints(cx=CX, cy=CY, scale=1.0):
    base = {
        "Nose": (0, 0), "L_Eye": (6, -6), "R_Eye": (-6, -6), "L_Ear": (15, -3), "R_Ear": (-15, -3),
        "L_Shoulder": (30, 35), "R_Shoulder": (-30, 35), "L_Elbow": (48, 80), "R_Elbow": (-48, 80),
        "L_Wrist": (60, 125), "R_Wrist": (-60, 125), "L_Hip": (20, 110), "R_Hip": (-20, 110),
        "L_Knee": (25, 160), "R_Knee": (-25, 160), "L_Ankle": (28, 208), "R_Ankle": (-28, 208),
    }
    return {name: [cx + dx * scale, cy + dy * scale] for name, (dx, dy) in base.items()}


def _scene(i, size=(320, 300)):
    image = Image.new("RGB", size, (225, 232, 240))
    draw = ImageDraw.Draw(image)
    joints = _joints()
    draw.ellipse([CX - 20, CY - 20, CX + 20, CY + 20], fill=COLOURS[i % 6])
    for a, b in (("L_Shoulder", "L_Wrist"), ("R_Shoulder", "R_Wrist"), ("L_Hip", "L_Ankle"), ("R_Hip", "R_Ankle"), ("L_Shoulder", "R_Hip")):
        draw.line([tuple(joints[a]), tuple(joints[b])], fill=(40, 60, 140), width=12)
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    return image


def _record(i, labelled=17, category=None):
    joints = dict(list(_joints().items())[:labelled])
    return {"id": f"r{i:03d}", "image": _scene(i), "box": [CX - 70.0, CY - 25.0, CX + 70.0, CY + 220.0], "keypoints": joints, "category": category or COLOURS[i % 6]}


def _records(n=12):
    return [_record(i) for i in range(n)]


class _ScriptedPose:
    """Returns every joint at the drawn position shifted right by `shift` px (records the calls)."""

    def __init__(self, shift=0.0):
        self.calls = 0
        self.shift = shift

    def __call__(self, image, boxes):
        self.calls += 1
        joints = _joints()
        return [[{"name": n, "x": joints[n][0] + self.shift, "y": joints[n][1], "score": 0.9} for n in KEYPOINT_NAMES] for _ in boxes]


def _no_detect(image, threshold):
    return []


# --- record contract -----------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    manifest = validate_dataset(_records())
    assert manifest["n_records"] == 12 and manifest["category_counts"] == {c: 2 for c in COLOURS} and manifest["n_images"] == 12
    assert manifest["box_height"] == {"min": 245.0, "max": 245.0} and manifest["labelled_joints"] == {"min": 17, "max": 17}
    assert len(manifest["digest"]) == 64 and manifest["model_id"] == MODEL_ID


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.__setitem__("id", "bad id"), "id must match"),
        (lambda r: r.__setitem__("box", [10.0, 10.0, 5.0, 50.0]), "not a non-empty box"),
        (lambda r: r.__setitem__("keypoints", {"Nose": [1.0, 1.0], "Toe": [2.0, 2.0]}), "unknown keypoint names"),
        (lambda r: r.__setitem__("keypoints", dict(list(_joints().items())[:5])), "MIN_LABELLED"),
        (lambda r: r.__setitem__("keypoints", {**_joints(), "Nose": [999.0, 1.0]}), "outside"),
        (lambda r: r.__setitem__("keypoints", {**_joints(), "Nose": "x"}), "must be \\[x, y\\]"),
        (lambda r: r.__setitem__("keypoints", [1, 2]), "must map joint names"),
        (lambda r: r.__setitem__("image", Image.new("RGB", (8, 8))), "MIN_IMAGE_SIDE"),
        (lambda r: r.__setitem__("category", "x" * 40), "category must be"),
        (lambda r: r.pop("box"), "missing 'box'"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[0])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_and_unique_ids():
    with pytest.raises(ValueError, match="8..5000"):
        validate_dataset(_records(4))
    records = _records()
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset(_records(3))
    validate_dataset(_records())


def test_digests_and_split_disjointness():
    records = _records(24)
    assert dataset_digest(records) == dataset_digest(list(reversed(records)))
    assert image_digest(records[0]["image"]) != image_digest(records[1]["image"])
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 24 and all(splits.values())
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    leaked = {**splits, "test": [*splits["test"], splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)
    two_persons = [*records, {**records[0], "id": "second-person", "box": [10.0, 10.0, 60.0, 200.0]}]
    both = split_dataset(two_persons, seed=1)
    assert sum(len(v) for v in both.values()) == 25
    home = next(name for name, part in both.items() if any(r["id"] == "second-person" for r in part))
    assert any(r["id"] == records[0]["id"] for r in both[home])  # persons follow their image
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.9)


# --- degradation and pinned corpus -----------------------------------------------------------------------------------


def test_degrade_scales_image_box_and_joints_together():
    record = _record(0)
    image, box, joints, scale = degrade(record["image"], record["box"], record["keypoints"], target_height=49, quality=30)
    assert scale == pytest.approx(0.2) and image.size == (64, 60)
    assert box[3] - box[1] == pytest.approx(49.0, abs=0.5) and joints["Nose"][0] == pytest.approx(32.0, abs=0.5)
    same, box2, joints2, scale2 = degrade(record["image"], record["box"], record["keypoints"], target_height=1000)
    assert scale2 == 1.0 and same.size == record["image"].size and box2 == list(record["box"]) and joints2 == record["keypoints"]
    with pytest.raises(ValueError, match="quality"):
        degrade(record["image"], record["box"], record["keypoints"], quality=0)
    with pytest.raises(TypeError):
        degrade("x", record["box"], record["keypoints"])


def test_pins_categories_and_split_sizes():
    assert len(SAMPLE_IMAGES) == sm.CORPUS_IMAGES == 300 and sum(len(e[7]) for e in SAMPLE_IMAGES) == sm.CORPUS_PERSONS == 498
    assert sum(e[1] for e in SAMPLE_IMAGES) == sm.CORPUS_BYTES and all(len(e[2]) == 64 and e[5] in (4, 5) for e in SAMPLE_IMAGES)
    assert all(len(person[2].split()) == 51 for e in SAMPLE_IMAGES for person in e[7])
    assert category_for(17) == "full" and category_for(15) == "partial" and category_for(12) == "sparse"
    assert sum(SAMPLE_SPLIT.values()) == 300


def _fake_photo(entry):
    image = _scene(entry[0] % 6, (entry[3], entry[4]))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_fetch_corpus_refuses_a_photo_that_does_not_match_its_pin(tmp_path):
    first = SAMPLE_IMAGES[0]
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: b"not-the-photo")
    good = _fake_photo(first)
    with pytest.raises(ValueError, match="bytes, pinned"):
        fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: good)
    assert not list(tmp_path.iterdir())


def test_read_corpus_builds_degraded_records_from_verified_bytes():
    entries = SAMPLE_IMAGES[:4]
    files = {e[0]: _fake_photo(e) for e in entries}
    records = read_corpus(files)
    assert len(records) == sum(len(e[7]) for e in entries) and records[0]["id"].startswith("coco-")
    assert all(39 <= r["box"][3] - r["box"][1] <= 41 for r in records) and all(r["scale"] < 1.0 for r in records)
    assert all(r["category"] == category_for(len(r["keypoints"])) and r["license"].startswith("Attribution") for r in records)
    clean = read_corpus(files, target_height=None)
    assert all(r["scale"] == 1.0 and r["image"].size == (e[3], e[4]) for r, e in zip(clean[:1], entries[:1], strict=False))
    manifest = validate_dataset(records, min_records=1)
    assert manifest["n_records"] == len(records)
    with pytest.raises(ValueError, match="decoded size"):
        read_corpus({entries[0][0]: _fake_photo((*entries[0][:3], 50, 50))})


def test_default_draw_matches_the_pinned_digest_when_the_photos_are_cached():
    cached = sm.DEFAULT_CACHE_DIR
    if not all((cached / f"{e[0]:012d}.jpg").is_file() for e in SAMPLE_IMAGES):
        pytest.skip("COCO photographs not cached")
    splits = build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cached)))
    assert {k: len({r["image_id"] for r in v}) for k, v in splits.items()} == SAMPLE_SPLIT
    assert check_split_disjoint(splits)
    assert dataset_digest([r for part in splits.values() for r in part]) == sm.SAMPLE_DIGEST


# --- BYOD ---------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_images_with_a_keypoints_csv_from_a_zip_or_directory(tmp_path):
    header = ["file", "person", "x0", "y0", "x1", "y1", "category", *[f"{n}_{a}" for n in KEYPOINT_NAMES for a in ("x", "y")]]
    rows = []
    for i in range(9):
        joints = _joints()
        values = [f"photo{i}.png", "0", CX - 70, CY - 25, CX + 70, CY + 220, COLOURS[i % 6]]
        for name in KEYPOINT_NAMES:
            if name == "R_Ankle":
                values += ["", ""]
            else:
                values += [joints[name][0], joints[name][1]]
        rows.append(",".join(str(v) for v in values))
    csv_text = ",".join(header) + "\n" + "\n".join(rows) + "\n"
    zip_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for i in range(9):
            buffer = io.BytesIO()
            _scene(i).save(buffer, format="PNG")
            archive.writestr(f"photo{i}.png", buffer.getvalue())
        archive.writestr("keypoints.csv", csv_text)
    records = load_byod_dataset(zip_path)
    assert len(records) == 9 and all(len(r["keypoints"]) == 16 and "R_Ankle" not in r["keypoints"] for r in records)
    assert validate_dataset(records)["n_records"] == 9 and records[0]["id"] == "photo0-0"
    folder = tmp_path / "folder"
    folder.mkdir()
    for i in range(9):
        _scene(i).save(folder / f"photo{i}.png")
    (folder / "keypoints.csv").write_text(csv_text, encoding="utf-8")
    assert len(load_byod_dataset(folder)) == 9
    (folder / "keypoints.csv").write_text(csv_text.replace("photo8.png", "missing.png"), encoding="utf-8")
    with pytest.raises(ValueError, match="not among the uploaded files"):
        load_byod_dataset(folder)
    (folder / "keypoints.csv").unlink()
    with pytest.raises(ValueError, match="needs a keypoints.csv"):
        load_byod_dataset(folder)
    with pytest.raises(ValueError, match="neither"):
        load_byod_dataset(tmp_path / "nothing")
    csv_path = write_dataset_csv(records, tmp_path / "train.csv")
    assert csv_path.read_text(encoding="utf-8").startswith("id,category,width,height,x0,y0,x1,y1,labelled_joints")


# --- metrics, baselines, evaluate ----------------------------------------------------------------------------------


def test_pose_metrics_and_baselines():
    records = _records()
    exact = pose_metrics([_joints() for _ in records], records)
    assert exact["pck"] == 1.0 and exact["oks"] == pytest.approx(1.0) and set(exact["per_category"]) == set(COLOURS)
    shifted = pose_metrics([_joints(cx=CX + 30) for _ in records], records)  # 30 px > 0.1 * 245 px
    assert shifted["pck"] == 0.0 and 0.0 < shifted["oks"] < 1.0 and shifted["per_record"][0]["mean_error_px"] == pytest.approx(30.0)
    centre = box_centre_baseline(records)
    prior = mean_pose_baseline(records, records)
    assert centre["pck"] < prior["pck"] == 1.0  # the prior is exact on identical drawings
    with pytest.raises(ValueError, match="same length"):
        pose_metrics([_joints()], records)
    with pytest.raises(ValueError, match="lacks joint"):
        pose_metrics([{"Nose": [1.0, 1.0]} for _ in records], records)
    with pytest.raises(ValueError, match="must not be empty"):
        mean_pose_baseline([], records)


def test_evaluate_runs_every_record_through_estimate_and_scores_it():
    pose = _ScriptedPose()
    pipe = VitPoseKeypointPipeline(_no_detect, pose, "cpu")
    result = pipe.evaluate(_records())
    assert pose.calls == 12 and result["n"] == 12 and result["adapted"] is False and result["pck"] == 1.0
    assert result["verdict"] == "measured-small-sample" and result["model_id"] == MODEL_ID
    worse = VitPoseKeypointPipeline(_no_detect, _ScriptedPose(shift=30.0), "cpu").evaluate(_records())
    assert worse["pck"] == 0.0
    with pytest.raises(ValueError):
        pipe.evaluate([{"id": "x"}] * 8)


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = VitPoseKeypointPipeline(_no_detect, _ScriptedPose(), "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records())
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"trainable_blocks": 2},
        "tensors": ["head.deconv.0.weight", "backbone.encoder.layer.11.attention.attention.query.weight", "backbone.layernorm.weight"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="heatmap head"):
        pl._check_artifact_manifest(_manifest(tmp_path, tensors=["backbone.embeddings.patch_embeddings.projection.weight"]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trainable_blocks"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"trainable_blocks": 99}), tmp_path, "base-digest")


def test_trainable_names_selects_the_head_and_the_last_blocks():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["backbone.embeddings.patch_embeddings.projection.weight", *[f"backbone.encoder.layer.{i}.attention.query.weight" for i in range(12)], "backbone.layernorm.weight", "head.deconv.0.weight", "head.conv.weight"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model(), 0) == ["head.deconv.0.weight", "head.conv.weight"]
    two = pl._trainable_names(_Model(), 2)
    assert two == ["head.deconv.0.weight", "head.conv.weight", "backbone.encoder.layer.10.attention.query.weight", "backbone.encoder.layer.11.attention.query.weight", "backbone.layernorm.weight"]
    assert len(pl._trainable_names(_Model(), 12)) == 15


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.003, "val": {"pck": 0.7, "oks": 0.65, "score": 0.675, "n": 72}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["score"] == 0.675
    assert np.isclose(payload["val"]["score"], (0.7 + 0.65) / 2)
