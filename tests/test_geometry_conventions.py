"""Known-joint regression tests for the two coordinate conventions of review finding VTP-M5.

(a) `degrade` must move a COCO pixel-index joint with the image content: a Gaussian blob centred on a joint, downscaled
    by `degrade`, has its centroid on the degraded label.
(b) `_heatmap_targets` must encode with the stride the pinned processor decodes with (UDP, ``(crop - 1) / (hm - 1)``):
    a target encoded here and decoded by the processor's own `transform_preds` (and, when SciPy is installed, by its
    full `post_process_pose_estimation`) comes back on the joint.
"""
# ruff: noqa: E501  -- known-joint fixtures and assertion messages are kept on single lines

import numpy as np
import pytest
from PIL import Image

from vitpose_keypoint_pipeline import KEYPOINT_NAMES, RECIPE_GRID, RECIPE_MARGIN, select_recipe
from vitpose_keypoint_pipeline.samples import degrade


def _blob_offsets(seed=0, draws=20):
    rng = np.random.default_rng(seed)
    offsets = []
    for _ in range(draws):
        width, height = 640, 480
        box_height = float(rng.uniform(120, 400))
        x, y = float(rng.integers(150, 490)), float(rng.integers(120, 360))
        sigma = 6.0 * box_height / 40.0
        yy, xx = np.mgrid[0:height, 0:width]
        blob = np.uint8(np.round(255 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))))
        image = Image.fromarray(blob).convert("RGB")
        box = [x - 20, y - box_height / 2, x + 20, y + box_height / 2]
        small, _box, joints, _scale = degrade(image, box, {"Nose": [x, y]}, target_height=40, quality=95)
        a = np.asarray(small.convert("L"), dtype=np.float64)
        a = np.clip(a - a.max() * 0.05, 0, None)
        gy, gx = np.mgrid[0 : a.shape[0], 0 : a.shape[1]]
        offsets.append([(a * gx).sum() / a.sum() - joints["Nose"][0], (a * gy).sum() / a.sum() - joints["Nose"][1]])
    return np.asarray(offsets)


def test_degrade_moves_a_pixel_index_joint_with_the_image_content():
    offsets = _blob_offsets()
    # x * s would leave every label 0.5 * (1 - s) ~ 0.42 px right of and below the content (review VTP-M5 (a)).
    assert np.abs(offsets.mean(axis=0)).max() < 0.12, offsets.mean(axis=0)
    assert np.abs(offsets).max() < 0.25


def test_degrade_clamps_an_edge_joint_into_the_image():
    image = Image.new("RGB", (400, 300), (90, 90, 90))
    small, _box, joints, scale = degrade(image, [0, 0, 100, 200], {"Nose": [0.0, 0.0], "L_Eye": [399.0, 299.0]}, target_height=40)
    assert scale == pytest.approx(0.2)
    assert joints["Nose"] == [0.0, 0.0]  # (0 + 0.5) * 0.2 - 0.5 = -0.4 -> clamped
    assert 0.0 <= joints["L_Eye"][0] <= small.width - 1 and 0.0 <= joints["L_Eye"][1] <= small.height - 1


def _targets_for(rng):
    from vitpose_keypoint_pipeline.pipeline import _heatmap_targets

    bw, bh = float(rng.uniform(12, 30)), float(rng.uniform(38, 41))
    x0, y0 = float(rng.uniform(5, 50)), float(rng.uniform(5, 30))
    # joints well inside the box, so no Gaussian is cut by the heatmap edge
    joints = {n: [x0 + float(rng.uniform(0.2, 0.8)) * bw, y0 + float(rng.uniform(0.2, 0.8)) * bh] for n in KEYPOINT_NAMES}
    targets, weights = _heatmap_targets({"box": [x0, y0, x0 + bw, y0 + bh], "keypoints": joints}, (256, 192), (64, 48))
    return [x0, y0, bw, bh], joints, targets, weights


def test_heatmap_targets_decode_back_to_the_joint_through_the_processor_transform():
    pytest.importorskip("transformers")
    from transformers.models.vitpose.image_processing_vitpose import box_to_center_and_scale, transform_preds

    rng = np.random.default_rng(1)
    errors = []
    for _ in range(20):
        box, joints, targets, weights = _targets_for(rng)
        assert weights.all()
        ys, xs = np.mgrid[0 : targets.shape[1], 0 : targets.shape[2]]
        mass = targets.sum(axis=(1, 2))
        coords = np.stack([(targets * xs).sum(axis=(1, 2)) / mass, (targets * ys).sum(axis=(1, 2)) / mass], axis=1)
        center, scale = box_to_center_and_scale(np.array(box, dtype=np.float32), image_width=192, image_height=256)
        decoded = transform_preds(coords, center=center, scale=scale, output_size=[64, 48])
        errors.append(np.hypot(*(decoded - np.array([joints[n] for n in KEYPOINT_NAMES])).T))
    # crop / heatmap would put every joint ~0.3 px right of and below its label at 40-px boxes (review VTP-M5 (b)).
    assert np.concatenate(errors).max() < 0.02


def test_heatmap_targets_round_trip_through_the_pinned_post_processing():
    pytest.importorskip("scipy")  # the processor's DARK refinement needs SciPy (not in CI's dependency budget)
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from transformers import VitPoseImageProcessor

    processor = VitPoseImageProcessor(size={"height": 256, "width": 192})
    rng = np.random.default_rng(2)
    errors = []
    for _ in range(20):
        box, joints, targets, _weights = _targets_for(rng)
        out = processor.post_process_pose_estimation(
            SimpleNamespace(heatmaps=torch.from_numpy(targets[None])), boxes=[np.array([box], dtype=np.float32)], threshold=0.0
        )
        decoded = out[0][0]["keypoints"].numpy()
        errors.append(np.hypot(*(decoded - np.array([joints[n] for n in KEYPOINT_NAMES])).T))
    assert np.concatenate(errors).max() < 0.05


def test_recipe_rule_prefers_fewer_blocks_and_the_frozen_model_within_the_margin():
    assert [name for name, _ in RECIPE_GRID] == ["G1", "G2", "G3"] and [b for _, b in RECIPE_GRID] == [0, 2, 4]
    assert RECIPE_MARGIN == 0.01

    def grid(g1, g2, g3):
        return [{"name": n, "trainable_blocks": b, "best_score": s} for (n, b), s in zip(RECIPE_GRID, (g1, g2, g3), strict=True)]

    assert select_recipe(grid(0.70, 0.75, 0.755), 0.65)["chosen"] == "G2"  # G3 within the margin of G2
    assert select_recipe(grid(0.70, 0.705, 0.72), 0.65)["chosen"] == "G3"  # G3 clears G1 by more than 0.01
    assert select_recipe(grid(0.70, 0.709, 0.709), 0.65)["chosen"] == "G1"
    assert select_recipe(grid(0.655, 0.659, 0.66), 0.65)["chosen"] == "frozen"  # nothing beats frozen by > 0.01
    picked = select_recipe(grid(0.655, 0.67, 0.665), 0.65)
    assert picked["chosen"] == "G2" and picked["eligible"] == ["G2", "G3"]


def test_target_centre_sits_on_the_processors_own_warped_pixels():
    """S1: a blob at a known joint, warped by the real processor (no weights), peaks where `_heatmap_targets` puts the
    Gaussian centre times the UDP stride, so the hand-copied warp arguments match the processor's."""
    pytest.importorskip("scipy")  # the processor's affine warp needs SciPy
    pytest.importorskip("transformers")
    from transformers import VitPoseImageProcessor

    from vitpose_keypoint_pipeline.pipeline import _heatmap_targets

    processor = VitPoseImageProcessor(size={"height": 256, "width": 192})
    rng = np.random.default_rng(3)
    for _ in range(6):
        jx, jy = float(rng.uniform(30, 50)), float(rng.uniform(40, 80))
        box = [jx - float(rng.uniform(6, 12)), jy - float(rng.uniform(15, 25)), jx + float(rng.uniform(6, 12)), jy + float(rng.uniform(15, 25))]
        yy, xx = np.mgrid[0:120, 0:90]
        blob = np.exp(-((xx - jx) ** 2 + (yy - jy) ** 2) / (2 * 1.5**2))
        image = Image.fromarray(np.uint8(np.round(255 * blob))).convert("RGB")
        xywh = np.array([[box[0], box[1], box[2] - box[0], box[3] - box[1]]], dtype=np.float32)
        crop = processor(image, boxes=[xywh], return_tensors="np")["pixel_values"][0][0]
        crop = crop - crop.min()
        crop = np.clip(crop - crop.max() * 0.2, 0, None)
        cy, cx = np.mgrid[0 : crop.shape[0], 0 : crop.shape[1]]
        peak = np.array([(crop * cx).sum() / crop.sum(), (crop * cy).sum() / crop.sum()])
        targets, _w = _heatmap_targets({"box": box, "keypoints": {"Nose": [jx, jy]}}, (256, 192), (64, 48))
        t = targets[0]
        hy, hx = np.mgrid[0 : t.shape[0], 0 : t.shape[1]]
        centre = np.array([(t * hx).sum() / t.sum(), (t * hy).sum() / t.sum()]) * np.array([(192 - 1) / (48 - 1), (256 - 1) / (64 - 1)])
        assert np.abs(peak - centre).max() < 0.5, (peak, centre)
