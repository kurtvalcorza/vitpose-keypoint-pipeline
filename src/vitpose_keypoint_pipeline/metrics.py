"""Corpus-level keypoint measures and two non-neural baselines, in numpy.

``pose_metrics`` scores one predicted joint set per record against ``record['keypoints']``: **PCK** — the share of
labelled joints within ``PCK_FRACTION`` (0.1) of the box's longest side of their reference, the package's own
``keypoint_pck`` convention — and **mean OKS** — the COCO object-keypoint-similarity kernel averaged over the
labelled joints with the COCO per-joint sigmas and the box area as the object scale (the primitive of the COCO
keypoint AP, without its thresholds) — overall and per ``category``. The baselines answer from the box alone: every
joint at the box centre, or every joint at the training split's mean position relative to the box.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .pipeline import KEYPOINT_NAMES, PCK_FRACTION

METRIC_DEFINITIONS = {
    "pck": f"share of labelled joints whose prediction lies within PCK_FRACTION ({PCK_FRACTION}) of the person box's longest side of the reference; in 0..1, higher is better",
    "oks": "mean over labelled joints of exp(-d^2 / (2 * area * (2 * sigma_joint)^2)) with the COCO keypoint sigmas and the box area — the COCO OKS kernel per joint; in 0..1, higher is better",
}
# COCO per-joint sigmas (pycocotools), in KEYPOINT_NAMES order.
OKS_SIGMAS = {
    name: value
    for name, value in zip(
        KEYPOINT_NAMES,
        (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089),
        strict=True,
    )
}


def _joint_errors(predicted: Mapping[str, Sequence[float]], record: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    names = list(record["keypoints"])
    errors = []
    for name in names:
        rx, ry = record["keypoints"][name]
        if name not in predicted:
            raise ValueError(f"prediction for {record['id']!r} lacks joint {name!r}")
        px, py = predicted[name][0], predicted[name][1]
        errors.append(float(np.hypot(px - rx, py - ry)))
    return np.asarray(errors, dtype=np.float64), names


def pose_metrics(predictions: Sequence[Mapping[str, Sequence[float]]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one joint mapping (name -> [x, y], all 17 joints) per record against `record['keypoints']`; per-record
    rows, means overall and per category. Raises when the lengths differ or nothing is scored."""
    if len(predictions) != len(records) or not predictions:
        raise ValueError("predictions and records must be non-empty and the same length")
    rows = []
    for predicted, record in zip(predictions, records, strict=True):
        errors, names = _joint_errors(predicted, record)
        x0, y0, x1, y1 = record["box"]
        radius = PCK_FRACTION * max(x1 - x0, y1 - y0)
        area = max(1.0, (x1 - x0) * (y1 - y0))
        sigmas = np.asarray([OKS_SIGMAS[n] for n in names])
        oks = float(np.mean(np.exp(-(errors**2) / (2.0 * area * (2.0 * sigmas) ** 2))))
        rows.append(
            {
                "id": record["id"],
                "category": record.get("category"),
                "pck": float(np.mean(errors <= radius)),
                "oks": oks,
                "mean_error_px": float(errors.mean()),
                "n_joints": len(names),
            }
        )

    def _mean(items: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {"n": len(items), "pck": float(np.mean([r["pck"] for r in items])), "oks": float(np.mean([r["oks"] for r in items]))}

    categories = sorted({r["category"] for r in rows if r["category"] is not None})
    return {
        **_mean(rows),
        "per_category": {c: _mean([r for r in rows if r["category"] == c]) for c in categories},
        "per_record": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def box_centre_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every joint at the centre of the person box — the floor."""
    predictions = []
    for record in records:
        x0, y0, x1, y1 = record["box"]
        centre = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        predictions.append({name: list(centre) for name in KEYPOINT_NAMES})
    return {**pose_metrics(predictions, records), "baseline": "every joint at the box centre"}


def mean_pose_baseline(train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Every joint at its mean position relative to the box over the training split — the layout prior a box alone
    gives away."""
    if not train:
        raise ValueError("train must not be empty")
    sums: dict[str, list[float]] = {name: [0.0, 0.0, 0.0] for name in KEYPOINT_NAMES}
    for record in train:
        x0, y0, x1, y1 = record["box"]
        w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
        for name, (px, py) in record["keypoints"].items():
            sums[name][0] += (px - x0) / w
            sums[name][1] += (py - y0) / h
            sums[name][2] += 1.0
    mean = {name: ((s[0] / s[2], s[1] / s[2]) if s[2] else (0.5, 0.5)) for name, s in sums.items()}
    predictions = []
    for record in records:
        x0, y0, x1, y1 = record["box"]
        predictions.append({name: [x0 + mean[name][0] * (x1 - x0), y0 + mean[name][1] * (y1 - y0)] for name in KEYPOINT_NAMES})
    return {**pose_metrics(predictions, records), "baseline": "every joint at the training split's mean position relative to the box"}
