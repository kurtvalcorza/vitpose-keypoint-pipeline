"""Fresh-test score-once rule (owner ruling, 2026-10-17): the fresh test is scored only when EVERY pre-registered
setting matches; any other run is exploratory (validation only, never the fresh test, a separate
``*_exploratory_<n>.json``); a later identical run is a reproduction. The notebook tests run the generated Section 4
settings block and the real Sections 6, 7 and 8 cells against a stand-in pipeline that records what it scores (stand-in
evidence, no model)."""

# ruff: noqa: E501  -- notebook source markers and stand-in fixtures are kept on single lines
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from vitpose_keypoint_pipeline import (
    PREREGISTERED_SETTINGS,
    TEST_LOOKS,
    check_preregistration,
    paired_bootstrap,
    plan_test_look,
)
from vitpose_keypoint_pipeline import metrics as mt
from vitpose_keypoint_pipeline import pipeline as pl
from vitpose_keypoint_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "vitpose_keypoint_colab.ipynb"
STEM = "vitpose_keypoint"
FRESH = "fresh-1"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and marker in c["source"]]
    assert len(found) == 1, marker
    return found[0]


# --- the package rule ---------------------------------------------------------------------------------------------


def test_preregistered_settings_match_the_package_constants():
    assert PREREGISTERED_SETTINGS["target_height_px"] == sm.TARGET_HEIGHT == 40
    assert PREREGISTERED_SETTINGS["jpeg_quality"] == sm.JPEG_QUALITY == 30
    assert PREREGISTERED_SETTINGS["split_seed"] == sm.SAMPLE_SEED == 42
    assert PREREGISTERED_SETTINGS["recipe_grid"] == [[n, b] for n, b in pl.RECIPE_GRID]
    assert PREREGISTERED_SETTINGS["recipe_margin"] == pl.RECIPE_MARGIN
    assert check_preregistration(dict(PREREGISTERED_SETTINGS)) == {
        "preregistered": True,
        "settings_differing": [],
        "settings": PREREGISTERED_SETTINGS,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("learning_rate", 1e-4),
        ("epochs", 6),
        ("batch_size", 4),
        ("adapt_seed", 1),
        ("split_seed", 7),
        ("target_height_px", 32),
        ("jpeg_quality", 50),
        ("data_source", "BYOD"),
        ("byod", True),
        ("run_recipe_grid", False),
        ("recipe_grid", [["G1", 0], ["G2", 2]]),
        ("recipe_margin", 0.02),
    ],
)
def test_any_changed_setting_is_exploratory_and_never_scores_the_fresh_test(name, value, tmp_path):
    check = check_preregistration({**PREREGISTERED_SETTINGS, name: value})
    assert check["preregistered"] is False and check["settings_differing"] == [name]
    plan = plan_test_look(check, tmp_path, STEM)
    assert plan["test_look"] == "not scored (exploratory)" and plan["fresh_test_scored"] is False
    assert plan["suffix"] == "_exploratory_1"
    (tmp_path / f"{STEM}_evaluation_report_exploratory_1.json").write_text("{}", encoding="utf-8")
    assert plan_test_look(check, tmp_path, STEM)["suffix"] == "_exploratory_2"


def test_a_setting_cannot_be_left_out_or_smuggled_in():
    with pytest.raises(ValueError, match="missing"):
        check_preregistration({k: v for k, v in PREREGISTERED_SETTINGS.items() if k != "epochs"})
    with pytest.raises(ValueError, match="unknown"):
        check_preregistration({**PREREGISTERED_SETTINGS, "warmup": 0})
    # a bool never stands in for a number, and an int equal to the float is the same setting
    assert check_preregistration({**PREREGISTERED_SETTINGS, "adapt_seed": False})["settings_differing"] == [
        "adapt_seed"
    ]
    assert check_preregistration({**PREREGISTERED_SETTINGS, "epochs": 8.0})["preregistered"] is True


def test_identical_rerun_is_labelled_a_reproduction(tmp_path):
    check = check_preregistration(dict(PREREGISTERED_SETTINGS))
    first = plan_test_look(check, tmp_path, STEM)
    assert first["test_look"] == TEST_LOOKS[0] == "pre-registered evaluation" and first["suffix"] == ""
    # a recorded hosted evaluation makes every later identical run a reproduction
    assert (
        plan_test_look(check, tmp_path, STEM, recorded_evaluation="release-verification run 1")["test_look"]
        == "reproduction"
    )
    # a second identical run in the same runtime is a reproduction and never overwrites the canonical report
    (tmp_path / f"{STEM}_evaluation_report.json").write_text("{}", encoding="utf-8")
    again = plan_test_look(check, tmp_path, STEM)
    assert (
        again["test_look"] == "reproduction"
        and again["suffix"] == "_reproduction_1"
        and again["fresh_test_scored"]
    )


# --- the notebook: Section 4 settings, Sections 6-8 with a recording stand-in ----------------------------------------


def _m(records):
    rows = [{"id": r["id"], "pck": 0.6, "oks": 0.55} for r in records]
    return {
        "pck": 0.6,
        "oks": 0.55,
        "n": len(rows),
        "verdict": "measured-small-sample",
        "definitions": {},
        "per_category": {"full": {"n": len(rows), "pck": 0.6, "oks": 0.55}},
        "baseline": "stand-in",
        "per_record": rows,
    }


class StandIn:
    def __init__(self):
        self.scored: list[str] = []

    def evaluate(self, records):
        self.scored.extend(r["id"] for r in records)
        return _m(records)

    def adapt(self, train, val, *, epochs, lr, batch_size, trainable_blocks, seed, progress):
        gain = 0.05 * (trainable_blocks + 1)
        history = [{"epoch": 0, "train_loss": None, "val": {"pck": 0.6, "oks": 0.6, "score": 0.6}}]
        history += [
            {
                "epoch": e,
                "train_loss": 0.1,
                "val": {"pck": 0.6 + gain, "oks": 0.6 + gain, "score": 0.6 + gain},
            }
            for e in range(1, epochs + 1)
        ]
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": float(lr),
            "seed": seed,
            "trainable_blocks": trainable_blocks,
            "history": history,
            "best_epoch": 1,
            "n_trainable": 1,
            "n_total": 2,
            "seconds": 0.0,
            "trainable_names": [],
        }

    def save_artifact(self, *a, **k):
        return None

    def load_artifact(self, *a, **k):
        return None

    def restore_base(self):
        return []


OVERRIDES = {
    "learning rate": (
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 8, 0",
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 1e-4, 8, 0",
    ),
    "epochs": (
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 8, 0",
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 4, 5e-5, 8, 0",
    ),
    "batch (OOM tip)": (
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 8, 0",
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 4, 0",
    ),
    "fine-tuning seed": (
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 8, 0",
        "EPOCHS, LEARNING_RATE, BATCH_SIZE, ADAPT_SEED = 8, 5e-5, 8, 1",
    ),
    "split seed": ("SPLIT_SEED = 42  # @param", "SPLIT_SEED = 7  # @param"),
    "size": ("TARGET_HEIGHT, JPEG_QUALITY = 40, 30", "TARGET_HEIGHT, JPEG_QUALITY = 32, 30"),
    "JPEG": ("TARGET_HEIGHT, JPEG_QUALITY = 40, 30", "TARGET_HEIGHT, JPEG_QUALITY = 40, 50"),
    "data (BYOD)": ("USE_BYOD = False  # @param", "USE_BYOD = True  # @param"),
    "grid off": ("RUN_RECIPE_GRID = True  # @param", "RUN_RECIPE_GRID = False  # @param"),
}


def _run_sections_4_to_8(notebook, tmp_path, override=None):
    section_4 = _cell(notebook, "USE_BYOD = False")
    head = section_4[section_4.index("USE_BYOD = False") : section_4.index("os.makedirs")]
    if override:
        assert head.count(override[0]) == 1, override[0]
        head = head.replace(*override)
    block = section_4[section_4.index("RUN_SETTINGS = {") : section_4.index("if len(val_records) < 20")]
    ns: dict = {}
    for module in (pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    pipe = StandIn()
    ns.update(
        {
            "pipe": pipe,
            "Path": Path,
            "json": json,
            "time": __import__("time"),
            "paired_bootstrap": paired_bootstrap,
            "image_digest": lambda image: "d",
            "train_records": [],
            "val_records": [{"id": "val-1", "image_id": 1}, {"id": "val-2", "image_id": 2}],
            "test_records": [{"id": FRESH, "image_id": 9}],
            "clean_test": [{"id": FRESH, "image_id": 9}],
            "dev_test_records": None,
            "clean_dev": None,
            "box_centre_baseline": lambda r: _m(r),
            "mean_pose_baseline": lambda a, b: _m(b),
            "MODEL_ID": "stand-in",
            "MODEL_REVISION": "0" * 40,
            "MODEL_KEY": "stand-in",
            "data_source": "stand-in",
            "dataset_manifests": {"test": {"digest": "d"}},
            "disjoint": {},
        }
    )
    exec(head + "\n" + block, ns)
    exec(_cell(notebook, "baseline_centre = box_centre_baseline("), ns)
    exec(_cell(notebook, "recipe_selection = select_recipe(grid_rows, frozen_val_score)"), ns)
    exec(_cell(notebook, "adapted_val = pipe.evaluate(val_records)"), ns)
    return ns, pipe


def test_default_settings_score_the_fresh_test_as_the_preregistered_evaluation(
    notebook, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns, pipe = _run_sections_4_to_8(notebook, tmp_path)
    assert ns["preregistration"]["preregistered"] and FRESH in pipe.scored
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert (
        report["test_look"] == "pre-registered evaluation"
        and report["settings_differing"] == []
        and report["preregistered"] is True
    )
    assert report["held_out"] == "fresh test" and report["test_metrics"] is not None
    # Run all again in the same runtime, settings unchanged: a reproduction, written beside the first report.
    ns2, _ = _run_sections_4_to_8(notebook, tmp_path)
    assert ns2["test_look"]["test_look"] == "reproduction"
    repro = json.loads(
        (tmp_path / "outputs" / f"{STEM}_evaluation_report_reproduction_1.json").read_text(encoding="utf-8")
    )
    assert repro["test_look"] == "reproduction"
    assert (
        json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))[
            "test_look"
        ]
        == "pre-registered evaluation"
    )


@pytest.mark.parametrize("label", sorted(OVERRIDES))
def test_changed_setting_runs_exploratory_and_never_scores_the_fresh_test(
    notebook, tmp_path, monkeypatch, label
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    ns, pipe = _run_sections_4_to_8(notebook, tmp_path, OVERRIDES[label])
    assert FRESH not in pipe.scored, f"{label}: the fresh test was scored in an exploratory run"
    assert ns["held_out_records"] is ns["val_records"] and ns["held_out_clean"] is None
    assert not (tmp_path / "outputs" / f"{STEM}_evaluation_report.json").exists()
    report = json.loads(
        (tmp_path / "outputs" / f"{STEM}_evaluation_report_exploratory_1.json").read_text(encoding="utf-8")
    )
    assert (
        report["test_look"] == "not scored (exploratory)"
        and report["preregistered"] is False
        and report["settings_differing"]
    )
    assert report["test_metrics"] is None and report["frozen_test"] is None


def test_section_7_code_edit_after_section_6_stops_section_8(notebook, tmp_path, monkeypatch):
    """Training with settings other than Section 4's (an edited Section 7) must not reach the fresh test."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    edited = notebook | {
        "cells": [
            dict(c, source=c["source"].replace("epochs=EPOCHS,", "epochs=3,"))
            if c["cell_type"] == "code"
            else c
            for c in notebook["cells"]
        ]
    }
    with pytest.raises(RuntimeError, match="run again from Section 4"):
        _run_sections_4_to_8(edited, tmp_path)


def test_no_cell_scores_the_fresh_test_outside_the_gate(notebook):
    code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    assert not re.search(r"evaluate\(test_records\)", code) and not re.search(r"evaluate\(clean_test\)", code)
    assert len(re.findall(r"(?<![\w])pipe\.evaluate\(held_out_records\)", code)) == 2
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert (
        "test_records" not in experiment
        and "experiment_pipe.evaluate(val_records)" in experiment
        and "_exploratory_" in experiment
    )
    section_9 = _cell(notebook, "for record in held_out_records[:4]:")
    assert "test_records" not in section_9
    text = json.dumps(notebook)
    for claim in ("scored once", "scored **once**", "scores the fresh test once", "one-off look"):
        assert claim not in text, claim
