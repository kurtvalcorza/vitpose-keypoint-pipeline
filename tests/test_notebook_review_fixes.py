"""Regression tests for the 2026-10-05 notebook review findings (VTP-M1..M4, VTP-m1, VTP-m2).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and restore_base() is exercised on a stand-in model, not the checkpoint. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vitpose_keypoint_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "vitpose_keypoint_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "vitpose_keypoint_pipeline" / "pipeline.py"
STEM = "vitpose_keypoint"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- VTP-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_vtp_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_vtp_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_vtp_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- VTP-M3: every adaptation starts from the pinned base -------------------------------------------------------


def test_vtp_m3_adapt_and_load_artifact_restore_the_base_first():
    """Torch-backed, so the order inside adapt is checked statically: pre-call state kept, base restored, then epoch 0."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("previous_state = {", "restored = self.restore_base()", "self._remember_base(names)", "frozen_state = {", '"note": "frozen model"', "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert failure.index("model.load_state_dict(previous_state, strict=False)") < failure.index("self.adapter = previous_adapter") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self._remember_base(sorted(tensors))") < load.index("model.load_state_dict(")


class _Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=float)

    def detach(self):
        return self

    def clone(self):
        return _Tensor(self.value.copy())


class _Model:
    def __init__(self):
        self.state = {"backbone.encoder.layer.11.w": _Tensor([1.0, 2.0]), "head.conv.w": _Tensor([3.0]), "backbone.embeddings.w": _Tensor([4.0])}

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, values, strict=True):
        assert strict is False
        for name, tensor in values.items():
            self.state[name] = _Tensor(tensor.value.copy())

    def eval(self):
        return self


def test_vtp_m3_restore_base_makes_a_head_only_rerun_start_from_base_blocks_stand_in():
    """After a two-block run, restore_base() puts the blocks back too, so a later head-only run and its head-only
    artifact see the base blocks (the reload-parity failure the review traced). Numpy stand-in, not torch."""
    from vitpose_keypoint_pipeline import VitPoseKeypointPipeline

    model = _Model()
    pipe = VitPoseKeypointPipeline(_detect=lambda *a: [], _pose=lambda *a: [], _model=model, _processor=object())
    assert pipe.restore_base() == []
    pipe._remember_base(["backbone.encoder.layer.11.w", "head.conv.w"])  # the default two-block run
    model.state["backbone.encoder.layer.11.w"] = _Tensor([9.0, 9.0])
    model.state["head.conv.w"] = _Tensor([9.0])
    pipe.adapter = {"trainable_blocks": 2}
    assert pipe.restore_base() == ["backbone.encoder.layer.11.w", "head.conv.w"]  # what a head-only adapt() does first
    assert model.state["backbone.encoder.layer.11.w"].value.tolist() == [1.0, 2.0] and model.state["head.conv.w"].value.tolist() == [3.0]
    assert model.state["backbone.embeddings.w"].value.tolist() == [4.0] and pipe.adapter is None


def test_vtp_m3_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_tensors = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = VitPoseKeypointPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR, device=pipe.device)" in experiment
    assert "EXPERIMENT_TRAINABLE_BLOCKS = 0" in experiment and "experiment_parity" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- VTP-M2: quality outcomes are reported verdicts -------------------------------------------------------------


def test_vtp_m2_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert asserts == ["assert parity['identical_persons'] == parity['of']"]


def _m(pck, oks):
    return {"pck": pck, "oks": oks, "n": 4, "verdict": "measured-small-sample", "definitions": {}, "per_category": {"full": {"n": 4, "pck": pck, "oks": oks}}, "baseline": "stand-in"}


def test_vtp_m2_a_run_without_a_gain_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 with stand-ins where validation kept epoch 0 (adapted == frozen) and the clean OKS falls:
    both cells complete and record `no gain` (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_m(0.6, 0.56), _m(0.957, 0.94), _m(0.6, 0.56), _m(0.6, 0.56), _m(0.959, 0.93)])

    class StandIn:
        def evaluate(self, records):
            return next(scores)

    ns = {
        "pipe": StandIn(), "train_records": [], "val_records": [], "test_records": [], "clean_test": [], "time": __import__("time"), "json": json,
        "box_centre_baseline": lambda r: _m(0.086, 0.165), "mean_pose_baseline": lambda a, b: _m(0.472, 0.44),
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "TARGET_HEIGHT": 40, "JPEG_QUALITY": 30, "disjoint": {}, "adapt_result": {"history": [], "trainable_names": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "baseline_centre = box_centre_baseline("), ns)
    assert ns["frozen_verdict"] == "frozen above the box-centre floor"
    exec(_cell(notebook, "adapted_test = pipe.evaluate(test_records)"), ns)
    verdicts = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]["verdicts"]
    assert verdicts["adapted_vs_frozen_pck"] == "no gain" and verdicts["adapted_vs_frozen_oks"] == "no gain"
    assert verdicts["clean_pck"] == "improved" and verdicts["clean_oks"] == "worse"


# --- VTP-M4: guided layer and infrastructure labelling ----------------------------------------------------------


def test_vtp_m4_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_vtp_m4_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_vtp_m4_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- VTP-m1: BYOD contract --------------------------------------------------------------------------------------


def _jpeg(i: int) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.random.default_rng(i).integers(0, 255, (120, 100, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def _row(name: str, joints: int = 12, bad: bool = False) -> str:
    from vitpose_keypoint_pipeline.pipeline import KEYPOINT_NAMES

    cells = [name, "0", "10", "10", "90" if not bad else "x", "110"]
    for k, _joint in enumerate(KEYPOINT_NAMES):
        cells += [str(20 + 3 * k), str(20 + 4 * k)] if k < joints else ["", ""]
    return ",".join(cells)


def _zip(path: Path, n: int, *, drop: str | None = None, bad_line: int | None = None, extra: dict[str, bytes] | None = None) -> Path:
    from vitpose_keypoint_pipeline.pipeline import KEYPOINT_NAMES

    header = ",".join(["file", "person", "x0", "y0", "x1", "y1", *[f"{j}_{a}" for j in KEYPOINT_NAMES for a in ("x", "y")]])
    rows = [_row(f"img{i}.jpg", bad=(bad_line == i + 2)) for i in range(n)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("keypoints.csv", "﻿" + header + "\n" + "\n".join(rows) + "\n")
        for i in range(n):
            if f"img{i}.jpg" != drop:
                archive.writestr(f"img{i}.jpg", _jpeg(i))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_vtp_m1_stated_minimum_is_what_the_split_accepts(tmp_path, notebook):
    assert sm.min_byod_records()["total"] == 12 and sm.min_byod_records()["images"] == 12
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"the train split holds 7 persons on 7 image\(s\).*supply at least 12 labelled persons"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 11)), seed=42)
    assert "**12 persons on 12 single-person images**" in _markdown(notebook)


def test_vtp_m1_refusals_name_the_line_and_litter_is_skipped(tmp_path):
    with pytest.raises(ValueError, match=r"keypoints.csv line 5 \(file 'img3.jpg'\): names an image that is not among the uploaded files"):
        sm.load_byod_dataset(_zip(tmp_path / "missing.zip", 12, drop="img3.jpg"))
    with pytest.raises(ValueError, match=r"keypoints.csv line 4 \(file 'img2.jpg'\): x0, y0, x1 and y1 must be present and numeric"):
        sm.load_byod_dataset(_zip(tmp_path / "bad.zip", 12, bad_line=4))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 12, extra={"__MACOSX/._keypoints.csv": b"\0", "__MACOSX/._img0.jpg": b"\0"}))) == 12


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from vitpose_keypoint_pipeline import metrics as mt
    from vitpose_keypoint_pipeline import pipeline as pl

    ns = {}
    for module in (pl, mt, sm):
        ns.update({k: getattr(module, k) for k in dir(module) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"best_epoch": 2}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_vtp_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 14)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_rows"] == {"byod": 14, "effective_minimum": 12}
    assert "held-out test persons" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_vtp_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


# --- VTP-m2: the prose names its runs and reads clean PCK and OKS together ----------------------------------


def test_vtp_m2_prose_names_its_runs_and_reads_clean_oks(notebook):
    markdown = _markdown(notebook)
    for stale in ("the adaptation did not cost the resolution the checkpoint was built for", "without costing the clean input"):
        assert stale not in markdown, stale
    assert "Kaggle T4 release run" in markdown and "0.94 to 0.93" in markdown and "0.718" in markdown
    access = next(line for line in markdown.splitlines() if line.startswith("- **External access:**"))
    assert "images.cocodataset.org" in access and "SHA-256" in access
