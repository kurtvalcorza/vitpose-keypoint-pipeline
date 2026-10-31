"""DAT19/DAT20: a refused BYOD input renders exactly one `Refused: <file>: <reason>` line, no traceback.

Executes the committed Section 4 cell with the real loader and splitter. Covers the scripted hosted refusal
input (`BYOD_PATH='/etc/hostname'`: neither a folder nor a zip), a folder without keypoints.csv, a missing
path, an empty path outside Colab, a cancelled and a several-file upload, the in-cell invalidation, and that
a genuine error keeps its own type.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from test_notebook_review_fixes import NOTEBOOK, _section_4, _section_4_namespace

REFUSAL_CELL = "vitpose_keypoint-17"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _refused(notebook, path: str, ns: dict | None = None) -> str:
    with pytest.raises(ValueError) as caught:
        exec(_section_4(notebook, path), ns if ns is not None else _section_4_namespace([]))
    assert caught.value.dimer_refusal is True and caught.value.__cause__ is None
    rendered = caught.value._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    return tmp_path


def _colab(monkeypatch, uploaded):
    google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
    files.upload = lambda: uploaded
    colab.files, google.colab = files, colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)


def test_scripted_hosted_input_neither_a_folder_nor_a_zip(notebook, workdir):
    (workdir / "hostname").write_text("colab-vm\n", encoding="utf-8")
    line = _refused(notebook, str(workdir / "hostname"))
    assert line.startswith("Refused: hostname: ") and "is neither a directory nor a zip file" in line, line


def test_a_folder_a_missing_path_and_an_empty_path(notebook, workdir):
    (workdir / "etc").mkdir()
    assert _refused(notebook, str(workdir / "etc")).startswith("Refused: etc: BYOD needs a keypoints.csv")
    assert _refused(notebook, "nowhere.zip").startswith("Refused: nowhere.zip: BYOD_PATH does not exist")
    assert _refused(notebook, "").startswith("Refused: (no file): USE_BYOD is True but BYOD_PATH is empty")


def test_cancelled_and_several_file_uploads(notebook, workdir, monkeypatch):
    _colab(monkeypatch, {})
    assert _refused(notebook, "").startswith("Refused: (no file): upload exactly one .zip file (received 0")
    _colab(monkeypatch, {"a.zip": b"", "b.zip": b""})
    assert _refused(notebook, "").startswith(
        "Refused: a.zip, b.zip: upload exactly one .zip file (received 2"
    )


def test_refusal_invalidates_earlier_results_in_the_cell(notebook, workdir):
    ns = _section_4_namespace([])
    ns.update({"train_records": ["stale"], "adapt_result": {"stale": True}})
    _refused(notebook, "nowhere.zip", ns)
    assert not {"train_records", "adapt_result"} & set(ns)


def test_genuine_error_keeps_its_type(notebook, workdir):
    (workdir / "mine").mkdir()
    ns = _section_4_namespace([])

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        exec(_section_4(notebook, str(workdir / "mine")), ns)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_refusal_cell_id(notebook):
    ids = [c["id"] for c in notebook["cells"] if "class ByodRefusal" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]
