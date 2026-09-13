# Release verification

`tutorials/vitpose_keypoint_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on
  the primary path; exactly one cell tagged `embedded_module` equal to
  `src/vitpose_keypoint_pipeline/pipeline.py` after the generator's documented rewrites; the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` and `DETECTOR_MODEL_ID`/`DETECTOR_REVISION` are bound only in the carried
  module cell (and repeated in the two inline manifests, which the notebook asserts against the module
  before fetching), both revisions are 40-hex immutable commits, and the same identity strings appear in
  `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `stage_missing_detector_files`,
  `verify_snapshot`, `verify_detector_snapshot`,
  `VitPoseKeypointPipeline.from_pretrained(weights_dir=..., detector_dir=...)`, `validate_inputs`,
  `detect_people`, `estimate`, `evaluation_report`), the ceiling print (`MIN_IMAGE_SIDE`,
  `MAX_IMAGE_SIDE`, `MAX_PERSONS`, `DETECTION_THRESHOLD`, `KEYPOINT_THRESHOLD`, the 17
  `KEYPOINT_NAMES`, `PCK_FRACTION`), the exports, the learner-facing statements (two caller-owned
  thresholds, heatmap maxima are not probabilities, stage 1 finds no person on the drawing, stage 2
  returns joints for any box, no OKS-AP, PCK as sanity check, capability exclusions) and the gated-off
  BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `VitPoseForPoseEstimation` / `RTDetrForObjectDetection` /
  `AutoImageProcessor` / `post_process_pose_estimation(` / `post_process_object_detection(` /
  `from huggingface_hub import` use **outside the carried module cell**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`;
injected runner, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (**no repository checkout is needed — the notebook is standalone**) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `detection_threshold = 0.3`,
   `keypoint_threshold = 0.3`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded
   in `metadata.dimer.generated_from` and that the installed core package versions equal the inline
   `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the carried module cell executes (defines `VitPoseKeypointPipeline`, `validate_inputs`,
     `evaluation_report`, `keypoint_pck`, `verify_snapshot`, `verify_detector_snapshot`,
     `stage_missing_files`, `stage_missing_detector_files`, the 17 `KEYPOINT_NAMES`) with no import of
     the repository package;
   - synthetic 640×640 cartoon person drawn in code with its RGB SHA-256 printed and the ceilings
     (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_PERSONS` 50, `DETECTION_THRESHOLD` 0.3,
     `KEYPOINT_THRESHOLD` 0.3, 17 keypoint names, `PCK_FRACTION` 0.1) surfaced;
   - pinned `usyd-community/vitpose-base` **and** `PekingU/rtdetr_r50vd` acquisition at their immutable
     revisions through the carried module: both inline manifests are asserted against the module
     identities and written to `weights/vitpose-base/` and `weights/rtdetr-r50vd/`, the two stagers
     report all four entries each on a clean runtime, both verifiers return their summary dicts, and
     `from_pretrained(weights_dir=WEIGHTS_DIR, detector_dir=DETECTOR_WEIGHTS_DIR)` loads from the verified
     directories;
   - `validate_inputs` writes `outputs/vitpose_keypoint_input_manifest.json` (verdict `accepted`,
     `box_source` caller with the drawn box, one recorded rejection finding from the box-outside-image
     probe);
   - `detect_people` on the drawing returning **no** `person` at 0.3 — the expected finding, recorded
     in the report — then `estimate` with the drawn box returning one pose with 17 joints; record the
     scores (the card-pass CPU smoke got all 17 at 0.85–0.97 with a mean error of 12.8 px against the
     drawn joints; a materially different result is a finding to record, not a failure by itself,
     because no metric is asserted);
   - `evaluation_report` writes `outputs/vitpose_keypoint_evaluation_report.json` with verdict
     `sample-sanity` and one `keypoint_pck` entry (17/17 within the 48 px radius in the smoke run) on the
     synthetic sample (`not-measurable` on BYOD), stated as such;
   - `outputs/vitpose_keypoint_result.json`, `outputs/vitpose_keypoint_keypoints.csv` and
     `outputs/vitpose_keypoint_annotated.png` written with `NOTEBOOK_SOURCE`, both model revisions, both
     licences, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/vitpose_keypoint_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/vitpose_keypoint_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Local pre-flight evidence (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `3fcfd7c0d6f7` (commit `e1cd703`, generated at `ebe84a2`; `NOTEBOOK_SOURCE.repository_revision` = `ebe84a2…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), both stagers fetched all 4 + 4 manifest entries (360 MB + 172 MB) from the Hub cache at the pinned revisions into the scratch `weights/`, both verifiers PASS, `detect_people` → 0 persons at 0.3 (finding recorded), `estimate` with the drawn box → 1 pose, 17/17 joints kept, mean keypoint score 0.930, `evaluation_report` `sample-sanity` (`keypoint_pck` 1.0 = 17/17 within 48 px, mean error 12.8 px), 5 outputs written | 60.1 s | PASS — pre-flight only; not promotion evidence |

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — no Colab/Kaggle run yet |

## Current status

No clean-runtime execution in a **supported** runtime (Colab or Kaggle) has been recorded yet; the run is
**pending**. What exists: static validation (`tools/validate_release_assets.py`), the generator parity
checks (`--check` OK), the offline unit suite, and one **local fresh-kernel execution** of the generated
notebook (table above) that exercised the standalone carrier, the real `hf_hub_download` staging path
into an empty `weights/` directory, verification, both stages, the evaluation report and every export —
which is necessary but not promotion evidence because the workstation is not a supported runtime. The
registry status remains **Candidate** until a reviewer confirms a recorded supported-runtime run against
the notebook blob under review and an integrator promotes it. Facts a reviewer should weigh: the CUDA
path has not been executed; the default path never exercises the detector-supplied-box branch of
`estimate` (stage 1 finds no `person` on the cartoon at 0.3, and seven low-confidence proposals at 0.05),
so the two-stage chain end to end is only proven on a photograph a BYOD run supplies — the
detector's contract itself is proven in the sibling `rtdetr-detection-pipeline`; a box that contains
no person still yields 17 joints (all scored below 0.04 on a blank box in the smoke run), so the
heatmap score is the only "no person" signal; and both snapshots declare slow image processors
(transformers prints `use_fast` notices), which is the configuration the smoke numbers were measured
with.
