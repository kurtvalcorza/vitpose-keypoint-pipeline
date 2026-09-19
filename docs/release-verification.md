# Release verification

`tutorials/vitpose_keypoint_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 4-entry ViTPose snapshot manifest
  (the detector's `DETECTOR_MANIFEST` travels in the same cell) and the inline `PINS` equal to the `pyproject.toml`
  runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision;
  the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned RT-DETR
  detector revision `df939e661d8c52e80608d1ec566561aabd25a4e7` is the one other 40-hex revision the documents cite);
- the profile-specific public-API calls (`stage_missing_files`, `stage_missing_detector_files`, `verify_snapshot`,
  `verify_detector_snapshot`, `VitPoseKeypointPipeline.from_pretrained(weights_dir=..., detector_dir=...)`,
  `fetch_corpus` from the pinned cache path, `read_corpus` (degraded, and the clean counterparts with
  `target_height=None`), `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the box-outside-image refusal probe, `detect_people`, `estimate` with the
  sanity checks and the per-image `evaluation_report` on the synthetic drawing, `box_centre_baseline`,
  `mean_pose_baseline`, `pipe.evaluate` on the frozen model over the small persons and the clean counterparts with
  the floor assertion, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the validation, test and
  clean splits after adaptation with the PCK assertion, `estimate` + `evaluation_report` on the drawing after
  adaptation, the skeleton panels against a freshly loaded frozen base, `pipe.save_artifact`,
  `VitPoseKeypointPipeline.from_artifact` and the joint-parity assertion, and the result fields `weight_file` /
  `weight_format` / `weight_sha256`, the detector identity and the `corpus` block), the eight expected `outputs/`
  paths, the learner-facing statements (Apache-2.0 weights, adaptation with labelled persons, the distant low-quality
  person, 40 px tall, the frozen model at 0.957 clean and 0.598 small, the two box-only baselines, PCK and mean OKS,
  the joint-weighted heatmap MSE, the mean-of-PCK-and-OKS selection, no dispersion estimate, the degradation and
  leakage guidance, the excluded tasks, the snapshot note, the detector's no-`person` finding) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on
  the primary path, a mutable `revision='main'`, direct `from transformers import` / `VitPoseForPoseEstimation` /
  `RTDetrForObjectDetection` / `AutoImageProcessor` / `post_process_*` / `torch.inference_mode(` / `from
  huggingface_hub import` / `urllib.request` / `safetensors` imports / `torch.optim` / `.backward(` /
  `requires_grad` / `pipe._model` / `extractall(` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `huggingface-hub`, `numpy` and
`pillow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`, `test_role_helpers.py`,
`test_notebook_parity.py`; injected runner and photo fetcher, no weights — `tests/test_model_backed.py` is skipped
without the snapshots). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshots `weights/vitpose-base/` and `weights/rtdetr-r50vd/` or the photograph cache `weights/coco-val-persons/`
   (the standalone path writes both manifests itself, stages the missing files from the Hub and fetches the pinned
   photographs from the COCO image host, so none of the three directories may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 8`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 8`,
   `TRAINABLE_BLOCKS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `VitPoseKeypointPipeline`, `verify_snapshot`,
     `verify_detector_snapshot`, `stage_missing_files`, `stage_missing_detector_files`, `validate_inputs`,
     `evaluation_report`, `keypoint_pck`, `pose_metrics`, `box_centre_baseline`, `mean_pose_baseline`,
     `fetch_corpus`, `read_corpus`, `degrade`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`,
     `split_dataset`, `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository
     package;
   - both inline manifests asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` and `stage_missing_detector_files(DETECTOR_WEIGHTS_DIR, allow_download=True)` reporting all
     4 + 4 manifest entries fetched at the immutable revisions on a clean runtime, both verifiers returning their dicts
     (the 360 MB and 172 MB `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR,
     detector_dir=DETECTOR_WEIGHTS_DIR)` loading from the verified directories;
   - Section 4: `fetch_corpus` fetching the 300 pinned photographs with every byte count and SHA-256 matching; 498
     persons degraded to 40-px boxes at JPEG 30 and the 136 clean counterparts built; the seeded split into 290 / 72 /
     136 persons from 180 / 45 / 75 photographs with `check_split_disjoint` reporting no shared photograph and the
     three dataset digests printed; `outputs/…_train.csv` written; the four dataset refusal probes each raising
     `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_PERSONS` 50, both thresholds 0.3,
     `PCK_FRACTION` 0.1, `MIN_LABELLED` 12, `MIN_RECORDS` 8, `MAX_RECORDS` 5000) surfaced; the drawing built;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from
     the box-outside-image probe); `detect_people` finding no `person` at 0.3 (recorded as a finding); `estimate` with
     the drawn box returning 17 joints in `KEYPOINT_NAMES` order with every sanity check `True`,
     `outputs/…_scene_frozen.png` written and the per-image `evaluation_report` verdict `sample-sanity` (the
     inference-only card recorded PCK 1.0 within the 48-px radius and a mean error of 12.8 px);
   - Section 6: the box-centre floor (≈ 0.09 PCK), the mean-pose prior (≈ 0.47 PCK / 0.44 OKS) and the frozen model's
     test scores (≈ 0.60 PCK / 0.57 OKS on the small persons, ≈ 0.96 / 0.94 on the clean counterparts in the RTX
     5070 Ti build record) with the per-category breakdown, and the cell's assertion that the frozen model is above
     the floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 18,376,977 trainable of 89,994,513 parameters, the
     preparation time, and an eight-epoch history with the validation score rising (build record: validation PCK
     0.663 → 0.743, `best_epoch` 7);
   - Section 8: `pipe.evaluate` on the validation, test and clean splits with the six-way comparison on both measures,
     the per-category breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test PCK
     on the small persons exceeds the frozen one — 0.721 versus 0.598 in the build record, OKS 0.565 → 0.684, the
     clean counterparts 0.957 → 0.958; the adapted model also clears the mean-pose prior, reported, not asserted);
   - Section 9: the drawing re-estimated by the adapted model with the `sample-sanity` report,
     `outputs/…_scene_adapted.png` and four skeleton panels under `outputs/…_examples/` written; `pipe.save_artifact`
     writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (42 tensors, 73,512,628 bytes) and
     `VitPoseKeypointPipeline.from_artifact` reloading it with 8/8 identical joint sets on eight test persons (the cell
     asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, both model identities and licences, the
     snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block with the degradation, the
     inference-contract reports, the comparison, the artifact digest, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), both model
   identifiers and immutable revisions, whether the model cache, the weights directories and the photograph cache
   were clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `vitpose_keypoint_colab.ipynb` (`E2E`) | pending | — | — | **PENDING** — no clean-runtime execution of the `E2E` blob has been recorded |
| `vitpose_keypoint_colab.ipynb` (`TASK-INFERENCE`, superseded) | `0c58343` / `840b7f899948` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-vitpose-keypoint` v1) | PASS — 8/8 code cells, 208.7 s, 20 files, 532 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/vitpose_keypoint_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/vitpose_keypoint_colab.ipynb`). Wall times, when recorded, are the sum of per-cell
times reported by the executor and include installs and the model download; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | committed template at the candidate revision (blob differs from the committed one in the recorded generating revision only) | Local WSL harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), both snapshots and the photographs pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), both stagers reported nothing to fetch, both verifiers PASS (4 + 4 files), 300 photographs re-hashed from the cache, 498 persons degraded and split 290 / 72 / 136, four refusal probes raised, the drawing estimated (stage 1 found no `person`, PCK 1.0 from the drawn box), baselines 0.086 / 0.472 PCK, frozen 0.598 / 0.565 on the small persons in 1.8 s and 0.957 / 0.940 on the clean counterparts, 290 crops prepared in 1.5 s, eight epochs 30.3 s (validation PCK 0.663 → 0.710 / 0.723 / 0.732 / 0.735 / 0.734 / 0.744 / 0.741 / 0.743, epoch 6 kept on the mean score), adapted 0.715 / 0.681 (full 0.604 → 0.782, partial 0.627 → 0.712, sparse 0.568 → 0.667), clean counterparts 0.959 / 0.930, the drawing re-estimated at PCK 1.0, four panels written, adapter 73,512,628 B / 42 tensors, reload parity 8/8 with 0 px difference, 8 outputs written | 59.4 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-14 | `0c58343` / `840b7f899948` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-vitpose-keypoint` v1) | Default sample path | 208.7 s | PASS — 8/8 ok code cells, 20 files, 532 MB staged; not evidence for the `E2E` blob |
| 2026-09-14 | notebook blob `3fcfd7c0d6f7` (`TASK-INFERENCE`, superseded; commit `e1cd703`, generated at `ebe84a2`) | Local Windows-venv harness (`run_nb_local.py`, fresh kernel, `CUDA_VISIBLE_DEVICES=-1`), Python 3.12.10, torch 2.14.0+cu130 | Default sample path, all 8 code cells; both snapshots staged from the Hub into an empty `weights/`; 17/17 joints, PCK 1.0 | 129.0 s | PASS — pre-flight only for the earlier notebook |

## Current status

No clean-runtime execution of the `E2E` notebook has been recorded yet; the run is **pending**. Static validation
(`tools/validate_release_assets.py`), the generator parity checks (`--check` OK) and the unit suite passed on the
tutorial source at the candidate revision, and the model-backed regressions (`tests/test_model_backed.py`, 7 tests
including the CUDA path) passed on the build workstation's RTX 5070 Ti, which is necessary but not sufficient. The
registry status remains **Candidate** until a reviewer confirms a recorded run against the notebook blob under review
and an integrator promotes it; promotion is not performed by the builder.

Facts a reviewer should weigh: the degradation is a stated recipe (40-px person boxes, JPEG 30) chosen because the
frozen checkpoint is already at 0.957 PCK on the clean photographs and nothing honest could be gained there; the gain
(0.598 → 0.721 PCK in the build record, past the 0.472 mean-pose prior) is a repair of that specific input shift and
says nothing about blur, viewpoint or crowding; the head alone recovers only a hundredth and four blocks at 32-px
persons overfit, so the recipe is a bounded compromise, not a converged one; the 72-person validation split selects
the epoch on the mean of PCK and OKS and the best epoch was the seventh of eight; the clean counterparts held
(0.957 → 0.958) under this recipe but would need re-measuring under any other; and the heatmap scores remain
uncalibrated after adaptation.
