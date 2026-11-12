# VoltVision AI — Project Status

Reconnaissance date: 2026-11-12. Sections 1–6 are the pre-implementation snapshot; §7 records what was implemented on 2026-11-13. Every statement below was checked against the repository, the local dataset, the artifacts, or a command run on that date. Nothing was carried over from earlier descriptions without checking it.

## 1. Summary

| Area | State |
|---|---|
| Git | One commit (`b16dc31`) that is already pushed to `origin/main` (github.com/ChandanHegde15/voltvision-ai). There is a large set of uncommitted work (schema-v2 artifacts, reference retrieval, research experiments). |
| Tests | `python -m pytest -q --basetemp=.pytest-tmp` gives **17 passed** in about 27 s. Bare `pytest` fails at collection with 7 import errors. |
| Dataset | MVTec AD `cable` is present and complete: 224 train/good images, 150 test images (58 good, 92 defect across 8 categories), 92 masks, no invalid images. Size on disk: 489 MB. |
| Artifacts | Schema v2, created 2026-11-09. They load and pass full validation (2.3 s including the dataset fingerprint). |
| Baseline result | `reports/evaluation.json`: P 1.000, R 0.8261, F1 0.9048, ROC-AUC 0.9936, CM `[[58,0],[16,76]]`, threshold 0.6180, mean latency **193.2 ms**. |
| Experiments | All four experiment JSONs are present (2026-11-09). There are 60 per-image chunks covering the 4 non-identity robustness conditions. |
| Company references | A company name appeared in `README.md` and the local developer notes. **Both have now been removed from the working tree.** It still appears in the pushed commit `b16dc31` (see F1). |

## 2. Environment (verified)

- Python 3.13.2 in `.venv` (1.3 GB).
- torch 2.14.1+cpu, torchvision 0.29.1+cpu, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6, streamlit 1.65.0, Pillow 12.3.0.
- `requirements.txt` uses only `>=` lower bounds. There is no lock file, so these exact versions are not recorded anywhere except this document.
- ResNet18 ImageNet weights are downloaded or cached by torchvision at runtime. `tests/test_preprocessing.py::test_cached_pretrained_embedding_shape_and_values` needs either the cached weights or network access.

## 3. Git status (verified)

Tracked files with uncommitted modifications: `README.md`, `app.py`, `config.py`, `scripts/build_reference_bank.py`, `src/anomaly_detector.py`, `src/artifacts.py`, `tests/test_anomaly_detector.py`, `tests/test_artifacts.py`, plus test-temp files under `.pytest-tmp/`.

Untracked files: `PROJECT_STATUS.md`, `docs/`, `reports/experiments/`, `scripts/{aggregate_robustness_experiment,run_research_experiments,run_robustness_chunk,run_robustness_condition}.py`, `src/research_experiments.py`, `tests/test_research_experiments.py`, `.pytest-tmp/<new dirs>`.

The uncommitted work introduces these changes:
- Artifact schema v2: dataset-relative image IDs, SHA-256 manifests, a dataset fingerprint, and recorded preprocessing settings.
- The closest-reference choice is now made over whole images (mean of per-patch minimum distances) instead of from a single nearest patch.
- `anomaly_score(percentile)` was added for experiments.
- There are four experiment runners.

## 4. Findings

Severity: **High** means a correctness, privacy or reproducibility risk. **Medium** means misleading or inconsistent. **Low** means hygiene.

### High

- **F1. Company name remains in published git history.** Commit `b16dc31`, which is pushed to `origin/main`, contains the original README line naming the company. The working tree is now clean of it. Removing it from history needs a history rewrite and a force-push, which requires your explicit approval (Plan item P1).
- **F2. `.pytest-tmp/` is committed to git.** It holds nine tracked files, and every test run modifies or deletes them, which makes `git status` noisy. It is not in `.gitignore`.
- **F3. README metrics section is stale.** It says "Verified local run (2026-10-13)" with mean latency `84.5 ms`. The current artifacts were rebuilt on 2026-11-09, and the current `evaluation.json` reports `193.2 ms`. The precision, recall, F1, ROC-AUC and confusion matrix still match.
- **F4. Artifacts are tied to one machine path.** `metadata.json` stores the absolute `dataset_path` (`D:\VOLTVISIONAI\...`), and `validate_metadata` requires it to match exactly. Moving or cloning the repo therefore invalidates the artifacts, even when the dataset content is byte-identical. The content fingerprint already guarantees dataset identity, so the path check adds nothing.

### Medium: scientific reporting

- **F5. The baseline split is unfavourable compared with other seeds.** In Exp 3 (seeds 0–9), F1 has mean 0.9387 ± 0.0133 (min 0.9112) and recall has mean 0.8946 ± 0.0319 (min 0.837). The seed-42 baseline (F1 0.9048, recall 0.8261) falls **below the minimum of all ten seeds**. Results should be reported as mean ± std across splits, not as the single baseline figure.
- **F6. The 30k row in Exp 1 is not the baseline.** At cap = 30,000, Exp 1 reports threshold 0.6033 and F1 0.9371, while the baseline has 0.6180 and 0.9048. The difference comes from patch sampling: Exp 1 uses a nested permutation with seed 2027, and the baseline continues the seed-42 RNG. Both are valid, but this needs documenting so the rows are not confused. Exp 2 at the 95th percentile reproduces the baseline exactly, which confirms that the split logic in `build_reference_bank.py` and `deterministic_split` is equivalent.
- **F7. Latency figures measure different things.** `evaluate.py` times feature extraction plus scoring (193 ms). The experiments time nearest-neighbour scoring only (`latency_scope` says so). Exp 4 runs were done in separate processes and their latencies vary from 42 to 82 ms for identical work, so they reflect machine load rather than the input condition.
- **F8. Exp 4 per-category metrics are partly meaningless.** Each defect category contains only positive images, so precision is trivially 1, ROC-AUC is null, and only recall (detection rate) carries information. The `good` entry uses a different, hand-written schema. Recall by category: bent_wire 1.0, combined 1.0, cut_inner_insulation 1.0, missing_cable 1.0, cut_outer_insulation 0.7, poke_insulation 0.7, cable_swap 0.583, missing_wire 0.5. The 16 false negatives are cable_swap 5, missing_wire 5, cut_outer_insulation 3 and poke_insulation 3. Their margins below the threshold range from 0.0003 to 0.047, so several are near-misses.
- **F9. Three scripts write the Exp 4 file with different schemas.** `run_research_experiments.py`, `run_robustness_condition.py` and `aggregate_robustness_experiment.py` all write `exp_04_robustness_failure_analysis.json`. The current file came from the aggregator: it has no `protocol` or `latency_scope` fields, and its identity run was copied from an earlier summary because there are no identity chunks. The per-image chunks also include overlapping ranges (for example `025_035` and `025_050`); the aggregator removes duplicates by image ID.
- **F10. Robustness is mostly stable.** With the baseline threshold left unchanged, F1 ranges from 0.898 to 0.924 across blur 0.5/1.0 and brightness 0.8/1.2, against 0.9048 for identity. There are no false positives under any condition.
- **F11. `evaluation.json` has no provenance.** It records no timestamp, artifact hash, or metadata reference, so it cannot be tied to a specific bank.

### Medium: code

- **F12. Duplicated logic.** The perturbation code appears three times (`run_research_experiments.py`, `run_robustness_chunk.py`, `run_robustness_condition.py`). The `CONDITIONS` tuple is defined four times. The 80/20 split is implemented twice (inline in `build_reference_bank.py` and in `src/research_experiments.deterministic_split`). Each test-label rule (`parent.name == "good"`) is repeated in five places.
- **F13. Full dataset hash on every load.** `load_reference_artifacts` hashes all 374 images and 92 masks each time it runs (about 2 s). This affects the first inspection in the app and every experiment run.
- **F14. Deprecated Streamlit API.** `app.py` uses `use_container_width`, which is deprecated in the installed Streamlit 1.65.0 according to its own docstring.

### Low

- **F15. Unused code.**
  - `config.DEFAULT_THRESHOLD` is never used.
  - `app.py` imports `Path` without using it.
  - `src/evaluation.py` imports `time` without using it.
  - `tests/test_demo_mode.py` imports `DATA_DIR` and `load_reference_artifacts` without using them, and its name promises a demo-mode test that it does not perform.
  - `InspectionResult.closest_index` is no longer used by the app.
- **F16. Test configuration and coverage.** There is no `pytest.ini`, so `pytest` only works when run as `python -m pytest`. Nothing tests the experiment scripts, the aggregator, dataset-fingerprint mismatches, tie-breaking in `closest_reference_image`, `save_results`, or the app.
- **F17. Job-interview framing.** `INTERVIEW_NOTES.md` is framed as interview preparation and mentions PLC/MQTT and "factory data". It names no company, but it does not fit an independent research project.
- **F18. Local inspection log.** `artifacts/inspection_events.csv` holds 5 local inspection rows with uploaded screenshot filenames. It is gitignored and not published.
- **F19. Runner logs contain absolute paths.** `reports/experiments/runner.stdout.log` and `runner.stderr.log` (empty) record absolute local paths. They would be committed with `reports/experiments/`, because `.gitignore` only covers top-level `reports/*.json` and `reports/*.png`.

## 5. File classification

### A. Protected research assets (do not modify or regenerate without approval)
- `data/mvtec_anomaly_detection/cable/**`: licensed dataset, gitignored.
- `artifacts/reference_patches.npy`, `artifacts/reference_paths.json`, `artifacts/metadata.json`: the baseline bank (schema v2).
- `reports/evaluation.json`, `reports/confusion_matrix.png`: the baseline evaluation.
- `reports/experiments/exp_01..exp_04*.json`, `reports/experiments/per_image/*.json`: experiment results.

### B. Active source code
- `config.py`, `app.py`
- `src/feature_extractor.py`, `src/anomaly_detector.py`, `src/artifacts.py`, `src/data_loader.py`, `src/evaluation.py`, `src/edge_simulator.py`, `src/research_experiments.py`, `src/__init__.py`
- `scripts/download_cable_dataset.py`, `inspect_dataset.py`, `build_reference_bank.py`, `evaluate.py`, `run_research_experiments.py`, `run_robustness_chunk.py`, `run_robustness_condition.py`, `aggregate_robustness_experiment.py`
- `tests/*.py`, `.streamlit/config.toml`, `requirements.txt`, `.gitignore`

### C. Documentation needing updates
- `README.md`: stale metrics section (F3); no description of the experiments or their findings (F5–F10).
- `INTERVIEW_NOTES.md`: reframe or remove (F17).
- Local developer notes (not tracked): update after the implementation changes.
- `data/README.md`: accurate, no change needed.

### D. Possibly obsolete
- `.pytest-tmp/**` (tracked; F2).
- `reports/experiments/runner.stdout.log`, `runner.stderr.log` (F19).
- `config.DEFAULT_THRESHOLD` (F15).
- `scripts/run_robustness_condition.py`: superseded by the chunk and aggregate flow, and writes a divergent schema (F9).

### E. Needs further investigation
- `artifacts/inspection_events.csv`: keep it as local history or reset it?
- Whether `origin/main` has other clones or forks that a force-push would affect (F1).
- Whether the overlapping per-image chunks hold identical scores for the same image (they should, since the computation is deterministic).
- Whether the remote MVTec mirror `foersben/mvtec-ad` (used by `download_cable_dataset.py`) remains available, and whether its licence terms match the official MVTec terms.

## 6. Changes made during this reconnaissance

1. `README.md` line 3: the company disclaimer sentence was replaced with "It is an independent project, uses only the public MVTec AD dataset, and is not deployed in production."
2. Local developer notes: the company mention was replaced with an instruction not to add company names or branding.

No other files were modified. No artifacts were rebuilt and no dependencies were installed.

## 7. Implementation outcome (2026-11-13)

All plan items were approved ("approve all") and implemented as separate commits. The full suite is now **43 passed** under both `pytest` and `python -m pytest`.

| Item | Result | Verification |
|---|---|---|
| 0 | The uncommitted work was committed as one baseline commit. The two runner logs and the new `.pytest-tmp` files were left out because they contain local paths or scratch output. | `git log` |
| P1 | The root commit was recreated with the neutral README sentence, keeping the original author and date. Force-pushed to `origin/main` with a lease on the old SHA. Backup ref deleted, reflog expired, gc run. | `git log --all -p` has 0 matches for the former company name; the old SHA no longer exists locally; `origin/main` equals local `main`. |
| P2 | `.pytest-tmp/` was untracked. `.gitignore` now covers `.pytest-tmp/` and runner logs. `pytest.ini` was added. | Bare `pytest` passes; `git status` stays clean after a test run. |
| P3 | `dataset_path` is now informational only. | New tests: a relocated identical dataset loads; changed content is rejected. The real artifacts still load. |
| P4 | Removed unused code. `use_container_width` was replaced with `width="stretch"`. The demo-mode test now exercises the missing-artifact path. | Headless `AppTest` render: no exceptions and no deprecation warnings. Two real test images inspected through the engine path; a known false negative reproduced its recorded score exactly. The upload button itself was not clicked (AppTest cannot drive file uploads). |
| P5 | `src/perturbations.py` and `data_loader.test_label` were added. `build_reference_bank.py` uses `deterministic_split`. | Tests pin the old pixel output and split/RNG behaviour. **A scratch rebuild of the bank was byte-identical** to `artifacts/` (same SHA-256 for the patches and paths files, same threshold). `artifacts/` hashes were checked before and after. |
| P6 | One Exp 4 schema with per-category detection rates. `run_robustness_condition.py` was removed. | The candidate was compared with the recorded file: all 5 runs and all 16 false negatives are identical within 1e-12, and detection rates equal the old recall values. The candidate then replaced the original. Overlapping chunks contain identical scores (this resolves item E3). |
| P7 | `evaluate.py` records provenance and accepts `--output-dir`. | A scratch re-run reproduced precision, recall, F1, ROC-AUC, the confusion matrix and the threshold exactly. |
| P8 | Added 26 tests (17 → 43). | Suite passes. |
| P9 | `requirements-lock.txt` from `pip freeze`. No local paths in it. | — |
| P10 | README results and `docs/EXPERIMENTS.md` are generated by `scripts/build_docs.py`. `INTERVIEW_NOTES.md` became `docs/PROJECT_NOTES.md`, with the interview framing and company-specific wording removed. Local developer notes updated. | `build_docs.py --check` passes. No company names in the repository. |

Deviations and new findings:

- **The recorded `reports/evaluation.json` was not replaced.** The plan made replacing it a separate step. The scratch re-run matched every metric, but its latency was 326.7 ms (median 319.2 ms) against the recorded 193.2 ms. Latency on this machine varies a lot with load. The recorded file therefore still has no provenance fields; running `scripts\evaluate.py` without `--output-dir` would add them.
- **Cause of the latency increase (F3).** A profile on 15 test images gave median times of: feature extraction 105 ms, 1-NN scoring 53 ms, closest-reference retrieval **127 ms**. The image-level retrieval added in the schema-v2 work is the most expensive step. That explains why end-to-end latency rose from the 84.5 ms reported earlier. Optimising it is listed as future work and was not done here, because it was not in the plan.
- **Exp 4 duplicate rows: last chunk wins.** Where chunks overlap, the aggregator keeps the row from the chunk that sorts last by name, matching the original behaviour. Only latency can differ between duplicates, and keeping the original rule reproduces the recorded blur_0_5 latency exactly.
- GitHub may keep cached views of the old commit SHA. Removing those requires a request to GitHub support, if needed.
