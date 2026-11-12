# VoltVision AI — Implementation Plan

Status: **awaiting approval**. This plan is based on the findings in `PROJECT_STATUS.md` (F-numbers below refer to those findings).

## Ground rules for every item

- **Scientific method stays the same:** ResNet18 `layer3` features, L2-normalised patches, Euclidean 1-NN, image score at the 95th percentile, threshold at the 99th percentile of the seed-42 normal validation split, and the `>=` decision rule.
- **Protected assets (class A) are never overwritten.** Any regeneration goes to a temporary directory and is compared with the original. Only after that comparison, and with approval, may the original be replaced.
- **Existing uncommitted work is kept.** Before implementation starts, the current working tree is committed as its own commit ("Schema-v2 artifacts and research experiments") so that later changes are easy to review against it. That commit itself needs your approval.
- After every item, the full suite must pass: `.\.venv\Scripts\python.exe -m pytest -q`.

## Execution order

| Order | Item | Depends on | Touches protected assets? |
|---|---|---|---|
| 0 | Commit the current working tree | none | No |
| 1 | P1: remove company name from git history | 0 | No (force-push) |
| 2 | P2: repository and test hygiene | 0 | No |
| 3 | P3: portable artifact validation | 2 | No (validation logic only) |
| 4 | P4: remove unused code and deprecated calls | 2 | No |
| 5 | P5: consolidate duplicated logic | 2, 3 | Read-only check |
| 6 | P6: single writer for Experiment 4 | 5 | Re-aggregates exp_04 into a temp file first |
| 7 | P7: evaluation provenance | 3 | Optional re-run into a temp file |
| 8 | P8: test coverage | 3–7 | No |
| 9 | P9: reproducibility record | none | No |
| 10 | P10: documentation | all | No |
| (deferred) | P11: speed up fingerprint checks | 3 | No |

---

## P1. Remove the company name from published git history (F1)

- **Files:** git history (commit `b16dc31`, `README.md` line 3), remote `origin/main`.
- **Why:** The project must carry no company history. The working tree is already clean, but the pushed initial commit still contains the former company name.
- **Current:** One commit, pushed to `github.com/ChandanHegde15/voltvision-ai`, whose README names the company.
- **Proposed:**
  1. Make a backup ref: `git branch backup/pre-rewrite b16dc31`.
  2. Rewrite history so the README in every commit contains the neutral sentence. With only one or two commits, this can be done by recreating the root commit from the cleaned tree; `git filter-repo --replace-text` is the alternative.
  3. Check with `git log --all -p | grep -ci <former-company-name>`, which must output `0` (except in the local backup ref, which is deleted after confirmation).
  4. `git push --force-with-lease origin main`.
  5. Delete the backup branch, then expire the reflog and run gc so the old objects are removed locally.
- **Dependencies:** item 0. You must explicitly approve the force-push.
- **Regression risk:** Medium. Existing clones or forks diverge. GitHub may keep cached views of the old commit SHA until GitHub support purges them.
- **Tests:** The full suite before and after (the content is unchanged apart from one sentence).
- **Acceptance:** `git log --all -p` contains no company name. `origin/main` matches local `main`. The suite passes.

## P2. Repository and test hygiene (F2, F16, F19)

- **Files:** `.gitignore`, new `pytest.ini`, `.pytest-tmp/**` (removed from the index only), `reports/experiments/runner.*.log`.
- **Why:** Test output is committed and changes on every run. Bare `pytest` fails. The runner logs contain absolute local paths.
- **Current:** Nine tracked files under `.pytest-tmp/`. No pytest configuration. `.gitignore` does not cover `.pytest-tmp/` or `*.log`.
- **Proposed:**
  - Add `.pytest-tmp/` and `reports/experiments/*.log` to `.gitignore`.
  - Run `git rm -r --cached .pytest-tmp`. The local files stay on disk.
  - Add `pytest.ini`:
    ```ini
    [pytest]
    testpaths = tests
    pythonpath = .
    addopts = --basetemp=.pytest-tmp
    ```
  - Keep the two runner logs on disk but untracked.
- **Dependencies:** item 0.
- **Regression risk:** Low.
- **Tests:** Both `pytest -q` and `python -m pytest -q` pass with 17 or more tests.
- **Acceptance:** `git status` stays clean after a test run. Bare `pytest` works.

## P3. Portable artifact validation (F4)

- **Files:** `src/artifacts.py` (`validate_metadata`), `tests/test_artifacts.py`.
- **Why:** The absolute `dataset_path` must match exactly, so cloning or moving the repo invalidates byte-identical data. The SHA-256 dataset fingerprint already proves the data is the same.
- **Current:** `expected["dataset_path"] = str(Path(dataset_root).resolve())` is part of the configuration-mismatch check.
- **Proposed:** Remove `dataset_path` from the equality check, but keep it in `REQUIRED_METADATA` and in the stored metadata as information only. All other checks stay: version, seed, cap, model, preprocessing, metric, manifests, and fingerprint. The schema is unchanged, so `ARTIFACT_FORMAT_VERSION` stays at 2 and the existing artifacts stay valid without a rebuild.
- **Dependencies:** P2 (so tests run cleanly).
- **Regression risk:** Low. Integrity still rests on the content hashes.
- **Tests:**
  - New: copy the dataset fixture to a different directory; loading still succeeds.
  - New: changing one byte of an image causes the fingerprint check to fail.
  - Existing tests still pass.
- **Acceptance:** The real `artifacts/` load against `data/.../cable` exactly as before. A relocated copy of the fixture loads. Tampered content is rejected.

## P4. Remove unused code and deprecated calls (F14, F15)

- **Files:** `config.py`, `app.py`, `src/evaluation.py`, `tests/test_demo_mode.py`.
- **Why:** Dead code and deprecated Streamlit calls.
- **Current:**
  - `DEFAULT_THRESHOLD` is never used.
  - `app.py` imports `Path` without using it and calls `use_container_width=True` five times. This is deprecated in Streamlit 1.65.
  - `evaluation.py` imports `time` without using it.
  - `test_demo_mode.py` has two unused imports and does not test demo mode.
- **Proposed:**
  - Delete `DEFAULT_THRESHOLD` and the unused imports.
  - Replace `use_container_width=True` with `width="stretch"`. First confirm that `st.image`, `st.button`, `st.pyplot` and `st.dataframe` accept it in 1.65.0. If any does not, keep `use_container_width` for that call.
  - Make `test_demo_mode.py` assert that `load_reference_artifacts(tmp_path)` raises `FileNotFoundError` with the rebuild hint. This is the path the app depends on in demo mode.
  - Keep `InspectionResult.closest_index`, because it is a public field and still used as the representative patch index.
- **Dependencies:** P2.
- **Regression risk:** Low. The UI must be checked by hand.
- **Tests:** The suite passes. Manual check: `streamlit run app.py` starts without deprecation warnings; inspecting one test image shows the score, heatmap and reference image; the history tab renders.
- **Acceptance:** No deprecation warnings in the Streamlit log. The UI looks and behaves the same.

## P5. Consolidate duplicated logic (F12)

- **Files:**
  - New `src/perturbations.py`.
  - `src/data_loader.py` (add `test_label(path)`).
  - `scripts/build_reference_bank.py` (use `deterministic_split`).
  - `scripts/run_research_experiments.py` and `scripts/run_robustness_chunk.py` (import the shared code).
- **Why:** The perturbations, `CONDITIONS` and the split are defined in several places. If one copy is changed and another is not, the experiments stop being comparable without anyone noticing.
- **Current:** Three copies of the perturbation code, four copies of `CONDITIONS`, two implementations of the split, and five inline label rules.
- **Proposed:**
  - `src/perturbations.py` exports `CONDITIONS` and `apply_condition(image, condition) -> PIL.Image`, with the radii and factors exactly as they are now.
  - Add `data_loader.test_label(path) -> int`.
  - `build_reference_bank.py` calls `deterministic_split(paths, SEED)` and keeps using the returned RNG for the patch cap, so RNG consumption is unchanged.
- **Dependencies:** P2 and P3.
- **Regression risk:** **Medium.** A change in RNG consumption would silently change the bank.
- **Tests:**
  - New unit test: for 224 synthetic paths, the old inline split and `deterministic_split` produce the same index sets for seed 42. Add this test before the refactor so it pins the current behaviour.
  - New unit test: each condition gives the same pixels as the previous inline code on a fixed image.
  - **Read-only equivalence check (needs approval, about 1 minute on CPU):** rebuild the bank into a temporary directory. The SHA-256 of `reference_patches.npy` and `reference_paths.json`, and the threshold, must match `artifacts/` exactly. Then delete the temporary directory. `artifacts/` is not touched.
- **Acceptance:** Byte-identical bank and identical threshold in the temporary rebuild. All tests pass.

## P6. One writer and correct per-category reporting for Experiment 4 (F8, F9)

- **Files:**
  - `scripts/aggregate_robustness_experiment.py`.
  - `scripts/run_robustness_condition.py` (delete).
  - `scripts/run_research_experiments.py` (Exp 4 section).
  - `src/research_experiments.py` (new `per_category_detection`).
- **Why:**
  - Three scripts write the same file with different schemas.
  - Precision and AUC for each defect category are meaningless because those categories contain only defects.
  - The current file is missing the `protocol` and `latency_scope` fields.
- **Current:** The aggregator output has no protocol fields, uses a hand-written `good` entry, and gives full binary metrics for each category.
- **Proposed:**
  - Use one schema, created by a helper in `src/research_experiments.py` that both remaining writers call. It contains `protocol`, `latency_scope`, `baseline_threshold`, `runs`, `per_category_original`, `false_negatives_original` and `limitations`.
  - For each defect category, report `{images, detected, detection_rate}`. For `good`, report `{images, false_positives, normal_acceptance_rate}`.
  - Remove `run_robustness_condition.py`. The chunk and aggregate flow replaces it.
  - Re-aggregation only reads existing per-image chunks and the existing identity summary. **It does not run any model.**
- **Dependencies:** P5.
- **Regression risk:** Low to medium. The protected JSON is rewritten, so the result goes to `exp_04....candidate.json` first.
- **Tests:**
  - Unit tests for `per_category_detection`.
  - An aggregator test on synthetic chunks, covering overlapping ranges and a missing chunk.
  - A comparison script asserting that every `runs[*]` metric in the candidate equals the current file to within 1e-12. Only the per-category fields may differ.
- **Acceptance:** All five runs have identical overall metrics. The detection rates per category match the current recall values (bent_wire 1.0 … missing_wire 0.5). After approval the candidate replaces the original, and the original is kept in git history.

## P7. Provenance for evaluation results (F7, F11)

- **Files:** `scripts/evaluate.py`, `src/evaluation.py`.
- **Why:** `evaluation.json` cannot be traced to a specific bank, and its latency measures something different from the experiments' latency.
- **Current:** It records metrics, the threshold and `average_latency_ms` (end-to-end) only.
- **Proposed:**
  - Add `created_at_utc`, `artifact_created_at_utc`, `reference_image_list_hash`, `dataset_fingerprint`, `latency_scope: "image load + preprocessing + feature extraction + scoring"`, `median_latency_ms`, plus the torch and torchvision versions.
  - Keep the metric computation unchanged.
- **Dependencies:** P3.
- **Regression risk:** Low.
- **Tests:** A unit test that `save_results` round-trips the new fields. Existing evaluation tests still pass.
- **Acceptance:** An optional re-run, written to a temporary path, gives the same P, R, F1, AUC and confusion matrix as the current `reports/evaluation.json`. Replacing the original is a separate approval step.

## P8. Test coverage (F16)

- **Files:** `tests/test_anomaly_detector.py`, `tests/test_artifacts.py`, `tests/test_evaluation.py`, new `tests/test_perturbations.py`, new `tests/test_aggregation.py`.
- **Why:** Paths that matter for correctness have no tests.
- **Proposed new tests:**
  - Tie-breaking in `closest_reference_image` (an exact tie must resolve to the lexicographically smaller ID).
  - `score` and `anomaly_score` agree at the 95th percentile.
  - Detection of a dataset-fingerprint mismatch.
  - Rejection of a non-finite patch array.
  - Rejection of an ownership mapping that refers to an unknown image.
  - `save_results` writes a PNG.
  - P5 and P6 tests as listed above.
- **Dependencies:** P3 to P7.
- **Regression risk:** None (tests only).
- **Acceptance:** All tests pass. Apart from the existing weight-cache test, none needs the network or the real dataset.

## P9. Reproducibility record (F14 environment)

- **Files:** New `requirements-lock.txt`, and `README.md` (setup section).
- **Why:** With `>=` bounds only, the exact environment that produced the results is not recorded.
- **Proposed:** Write `.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt`. This reads the environment and installs nothing. Document it as "the exact environment used for the reported results". Keep `requirements.txt` as the flexible install.
- **Dependencies:** none.
- **Regression risk:** None.
- **Acceptance:** The file lists torch 2.14.1+cpu, torchvision 0.29.1+cpu, and the other packages recorded in `PROJECT_STATUS.md` §2.

## P10. Documentation (F3, F5–F10, F17)

- **Files:** `README.md`, `INTERVIEW_NOTES.md` (to become `docs/PROJECT_NOTES.md`), new `docs/EXPERIMENTS.md`.
- **Why:** The README results are stale. The experiment findings are not documented. The interview framing does not fit an independent project.
- **Proposed:**
  - **README:**
    - Replace "Verified local run (2026-10-13)" with values read programmatically from `reports/evaluation.json` and the artifact metadata (dated 2026-11-09). This includes latency 193.2 ms, labelled end-to-end.
    - Add a short "Robustness across splits" line: F1 0.939 ± 0.013 and recall 0.895 ± 0.032 over 10 seeds, with the baseline at the low end.
    - Link to `docs/EXPERIMENTS.md`.
  - **docs/EXPERIMENTS.md:**
    - The protocol for each experiment, with tables generated from the JSON files.
    - The caveats: F6 (the 30k sampling seed), F7 (latency scopes) and F8 (per-category detection).
    - Must contain no hand-typed numbers. A small script generates the tables, and the docs are checked against the JSON.
  - **INTERVIEW_NOTES.md:** Move to `docs/PROJECT_NOTES.md` with the title "Project notes". Remove the interview framing and the "30-second introduction" wording. Keep the technical explanations and FAQ. Replace "PLC/MQTT integration, model validation on representative factory data" with generic future work such as "integration with inspection hardware, validation on representative production imagery".
  - Final check: no company names anywhere (`grep -rniI <former-company-name> .` returns nothing outside `.venv`).
- **Dependencies:** all other items, so the documentation describes the final state.
- **Regression risk:** None to code. The risk is in the numbers, which is why they are generated rather than typed.
- **Acceptance:** Every number in the docs matches its source JSON. No company references. Links work.

## P11 (deferred). Faster fingerprint validation (F13)

Hashing the whole dataset on each load takes about 2 s. A cache keyed on file size and modification time would make it faster, but it would weaken the content-integrity guarantee that the schema-v2 work introduced on purpose. **Recommendation: do not change this now.** The cost only affects the first inspection after the app starts (`st.cache_resource`) and the start of each script.

---

## Out of scope (would change the scientific method; needs a separate pre-registered proposal)

- Changing the backbone, layer, aggregation percentile, threshold percentile, bank cap or sampling seed.
- Using a different split than seed 42 for the baseline, even though F5 shows the baseline is a weak split. Report the multi-seed mean instead.
- Upsampling or overlaying the heatmap on the image. This is a UI change, but it could imply pixel-level localisation accuracy that has not been evaluated.
- Pixel-level metrics using the 92 ground-truth masks.

## Approval checklist

Please confirm each item, or reply "approve all":

- [ ] 0. Commit the current working tree as-is.
- [ ] P1. Rewrite history and **force-push** to `origin/main`.
- [ ] P2 to P4. Hygiene, portability and cleanup.
- [ ] P5. Includes a read-only temporary rebuild of the bank for the equivalence check (about 1 min CPU).
- [ ] P6. Re-aggregate Exp 4 (no model run). The candidate replaces the original after the check.
- [ ] P7. Optional evaluation re-run into a temporary path.
- [ ] P8 to P10. Tests, lock file and documentation.
