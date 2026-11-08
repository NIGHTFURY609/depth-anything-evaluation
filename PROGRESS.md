# Progress

Where the project stands, what has been done and what is still open. The reasoning behind each choice is in [DECISIONS.md](DECISIONS.md). Update both at the end of every phase.

Last updated: 2026-11-08.

## Status

| Phase (research plan section 8) | State | Result |
| --- | --- | --- |
| 0: environment, checks, novelty search | Done | Everything fits in 8 GB; no near-duplicate found in a limited search |
| 1: loader, projection, split, ground truth | Done, gate passed for gross geometry | Ground truth cached for all 13,556 frames; a calibration offset remains and is disclosed |
| 2: zero-shot baselines, `eval.py` | Done | Protocol frozen at `6b486bc`, test evaluated once. Tables 1, 3 and 4 (zero-shot) in `results/phase2/test/tables.md` |
| 3: fine-tuning | **In progress: the author is running the training queue** (started 2026-11-08) | Code, teacher cache and review done; a scale-collapse bug was found and fixed; the overfit check missed its bar (0.049 against 0.03). See the Phase 3 log |
| 4 onward | Not started | |

Nothing has been pushed. Commits are on local `main`.

## Where things are

| What | Where |
| --- | --- |
| Code, config, results, documents | `N:\coding\projects\DA-paper`, or `/mnt/n/coding/projects/DA-paper` in WSL |
| Decisions and their evidence | [DECISIONS.md](DECISIONS.md) |
| Phase 0 implementation plan, as approved; its checkboxes were never ticked | [docs/plans/2026-11-02-phase0-plan.md](docs/plans/2026-11-02-phase0-plan.md) |
| One-off diagnostics and their saved output | [scripts/diagnostics/](scripts/diagnostics/README.md), [results/diagnostics/](results/diagnostics/) |
| Research plan v2 (git-ignored, so it stays private) | Repo root |
| Virtual environment | WSL `Ubuntu`, `~/envs/da3ft` (Python 3.11, torch 2.11.0+cu128) |
| DA3 upstream | `~/src/Depth-Anything-3`, commit `3d835ec` |
| RELLIS data | `~/da3ft-data/rellis` (`Rellis-3D` 45 GB, `raw` 15 GB, `calib` 2.2 MB) |
| Ground truth caches | `~/da3ft-data/cache/gt` (582 MB), `~/da3ft-data/cache/gt_sensitivity` (574 MB); 13,556 frames each |
| Prediction cache | `~/da3ft-data/cache/pred/<model>/<seq>/`, val and test for the four zero-shot models (about 3 GB) |
| Paper material: every number so far, figure data, wording and limitations | [docs/paper-results.md](docs/paper-results.md) |
| Phase 2 results | `results/phase2/{val,test}/`: `tables.md`, `summary_*.csv`, `meta.json` (committed); `*_per_image.csv.gz` (git-ignored, rebuilt by `scripts/eval.py`) |
| Overlays (git-ignored, derived from RELLIS images) | `results/overlays/`: Phase 1 overlays and crops, low-agreement panels, edge-band and val grass panels |

## Rerunning everything

Run in WSL `Ubuntu` from the repo root. RELLIS is CC BY-NC-SA 3.0, so data and weights are never committed.

```bash
bash scripts/setup_wsl.sh                  # uv, venv, torch, DA3 at the pinned commit
source ~/envs/da3ft/bin/activate
bash scripts/download_rellis.sh            # the images zip is extracted by hand from the repo root copy
python scripts/verify_rellis.py            # exit 0 expected: 13,556 frames matched
python phase0_check.py                     # results/phase0.csv
export OMP_NUM_THREADS=1                   # GT build takes 3m18s with this, 6m41s without
python scripts/build_gt.py                 # official calibration, cache/gt
python scripts/phase1_gate.py              # overlays and results/phase1_*.csv
python scripts/refine_calibration.py       # results/phase1_refine.csv, calibration_estimates.yaml
python scripts/build_gt.py --sensitivity   # cache/gt_sensitivity, reporting only
python scripts/infer.py --split val        # 2.5 min; also --split test (writes frames_<split>.csv)
python scripts/eval.py --split val         # 2-3 min; --split test needs eval.frozen in a clean commit (6.7 min)
python scripts/grass_look.py               # val-only grass panels
python -m pytest -q                        # from the repo root: 150 passed on 2026-11-03
```

`setup_wsl.sh` installs current packages rather than installing from `requirements.lock`, so an exact rebuild has to install the lock by hand.

## Phase 0 log

- Environment built with uv in WSL `Ubuntu`. `Ubuntu-26.04` and `.wslconfig` were left alone.
- Trainable DA3 path built around the inner `DepthAnything3Net`. It matches the API within 1e-4, including on a real frame from each sequence.
- Fine-tuning modes built: head, LoRA (27,648 x rank parameters) and full.
- Peak training VRAM at batch 4: head 1214 MB, LoRA 1844 MB, full 2222 MB. No OOM; checkpointing is available but not needed.
- All data downloaded and matched by frame number: 2847, 2319, 4147, 2184 and 2059 frames across 00000-00004.
- Novelty search: three web searches, no near-duplicate. Repeat before posting.
- Independent code review done. It found two real bugs (below).

## Phase 1 log

- Projection built from the official RELLIS example. Controls confirm the transform direction: median agreement is 0.78-0.91 with the inverse and 0.09-0.28 without.
- The correct scan pairing is supported: a scan from 1 s later lowers agreement in every sequence. Distortion on or off is not distinguished by this metric (`results/phase1_controls.csv`).
- Split fixed: test 00002, val 00001, train 00000, 00003, 00004.
- Ground truth cached at 504x322 for every frame. No frames dropped.
- Calibration offset found. The author chose to refine and compare, with official calibration as the fallback. The pre-fixed rule accepted two of five sequences, so the fallback applies. Official calibration is primary; all five estimates build the sensitivity cache.
- Grass versus bush found as the largest label disagreement. LiDAR labels stay primary, with a merged vegetation group and a both-sources check.
- Close-out (2026-11-03): the last 8 sampled overlays viewed. Every frame below 0.3 agreement explained as grass versus bush with intact geometry. No frames dropped. Sequence ends compared: only val 00001 and train 00003 are continuous.

### Overlay gate: what was actually checked

- Viewed: all 20 sampled overlays (the last 8 on 2026-11-03), the worst-case overlays, full-resolution crops, and label panels for every frame still below 0.3 agreement.
- Trees, ground, people and barriers look right at the scale of the 504x322 overlays, except for the offset. The barrier crop (00001, frame 1545) shows it plainly.
- So the gate is passed for gross geometry and not passed for a few pixels of vertical alignment. No frames were dropped.

## Phase 2 log

- **Offroad-Nav edge mask read from code.** Otsu per image (not a fixed threshold), a 5 px kernel diameter, and Canny as the shipped default.
- **DA2 moved into `src/models.py`.** `src/infer.py` caches predictions at batch 1 with model ID and git hash.
- **`src/eval.py`, `src/report.py` and `scripts/eval.py` built test-first.** The design was critiqued by a planning agent before coding and reviewed independently before the freeze. The review found no arithmetic bug but showed the faithful Sobel mask removes most valid pixels, so the primary filtering comparison became ranked curves at fixed budgets, as the author chose.
- **Thresholds set from val ground truth:** depth range, edge band k and τ, subset minimum. The bootstrap block came from val autocorrelation. All are logged in DECISIONS.md.
- **First look at grass errors, on val only:** far grass predicted too close.
- **Protocol frozen** at `6b486bc` (`eval.frozen: true`, clean tree, test frame list committed). Test was evaluated once, from that commit: 830 frames, none skipped.

### Phase 2 results on test 00002 (zero-shot)

Official GT, LiDAR labels, primary alignment (DA3 median scale, DA2 least-squares scale and shift), AbsRel with 95% block-bootstrap intervals. Full tables are in `results/phase2/test/tables.md`.

| Model | all | grass | grass 20-80 m | edge_depth |
| --- | --- | --- | --- | --- |
| DA3-S | 0.155 [0.133, 0.178] | 0.111 [0.094, 0.137] | 0.313 [0.274, 0.360] | 0.370 [0.281, 0.489] |
| DA3-B | 0.142 [0.123, 0.165] | 0.098 [0.080, 0.124] | 0.238 [0.201, 0.288] | 0.400 [0.302, 0.538] |
| DA2-S | 0.165 [0.146, 0.186] | 0.102 [0.092, 0.113] | 0.327 [0.277, 0.376] | 0.435 [0.357, 0.530] |
| DA2-B | 0.169 [0.148, 0.194] | 0.100 [0.090, 0.110] | 0.312 [0.267, 0.358] | 0.454 [0.363, 0.556] |

- **DA3 beats DA2 at equal alignment.** With LS on both sides, DA2-S is worse than DA3-S by 0.030 [0.021, 0.039] AbsRel overall and by 0.108 [0.076, 0.146] on `edge_depth`.
- **DA3-B against DA3-S (median):** slightly better overall (-0.012) and on grass (-0.014), but worse on `edge_depth` (+0.030 [0.007, 0.058]).
- **Far grass is hard for every model** (AbsRel 0.24-0.33 at 20-80 m), as on val.
- **Test is harder than val:** DA3-S all is 0.155 against 0.113.
- **Confidence (DA3-S, median):** AUROC 0.882 [0.853, 0.901], AUSE 0.033, AURG 0.064.
- **Filtering (DA3-S):** removing the lowest-confidence 20% of valid pixels cuts AbsRel from 0.155 to 0.100. Sobel-ranked removal gives 0.151, a paired difference of -0.052 [-0.065, -0.039], and confidence wins at every budget. Sobel-ranked removal at 50% is worse than none (0.173).
- **The faithful Sobel mask** removes 78-79% of valid pixels for DA3 and 51-57% for DA2.
- **Calibration sensitivity:**
  - Overall and grass AbsRel barely move.
  - For DA3, edge bands are about 6-8% lower with the corrected calibration (DA3-S `edge` 0.343 to 0.322) and far grass is about 13% higher (DA3-S 0.313 to 0.354).
  - DA2's numbers barely move (DA2-S `edge` 0.400 to 0.399), so the calibration choice also shifts the DA2/DA3 gap at edges.
  - So edge-band and far-range numbers depend on calibration, which is the careful wording the open item anticipated.

### What has been seen from test sequence 00002

Kept so the claim of a clean test set can be checked.

- Class counts, label agreement and overlays, none of which involve a depth model.
- Its calibration estimate (+0.88, +4.36), fitted on labels only and used only for sensitivity reporting.
- From the zero-shot DA3 diagnostic: the location of the best shift only, printed without error values. See `results/diagnostics/depthalign.txt`. These locations did not set any threshold or decision.
- Until the freeze at `6b486bc`, no depth accuracy for 00002 had been computed or viewed for any model being compared.
- After the freeze, the zero-shot test results (`results/phase2/test/`) were computed once and read, as summarised above. No test frame was drawn; the grass panels are val only. From here on, Phase 3 choices must rest on val alone.

## Bugs found and fixed

The first three fixes went test-first, red then green.

| Problem | How it was found | Fix |
| --- | --- | --- |
| `scratch.output_conv1` was frozen as unused, but it is used | Direct gradient check | Kept trainable; `test_no_frozen_head_param_is_used_for_depth` |
| The download script could delete a valid zip if the checker itself crashed | Phase 0 review | Deletes only on checker exit code 3 |
| DA2 ran in bf16, which quantised its baselines by roughly 0.4% | Phase 0 review | DA2 runs in float32; Phase 0 rerun |
| `addict` is undeclared upstream | Import failure | Added to setup and lock |
| GT build oversubscribed threads (6m41s) | Timing | `cv2.setNumThreads(0)`, `OMP_NUM_THREADS=1` (3m18s) |
| Four mistakes in the tests themselves: a point outside the FOV, wrong occlusion spacing, a tie in the shift test, and a rotation tolerance too tight for the real 1+tan²θ effect | Test failures | Test corrections only |
| First commit had a hand-set author | Checked after commit | Amended to the user's git identity |

## Open items

### Must settle before the paper

- **Cause of the calibration offset is unknown.** Per-frame pitch or sync was not investigated. Phase 2 shows DA3's edge-band (about 7%) and far-grass (about 13%) AbsRel shift between the two caches, while DA2's barely move, so those numbers need careful wording.
- **Phase 2 eval was reviewed, not proven.** The review re-derived the summaries from the per-image rows, but the final pre-release review should cover `scripts/eval.py`'s table assembly again.
- **Phase 1 code has had no independent review.** The final pre-release review must cover it.
- **Repeat the novelty search** before posting, and check the three arXiv and CVPR references.
- **Say "unseen sequences", not "unseen place".** All sequences come from one site.
- **Val and train touch at one boundary.** 00001 ends 53 s before 00003 starts, at the same spot. State it as a limitation of model selection; the test sequence is isolated.

### Small code issues, deferred

- `has_conf` cannot truly detect missing confidence. Fix before any DA3Mono-Large check.
- GPU memory is only freed by the next run's cleanup.
- The OOM handling test triggers only at model load.
- `setup_wsl.sh` does not install from the lock.
- The LoRA regex is duplicated in `src/models.py` and `config.yaml`.
- pytest must run from the repo root.
- `verify_rellis.py` does not check that calibration files exist.
- `scripts/diagnostics/` has no tests and has not been rerun from its new location.
- Test predictions are stamped `01eda9a…-dirty`; the only dirty file was the then-untracked `frames_test.csv`. `infer.py` now reads the hash first.
- An external tool keeps stripping `pytest.ini`'s final newline; the stripped version is committed so the tree stays clean.

### Disk

- **N:** 57 GB free, against the plan's 50 GB margin. Once the author is happy with Phase 1, the 11.6 GB images zip in the repo root and the 15 GB `raw/` folder can be deleted. Neither has been deleted.
- **C:** 10.3 GB free, down from 14.9 GB at the start, so the Phase 0 goal of leaving C: unchanged was not met. It dipped to 6.8 GB during the work. The WSL swap file is only 0.04 GB and no large new files turned up in the folders scanned. The Windows pagefile is suspected but unproven.

## Phase 3 log

Settings, the review, the bug and the sanity outcomes are in DECISIONS.md (2026-11-06 entries).

- **Teacher cached:** `pred/da3_small/` for all 7,090 train frames, 27 min, 5.7 GB.
- **Distillation space:** linear, by the pre-fixed rule (far-10% share 0.187 < 0.3).
- **Code:** `src/losses.py`, `src/train_data.py`, `src/val.py`, `src/train.py`, `scripts/train.py`, `scripts/phase3_runs.py` and `scripts/phase3_summary.py`. Also `scripts/infer.py --checkpoint/--name` and `scripts/eval.py --out`. 201 tests pass (2026-11-06).
- **Validator check:** on the untrained model it reproduces Phase 2's zero-shot val numbers exactly: AbsRel all 0.113095, grass 0.075511, vegetation 0.115414, `edge_depth` 0.372562, grass 20-80 m 0.297053, with the same image counts.
- **Independent review:** six real bugs, all fixed with tests.
- **Overfit check:**
  - It found a real bug: the detached scale made depth collapse to NaN. Fixed.
  - It then missed its bar, 0.049 against 0.03, explained by the confidence-weighted loss form. Overlays OK.
  - Not rerun or relaxed.
- **One-epoch check:** not run separately. Epoch 1 of the first S1 run stands in for it (bar: val AbsRel below 0.1131).
- **Queue:** committed at `32272ce`, started by the author on 2026-11-08 at 11:52 (an attempt at 11:48 was restarted before finishing an epoch). Epoch 0 gives val AbsRel 0.1131, matching Phase 2.

### Next

1. The author finishes the queue (`docs/phase3-runbook.md`), runs `scripts/phase3_summary.py`, commits `results/phase3/`, and hands back with "Phase 3 runs finished. Read results/phase3/summary.md."
2. Claude checks the summary against the logs and caches val predictions for the selected checkpoints. Then it runs `scripts/eval.py --split val --out results/phase3/val`, and checks on val whether fine-tuning removes the far-grass compression or only moves it.
3. In Phase 4, Claude runs the frozen `scripts/eval.py` on test for the checkpoints, with `--out` so the Phase 2 tables stay as they are.
