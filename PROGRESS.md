# Progress

Where the project stands, what has been done and what is still open. The reasoning behind each choice is in [DECISIONS.md](DECISIONS.md). Update both at the end of every phase.

Last updated: 2026-11-03.

## Status

| Phase (research plan section 8) | State | Result |
| --- | --- | --- |
| 0: environment, checks, novelty search | Done | Everything fits in 8 GB; no near-duplicate found in a limited search |
| 1: loader, projection, split, ground truth | Done, gate passed for gross geometry | Ground truth cached for all 13,556 frames; a calibration offset remains and is disclosed |
| 2: zero-shot baselines, `eval.py` | **In progress** | Plan: Step 0 done |
| 3 onward | Not started | |

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
| Overlays (git-ignored, derived from RELLIS images) | `results/overlays/`, 29 files |

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
python -m pytest -q                        # from the repo root: 94 passed on 2026-11-02
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

### What has been seen from test sequence 00002

Kept so the claim of a clean test set can be checked.

- Class counts, label agreement and overlays, none of which involve a depth model.
- Its calibration estimate (+0.88, +4.36), fitted on labels only and used only for sensitivity reporting.
- From the zero-shot DA3 diagnostic: the location of the best shift only, printed without error values. See `results/diagnostics/depthalign.txt`. These locations did not set any threshold or decision.
- No depth accuracy for 00002 has been computed or viewed for any model being compared.

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

- **Cause of the calibration offset is unknown.** Per-frame pitch or sync was not investigated. If the two caches give noticeably different results in Phase 2, the edge-band and ground numbers need careful wording.
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

### Disk

- **N:** 57 GB free, against the plan's 50 GB margin. Once the author is happy with Phase 1, the 11.6 GB images zip in the repo root and the 15 GB `raw/` folder can be deleted. Neither has been deleted.
- **C:** 10.3 GB free, down from 14.9 GB at the start, so the Phase 0 goal of leaving C: unchanged was not met. It dipped to 6.8 GB during the work. The WSL swap file is only 0.04 GB and no large new files turned up in the folders scanned. The Windows pagefile is suspected but unproven.

## Next: Phase 2

1. Get the Sobel threshold from the Offroad-Nav code.
2. Run every zero-shot model on the validation and test frames, caching depth and confidence to `.npz`.
3. Build `eval.py`: alignment, all metrics, filtering curves, bootstrap, reused unchanged for fine-tuned models. It must report against both `cache/gt` and `cache/gt_sensitivity`, plus the merged vegetation group and the both-label-sources check.
4. Fix thresholds before viewing test results.
5. Deliver results tables 1 and 3 (zero-shot part) and a first look at grass errors, then a phase review.
