# Decisions

Dated log of every choice made before seeing test results.

## 2026-11-02: environment

- WSL2 `Ubuntu` with uv and Python 3.11. The venv, weights and data live on WSL ext4, whose disk is on N:. C: has only 14.9 GB free and is kept out of it.
- Torch 2.11.0+cu128. DA3 pinned at commit `3d835ec`. Full versions are in `requirements.lock`.
- Upstream imports `addict` without declaring it, so `scripts/setup_wsl.sh` installs it.

## 2026-11-02: corrections to research plan v2

1. **Input is 504x322, not 504x315 or a 504x308 crop.** `upper_bound_resize` gives 504x315, and the tie at 315 rounds up to 322. Every DA3 and DA2 condition uses 504x322. LiDAR projection must scale K by sx=504/1920 and sy=322/1200, which are slightly different. Evidence: `test_output_size`, `test_forward_matches_api_on_real_frame`.
2. **`lambda_c` is not in the DA3 paper.** Section 3.3 gives the equation, `c*|d_hat - d| - lambda_c*log(c)`, with no value. No official training code exists (issues #35, #112 and #251 are open; #275 reports the same at this commit). It becomes a Phase 3 hyperparameter. Start at 1.0, the default in the third-party reimplementation valeoai/OccAny, which is a reference and not an official value.
3. **"Loss falls to near zero" cannot be the overfit check.** The loss is unbounded below as e goes to 0. Judge the overfit check on AbsRel.
4. **`lambda_c` limits what confidence can rank.** `depth_conf` is `exp(x)+1`, so c >= 1 (`test_conf_floor`). The optimum is c = max(1, lambda_c/e), so pixels with relative error above `lambda_c` all sit at the floor. `lambda_c` must stay above the 0.25 high-error threshold. Check the confidence histogram on validation.
5. **DA3's ViT has no gradient checkpointing.** `enable_checkpointing` adds it, though the Phase 0 numbers show it is not needed.
6. **Terrain labels come from LiDAR points first.** Image annotations cover 6,234 of 13,556 frames (1200, 1161, 2074, 900, 899 across 00000-00004; earlier notes said 6,235, and 6,236 annotation files exist on disk, the extras unmatched and not looked into), so every-5th-frame evaluation would often have no 2D labels. The separate Ouster label download (174 MB per the README) labels every valid GT pixel. Phase 1 cross-checks them against the image annotations where both exist.
7. **uv, not conda**, as conda is absent from WSL.

## 2026-11-02: traps in the DA3 source

- A list passed to `api.inference()` is treated as multi-view. Frames must be batched as `(B, 1, 3, H, W)`; `forward_da3` does this. Evidence: `test_batch_is_independent`, `test_multiview_differs`.
- `DepthAnything3.forward` runs under `inference_mode`, so training uses the inner `DepthAnything3Net`.
- `patch_embed.proj` matches a bare "proj", so the LoRA target is a regex scoped to `blocks.N.attn`. Evidence: `test_lora_skips_patch_embed`, `test_lora_param_count`.
- The ray branch (`*_aux`) never receives gradients and is frozen by name. `scratch.output_conv1` looks unused, as its call in `_forward_impl` is commented out, but it does receive gradients and must stay trainable. Evidence: `test_no_frozen_head_param_is_used_for_depth`.
- The head runs with autocast disabled, so bf16 covers the backbone only.

## 2026-11-02: Phase 0 measurements

From `results/phase0.csv`, at 504x322 with about 7.0 GB of VRAM free. Latencies use a seeded synthetic image.

| Model | Inference peak (MB) | Median latency (ms) | Confidence |
| --- | --- | --- | --- |
| DA3-Small | 285 | 25.3 | Yes |
| DA3-Base | 896 | 43.3 | Yes |
| DA2-Small | 170 | 14.5 | No |
| DA2-Base | 490 | 34.7 | No |

Precision differs by design, so these latencies are not like-for-like. DA3 uses bf16 on the backbone and float32 on the head, as upstream does, and skips the camera decoder. DA2 runs entirely in float32, the official pipeline, because bf16 would quantise its disparity baselines by roughly 0.4%. Do not report this table as a speed comparison without saying so.

DA3-Small training, one full optimiser step:

| Mode | Batch 2 peak (MB) | Batch 4 peak (MB) | Batch 4 step (ms) |
| --- | --- | --- | --- |
| Head | 694 | 1214 | 80.0 |
| LoRA | 1033 | 1844 | 108.5 |
| Full | 1308 | 2222 | 118.5 |

Decisions:

- VRAM is not a constraint: no mode hit OOM and checkpointing was not needed. Batch 4 stays the starting point, with accumulation to an effective 8 as planned. Batch 8 looks feasible and can be measured in Phase 3 if it helps.
- These step times exclude data loading, so they are GPU-side lower bounds. Real epoch times are measured in Phase 3, along with whether a resized image cache is worthwhile.

## 2026-11-02: novelty search

Queries (web search, not Google Scholar directly): "fine-tuning Depth Anything RELLIS-3D off-road monocular depth LiDAR"; "Depth Anything 3 fine-tune off-road vegetation depth estimation confidence evaluation 2026 arXiv"; "monocular depth foundation model domain adaptation unstructured terrain GOOSE WildScenes RELLIS grass vegetation LoRA fine-tuned".

**Verdict: no near-duplicate found**, so the angle stands. This is a limited search, so state the claim cautiously in the paper and repeat it before posting.

New related work worth reading and citing, IDs to be checked first:

- arXiv 2505.06905: monocular height estimation. Fine-tuning DA2 gave only modest gains there, and a Random Forest correction from sparse LiDAR worked better. This is the closest warning that fine-tuning may not help.
- arXiv 2408.04523, Depth Any Canopy: fine-tuned DA2 for canopy height, a nearby vegetation domain.
- Liang et al., CVPR 2025: distils a monocular foundation model for depth completion, with fine-tuning on sparse LiDAR.

## 2026-11-02: data

`scripts/verify_rellis.py` passes: 13,556 frames (2847, 2319, 4147, 2184, 2059 across 00000-00004), with every image matched to a scan and every scan to a label by frame number. This matching is by name only; the Phase 1 overlay gate still has to confirm it. Calibration is per sequence: `transforms.yaml` and `camera_info.txt` exist for each of the five.

Disk: `Rellis-3D/` is 45 GB and `raw/` 15 GB, more than the plan's 26 GB estimate because the scans unzip larger. N: has 57.7 GB free, close to the plan's 50 GB margin. The raw scan zip can be removed for space, which WSL can then reuse for caches.

## 2026-11-02: Phase 1 projection facts

From RELLIS-3D `utils/lidar2img.ipynb`, the official example:

- `transforms.yaml` stores camera->LiDAR and must be inverted. Controls agree: median label agreement is 0.78-0.91 with the inverse and 0.09-0.28 without it (`results/phase1_controls.csv`).
- Distortion is `[-0.134313, -0.025905, 0.002181, 0.00084, 0]`, found only in that notebook, not in `camera_info.txt`.
- The example pairs `frame000104` with `000104.bin`, which supports matching by frame number. Taking the scan from 1 s later lowers agreement in every sequence (0.72-0.83).
- Extrinsics differ per sequence: 00000, 00001 and 00003 share one file; 00002 and 00004 each have their own.
- Label agreement cannot distinguish distortion on from off (for example 0.7753 against 0.7752 in 00000). The official values are used because the official example uses them, not because this control confirmed them.

Our own choices:

- Pixels scale through the half-pixel offset, `(u + 0.5) * s - 0.5`, not by scaling K alone.
- Points beyond 1.2x the image's half-FOV are dropped before distortion, because the polynomial can fold them back into the image.
- The occlusion filter checks the pixel's row and the rows above it, so distant ground is not treated as occluded. It keeps 94.5-98.2% of points per sequence (medians).

The ontology has no gravel class, so the ground group is dirt, mud, asphalt and concrete.

## 2026-11-02: split

The official RELLIS splits mix frames of 00000-00002 across train, val and test, so they are not used.

- **Test: 00002**, the most grass and vegetation pixels (18.6M grass, 9.6M trees/bushes), per the plan's rule.
- **Val: 00001.** The plan gives no rule, so we take the sequence whose grass and tree/bush fractions are closest (L1) to the test sequence's: 00001 is 0.116 away, 00000 is 0.137.
- **Train: 00000, 00003, 00004** (7,090 frames).

Counts are in `results/phase1_class_counts.csv`. No model accuracy for 00002 was viewed when choosing. All sequences come from one site, so "unseen place" is not established and should be stated as unseen sequences.

## 2026-11-02: calibration offset

LiDAR points project slightly too high in the image. **Decision: the official calibration stays the primary ground truth, the offset is disclosed in the paper, and every Phase 2 and 4 result is also reported against `cache/gt_sensitivity`.**

The author chose to attempt a refinement, with the official calibration as the fallback if it did not converge cleanly. It did not:

| Sequence | Role | dx | dy | dy 95% CI | Held-out precision (official to corrected) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 00000 | train | -0.5 | +5.0 | 4.3 to 6.6 | 0.277 to 0.356 | accepted |
| 00001 | val | +2.5 | +6.0 | 5.5 to 6.5 | 0.791 to 0.911 | accepted |
| 00002 | test | +0.9 | +4.4 | 1.0 to 7.3 | 0.876 to 0.907 | unstable |
| 00003 | train | -2.9 | +2.5 | -0.7 to 4.3 | 0.755 to 0.760 | unstable |
| 00004 | train | +4.6 | +1.3 | -0.7 to 1.7 | 0.912 to 0.926 | unstable |

The fit uses labels of sharply bounded classes (pole, vehicle, fence, concrete, barrier) and no depth model, so it cannot favour any model being compared. Person is excluded because its labels barely overlap between sources. The acceptance rule was fixed in `config.yaml` before fitting: 95% interval no wider than 3 px on both axes, held-out gain, at least 2,000 points (`results/phase1_refine.csv`). The estimate is the centre of the 98% plateau of label precision, and the interval comes from a 300-resample bootstrap over blocks of 50 frames, because neighbouring frames are not independent.

- Thresholds were not loosened after seeing these results.
- The 00000 acceptance is the weaker of the two: 19,163 points, with precision only 0.277 to 0.356.
- Applying only the two accepted corrections would leave training partly corrected and the test sequence not, which is worse than a consistent offset.
- 00003 shares an extrinsics file with 00000 and 00001 but gives a different estimate, so a constant rotation per file does not explain the offset. Per-frame effects such as robot pitch or sync are possible; not investigated.
- `results/calibration_estimates.yaml` holds all five point estimates, accepted or not, and `cache/gt_sensitivity` is built from them. That cache exists only to show how much the results depend on calibration. Training uses the official one.
- For 00001, a side-by-side crop confirms the correction: the top ring floats above the barriers with the official calibration and lands on them when corrected.

Evidence for a real offset (one-off scripts in `scripts/diagnostics/`, saved output in `results/diagnostics/`; clean up and re-verify them before the paper cites these numbers):

- A full-resolution crop (00001, frame 1545) shows the top ring floating 12-15 source px above a barrier, about 3-4 px at 322. At that range this suggests a rotation error near 0.3 degrees rather than parallax.
- Shifting labels: the best dy is +5 to +6 in every sequence, at the search limit of 6 (`results/phase1_shift.csv`).
- Barriers peak at dy +2 to +6 and vehicles near +3. Barrier precision rises from 0.70 to 0.88 at dy +6, dx +2 in 00001.
- Non-ground points against zero-shot DA3 give dy +2, +6, +6, +8 for 00000, 00001, 00003, 00004.

Why the size is not settled:

- 00001 and 00003 share an extrinsics file yet barriers give +6 and +2, so label bias or per-frame effects are mixed in.
- Label agreement is biased by the scene's layering, and DA3 against all points is biased by its ground slope (it flips to dy -3 to -10).
- LiDAR depth edges against image edges gave no usable peak.
- 00002 may also have a horizontal offset (dx +10 to +12 against DA3, at the search limit; barriers say dx 0). Only the location of its optimum was printed, no error values.

Why it matters: on ground at 20 m, 4 px is about a 9% depth error, comparable to the effect the paper measures, and fine-tuning would learn it.

## 2026-11-02: grass and bush labels disagree

LiDAR point labels and image annotations disagree most on grass versus bush, the boundary the terrain breakdown relies on. Where LiDAR says grass, the image annotation agrees 75-93% of the time and says bush 4-14% of the time, depending on sequence. Some stretches are extreme: in 00002, frames 1215-1277 score below 0.3, with one field labelled grass in LiDAR and bush in the image, though the geometry looks right. That cause was confirmed on frames 1255 and 1265. Frames also below 0.3 but not inspected: 00002 frames 102-108 and 1327-1345, and 00001 frame 896.

Decisions:

- Keep LiDAR labels as primary, since they cover every valid pixel.
- Also report a merged vegetation group (grass + trees/bushes), which this disagreement cannot affect.
- On annotated frames, report the terrain breakdown under both label sources as a sensitivity check.
- No frames were dropped for label disagreement: it says nothing about the depth values. The plan's rule drops only visibly misaligned frames, and none of the overlays viewed so far qualified on their own beyond the offset above.

## 2026-11-02: Phase 1 gate outcome

The plan's gate is that overlays line up with trunks and ground on several frames from every sequence. **Passed for gross geometry, not passed for a few pixels of vertical alignment**, which is the offset above. The author accepted proceeding on that basis by choosing official calibration with disclosure as the fallback. What was viewed is listed in [PROGRESS.md](PROGRESS.md).

## 2026-11-03: Phase 1 close-out

- **All 20 sampled overlays have now been viewed.** All pass for gross geometry; the only visible error is the known small upward offset (head-top points sit slightly high on the walkers in 00003 frame 2183 and 00004 frame 2058).
- **The remaining low-agreement frames are label disagreements, not misalignment.** 00002 frames 102-108 and 1327-1345 and 00001 frame 896 all disagree on grass (3) against bush (19), with geometry intact on the panels viewed (`scripts/diagnostics/lowagree.py`, `results/diagnostics/lowagree.txt`).
  - In 00002 frames 102-108, LiDAR says bush and the image says grass; from 1325 on, the reverse.
  - 00001 frame 896 scores 0.17 between neighbours at 0.95. A single image annotation labels the near field bush.
- **No frames dropped.** The rule, fixed before looking, was to drop only frames visibly misaligned beyond the known global offset. `gt.dropped_frames` exists in `config.yaml` and is empty.
- **The test sequence is temporally isolated; val and train touch** (`scripts/diagnostics/seqends.py`). Timestamps show:
  - 00001, 00003 and 00000 were recorded within about 19 minutes on 13 Feb 2020;
  - 00004 and 00002 were recorded 1.5 h apart on 15 Feb.
  - Val 00001 ends 53 s before train 00003 starts, at the same spot: the mean grey-level difference is 9.1, against 20 or more for every other pair.
  - Test 00002 shares no boundary with any sequence.
  - This affects only model selection on val. It is not corrected, and is stated as a limitation.

## 2026-11-03: the Offroad-Nav edge mask

Read from LARIAD/Offroad-Nav at commit `69640cf`, `mono_depth/src/depth_rescaling/src/depth_estimation/utils.py::edge_detection` and `depth_rescaling_node.py`. Three corrections to research plan v2:

1. **There is no fixed Sobel threshold.** The steps are:
   1. Min-max scale the DA2 disparity to uint8, at camera resolution.
   2. Gaussian blur, 5x5.
   3. Sobel, ksize 3, magnitude via `hypot`.
   4. Scale the magnitude by its maximum to 0-255.
   5. **Otsu threshold, per image.**
   6. Dilate with a `MORPH_ELLIPSE` kernel of size `thickness`.
2. **"5 px" is the kernel diameter**, about ±2 px, not "within 5 px". The config comment reads "10 for canny, 5 for sobel".
3. **The shipped default is Canny**, thickness 10. We use Sobel, as the paper describes.

Their camera resolution is not in the repo, so the kernel cannot be scaled to 504 px. We sweep thickness {3, 5, 9} at 504x322 with 5 as primary, and confidence is compared at each Sobel budget.

## 2026-11-03: Phase 2 alignment

The author chose to report **both least squares and Huber** for every scale-and-shift fit (DA2 in disparity, and DA3 in inverse depth as the fairness check). This replaces plan v2's single "robust fit".

- Huber uses δ = 1.345·σ̂, where σ̂ = 1.4826·MAD of the residuals, re-estimated at each iteration. These are textbook constants, so nothing is tuned on data.
- DA3's main alignment stays scale-only median ratio, the same scale the Phase 3 loss uses.

## 2026-11-03: Phase 2 evaluation protocol, fixed on val before any test number

All values are in `config.yaml` under `eval:`. They were set from val ground truth (`scripts/diagnostics/valgt_stats.py`, `results/diagnostics/valgt_stats.txt`) or from val per-image statistics. No test depth number existed when they were set.

**Valid set.** `keep` (occlusion-filtered) and 1-80 m. It depends only on ground truth, so every model is scored on the same pixels. In val GT nothing is below 3.4 m and 0.008% is beyond 80 m. Images with under 200 valid pixels are skipped and logged; none were on val.

**Alignment.** Fitted once per image on all valid pixels, then applied to every subset. Every model's depth is clamped to [1, 80] m, and the clamped share is reported.
- DA3: `median` (scale only, main), plus `ls` and `huber` scale and shift in inverse depth (the fairness check).
- DA2: `ls` and `huber` in disparity. Non-positive disparity becomes 80 m.
- Huber constants are as logged above. Both estimators are reported everywhere, as the author chose.

**Metrics.** AbsRel, RMSE, δ1 (< 1.25), phantom rate (pred < 0.8 g, strict), overshoot (pred > 1.25 g). Phantom rate is compared only within one alignment, because median scaling puts about half the pixels below ratio 1 by construction.

**Subsets.**
- all, grass, trees_bushes, ground, other, vegetation (grass ∪ trees_bushes).
- Distance bins 0-10, 10-20 and 20-80 m for all, grass, trees_bushes, ground and vegetation. Val GT puts 62%, 29% and 9% of points in them; only 0.4% are beyond 40 m, which is why there is no 40-80 m bin.
- A subset under 50 pixels in an image gives NaN for that image, and the number of images is reported. The value 50 was lowered from a proposed 100 because `edge_depth` has a median of 71 pixels per val image, and 100 would have kept only 30% of images.
- LiDAR labels are primary. On annotated frames the same subsets are also computed from image labels, alongside LiDAR labels restricted to the same frames (`lidar_annotated`). Image labels sampled at LiDAR pixels inherit the calibration offset.

**Edge band.** A valid pixel is in the band if a LiDAR point in range lies within row ±1 and column ±4 of it, and across either a depth jump (ratio > 1.2, `edge_depth`) or a boundary between merged groups (grass and trees together, ground, other; void ignored, `edge_semantic`). `edge` is the union, the plan's definition.
- Jumps are searched among all in-range points, not only `keep`, because the occlusion filter removes exactly the far points beside a near edge.
- Rows ±1 never reach the next ring, about 7 rows away, so flat ground is not flagged. Horizontal depth edges between rings cannot be detected.
- k = 4 is about two point spacings (the median gap along a ring is 2 px) and equals the occlusion filter's radius. τ = 0.2 equals the filter's threshold.
- On val, `edge_semantic` is dominated by flat grass/mud trail edges, which carry no depth jump. That is why both parts are reported separately.

**Confidence (DA3).**
- AUROC per image: positive means per-pixel AbsRel > 0.25 after alignment, and the score is -conf. Images with fewer than 20 positives or 20 negatives are excluded and counted.
- Sparsification of AbsRel over fractions 0 to 0.95 in steps of 0.05, removing the lowest confidence first, with ties broken randomly (seeded per frame). Alignment is never refitted.
- AUSE is against the oracle. AURG is against random removal, whose expected curve is flat; it is reported because AUSE scales with base error.

**Filtering.** The Sobel signal is disparity for DA2 and 1/depth for DA3. AbsRel and phantom rate are taken on the valid pixels left. Random removal is the mean of 10 seeded removals. There are two comparisons.

- **Primary: ranked removal at fixed budgets of 10, 20, 30 and 50% of valid pixels.** At each budget the filters remove:
  - `sobelrank`: the highest Sobel scores, where the score is Offroad-Nav's gradient magnitude dilated with their kernel at diameter 5;
  - `confrank`: the lowest confidence;
  - `randomrank`: pixels at random.

  Each comparison has a paired per-image interval.
- **Secondary: the faithful Offroad-Nav mask (Otsu) at its own budget**, at thickness 3, 5 and 9:
  - `conf_valid` removes as many valid pixels as Sobel did;
  - `conf_all` removes as many pixels of the whole image, which is the deployment view;
  - `random_valid` removes at random.
- **Why the primary comparison changed.** The independent review found that the faithful mask removes 73-86% of valid pixels on val on average (median 94-96%). Otsu splits the zero-gradient sky from everything else, so sloping ground counts as edge, and this holds at camera resolution too. A comparison at that budget leaves a handful of pixels per image. The author chose the ranked curves, keeping the faithful mask as a reported point. This replaces the `conf_valid` vs `sobel5` primary set earlier the same day, before any test number.
- **Guard.** If any filter in a comparison group (one Sobel thickness, or one budget) leaves fewer than 50 valid pixels in an image, every filter in that group is NaN for that image. Comparisons therefore stay over the same images.

**Statistics.** Per-image metrics are averaged over images, with 95% percentile intervals from a block bootstrap: 1,000 resamples, seed 0, NaN images skipped. Blocks are cut on each image's position in the split's frame list, so subsets on every other frame (image labels) get blocks of the same duration. Paired differences against DA3-S use images valid for both and the same alignment, for AbsRel and phantom rate per subset, and for AUROC, AUSE and AURG.
- **The block is 30 evaluated frames, not plan v2's 10.** On val, interval width kept growing with block size (DA3-S AbsRel ±0.015 at 10, ±0.022 at 40), and per-image AbsRel stays autocorrelated to lag 20-30 for DA2 (lag 5-10 for DA3). 30 frames is about 15 s of driving, giving about 28 blocks on test.

**Frames.** Every 5th frame of the sequence, minus `gt.dropped_frames` (none). The lists are committed as `results/phase2/frames_<split>.csv`, and `scripts/eval.py` refuses to run if they differ from the config's selection.

**Primary comparisons for Phase 4, registered now.**
- DA3-S-zs against DA3-S-ft-lora, `median` alignment, official GT, LiDAR labels: AbsRel on grass, vegetation, `edge`, `edge_depth` and all, with phantom rate on grass.
- For confidence, AUROC and AURG.
- For filtering, AbsRel of `confrank` against `sobelrank` at every budget, with 20% as the headline, for zero-shot and fine-tuned DA3-S.
- All other cells, including both alignments and the sensitivity GT, are secondary.

**Independent review (2026-11-03, before the freeze).** A fresh reviewer re-derived the summaries from the per-image rows and found no arithmetic bug. Its findings, all fixed before freezing:
- The Sobel budget problem above.
- Registered comparisons lacked paired statistics; now added.
- Model names and primary alignments were hardcoded. They now come from the config or the command line and the prediction's kind, so fine-tuned models need no code change.
- Cached predictions are now checked against the config's model ID and the GT shape, and their provenance is written to `meta.json`.
- A DA2-only run crashed.
- Image-label subsets had blocks twice as long in time.
- Tests added: brute-force sparsification, equal removal counts, the group guard, determinism, position-based blocks.

**First look at grass errors (val only, `scripts/grass_look.py`).**
- DA3-S with `median` has grass AbsRel of 0.05 at 0-10 m and 0.30 at 20-80 m, and 84% of far-grass pixels are phantoms (predicted under 0.8x the LiDAR depth). In the worst frames, the far band of tall grass is predicted too close.
- Scale-and-shift in inverse depth cuts far-grass AbsRel to 0.19 (LS) or 0.15 (Huber), and the phantom rate to 0.41 or 0.25. Much of the error is therefore a distance-dependent compression that one scale cannot fix. DA3-B (0.18 with `median`) and DA2-S (0.16 with LS) show it less.
- Phase 3 should check whether fine-tuning removes it or only moves it.

## 2026-11-06: Phase 3 training setup, fixed before any training run

All values are in `config.yaml` under `train:`. The `eval:`, `split:`, `gt:`, `input:` and `models:` sections are unchanged from the freeze at `6b486bc`, and `test_eval_section_matches_freeze` checks that.

**Division of work (the author's choice).** Claude builds and tests the code, caches the teacher and runs the sanity checks. The author runs the long training queue (`scripts/phase3_runs.py`) unattended and hands back `results/phase3/summary.md`. The steps are in [docs/phase3-runbook.md](docs/phase3-runbook.md).

**Selection (the author's choice).** Mean per-image val AbsRel over all valid pixels, with median alignment and official GT, computed by the same functions as the frozen eval (`src/val.py`, checked against `evaluate_frame` in `tests/test_val.py`).
- It picks both the best epoch (early stopping) and the hyperparameters.
- Grass, vegetation, `edge_depth` and grass at 20-80 m are logged every epoch but never decide anything.
- Epoch 0, the untrained model, is logged for reference and is not a candidate.
- The test sequence is not loaded in Phase 3: `TrainFrames` refuses non-train sequences and `Validator` refuses non-val ones.

**Loss.**
- `L = L_lidar + lambda_d * L_distill`, as in plan v2 §6.
- `L_lidar`:
  - is computed on `keep` & 1-80 m, the eval's valid set;
  - takes s = median(g/d) per image, with the median interpolated like `np.median`. **s is not detached**, contrary to plan v2 and to the first version of this entry: see the sanity-check entry below;
  - is `mean(c*e - lambda_c*log c)` per image, then averaged over images, as eval averages over images.
- **`lambda_c = 1.0`, fixed, not tuned.** It must stay above the 0.25 high-error threshold (2026-11-02 item 4); a test checks this. The val confidence median and the share at the floor (below 1.05) are logged every epoch.
- `L_distill` is the L1 between median-normalised student and teacher over all pixels. The teacher's median is a constant; the student's stays in the graph, for the same reason as s.
- **Distillation stays linear, not log.** The rule was fixed before running: use log space if, on zero-shot DA3-S val predictions, the farthest 10% of pixels hold more than 0.3 of Σ d/med(d).
  - They hold 0.187 (median; p10 0.172, p90 0.203). The largest normalised depth is only 2.2 times the median, so DA3-S has no heavy sky tail.
  - The deferred worry below therefore did not hold (`scripts/diagnostics/distill_space.py`, `results/diagnostics/distill_space.txt`).

**Optimisation.**
- AdamW with weight decay 0.01.
- Batch 4, accumulated over 2 steps to an effective 8, with `drop_last`.
- Warmup over 300 optimiser steps, then cosine to 0 over 20 epochs. The schedule always spans 20 epochs, so a 1-epoch sanity run follows the same schedule as a real run's first epoch.
- Early stopping with patience 3. Gradient-norm clip 1.0. The backbone runs in bf16 and the head in fp32, as in `forward_da3`.
- Learning rates: the head always gets `lr_head = 5e-5`, and the LoRA parameters get the run's `lr`. LoRA rank 8 on attention qkv/proj, from `config.yaml`.
- A non-finite loss or val metric ends the run, recorded as `stopped: nonfinite` with its best epoch so far (or none, ranked last), and the queue moves on. Resume is deterministic, so retrying would only diverge again.
- `train()` mode is used. `test_train_mode_matches_eval_mode` shows it gives the identical forward, as dropout and drop-path are 0 upstream.
- Each epoch's shuffle, worker seeds and augmentation are seeded from (seed, epoch). Together with the optimiser and scheduler state in `last.pt`, this means a resumed run repeats an uninterrupted one (`test_resume_reproduces_an_uninterrupted_run`, on CPU).

**Data.**
- All 7,090 train frames with stride 1, minus `drop_last` remainders.
- Images go through upstream's own resize steps, so with augmentation off a training image equals `preprocess()` exactly (`test_unaugmented_item_matches_preprocess_and_caches`).
- Colour jitter (brightness, contrast and saturation 0.2, hue 0.02) is applied to the image only, before normalisation.
- A horizontal flip (p = 0.5) is applied to image, GT, mask and teacher together.
- The teacher is zero-shot DA3-S, cached by `scripts/infer.py --split train --stride 1` into `pred/da3_small/`, at batch 1, like every other prediction.

**Checkpoints.**
- `best_val.pt` and `last.pt` live in `~/da3ft-data/checkpoints/<run>/` and hold only the trainable parameters.
- Their metadata records the base model, mode, LoRA rank and regex, run settings, config SHA-256, git hash and teacher.
- `load_finetuned` rebuilds the net and refuses a state that does not cover exactly its trainable parameters.

**The queue (`train.stages`).** The tuning budget is lean, at 9 runs:
- **S1:** LoRA learning rate in {5e-5, 1e-4, 2e-4} at `lambda_d` 0.5.
- **S2:** `lambda_d` in {0.1, 0.5, 1.0} at S1's best rate. 0.5 is S1's own run, reused.
- **S3:** head-only at the best `lambda_d`, no-distillation (`lambda_d` 0) at the best rate, and seeds 1 and 2 of the main model.

Plan v2's 3x3 grid is not run. The selection is made by the script and written to `results/phase3/selection.json`.

**Sanity criteria, fixed before running.**
- Overfit 10 evenly spaced train frames with LoRA at 1e-4 (head 5e-5), no augmentation and no distillation: their AbsRel must be below 0.03 by step 300, and the step-0 and final overlays are viewed.
- One LoRA epoch at 1e-4 with `lambda_d` 0.5: val AbsRel (all) must fall below the untrained model's val AbsRel, 0.1131 in `results/phase2/val`.

## 2026-11-06: Phase 3 independent review and sanity checks

**Independent review.** A fresh reviewer read the training code against this spec and reran the tests. It found no leakage and confirmed that preprocessing and validation match eval. Its real findings, all fixed with tests before the queue started:
- A resume trusted any `last.pt` with the same run name. It now also requires the same fingerprint: config SHA-256, a hash of the training code files, the frame counts and the steps per epoch. Debug flags (`--limit-*`, `--epochs`) must write to separate roots.
- A single NaN would have blocked the queue for good, because a deterministic resume replays it. It now ends that run (see Optimisation).
- A crash between saving `best_val.pt` and `last.pt` could leave a best checkpoint that disagrees with the log. Now the best is written as `best_pending.pt`, then `last.pt` commits, then the pending file is renamed; resume promotes or discards it.
- `log.csv` could end up one row short after a crash. It is now rewritten on resume and at the end.
- A NaN val metric would have run all 20 epochs and then crashed.
- The queue now resumes a failed run once by itself before stopping.
- Smaller fixes: images with fewer than `eval.min_valid` LiDAR pixels are skipped in the loss as in eval; `run_name` refuses rates that would collide; `infer.py --name` can't overwrite another checkpoint's cache.

The reviewer also noted that GPU training is not bit-deterministic (gradients differ by about 1e-4 between identical passes), so "a resumed run repeats an uninterrupted one" holds exactly only on CPU. On GPU it holds up to that noise.

**Bug found by the overfit check: a detached scale made the depth collapse.** The first overfit run improved to AbsRel 0.07 by step 50, then diverged to NaN by step 100.
- `scripts/diagnostics/overfit_diverge.py` logged the per-step evidence. The raw predicted depth shrank every step (median 1.06 to 1e-33) with or without the confidence term.
- The cause: with s = median(g/d) detached, the gradient is not scale-invariant. The pixels predicted too far have larger d/g and so carry more weight than those predicted too near, which pushes the overall scale down every step with nothing to stop it.
- With s kept in the graph, L(k·d) = L(d) exactly, so the gradient has no scale component. `test_lidar_gradient_has_no_component_along_scale` and its distillation twin check sum(grad·d) = 0, and both failed before the fix.
- Rejected explanations: the confidence weights (removing them made it diverge faster), and step size or bf16 (the drift is systematic, not noise).
- After the fix the depth scale stays between 0.96 and 1.67 over 150 steps.

**The overfit check then failed its fixed bar.** The bar was AbsRel below 0.03 by step 300. The result was 0.049 (0.117 at step 0, a plateau of 0.045-0.05 from step 150). The bar was not changed after the fact.
- Most of the gap comes from the loss form. Minimising c·e − λ·log c over c gives λ·log(e) plus a constant, so the confidence-weighted loss keeps shrinking already-small errors and does little for large ones. The confidence median rose to about 55.
- With the confidence term off, the same setup reaches 0.027-0.034 (train-batch AbsRel). So the pipeline can fit, and 0.03 in 300 steps was a tight bar even for plain relative error.
- The overlays (`results/overlays/phase3_overfit/`, git-ignored) pass. The prediction sits on the LiDAR rows, and the horizon and field band improve from step 0 to step 300. Bare trees stay wrong, as they are for the zero-shot model, because LiDAR passes through the branches.
- An alternative was raised: train depth on plain relative error and the confidence head separately on the detached error. It was not adopted.

**The queue runs the plan's confidence-weighted loss unchanged**, with the scale fix. It was committed at `32272ce` and started on 2026-11-08. The one-epoch sanity check was not run separately. Epoch 1 of the first S1 run plays its role, against the same bar of val AbsRel (all) below 0.1131.

Consequence for the paper: the depth objective is effectively a log-error objective, while selection and reporting use AbsRel. Say so when describing the method.

**Small supporting changes.**
- `scripts/eval.py --out` writes elsewhere than `results/phase2/<split>`, so Phase 3 and 4 never overwrite the Phase 2 tables. No metric code changed.
- `scripts/infer.py --checkpoint --name` caches a fine-tuned model's predictions under `pred/<name>/`, with model ID `<base>+<name>@<checkpoint sha256[:12]>`.

## 2026-11-02: working rules

- Never touch WSL distro `Ubuntu-26.04` or change `.wslconfig`.
- Never commit RELLIS data or weights. Overlays are derived from RELLIS images, so they are git-ignored too.
- Fix thresholds before viewing test results. Log anything seen from test sequence 00002 in [PROGRESS.md](PROGRESS.md).
- Commit as work completes; do not push without being asked.
- Each phase ends with a results review before the next starts.
- Record every decision here and all progress in [PROGRESS.md](PROGRESS.md) as the work happens. Save one-off scripts and their output in the repo if a decision relies on them.

## Deferred to later phases

- ~~Phase 3: the distillation L1 on median-normalised linear depth is likely to be dominated by far and sky pixels.~~ Checked 2026-11-06 on val: it is not (far-10% share 0.187), so distillation stays linear. See the Phase 3 entry.
- Phase 3: fine-tuning changes what confidence targets, from L1 on normalised depth to relative error. Say so when comparing confidence before and after.
- Optional: whether DA3Mono-Large returns `conf`. Not checked, as the model is optional.
- Code issues and other open items are listed in [PROGRESS.md](PROGRESS.md).
