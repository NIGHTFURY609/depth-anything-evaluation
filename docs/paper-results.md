# Paper material: everything measured so far

Compiled 2026-11-03 from the repo at `8b1c167`. Every number below is copied from a file in `results/`, and the source is named in each section. Test numbers come from the single frozen run at `6b486bc`.

## 0. Read this first: what this file can and cannot support

The paper planned in research plan v2 is a **before-and-after fine-tuning paper**. Only the "before" half exists.

| Planned contribution (plan v2 §1) | Status | Where |
| --- | --- | --- |
| 1. Zero-shot evaluation of DA2 and DA3 on RELLIS-3D, by terrain class | **Done** | §4.1-4.5 |
| 2. Fine-tuned DA3-Small, compared before and after on a held-out sequence | **Not started** (Phase 3-4) | - |
| 3. Confidence vs Sobel filtering at equal budgets, before and after fine-tuning | **"Before" done**, "after" missing | §4.6-4.7 |
| 4. Open code for projection, training and evaluation | Projection and evaluation done, training not written | §9 |

| Planned table (plan v2 §7) | Status |
| --- | --- |
| Table 1: zero-shot accuracy | Complete |
| Table 2: zero-shot vs fine-tuned DA3-S, with the distillation ablation | **Missing**: needs Phase 3-4 |
| Table 3: confidence quality before and after fine-tuning | Zero-shot rows only |
| Table 4: equal-budget filtering | Zero-shot rows only |
| Table 5: second dataset (optional) | Not planned yet |
| Figures (plan v2 Phase 5) | Data for the plots is in §5, but no figure has been drawn yet |

So there are two ways forward:

- **A. Continue as planned.** Write the introduction, related work, method, setup and zero-shot results now, and add the fine-tuning results after Phases 3-4. §4 and §5 slot in unchanged; the Phase 4 primary comparisons are already registered (§3.9).
- **B. Publish a zero-shot evaluation paper now.** The material below is enough for that paper: DA2 against DA3 on off-road LiDAR, the far-vegetation compression, and confidence against the Offroad-Nav Sobel mask. It is a different and smaller paper than plan v2, and the title, research question and contributions would need rewriting.

Two rules carry over whichever you pick:

- **Validation numbers are for model selection only.** Report test numbers, not validation numbers, as results. Validation appears below only where it justified a protocol choice.
- **Test sequence 00002 has now been evaluated once.** Phase 3 choices must use validation only. Draw qualitative test figures after Phase 3, so that seeing test failure cases cannot steer training.

---

## 1. Framing (from plan v2, still valid)

**Research question (plan v2).** Does fine-tuning DA3-Small on real off-road LiDAR from RELLIS-3D reduce depth errors in grass, vegetation and at object edges, on sequences it never saw? And does DA3's per-pixel confidence flag the remaining errors better than a Sobel edge mask?

**Motivation.**
- An open-source off-road navigation stack (arXiv 2604.03096, code LARIAD/Offroad-Nav) uses zero-shot Depth Anything V2 with Sobel edge masking, and sells itself on needing no fine-tuning.
- In its simulated hard scene, tall grass gave blurry depth and outlier points that created phantom obstacles the edge mask did not remove.

**What the zero-shot results already say about that motivation (§4):**
- Far vegetation is predicted too close by every model. Under each model's main alignment, 49-70% of vegetation points beyond 20 m are "phantoms", predicted at under 0.8 × the LiDAR depth.
- Ranking pixels by the Offroad-Nav Sobel signal removes errors barely better than random at 10-20% budgets, and worse than random at 50%.
- DA3's confidence removes them far better at every budget.

---

## 2. Experimental setup

### 2.1 Hardware and software

| Item | Value |
| --- | --- |
| Machine | Laptop, i7-13620H, 16 GB RAM, RTX 4060 Laptop 8 GB |
| OS | WSL2 Ubuntu on Windows 11 |
| Python, PyTorch | 3.11, torch 2.11.0+cu128 (full lock in `requirements.lock`) |
| DA3 code | ByteDance-Seed/Depth-Anything-3 at commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| DA2 code | Hugging Face `transformers` checkpoints (`-hf`) |
| Seed | 0 everywhere |

### 2.2 Models

| ID | Checkpoint | Params | Licence | Output | Confidence | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| DA3-S | `depth-anything/DA3-SMALL` | 0.08B (model card) | Apache 2.0 | depth up to scale | yes | bf16 backbone, fp32 head (as upstream) |
| DA3-B | `depth-anything/DA3-BASE` | 0.12B (model card) | Apache 2.0 | depth up to scale | yes | as above |
| DA2-S | `depth-anything/Depth-Anything-V2-Small-hf` | 24.8M (DA2 paper; verify) | Apache 2.0 (verify) | relative disparity | no | fp32 |
| DA2-B | `depth-anything/Depth-Anything-V2-Base-hf` | 97.5M (DA2 paper; verify) | CC BY-NC 4.0 (verify) | relative disparity | no | fp32 |

Footnotes for the paper:
- The DA3 paper tables give different parameter counts (Small 0.03B, Base 0.11B). Cite the model card numbers and footnote the difference.
- Small and Base have no "-1.1" retrained checkpoints (DA3 issue #279).
- DA2 runs in fp32 because bf16 quantised its disparity by about 0.4%.

**Input.** Every model gets 504×322: `upper_bound_resize` to 504 on the long side gives 504×315, which rounds up to 322. That makes it a correction to plan v2, which assumed 504×315 or a 504×308 crop.

**Batching.** Batch 1. DA3 frames go in as `(B, 1, 3, H, W)`, because a list passed to `api.inference()` is treated as multi-view.

### 2.3 Inference cost (Phase 0, `results/phase0.csv`, 504×322)

| Model | Peak VRAM (MB) | Median latency (ms) |
| --- | --- | --- |
| DA3-S | 285 | 25.3 |
| DA3-B | 896 | 43.3 |
| DA2-S | 170 | 14.5 |
| DA2-B | 490 | 34.7 |

Latency is the median of 30 runs after 5 warm-up runs, on a seeded synthetic image. The precision differs (DA3 runs bf16 and fp32, DA2 runs fp32), so **do not present this as a like-for-like speed comparison** without saying so.

DA3-S training memory, for the method or appendix (one optimiser step, batch 4): head-only 1214 MB (80 ms), LoRA rank 8 1844 MB (108.5 ms), full fine-tune 2222 MB (118.5 ms).

### 2.4 Dataset: RELLIS-3D

- **Sensors:** Basler acA1920-50gc camera (1920×1200, 10 Hz) and Ouster OS1-64 LiDAR.
- **Licence:** CC BY-NC-SA 3.0.
- **Size:** 5 sequences, 13,556 frames, each image matched to a scan by frame number.
- **Annotations:** 6,234 frames have 2D image annotations. The Ouster point labels cover every scan.

| Seq | Frames | Recorded | Role | Image-annotated | LiDAR grass px | LiDAR trees/bushes px | Grass share | Trees/bushes share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00000 | 2847 | 13 Feb 2020 | train | 1200 | 11.80M | 8.25M | 0.565 | 0.395 |
| 00001 | 2319 | 13 Feb 2020 | val | 1161 | 9.05M | 3.36M | 0.672 | 0.250 |
| 00002 | 4147 | 15 Feb 2020 | **test** | 2074 | 18.57M | 9.60M | 0.636 | 0.329 |
| 00003 | 2184 | 13 Feb 2020 | train | 900 | 9.16M | 2.35M | 0.726 | 0.186 |
| 00004 | 2059 | 15 Feb 2020 | train | 899 | 7.97M | 2.27M | 0.662 | 0.188 |

Source: `results/phase1_class_counts.csv`, counted over valid ground-truth pixels at 504×322.

**Split**, fixed before any model accuracy was seen:
- **Test 00002:** the most grass and vegetation pixels, which is the plan's rule.
- **Val 00001:** its class mix is closest to test (L1 distance 0.116, against 0.137 for 00000).
- **Train:** 00000, 00003 and 00004, 7,090 frames.
- **Official splits not used:** they mix frames of 00000-00002 across train, val and test.

**Evaluation frames.** Every 5th frame, with none dropped: **830 test frames** and 464 val frames. The lists are committed as `results/phase2/frames_{val,test}.csv`. None of the 830 test frames was skipped.

**Temporal isolation** (`results/diagnostics/seqends.txt`):
- 00001, 00003 and 00000 were recorded within about 19 minutes; 00004 and 00002 were recorded 1.5 h apart, two days later.
- Test 00002 shares no boundary with any other sequence.
- Val 00001 ends 53 s before train 00003 starts, at the same spot: the mean grey-level difference between the two frames is 9.1, against 20 or more for every other pair. This affects model selection only.
- All five sequences come from one site, so write **"unseen sequences", not "unseen place"**.

**Terrain groups.**

| Group | Classes |
| --- | --- |
| Grass | grass |
| Trees/bushes | tree, bush |
| Ground | dirt, mud, asphalt, concrete (the ontology has no gravel class) |
| Other | all remaining non-void classes |
| Vegetation | grass ∪ trees/bushes |

### 2.5 Ground truth: LiDAR projection

1. **Extrinsics.** Use the per-sequence Basler→Ouster file (`transforms.yaml`), which stores camera→LiDAR and must be **inverted**. The SemanticKITTI-format `calib.txt` holds identity matrices (RELLIS issue #22).
2. **Front points.** Keep points with z > 0. Drop points beyond 1.2× the half field of view before distortion, because the polynomial can fold them back into the image.
3. **Distortion.** Apply `[-0.134313, -0.025905, 0.002181, 0.00084, 0]`, which appears only in the official `lidar2img.ipynb`, not in `camera_info.txt`.
4. **Projection.** Project directly at 504×322, with pixel coordinates scaled through the half-pixel offset, `(u+0.5)·s − 0.5`, where sx = 504/1920 and sy = 322/1200. Use a z-buffer so the nearest point wins. Never interpolate depth.
5. **Occlusion filter.** Within a ±4 px window, covering the pixel's row and the rows above, drop a point if it is farther than the nearest point by more than 20% (relative) and 0.5 m (absolute). It keeps 94.5-98.2% of points per sequence (medians). Checking only the rows above means distant ground is not treated as occluded.
6. **Labels.** Labels come from the Ouster point labels. Image annotations are also sampled at the LiDAR pixels where they exist.

**Checks that the projection is right** (`results/phase1_controls.csv`, 40 frames per sequence, median label agreement between LiDAR labels and image annotations):

| Seq | Correct | Transform not inverted | No distortion | Scan from 1 s later |
| --- | --- | --- | --- | --- |
| 00000 | 0.775 | 0.282 | 0.775 | 0.719 |
| 00001 | 0.913 | 0.089 | 0.917 | 0.815 |
| 00002 | 0.885 | 0.194 | 0.884 | 0.833 |
| 00003 | 0.839 | 0.270 | 0.839 | 0.753 |
| 00004 | 0.787 | 0.240 | 0.791 | 0.724 |

- Inverting the transform and pairing scans by frame number are both confirmed.
- Distortion on or off cannot be told apart by this metric. The official values are used because the official example uses them.
- Median agreement over all annotated frames (`results/phase1_agreement.csv`) is 0.772, 0.911, 0.886, 0.831 and 0.782 for 00000-00004, with a median of 5.9k-8.2k points per frame.

**Test ground-truth density.** Valid points (occlusion-filtered, 1-80 m) per test image:

| Subset | Median | 5th percentile | Total over 830 images |
| --- | --- | --- | --- |
| all | 6,937 | 5,205 | 5.85M |
| grass | 4,705 | 1,460 | 3.72M |
| trees_bushes | 2,204 | 0 | 1.92M |
| vegetation | 6,707 | 4,379 | 5.64M |
| ground | 2 | 0 | 0.15M |
| other | 4 | 0 | 0.06M |
| edge (union) | 269 | 61 | 0.24M |
| edge_depth | 247 | 36 | 0.22M |
| edge_semantic | 14 | 0 | 0.03M |

So on test, ground and other are sparse, and edge_semantic is rare: the test sequence is overwhelmingly vegetation.

### 2.6 The calibration offset (must be disclosed)

LiDAR points project a few pixels too high. Evidence:
- A full-resolution crop (00001, frame 1545) shows the top ring floating 12-15 source pixels (3-4 px at 322) above a barrier, which suggests a rotation error of about 0.3°.
- Shifting labels peaks at dy = +5 to +6 px in every sequence (`results/phase1_shift.csv`).

A refinement fitted on sharply bounded classes (pole, vehicle, fence, concrete, barrier), using labels only and no depth model, gave the estimates below (`results/phase1_refine.csv`). The acceptance rule was fixed before fitting: a 95% interval no wider than 3 px on both axes, a held-out gain, and at least 2,000 points.

| Seq | dx | dy | dy 95% CI | Held-out precision (official → corrected) | Verdict |
| --- | --- | --- | --- | --- | --- |
| 00000 | −0.5 | +5.0 | 4.3 to 6.6 | 0.277 → 0.356 | accepted |
| 00001 | +2.5 | +6.0 | 5.5 to 6.5 | 0.791 → 0.911 | accepted |
| 00002 | +0.9 | +4.4 | 1.0 to 7.3 | 0.876 → 0.907 | unstable |
| 00003 | −2.9 | +2.5 | −0.7 to 4.3 | 0.755 → 0.760 | unstable |
| 00004 | +4.6 | +1.3 | −0.7 to 1.7 | 0.912 → 0.926 | unstable |

**Protocol.** Official calibration is the primary ground truth. Every result is also computed against a "sensitivity" ground truth built from all five estimates (§4.8). The cause is unknown: sequences sharing one extrinsics file give different estimates, so per-frame pitch or sync is possible. Why it matters: 4 px on ground at 20 m is about a 9% depth error.

---

## 3. Evaluation protocol (frozen at commit `6b486bc`, before any test number)

### 3.1 Valid set

- **Valid pixels:** occlusion-filtered LiDAR pixels with depth in [1, 80] m. The set depends only on ground truth, so every model is scored on the same pixels.
- **Range evidence:** in val ground truth nothing is below 3.4 m and 0.008% of points are beyond 80 m.
- **Minimum:** an image needs at least 200 valid pixels; none fell below.

### 3.2 Alignment

Fitted once per image on all valid pixels, then applied to every subset.

| Model family | Primary | Also reported |
| --- | --- | --- |
| DA3 (depth) | `median`: scale only, s = median(g / d̂) | `ls`, `huber`: scale and shift in inverse depth (fairness check against DA2) |
| DA2 (disparity) | `ls`: scale and shift in disparity against 1/g | `huber` |

- **Huber:** iteratively reweighted least squares with δ = 1.345·σ̂, where σ̂ = 1.4826·MAD of the residuals, re-estimated at each iteration. These are textbook constants, not tuned.
- **Clamping:** after alignment, depth is clamped to [1, 80] m. Non-positive disparity becomes 80 m. The share of valid pixels clamped on test is 0 for `median`, 0.08-0.18% for DA3 `ls` and `huber`, and 0.18-0.61% for DA2.
- **Limitation:** per-image alignment uses ground truth that a robot would not have.

### 3.3 Metrics (per image, then averaged over images)

- AbsRel, RMSE (m), δ1 (max ratio < 1.25).
- **Phantom rate:** the share of pixels with pred < 0.8·g, strictly.
- **Overshoot rate:** the share of pixels with pred > 1.25·g.
- Phantom rate is compared only within one alignment, because median scaling puts about half the pixels below ratio 1 by construction.

### 3.4 Subsets

- **Groups:** all, grass, trees_bushes, vegetation, ground, other.
- **Distance bins:** 0-10, 10-20 and 20-80 m. Val ground truth puts 62%, 29% and 9% of points in them; only 0.4% are beyond 40 m, so there is no 40-80 m bin.
- **Minimum size:** a subset with fewer than 50 pixels in an image is NaN for that image, and the image count is reported as `(n=…)`.
- **Labels:** LiDAR labels are primary. On annotated frames the subsets are also computed from image labels, alongside LiDAR labels restricted to the same frames (`lidar_annotated`).

### 3.5 Edge band

A valid pixel is in the band if some in-range LiDAR point within row ±1 and column ±4 of it lies across:
- a **depth jump** (ratio > 1.2): `edge_depth`; or
- a **boundary between merged groups** (grass and trees together, ground, other): `edge_semantic`.

`edge` is their union, which is plan v2's definition. The settings:
- Jumps are searched among all in-range points, not only occlusion-kept ones, because the filter removes exactly the far points beside a near edge.
- **Rows:** ±1 never reaches the next LiDAR ring, about 7 rows away, so flat ground is not flagged. The cost is that horizontal edges between rings are invisible.
- **k = 4:** about two point spacings along a ring (median gap 2 px), and equal to the occlusion radius.
- **τ = 0.2:** equal to the occlusion filter's threshold.

### 3.6 Confidence (DA3 only)

- **AUROC per image.** A positive is a pixel with AbsRel > 0.25 after alignment, and the score is −conf. Images with fewer than 20 positives or 20 negatives are excluded; on test that is 0-1 of 830.
- **Sparsification.** AbsRel is recomputed after removing the lowest-confidence 0%, 5%, … 95% of pixels. Ties are broken at random with a per-frame seed, which matters because conf ≥ 1. Alignment is never refitted.
- **AUSE** is the area against the oracle (removal by true error). **AURG** is the gain over random removal; it is reported because AUSE scales with base error.

### 3.7 Filtering

**The Offroad-Nav edge mask, re-implemented exactly** (LARIAD/Offroad-Nav @ `69640cf`, `utils.py::edge_detection`):
1. Min-max scale the disparity to uint8.
2. Gaussian blur, 5×5.
3. Sobel with ksize 3, magnitude via `hypot`.
4. Scale the magnitude by its maximum.
5. **Otsu threshold, per image.**
6. Dilate with a `MORPH_ELLIPSE` kernel whose diameter is the "thickness".

The signal is disparity for DA2 and 1/depth for DA3. Three facts correct plan v2 and are worth stating in the paper:
- The threshold is Otsu, not a fixed value.
- "5 px" is the kernel **diameter**, about ±2 px.
- The shipped default is Canny (thickness 10), not Sobel.

There are two comparisons:
- **Primary: ranked removal at fixed budgets of 10, 20, 30 and 50% of valid pixels.**
  - `sobelrank` removes the highest Sobel magnitude, dilated with the thickness-5 kernel.
  - `confrank` removes the lowest confidence.
  - `randomrank` removes pixels at random, averaged over 10 seeds.
- **Secondary: the faithful Otsu mask at its own budget**, thickness 3, 5 and 9, compared with:
  - `conf_valid`, which removes as many valid pixels as Sobel did;
  - `conf_all`, which removes as many pixels of the whole image (the deployment view);
  - `random_valid`, which removes at random.
- **Why the primary changed before the freeze.** On val, the Otsu mask removes 73-86% of valid pixels, so a comparison at its own budget leaves a handful of pixels per image.
- **Group guard.** If any filter in a group leaves fewer than 50 pixels in an image, every filter in that group is NaN for that image.

### 3.8 Statistics

- **Interval:** 95% percentile interval from a block bootstrap, 1,000 resamples, seed 0.
- **Blocks:** 30 consecutive evaluated frames (about 15 s of driving, about 28 blocks on test), cut on each frame's position in the list.
- **Why 30, not plan v2's 10:** on val, interval width kept growing with block size, and per-image AbsRel stays autocorrelated to lag 20-30 for DA2 (lag 5-10 for DA3).
- **Paired differences** are taken against DA3-S over images valid for both models, with the same alignment on both sides.

### 3.9 Pre-registered primary comparisons for Phase 4

- **Accuracy:** DA3-S-zs against DA3-S-ft-lora, `median` alignment, official GT, LiDAR labels. AbsRel on grass, vegetation, `edge`, `edge_depth` and all, plus phantom rate on grass.
- **Confidence:** AUROC and AURG.
- **Filtering:** `confrank` against `sobelrank` at every budget, with 20% as the headline, for both zero-shot and fine-tuned DA3-S.
- **Everything else is secondary:** other alignments, the sensitivity ground truth, and every other cell.

---

## 4. Results on test sequence 00002 (zero-shot, 830 frames)

Unless a section says otherwise:
- ground truth is official, labels are from LiDAR, and alignment is primary (DA3 `median`, DA2 `ls`);
- values are means over images with 95% block-bootstrap intervals;
- `(n=…)` marks cells covering fewer than 90% of images.

Sources: `results/phase2/test/tables.md`, plus `summary_accuracy.csv`, `summary_confidence.csv`, `summary_filtering.csv`, `summary_filter_differences.csv` and `summary_curves.csv`.

### 4.1 Table 1 (headline): primary alignment

| Model | AbsRel all | AbsRel grass | AbsRel trees/bushes | AbsRel vegetation | AbsRel edge | AbsRel edge_depth | δ1 all | RMSE all (m) | Phantom all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | 0.155 [0.133, 0.178] | 0.111 [0.094, 0.137] | 0.288 [0.241, 0.358] | 0.153 [0.132, 0.177] | 0.343 [0.268, 0.443] | 0.370 [0.281, 0.489] | 0.816 [0.779, 0.844] | 3.72 [3.32, 4.19] | 0.106 [0.087, 0.130] |
| DA3-B | 0.142 [0.123, 0.165] | 0.098 [0.080, 0.124] | 0.277 [0.234, 0.333] | 0.140 [0.121, 0.162] | 0.367 [0.282, 0.478] | 0.400 [0.302, 0.538] | 0.853 [0.821, 0.878] | 3.43 [3.04, 3.91] | 0.070 [0.052, 0.093] |
| DA2-S | 0.165 [0.146, 0.186] | 0.102 [0.092, 0.113] | 0.364 [0.303, 0.432] | 0.161 [0.140, 0.183] | 0.400 [0.329, 0.480] | 0.435 [0.357, 0.530] | 0.822 [0.793, 0.852] | 4.59 [4.00, 5.22] | 0.106 [0.085, 0.127] |
| DA2-B | 0.169 [0.148, 0.194] | 0.100 [0.090, 0.110] | 0.380 [0.316, 0.453] | 0.165 [0.143, 0.191] | 0.417 [0.337, 0.511] | 0.454 [0.363, 0.556] | 0.828 [0.797, 0.856] | 4.89 [4.22, 5.68] | 0.103 [0.082, 0.125] |

Trees/bushes covers n = 722 images. DA3 uses `median` (scale only), which is one degree of freedom fewer than DA2's `ls`, so the fair DA2 against DA3 comparison is in §4.2.

### 4.1b Table 1 (full): AbsRel, every alignment

| Model | Align | all | grass | trees_bushes | vegetation | ground | other | edge | edge_depth | edge_semantic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | median | 0.155 [0.133, 0.178] | 0.111 [0.094, 0.137] | 0.288 [0.241, 0.358] (n=722) | 0.153 [0.132, 0.177] | 0.227 [0.169, 0.301] (n=231) | 0.163 [0.109, 0.232] (n=189) | 0.343 [0.268, 0.443] | 0.370 [0.281, 0.489] | 0.169 [0.118, 0.242] (n=184) |
| DA3-S | ls | 0.135 [0.116, 0.157] | 0.092 [0.083, 0.103] | 0.258 [0.221, 0.299] (n=722) | 0.133 [0.113, 0.155] | 0.236 [0.177, 0.297] (n=231) | 0.149 [0.101, 0.213] (n=189) | 0.305 [0.251, 0.374] | 0.326 [0.266, 0.401] | 0.163 [0.106, 0.284] (n=184) |
| DA3-S | huber | 0.140 [0.118, 0.167] | 0.085 [0.077, 0.094] | 0.285 [0.240, 0.337] (n=722) | 0.138 [0.116, 0.165] | 0.262 [0.174, 0.381] (n=231) | 0.146 [0.096, 0.214] (n=189) | 0.362 [0.282, 0.466] | 0.390 [0.301, 0.502] | 0.181 [0.100, 0.384] (n=184) |
| DA3-B | median | 0.142 [0.123, 0.165] | 0.098 [0.080, 0.124] | 0.277 [0.234, 0.333] (n=722) | 0.140 [0.121, 0.162] | 0.241 [0.184, 0.306] (n=231) | 0.131 [0.086, 0.201] (n=189) | 0.367 [0.282, 0.478] | 0.400 [0.302, 0.538] | 0.134 [0.094, 0.204] (n=184) |
| DA3-B | ls | 0.126 [0.110, 0.146] | 0.084 [0.077, 0.093] | 0.238 [0.207, 0.272] (n=722) | 0.124 [0.108, 0.144] | 0.220 [0.164, 0.285] (n=231) | 0.126 [0.083, 0.184] (n=189) | 0.308 [0.251, 0.382] | 0.332 [0.267, 0.412] | 0.143 [0.092, 0.249] (n=184) |
| DA3-B | huber | 0.131 [0.112, 0.154] | 0.075 [0.069, 0.081] | 0.266 [0.226, 0.309] (n=722) | 0.128 [0.109, 0.152] | 0.241 [0.161, 0.349] (n=231) | 0.121 [0.078, 0.181] (n=189) | 0.378 [0.287, 0.496] | 0.410 [0.307, 0.542] | 0.163 [0.086, 0.355] (n=184) |
| DA2-S | ls | 0.165 [0.146, 0.186] | 0.102 [0.092, 0.113] | 0.364 [0.303, 0.432] (n=722) | 0.161 [0.140, 0.183] | 0.291 [0.219, 0.354] (n=231) | 0.208 [0.116, 0.337] (n=189) | 0.400 [0.329, 0.480] | 0.435 [0.357, 0.530] | 0.174 [0.114, 0.285] (n=184) |
| DA2-S | huber | 0.194 [0.165, 0.228] | 0.088 [0.081, 0.096] | 0.498 [0.391, 0.614] (n=722) | 0.191 [0.160, 0.226] | 0.297 [0.212, 0.377] (n=231) | 0.203 [0.110, 0.343] (n=189) | 0.571 [0.438, 0.720] | 0.631 [0.484, 0.816] | 0.181 [0.100, 0.382] (n=184) |
| DA2-B | ls | 0.169 [0.148, 0.194] | 0.100 [0.090, 0.110] | 0.380 [0.316, 0.453] (n=722) | 0.165 [0.143, 0.191] | 0.287 [0.209, 0.358] (n=231) | 0.219 [0.119, 0.358] (n=189) | 0.417 [0.337, 0.511] | 0.454 [0.363, 0.556] | 0.183 [0.117, 0.308] (n=184) |
| DA2-B | huber | 0.207 [0.172, 0.247] | 0.085 [0.078, 0.093] | 0.536 [0.417, 0.664] (n=722) | 0.203 [0.168, 0.244] | 0.313 [0.204, 0.433] (n=231) | 0.223 [0.118, 0.373] (n=189) | 0.620 [0.469, 0.798] | 0.687 [0.514, 0.898] | 0.213 [0.101, 0.484] (n=184) |

δ1, phantom, RMSE and overshoot for every alignment are in `results/phase2/test/tables.md` (δ1, phantom) and `summary_accuracy.csv` (RMSE, overshoot). Primary-alignment RMSE (m) per subset:

| Model | all | grass | trees_bushes | vegetation | ground | edge | edge_depth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | 3.72 | 2.77 | 6.11 | 3.56 | 7.83 | 4.85 | 4.92 |
| DA3-B | 3.43 | 2.30 | 5.86 | 3.26 | 8.33 | 4.97 | 5.11 |
| DA2-S | 4.59 | 2.78 | 8.09 | 4.22 | 10.80 | 5.84 | 6.04 |
| DA2-B | 4.89 | 2.71 | 8.72 | 4.50 | 10.76 | 6.29 | 6.52 |

**Huber against LS.** Huber trades minority pixels for the majority. For DA2-S it improves grass (0.102 to 0.088) but worsens trees/bushes (0.364 to 0.498) and edge_depth (0.435 to 0.631). Grass is about 64% of the valid pixels, so a robust fit follows the grass and gives up on the rest. That is why both estimators are reported.

### 4.2 DA2 against DA3 at equal alignment (paired, against DA3-S)

A positive value means higher error than DA3-S. Both sides use the same alignment.

| Model | Align | all | grass | trees_bushes | vegetation | edge | edge_depth | edge_semantic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DA2-S | ls | **+0.030** [0.021, 0.039] | +0.010 [0.003, 0.016] | +0.106 [0.061, 0.157] | +0.028 [0.019, 0.037] | +0.094 [0.066, 0.125] | **+0.108** [0.076, 0.146] | +0.010 [−0.008, 0.029] |
| DA2-S | huber | +0.054 [0.037, 0.071] | +0.004 [−0.002, 0.009] | +0.213 [0.122, 0.308] | +0.053 [0.034, 0.070] | +0.208 [0.138, 0.283] | +0.241 [0.159, 0.328] | −0.001 [−0.008, 0.007] |
| DA2-B | ls | +0.034 [0.025, 0.044] | +0.007 [0.002, 0.012] | +0.122 [0.072, 0.177] | +0.032 [0.022, 0.043] | +0.111 [0.077, 0.149] | +0.128 [0.088, 0.171] | +0.019 [0.004, 0.036] |
| DA2-B | huber | +0.067 [0.046, 0.088] | +0.000 [−0.004, 0.005] | +0.250 [0.149, 0.355] | +0.065 [0.043, 0.087] | +0.258 [0.175, 0.351] | +0.297 [0.200, 0.404] | +0.032 [−0.003, 0.101] |
| DA3-B | median | −0.012 [−0.018, −0.007] | −0.014 [−0.018, −0.009] | −0.011 [−0.029, 0.005] | −0.013 [−0.018, −0.008] | +0.024 [0.004, 0.047] | **+0.030** [0.007, 0.058] | −0.035 [−0.061, −0.012] |
| DA3-B | ls | −0.009 [−0.014, −0.004] | −0.008 [−0.012, −0.005] | −0.019 [−0.030, −0.009] | −0.009 [−0.014, −0.004] | +0.003 [−0.010, 0.017] | +0.005 [−0.009, 0.020] | −0.021 [−0.037, −0.008] |
| DA3-B | huber | −0.009 [−0.016, −0.004] | −0.010 [−0.014, −0.006] | −0.020 [−0.033, −0.006] | −0.010 [−0.016, −0.004] | +0.016 [−0.006, 0.041] | +0.020 [−0.004, 0.050] | −0.018 [−0.030, −0.010] |

Readings:
- **DA3-S beats both DA2 models at equal alignment overall, on trees/bushes and on depth edges.** On grass alone the gap is small (+0.010 and +0.007) or zero.
- **DA3-B against DA3-S:** better on all, grass and vegetation, but worse on depth edges under `median` (+0.030). Under `ls` the edge difference vanishes (+0.005, with an interval spanning 0).

Paired phantom-rate differences are in `tables.md`, under "Paired phantom difference".

### 4.3 Error by distance: the far-vegetation compression

AbsRel, primary alignment:

| Model | veg 0-10 m | veg 10-20 m | veg 20-80 m | grass 0-10 m | grass 10-20 m | grass 20-80 m | trees 0-10 m | trees 10-20 m | trees 20-80 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | 0.126 [0.102, 0.152] | 0.205 [0.183, 0.232] | 0.322 [0.288, 0.358] | 0.081 [0.066, 0.108] | 0.170 [0.148, 0.195] | 0.313 [0.274, 0.360] | 0.253 [0.192, 0.333] | 0.260 [0.232, 0.292] | 0.376 [0.341, 0.410] |
| DA3-B | 0.119 [0.098, 0.143] | 0.184 [0.163, 0.210] | 0.262 [0.229, 0.301] | 0.076 [0.060, 0.103] | 0.130 [0.109, 0.154] | 0.238 [0.201, 0.288] | 0.255 [0.204, 0.319] | 0.251 [0.222, 0.282] | 0.305 [0.271, 0.338] |
| DA2-S | 0.126 [0.108, 0.147] | 0.243 [0.215, 0.271] | 0.333 [0.290, 0.374] | 0.069 [0.064, 0.075] | 0.186 [0.164, 0.208] | 0.327 [0.277, 0.376] | 0.338 [0.273, 0.410] | 0.377 [0.309, 0.463] | 0.373 [0.338, 0.410] |
| DA2-B | 0.128 [0.108, 0.152] | 0.256 [0.224, 0.288] | 0.326 [0.286, 0.369] | 0.067 [0.062, 0.073] | 0.186 [0.164, 0.208] | 0.312 [0.267, 0.358] | 0.350 [0.281, 0.429] | 0.409 [0.331, 0.511] | 0.382 [0.342, 0.430] |

Image counts: vegetation 830 / 801 / 749; grass 823 / 766 / 659; trees 649 / 670 / 573.

All-pixel AbsRel by distance (0-10 / 10-20 / 20-80 m):

| Model | 0-10 m | 10-20 m | 20-80 m |
| --- | --- | --- | --- |
| DA3-S | 0.126 | 0.202 | 0.316 |
| DA3-B | 0.119 | 0.182 | 0.258 |
| DA2-S | 0.126 | 0.239 | 0.345 |
| DA2-B | 0.128 | 0.252 | 0.341 |

**Phantom and overshoot rates by distance**, primary alignment. This is the signature: near pixels are predicted too far, and far pixels too near.

| Model | Rate | veg 0-10 m | veg 10-20 m | veg 20-80 m | grass 0-10 m | grass 10-20 m | grass 20-80 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | phantom | 0.009 | 0.283 [0.223, 0.354] | **0.702** [0.617, 0.784] | 0.008 | 0.289 | **0.724** [0.629, 0.823] |
| DA3-S | overshoot | 0.083 | 0.072 | 0.027 | 0.035 | 0.022 | 0.015 |
| DA3-B | phantom | 0.009 | 0.167 | 0.489 [0.397, 0.594] | 0.005 | 0.144 | 0.450 [0.341, 0.566] |
| DA3-B | overshoot | 0.072 | 0.097 | 0.053 | 0.032 | 0.034 | 0.044 |
| DA2-S | phantom | 0.015 | 0.308 | 0.665 [0.563, 0.756] | 0.015 | 0.381 | 0.696 [0.569, 0.821] |
| DA2-S | overshoot | 0.058 | 0.097 | 0.042 | 0.008 | 0.019 | 0.025 |
| DA2-B | phantom | 0.017 | 0.306 | 0.621 [0.511, 0.724] | 0.015 | 0.370 | 0.637 [0.500, 0.773] |
| DA2-B | overshoot | 0.052 | 0.098 | 0.060 | 0.007 | 0.023 | 0.047 |

δ1 by distance, primary alignment:

| Model | veg 0-10 m | veg 10-20 m | veg 20-80 m | grass 0-10 m | grass 10-20 m | grass 20-80 m |
| --- | --- | --- | --- | --- | --- | --- |
| DA3-S | 0.908 | 0.645 | 0.271 | 0.958 | 0.688 | 0.262 |
| DA3-B | 0.920 | 0.736 | 0.457 | 0.963 | 0.821 | 0.506 |
| DA2-S | 0.927 | 0.595 | 0.293 | 0.976 | 0.600 | 0.278 |
| DA2-B | 0.932 | 0.596 | 0.319 | 0.978 | 0.607 | 0.316 |

**Grass by distance under every alignment**, showing how much of the error a shift term can absorb:

| Model | Align | AbsRel 0-10 m | AbsRel 10-20 m | AbsRel 20-80 m | Phantom 0-10 m | Phantom 10-20 m | Phantom 20-80 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | median | 0.081 | 0.170 | 0.313 | 0.008 | 0.289 | 0.724 |
| DA3-S | ls | 0.065 | 0.166 | 0.287 | 0.013 | 0.311 | 0.579 |
| DA3-S | huber | 0.061 | 0.143 | 0.253 | 0.008 | 0.219 | 0.497 |
| DA3-B | median | 0.076 | 0.130 | 0.238 | 0.005 | 0.144 | 0.450 |
| DA3-B | ls | 0.060 | 0.141 | 0.260 | 0.008 | 0.229 | 0.581 |
| DA3-B | huber | 0.057 | 0.115 | 0.210 | 0.004 | 0.122 | 0.431 |
| DA2-S | ls | 0.069 | 0.186 | 0.327 | 0.015 | 0.381 | 0.696 |
| DA2-S | huber | 0.065 | 0.145 | 0.253 | 0.007 | 0.192 | 0.534 |
| DA2-B | ls | 0.067 | 0.186 | 0.312 | 0.015 | 0.370 | 0.637 |
| DA2-B | huber | 0.062 | 0.143 | 0.232 | 0.007 | 0.171 | 0.412 |

Interpretation, which is a hypothesis to state carefully:
- The error grows steadily with distance, and far pixels are predicted too close, so the shape looks like a distance-dependent compression.
- No single per-image scale or scale-and-shift removes it: even the best case (DA3-B, Huber) leaves 43% far-grass phantoms.
- This is the zero-shot analogue of the "phantom obstacle" failure. Whether fine-tuning removes it is the Phase 3 question.
- **Caveat 1, label errors:** LiDAR grass labels include tall-grass returns, and RELLIS labels disagree on grass versus bush.
- **Caveat 2, calibration:** the far-grass numbers move with calibration (§4.8).

### 4.4 Label-source sensitivity (annotated frames only)

AbsRel, primary alignment:

| Model | Labels | grass | trees_bushes | vegetation | ground |
| --- | --- | --- | --- | --- | --- |
| DA3-S | LiDAR (same frames) | 0.115 (n=413) | 0.296 (n=362) | 0.153 (n=415) | 0.228 (n=118) |
| DA3-S | image | 0.106 (n=414) | 0.229 (n=375) | 0.138 (n=415) | 0.470 (n=270) |
| DA3-B | LiDAR (same frames) | 0.101 | 0.279 | 0.140 | 0.249 |
| DA3-B | image | 0.095 | 0.216 | 0.125 | 0.457 |
| DA2-S | LiDAR (same frames) | 0.102 | 0.371 | 0.161 | 0.286 |
| DA2-S | image | 0.100 | 0.257 | 0.136 | 0.534 |
| DA2-B | LiDAR (same frames) | 0.099 | 0.389 | 0.165 | 0.283 |
| DA2-B | image | 0.097 | 0.259 | 0.136 | 0.607 |

The intervals are in `tables.md`, Table 1c. Readings:
- The model ranking is the same under both label sources.
- Grass and vegetation numbers are stable.
- Trees/bushes and especially ground depend on the label source. Image labels sampled at LiDAR pixels inherit the calibration offset, and the grass/bush disagreement moves pixels between groups.
- **Report vegetation (grass ∪ trees) as the robust group**, and per-class ground numbers with care.

**Label agreement facts for the paper:**
- Where LiDAR says grass, the image annotation agrees 75-93% of the time and says bush 4-14% of the time, depending on the sequence.
- Long stretches disagree completely; for example, in 00002 frames 1215-1277 one field is grass in the LiDAR labels and bush in the image annotations.

### 4.5 Table 3: confidence quality (DA3, zero-shot)

| Model | Align | AUROC (AbsRel > 0.25) | AUSE | AURG |
| --- | --- | --- | --- | --- |
| DA3-S | median | **0.882** [0.853, 0.901] | 0.033 [0.027, 0.041] | **0.064** [0.051, 0.080] |
| DA3-S | ls | 0.906 [0.895, 0.917] | 0.022 [0.020, 0.023] | 0.062 [0.050, 0.078] |
| DA3-S | huber | 0.900 [0.889, 0.911] | 0.023 [0.021, 0.024] | 0.068 [0.053, 0.089] |
| DA3-B | median | 0.894 [0.859, 0.913] | 0.032 [0.025, 0.045] | 0.060 [0.045, 0.076] |
| DA3-B | ls | 0.916 [0.906, 0.924] | 0.021 [0.019, 0.022] | 0.059 [0.048, 0.074] |
| DA3-B | huber | 0.909 [0.900, 0.919] | 0.022 [0.020, 0.023] | 0.065 [0.051, 0.084] |

- **Image counts.** AUROC covers 829 or 830 of 830 images; AUSE and AURG cover all 830.
- **DA3-B against DA3-S, paired:**
  - AUROC +0.012 [0.002, 0.021] (median), +0.009 [0.003, 0.016] (ls), +0.009 [0.002, 0.017] (huber);
  - AURG −0.004 [−0.009, −0.000] (median);
  - AUSE: no difference.
- **Wording.** Base ranks errors slightly better, but gains slightly less from filtering.
- **Context.** The DA3 confidence head is trained with a confidence-weighted L1 loss (DA3 paper §3.3), so it targets data noise rather than model uncertainty. Its output is exp(x)+1, so conf ≥ 1.

### 4.6 Table 4: filtering at fixed budgets (primary comparison)

AbsRel on the valid pixels left after each filter removes a share of them; primary alignment.

| Model | Budget | None | Sobel-ranked | Confidence-ranked | Random | Confidence − Sobel (paired) |
| --- | --- | --- | --- | --- | --- | --- |
| DA3-S | 10% | 0.155 | 0.148 | 0.116 | 0.155 | −0.032 [−0.042, −0.023] |
| DA3-S | **20%** | 0.155 | 0.151 | **0.100** | 0.155 | **−0.052 [−0.065, −0.039]** |
| DA3-S | 30% | 0.155 | 0.157 | 0.090 | 0.155 | −0.067 [−0.083, −0.049] |
| DA3-S | 50% | 0.155 | 0.173 | 0.078 | 0.155 | −0.095 [−0.118, −0.070] |
| DA3-B | 10% | 0.142 | 0.136 | 0.101 | 0.142 | −0.034 [−0.044, −0.025] |
| DA3-B | 20% | 0.142 | 0.139 | 0.086 | 0.142 | −0.052 [−0.066, −0.040] |
| DA3-B | 30% | 0.142 | 0.144 | 0.078 | 0.142 | −0.067 [−0.084, −0.051] |
| DA3-B | 50% | 0.142 | 0.163 | 0.069 | 0.142 | −0.094 [−0.117, −0.070] |
| DA2-S | 10 / 20 / 30 / 50% | 0.165 | 0.149 / 0.152 / 0.158 / 0.176 | - | 0.165 | - |
| DA2-B | 10 / 20 / 30 / 50% | 0.169 | 0.148 / 0.150 / 0.155 / 0.172 | - | 0.169 | - |

Phantom rate after filtering, DA3-S:

| Budget | None | Sobel-ranked | Confidence-ranked | Confidence − Sobel |
| --- | --- | --- | --- | --- |
| 10% | 0.106 | 0.105 | 0.082 | −0.024 [−0.031, −0.018] |
| 20% | 0.106 | 0.110 | 0.049 | −0.060 [−0.074, −0.049] |
| 30% | 0.106 | 0.115 | 0.027 | −0.088 [−0.107, −0.071] |
| 50% | 0.106 | 0.126 | 0.010 | −0.116 [−0.143, −0.091] |

Each filter against random, paired AbsRel difference (negative means better than random):

| Model | Comparison | 10% | 20% | 30% | 50% |
| --- | --- | --- | --- | --- | --- |
| DA3-S | Sobel − random | −0.006 [−0.010, −0.004] | −0.003 [−0.008, 0.000] | +0.002 [−0.005, 0.006] | **+0.018** [0.008, 0.026] |
| DA3-S | Confidence − random | −0.038 [−0.051, −0.028] | −0.055 [−0.069, −0.042] | −0.065 [−0.081, −0.051] | −0.077 [−0.094, −0.061] |
| DA3-B | Sobel − random | −0.007 [−0.010, −0.004] | −0.004 [−0.008, −0.001] | +0.002 [−0.002, 0.005] | +0.021 [0.014, 0.027] |
| DA3-B | Confidence − random | −0.041 [−0.054, −0.030] | −0.056 [−0.072, −0.043] | −0.065 [−0.082, −0.050] | −0.073 [−0.092, −0.055] |
| DA2-S (ls) | Sobel − random | −0.016 [−0.020, −0.011] | −0.013 [−0.017, −0.009] | −0.007 [−0.011, −0.003] | +0.011 [0.008, 0.015] |
| DA2-B (ls) | Sobel − random | −0.021 [−0.027, −0.015] | −0.019 [−0.025, −0.013] | −0.014 [−0.020, −0.009] | +0.003 [−0.003, 0.009] |

Readings:
- **Confidence beats Sobel at every budget**, for both DA3 models, under every alignment (`ls` and `huber` give the same picture).
- **The Sobel signal barely beats random at small budgets and is worse than random at 50%.** It removes pixels next to depth discontinuities, but most of the error sits in smooth, far vegetation (§4.3).

**Removing the same budget under the other alignments** (DA3, AbsRel):

| Model | Align | None | 10% S / C / R | 20% S / C / R | 30% S / C / R | 50% S / C / R |
| --- | --- | --- | --- | --- | --- | --- |
| DA3-S | median | 0.155 | 0.148 / 0.116 / 0.155 | 0.151 / 0.100 / 0.155 | 0.157 / 0.090 / 0.155 | 0.173 / 0.078 / 0.155 |
| DA3-S | ls | 0.135 | 0.129 / 0.101 / 0.135 | 0.133 / 0.087 / 0.135 | 0.139 / 0.077 / 0.135 | 0.152 / 0.063 / 0.135 |
| DA3-S | huber | 0.140 | 0.132 / 0.098 / 0.140 | 0.136 / 0.084 / 0.140 | 0.142 / 0.074 / 0.140 | 0.157 / 0.062 / 0.140 |
| DA3-B | median | 0.142 | 0.136 / 0.101 / 0.142 | 0.139 / 0.086 / 0.142 | 0.144 / 0.078 / 0.142 | 0.163 / 0.069 / 0.142 |
| DA3-B | ls | 0.126 | 0.120 / 0.091 / 0.126 | 0.123 / 0.077 / 0.126 | 0.128 / 0.068 / 0.126 | 0.143 / 0.057 / 0.126 |
| DA3-B | huber | 0.131 | 0.122 / 0.086 / 0.131 | 0.124 / 0.073 / 0.131 | 0.130 / 0.065 / 0.131 | 0.146 / 0.055 / 0.131 |

S = Sobel-ranked, C = confidence-ranked, R = random.

### 4.7 Table 4b: the faithful Offroad-Nav mask at its own budget

| Model | Thickness | Valid px removed by Sobel | Whole image removed by Sobel | AbsRel none | AbsRel Sobel (kept px) | conf_valid | conf_all | random_valid | conf_valid − Sobel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DA3-S | 3 | 0.760 | 0.466 | 0.155 | 0.407 | 0.084 | 0.121 | 0.158 | −0.323 [−0.458, −0.201] |
| DA3-S | 5 | 0.779 | 0.490 | 0.155 | 0.445 (n=742) | 0.085 | 0.119 | 0.161 | −0.360 [−0.520, −0.218] |
| DA3-S | 9 | 0.802 | 0.524 | 0.155 | 0.485 (n=715) | 0.084 | 0.114 | 0.163 | −0.401 [−0.598, −0.233] |
| DA3-B | 3 | 0.765 | 0.459 | 0.142 | 0.348 | 0.079 | 0.113 | 0.145 | −0.269 [−0.366, −0.173] |
| DA3-B | 5 | 0.792 | 0.488 | 0.142 | 0.391 | 0.079 | 0.111 | 0.147 | −0.312 [−0.437, −0.194] |
| DA3-B | 9 | 0.822 | 0.528 | 0.142 | 0.447 (n=735) | 0.078 | 0.108 | 0.148 | −0.368 [−0.528, −0.221] |
| DA2-S | 3 | 0.549 | 0.330 | 0.165 | 0.296 | - | - | 0.166 | - |
| DA2-S | 5 | 0.569 | 0.349 | 0.165 | 0.328 | - | - | 0.168 | - |
| DA2-S | 9 | 0.593 | 0.375 | 0.165 | 0.362 (n=737) | - | - | 0.172 | - |
| DA2-B | 3 | 0.487 | 0.297 | 0.169 | 0.273 | - | - | 0.171 | - |
| DA2-B | 5 | 0.505 | 0.315 | 0.169 | 0.302 | - | - | 0.174 | - |
| DA2-B | 9 | 0.525 | 0.339 | 0.169 | 0.309 (n=722) | - | - | 0.178 | - |

Intervals on the removal shares are wide, for example DA3-S thickness 5: 0.779 [0.701, 0.842].

- **The mask removes about half of the whole image but about 78% of the LiDAR-valid pixels.** The LiDAR-valid pixels sit in the textured lower half of the frame, and Otsu separates smooth sky from everything else.
- **The pixels it keeps are much worse than average** (AbsRel 0.41-0.49 for DA3, against 0.155 with no filter). One plausible reason is that the low-gradient pixels left behind are mostly far vegetation, the worst region in §4.3. That is untested.
- **conf_all:** removing the same share of the whole image by lowest confidence takes only 16-24% of valid pixels, because confidence spends its budget on the sky, and still leaves AbsRel 0.11-0.12.
- **This is a property of RELLIS-like scenes at 504×322.** It does not show that the Offroad-Nav stack performs badly in its own setting: their camera resolution is not in the repo, and their default edge detector is Canny.

### 4.8 Calibration sensitivity (official GT against the corrected-calibration GT)

Primary alignment:

| Model | Metric | Official | Sensitivity | Change |
| --- | --- | --- | --- | --- |
| DA3-S | AbsRel all | 0.155 | 0.155 | 0% |
| DA3-S | AbsRel grass | 0.111 | 0.116 | +5% |
| DA3-S | AbsRel edge | 0.343 | 0.322 | −6% |
| DA3-S | AbsRel edge_depth | 0.370 | 0.345 | −7% |
| DA3-S | AbsRel grass 20-80 m | 0.313 | 0.354 | +13% |
| DA3-S | Phantom all | 0.106 | 0.119 | +12% |
| DA3-S | AUROC | 0.882 | 0.879 | |
| DA3-S | Confidence − Sobel at 20% | −0.052 | −0.050 | |
| DA3-B | AbsRel all / edge / grass 20-80 m | 0.142 / 0.367 / 0.238 | 0.141 / 0.341 / 0.268 | 0% / −7% / +13% |
| DA2-S | AbsRel all / edge / grass 20-80 m | 0.165 / 0.400 / 0.327 | 0.170 / 0.399 / 0.334 | +3% / 0% / +2% |
| DA2-B | AbsRel all / edge / grass 20-80 m | 0.169 / 0.417 / 0.312 | 0.173 / 0.414 / 0.315 | +2% / −1% / +1% |

Paired DA2-S − DA3-S, both with `ls`: all +0.030 (official) against +0.033 (sensitivity); edge_depth +0.108 against +0.109.

**How to word it:**
- The overall ranking, the confidence results and the filtering results do not depend on calibration.
- DA3's edge-band and far-grass numbers do: they move 6-13% under DA3's primary alignment.
- The paired DA2/DA3 edge gap at equal alignment is stable.

All sensitivity tables are in `results/phase2/test/tables.md`, lower half.

### 4.9 Validation context (not results)

- **Val is easier than test.** DA3-S AbsRel all is 0.113 on val against 0.155 on test.
- **The first look at grass errors used val only.** For DA3-S with `median`:
  - far-grass AbsRel is 0.30, and 84% of far-grass pixels are phantoms;
  - scale and shift in inverse depth cut far-grass AbsRel to 0.19 (LS) or 0.15 (Huber), and phantoms to 41% or 25%.
- **Test confirmed the same pattern** (§4.3), which is the evidence that the protocol was not tuned to test.

---

## 5. Data for figures

No figure has been drawn yet (plan v2 Phase 5). Suggested figures and their data:

| # | Figure | Data | Status |
| --- | --- | --- | --- |
| F1 | Pipeline diagram: RGB → model → alignment → metrics; LiDAR → projection → occlusion filter → GT | §2.5, §3 | Draw by hand |
| F2 | Qualitative panels: RGB, LiDAR overlay, depth (DA2-S, DA3-S, later fine-tuned), confidence, error map, terrain labels, failure cases | Prediction caches `~/da3ft-data/cache/pred/`, GT caches | **Not drawn.** Make the test panels after Phase 3; val panels exist in `results/overlays/grass_val_*` (git-ignored) |
| F3 | Sparsification curves (confidence against oracle) for DA3-S and DA3-B | Table 5.1 | Ready to plot |
| F4 | AbsRel against % of valid pixels removed: confidence-ranked, Sobel-ranked and random at 10/20/30/50%, plus the faithful Otsu mask as a point at about 78% | §4.6, §4.7 | Ready to plot |
| F5 | Phantom and overshoot rate by distance bin, vegetation, four models | §4.3 | Ready to plot |
| F6 | The calibration offset: crop of 00001 frame 1545, official against corrected | `results/overlays/` | Exists (git-ignored) |
| F7 | Edge-band definition illustration | `results/overlays/` edge-band panels | Exists (git-ignored) |

Plotting notes from plan v2: use perceptually uniform colormaps, and the same colour range across panels that are compared.
- **Licence of figure images:** RELLIS images are CC BY-NC-SA 3.0, so figures built from them need attribution, and the preprint licence must be compatible.

### 5.1 Sparsification curves, test, official GT

Mean over images of AbsRel after removing a fraction of valid pixels. `conf` removes the lowest confidence first; `oracle` removes the highest true error first. In expectation, random removal stays flat at the fraction-0 value, so plot it as a horizontal line at that value. Source: `results/phase2/test/summary_curves.csv` (the sensitivity GT rows are there too).

| Fraction | DA3-S median conf | DA3-S median oracle | DA3-S ls conf | DA3-S ls oracle | DA3-S huber conf | DA3-S huber oracle | DA3-B median conf | DA3-B median oracle | DA3-B ls conf | DA3-B ls oracle | DA3-B huber conf | DA3-B huber oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.1545 | 0.1545 | 0.1348 | 0.1348 | 0.1401 | 0.1401 | 0.1424 | 0.1424 | 0.1259 | 0.1259 | 0.1306 | 0.1306 |
| 0.05 | 0.1306 | 0.1168 | 0.1116 | 0.1000 | 0.1110 | 0.0966 | 0.1174 | 0.1021 | 0.1022 | 0.0912 | 0.1001 | 0.0860 |
| 0.10 | 0.1161 | 0.1007 | 0.1007 | 0.0879 | 0.0977 | 0.0833 | 0.1014 | 0.0862 | 0.0909 | 0.0793 | 0.0860 | 0.0729 |
| 0.15 | 0.1069 | 0.0894 | 0.0933 | 0.0787 | 0.0901 | 0.0742 | 0.0924 | 0.0759 | 0.0833 | 0.0706 | 0.0784 | 0.0646 |
| 0.20 | 0.0999 | 0.0802 | 0.0870 | 0.0711 | 0.0839 | 0.0669 | 0.0863 | 0.0679 | 0.0773 | 0.0636 | 0.0730 | 0.0580 |
| 0.25 | 0.0943 | 0.0724 | 0.0815 | 0.0646 | 0.0789 | 0.0607 | 0.0814 | 0.0612 | 0.0724 | 0.0576 | 0.0687 | 0.0525 |
| 0.30 | 0.0896 | 0.0655 | 0.0767 | 0.0589 | 0.0745 | 0.0552 | 0.0776 | 0.0554 | 0.0683 | 0.0523 | 0.0652 | 0.0477 |
| 0.35 | 0.0858 | 0.0593 | 0.0725 | 0.0536 | 0.0706 | 0.0503 | 0.0745 | 0.0503 | 0.0647 | 0.0476 | 0.0620 | 0.0434 |
| 0.40 | 0.0825 | 0.0536 | 0.0688 | 0.0487 | 0.0673 | 0.0457 | 0.0720 | 0.0456 | 0.0616 | 0.0432 | 0.0592 | 0.0395 |
| 0.45 | 0.0798 | 0.0484 | 0.0656 | 0.0442 | 0.0643 | 0.0414 | 0.0702 | 0.0412 | 0.0589 | 0.0391 | 0.0568 | 0.0358 |
| 0.50 | 0.0777 | 0.0434 | 0.0627 | 0.0398 | 0.0615 | 0.0373 | 0.0692 | 0.0371 | 0.0566 | 0.0353 | 0.0548 | 0.0323 |
| 0.55 | 0.0759 | 0.0386 | 0.0599 | 0.0357 | 0.0590 | 0.0335 | 0.0686 | 0.0331 | 0.0545 | 0.0316 | 0.0530 | 0.0289 |
| 0.60 | 0.0744 | 0.0341 | 0.0574 | 0.0317 | 0.0565 | 0.0297 | 0.0683 | 0.0293 | 0.0526 | 0.0280 | 0.0513 | 0.0256 |
| 0.65 | 0.0731 | 0.0297 | 0.0550 | 0.0278 | 0.0543 | 0.0261 | 0.0682 | 0.0256 | 0.0508 | 0.0245 | 0.0497 | 0.0224 |
| 0.70 | 0.0722 | 0.0254 | 0.0528 | 0.0239 | 0.0522 | 0.0225 | 0.0683 | 0.0219 | 0.0492 | 0.0210 | 0.0482 | 0.0192 |
| 0.75 | 0.0712 | 0.0212 | 0.0507 | 0.0201 | 0.0501 | 0.0189 | 0.0683 | 0.0182 | 0.0477 | 0.0176 | 0.0467 | 0.0161 |
| 0.80 | 0.0702 | 0.0170 | 0.0485 | 0.0163 | 0.0479 | 0.0153 | 0.0685 | 0.0146 | 0.0464 | 0.0141 | 0.0454 | 0.0129 |
| 0.85 | 0.0690 | 0.0128 | 0.0464 | 0.0123 | 0.0458 | 0.0116 | 0.0690 | 0.0110 | 0.0453 | 0.0107 | 0.0443 | 0.0098 |
| 0.90 | 0.0678 | 0.0086 | 0.0443 | 0.0083 | 0.0436 | 0.0078 | 0.0699 | 0.0073 | 0.0445 | 0.0072 | 0.0432 | 0.0066 |
| 0.95 | 0.0669 | 0.0043 | 0.0429 | 0.0042 | 0.0420 | 0.0040 | 0.0716 | 0.0037 | 0.0442 | 0.0036 | 0.0427 | 0.0033 |

- **Under `median`, the confidence curve flattens after about 50-60% removal**, at about 0.07, and rises slightly for DA3-B. Under `ls` and `huber` it keeps falling, to about 0.043.
- **Reading:** after scale-only alignment, a residual error remains that confidence cannot rank. That fits a global, distance-dependent distortion (§4.3) rather than per-pixel noise. This is an interpretation, not a tested claim.

### 5.2 F4 points (DA3-S, primary alignment, AbsRel)

| % valid removed | Confidence-ranked | Sobel-ranked | Random |
| --- | --- | --- | --- |
| 0 | 0.155 | 0.155 | 0.155 |
| 10 | 0.116 | 0.148 | 0.155 |
| 20 | 0.100 | 0.151 | 0.155 |
| 30 | 0.090 | 0.157 | 0.155 |
| 50 | 0.078 | 0.173 | 0.155 |
| 77.9 (faithful Otsu, thickness 5; n=742) | 0.085 (conf_valid) | 0.445 | 0.161 |

The faithful-mask row is averaged over the 742 images where every filter left at least 50 pixels, so plot it as a separate marker, not as part of the curve. The DA3-B and DA2 rows are in §4.6 and §4.7.

---

## 6. Key findings, in claim order, with the wording they support

1. **DA3 is more accurate than DA2 on off-road LiDAR at equal alignment** (§4.2):
   - AbsRel −0.030 overall;
   - −0.108 on depth edges;
   - −0.106 on trees/bushes (DA3-S against DA2-S, `ls`).

   On grass alone the difference is small, at most 0.010.
2. **Error grows with distance for every model, and far vegetation is predicted too close** (§4.3):
   - 49-70% of vegetation pixels beyond 20 m are phantoms under the primary alignment;
   - near pixels err the other way (overshoot).

   No per-image alignment removes this. Bigger is better here: DA3-B is the least affected.
3. **DA3's confidence ranks its own errors well** (§4.5): AUROC 0.88-0.92 for high-error pixels, consistent across alignments and both GT sources.
4. **At equal budgets, confidence removes far more error than the Offroad-Nav Sobel signal** (§4.6):
   - at 20%: AbsRel 0.100 against 0.151 (paired −0.052 [−0.065, −0.039]); phantom rate 0.049 against 0.110;
   - the Sobel signal is barely better than random at 10-20% and worse at 50%.
5. **The faithful Offroad-Nav mask (Otsu) removes most LiDAR-covered pixels on RELLIS at 504×322** (76-82% for DA3, 49-59% for DA2), and the pixels it keeps have more than twice the average error (§4.7).
6. **Scale-only alignment leaves an error that confidence cannot rank** (§5.1). The confidence curve flattens near 0.07. *(Interpretation.)*
7. **Robustness checks:**
   - the conclusions hold under LS and Huber alignment, both calibrations, and both label sources;
   - the exceptions are DA3's edge and far-grass magnitudes (±6-13%) and per-class ground numbers.

**Do not claim:**
- "unseen place": write "unseen sequences";
- that Offroad-Nav fails: our resolution and scenes differ;
- that grass ground truth is exact, or that the grass problem is solved;
- that confidence is calibrated: only its ranking is evaluated.

---

## 7. Limitations to state

- **One dataset, one site.** All five sequences come from one campus, recorded over two days. There is no second dataset yet.
- **Calibration.** A residual offset of a few pixels, cause unknown, is disclosed, and the results are bounded by the sensitivity GT (§2.6, §4.8).
- **Ground-truth alignment.** Per-image alignment uses LiDAR ground truth that a robot would not have.
- **Labels.**
  - LiDAR returns in tall grass may hit the grass top or the ground, so grass "ground truth" is itself uncertain.
  - Grass and bush labels disagree between sources (§4.4).
- **Edge band.** Defined on sparse LiDAR (rows ±1), so horizontal edges between rings are invisible.
- **Val and train touch.** Val 00001 ends 53 s before train 00003 starts. This affects model selection only; test is isolated.
- **Sparse ground truth.** Ground and "other" have medians of 2-4 pixels per test image, so their per-class numbers rest on few images (n = 231 and 189).
- **No closed-loop navigation test.** The original grass failure was shown only in simulation.
- **The Offroad-Nav comparison** is a re-implementation at our resolution, on DA2 disparity or DA3 inverse depth, and not their full pipeline.
- **Latency** is not like-for-like, because the precisions differ (§2.3).

---

## 8. Related work and references to verify before citing

From plan v2 §2 and the Phase 0 novelty search. **Check every ID before citing.**

- **Depth models:**
  - DA3: arXiv 2511.10647, listed for ICLR 2026; confirm acceptance on OpenReview before citing it as ICLR.
  - DA2: arXiv 2406.09414.
- **Off-road stack:** arXiv 2604.03096, code LARIAD/Offroad-Nav @ `69640cf`.
- **Dataset:** RELLIS-3D, arXiv 2011.12954. Cite issue #22 (calib.txt holds identity matrices).
- **Confidence audits of 3D foundation models:**
  - Hillemann et al. 2606.16479;
  - Deng et al. 2608.29705;
  - Trust3R 2605.19539;
  - ReVar3R 2610.07883;
  - Landgraf et al. 2501.08188 (DA2 uncertainty, indoors).
- **Sparsification and AUSE:** Ilg et al. 2018.
- **Domain fine-tuning of Depth Anything:**
  - underwater 2507.02148;
  - endoscopy 2409.07723;
  - Depth Any Canopy 2408.04523;
  - monocular height estimation 2505.06905, where fine-tuning DA2 gave only modest gains;
  - Liang et al., CVPR 2025 (distilling a monocular model for depth completion).
- **Huber IRLS constants:** a standard robust-statistics text (1.345, 1.4826·MAD).
- **Otsu thresholding:** Otsu 1979.
- **Novelty claim.** Three web searches found no paper that fine-tunes Depth Anything on RELLIS-3D or evaluates DA3 confidence on off-road LiDAR. This is a limited search: **repeat it before posting** and state the claim cautiously.

---

## 9. Reproducibility facts for the paper and README

| Item | Value |
| --- | --- |
| Frozen protocol commit | `6b486bc` (`eval.frozen: true`, clean tree) |
| Config hash of the test run | sha256 `ab9b0199…a72a698` (`results/phase2/test/meta.json`) |
| Prediction provenance | Predictions cached at `01eda9a`. The only uncommitted file at the time was the then-untracked frame list, so the `-dirty` stamp is harmless |
| Test run | once, 830 frames, 0 skipped, about 6.7 min |
| Tests | 150 passing (`python -m pytest -q`) |
| Commands | `scripts/infer.py --split test`, then `scripts/eval.py --split test` (refuses unless frozen and clean) |
| Code licence planned | Apache 2.0; weights CC BY-NC-SA, because the training data is CC BY-NC-SA 3.0 |

---

## 10. What is still needed for the planned paper

- [ ] Phase 3: cache teacher outputs, write `train.py`, run the sanity checks, then train the head-only, LoRA (main) and no-distillation models, tuned on val only.
- [ ] Phase 4: evaluate the checkpoints with the frozen `scripts/eval.py`, giving Table 2 and the "after" rows of Tables 3 and 4.
- [ ] Phase 5: draw figures F1-F7.
- [ ] Before posting:
  - repeat the novelty search;
  - verify every reference ID;
  - verify the DA2 licences and parameter counts;
  - do an independent review of the Phase 1 code and of `scripts/eval.py`'s table assembly.
- [ ] Optional: a second dataset (GOOSE or WildScenes), a second split fold, and DA3-Large-1.1 as a zero-shot reference.
