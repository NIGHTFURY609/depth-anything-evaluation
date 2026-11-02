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
6. **Terrain labels come from LiDAR points first.** Image annotations cover 6,235 of 13,556 frames, so every-5th-frame evaluation would often have no 2D labels. The separate Ouster label download (174 MB per the README) labels every valid GT pixel. Phase 1 cross-checks them against the image annotations where both exist.
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

## 2026-11-02: OPEN, calibration offset

LiDAR points project slightly too high in the image, and this is unresolved. Phase 2 and 3 should not start until it is decided.

Evidence for a real offset:

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

LiDAR point labels and image annotations disagree most on grass versus bush, the boundary the terrain breakdown relies on. Where LiDAR says grass, the image annotation agrees 75-93% of the time and says bush 4-14% of the time, depending on sequence. Some stretches are extreme: in 00002, frames 1215-1277 score below 0.3, with one field labelled grass in LiDAR and bush in the image, though the geometry looks right.

Decisions:

- Keep LiDAR labels as primary, since they cover every valid pixel.
- Also report a merged vegetation group (grass + trees/bushes), which this disagreement cannot affect.
- On annotated frames, report the terrain breakdown under both label sources as a sensitivity check.
- No frames were dropped for label disagreement: it says nothing about the depth values. The plan's rule drops only visibly misaligned frames, and none of the overlays viewed so far qualified on their own beyond the offset above.

## Deferred to later phases

- Phase 3: the distillation L1 on median-normalised linear depth is likely to be dominated by far and sky pixels, as DA3-Small has no sky clamp. Log-space is the probable fix. Check on validation.
- Phase 3: fine-tuning changes what confidence targets, from L1 on normalised depth to relative error. Say so when comparing confidence before and after.
- Optional: whether DA3Mono-Large returns `conf`. Not checked, as the model is optional.
