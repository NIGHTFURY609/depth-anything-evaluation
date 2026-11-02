# One-off Phase 1 diagnostics

These scripts produced the supporting evidence in the calibration and label sections of [DECISIONS.md](../../DECISIONS.md). They were throwaway investigations, kept unchanged so the numbers can be traced. They have no tests and have not been rerun from this folder. Clean up and re-verify any script before the paper cites its numbers.

Run from the repo root, for example `python scripts/diagnostics/shiftclass.py`. They need the `cache/gt` build. Saved output is in [results/diagnostics/](../../results/diagnostics/), and some files contain stray log lines from the same command. For example, `edgealign.txt` starts with `crop.py`'s output.

| Script | Question | What it showed | Output |
| --- | --- | --- | --- |
| `shiftclass.py` | How does per-class label precision change with pixel shift? | Barrier and vehicle peak at dy +2 to +6. Person labels barely overlap, so person was excluded from refinement | `shiftclass.txt` |
| `depthalign.py` | Which shift best matches zero-shot DA3? | Non-ground points give dy +2 to +8. All points flip to negative dy, biased by DA3's ground slope. For 00002 it prints only locations, no errors | `depthalign.txt` |
| `edgealign.py` | Do LiDAR depth edges align with image edges? | No usable peak: optima sit at the search limit and move between splits | `edgealign.txt` |
| `worst.py` | Which frames agree least, and which have few points? | Agreement quantiles. One frame has under 2,000 points (00002, frame 893). Writes worst-case overlays | `worst.txt` |
| `worst2.py` | Why do the worst frames disagree? | Grass in LiDAR versus bush in the image is the top disagreement pair | `worst2.txt` |
| `crop.py` | Does the alignment look right at full resolution? | Top ring floats above the barrier in 00001 frame 1545 | Images in `results/overlays/` (git-ignored) |
| `crop2.py` | Does the correction fix that crop? | Official on top, corrected below: the ring lands on the barrier | Images in `results/overlays/` (git-ignored) |
