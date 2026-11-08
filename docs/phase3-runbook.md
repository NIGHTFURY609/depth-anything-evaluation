# Phase 3 runbook: running the training queue yourself

This trains the Phase 3 models without any AI involved, then produces one short file to hand back to Claude. Everything was set up and sanity-checked first; see the Phase 3 entries in [PROGRESS.md](../PROGRESS.md) and [DECISIONS.md](../DECISIONS.md).

## What it runs

`scripts/phase3_runs.py` reads the queue from `config.yaml` (`train.stages`) and works through it on its own:

| Stage | Runs | What it decides |
| --- | --- | --- |
| S1 | LoRA, learning rate 5e-5, 1e-4, 2e-4 (λd = 0.5) | the best learning rate |
| S2 | LoRA at that rate, λd = 0.1 and 1.0 (0.5 is reused from S1) | the best λd, which gives the main model |
| S3 | head-only; no distillation (λd = 0); main model with seeds 1 and 2 | nothing; these are the ablations and repeats |

That is 9 training runs. Each run trains for up to 20 epochs and stops early once validation AbsRel hasn't improved for 3 epochs. The script picks the winner of S1 and S2 by itself, using the rule fixed in DECISIONS.md, so you don't need to check in between stages.

**Time:** an epoch's time wasn't measured before the hand-off. The first epoch line shows it (the last number, in seconds), and `time_s` in `results/phase3/runs/<run>/log.csv` keeps it. The estimate was 4-6 min per epoch, so up to about 2 h per run and 10-15 h for the queue. Runs that stop early are shorter.

## Before you start

1. Plug the laptop into mains power.
2. Stop Windows sleeping: Settings → System → Power & battery → Screen, sleep & hibernate timeouts → set "When plugged in, put my device to sleep after" to **Never**. Turning the screen off is fine.
3. Close games and anything else using the GPU.
4. Check the repo is committed: run `git status` in the repo; it should say "nothing to commit". Each checkpoint records the commit it came from.

## Start the queue

Open Windows Terminal (or PowerShell) and run:

```bash
wsl -d Ubuntu
tmux new -s p3
cd /mnt/n/coding/projects/DA-paper
source ~/envs/da3ft/bin/activate
python scripts/phase3_runs.py
```

You'll see one line per epoch, for example:

```
[2026-11-07 01:12:40] lora_lr1e-04_ld0.5_s0: epoch 3/20 loss 0.0123 val absrel_all 0.1012 (best 0.1012 @ 3) 310s
```

**Keep that terminal window open.** You can minimise it, but closing it can shut WSL down and stop training. If you do close it by accident, nothing is lost: see "If it stops" below.

- To leave tmux without stopping anything: press `Ctrl+b`, then `d`.
- To get back into it: `wsl -d Ubuntu`, then `tmux attach -t p3`.

## Checking on it

From any WSL shell in the repo:

```bash
tail -f results/phase3/queue.log                  # progress; Ctrl+c to stop watching
python scripts/phase3_runs.py --plan              # which runs are done and which are left
nvidia-smi                                        # the GPU should show about 2 GB in use
python scripts/phase3_summary.py                  # the summary so far; safe to run at any time
```

For curves, run `tensorboard --logdir runs --port 6006` in a second tmux window (`Ctrl+b`, then `c`), and open http://localhost:6006 in your Windows browser.

## If it stops

Power cut, reboot, closed window or a crash: **run the same command again**.

```bash
wsl -d Ubuntu
tmux new -s p3          # or: tmux attach -t p3
cd /mnt/n/coding/projects/DA-paper && source ~/envs/da3ft/bin/activate
python scripts/phase3_runs.py
```

Finished runs are skipped. The interrupted run continues from its last completed epoch, so you lose at most one epoch, and it gives the same result as an uninterrupted run.

If a run fails, the script prints `FAILED` and the last 40 lines of output, then stops. Run the command once more. If the same run fails again, stop and send Claude the lines it printed. They're also at the end of `results/phase3/queue.log`.

## If you're short on time

- `python scripts/phase3_runs.py --stages S1` runs only S1. Later stages can be run afterwards with the plain command, and finished runs are skipped.
- The minimum useful result is S1 and S2: they give the main model. S3 runs in priority order (head-only, then no distillation, then the two extra seeds), so you can stop partway through S3 and still have the most important ablations.

## Don't

- Don't edit `config.yaml` while the queue is running, or between stages.
- Don't run `scripts/eval.py --split test`. The test sequence stays untouched until Phase 4.
- Don't delete `~/da3ft-data/checkpoints/`. Phase 4 needs the `best_val.pt` files.

## When it finishes

The log ends with `queue end`. Then run:

```bash
python scripts/phase3_summary.py
git add results/phase3
git commit -m "results: Phase 3 training runs"
```

Then start a Claude session in this repo and send:

> Phase 3 runs finished. Read results/phase3/summary.md.

That's all Claude needs. If something failed, also paste the last 30 lines of `results/phase3/queue.log`. Claude will then cache val predictions for the chosen checkpoints, run the full val evaluation, check whether fine-tuning fixed the far-grass compression, and record everything before Phase 4.
