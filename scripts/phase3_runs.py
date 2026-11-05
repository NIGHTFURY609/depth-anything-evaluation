"""Runs the Phase 3 training queue from config.yaml (train.stages), unattended.

    python scripts/phase3_runs.py                # everything; rerun the same command to resume
    python scripts/phase3_runs.py --stages S1    # only stage S1
    python scripts/phase3_runs.py --plan         # print what would run, train nothing

Stages run in order. A finished run (results/phase3/runs/<name>/done.json) is never rerun, and an
interrupted one resumes from its last.pt. After a stage marked `select`, the run with the lowest
val AbsRel (all pixels, median alignment) becomes "best" for later stages, and the choice is
written to results/phase3/selection.json. Everything printed also goes to results/phase3/queue.log.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.train import resolve_stage, select_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", help="stage names to train; default all")
    parser.add_argument("--only", help="train only the run with this name")
    parser.add_argument("--plan", action="store_true", help="print the resolved queue and exit")
    parser.add_argument("--out-root", help="debug: passed to scripts/train.py")
    parser.add_argument("--ckpt-root", help="debug: passed to scripts/train.py")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="debug: further arguments for scripts/train.py, e.g. --extra --limit-train 40")
    args = parser.parse_args()

    cfg = load_config(str(ROOT / "config.yaml"))
    if args.out_root:  # dry runs keep their queue log and selection beside their run logs
        runs_root = Path(args.out_root)
        phase3 = runs_root.parent
    else:
        phase3 = ROOT / cfg["paths"]["results_root"] / "phase3"
        runs_root = phase3 / "runs"
    phase3.mkdir(parents=True, exist_ok=True)
    log_file = open(phase3 / "queue.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    def done(name: str):
        path = runs_root / name / "done.json"
        return json.loads(path.read_text()) if path.exists() else None

    selection_path = phase3 / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    best = None
    log(f"queue start: stages {args.stages or 'all'}, only {args.only or '-'}{' (plan only)' if args.plan else ''}")
    for stage in cfg["train"]["stages"]:
        try:
            specs = resolve_stage(stage, best)
        except ValueError as e:
            log(f"stop: {e}. Finish the earlier stages first.")
            sys.exit(1)
        wanted = args.stages is None or stage["name"] in args.stages
        for spec in specs:
            status = "done" if done(spec["name"]) else "to run"
            if args.plan or not wanted or (args.only and spec["name"] != args.only):
                log(f"{stage['name']} {spec['name']}: {status}{'' if wanted else ' (stage not requested)'}")
                continue
            if status == "done":
                log(f"{stage['name']} {spec['name']}: already done, skipped")
                continue
            cmd = [sys.executable, "scripts/train.py", "--mode", spec["mode"], "--lambda-d", str(spec["lambda_d"]),
                   "--seed", str(spec["seed"])]
            if "lr" in spec:
                cmd += ["--lr", str(spec["lr"])]
            if args.out_root:
                cmd += ["--out-root", args.out_root]
            if args.ckpt_root:
                cmd += ["--ckpt-root", args.ckpt_root]
            cmd += args.extra
            log(f"{stage['name']} {spec['name']}: start")
            start = time.time()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tail = []
            for line in proc.stdout:
                line = line.rstrip()
                tail = (tail + [line])[-40:]
                if line.startswith(spec["name"]) or line.startswith("{") or "Error" in line or "Traceback" in line:
                    log(line)
            if proc.wait() != 0:
                log(f"{spec['name']}: FAILED (exit {proc.returncode}). Last output:")
                for line in tail:
                    log("  " + line)
                log("Rerun the same command to resume it; send the lines above to Claude if it fails again.")
                sys.exit(1)
            log(f"{stage['name']} {spec['name']}: finished in {(time.time() - start) / 3600:.2f} h")

        if stage.get("select"):
            results = {s["name"]: done(s["name"]) for s in specs}
            try:
                best = select_best(specs, results, cfg["train"]["selection"])
            except ValueError as e:
                log(f"{stage['name']}: no selection yet ({e})")
                if args.plan or not wanted or args.only:
                    break  # later stages need this stage's choice
                log("stop: a selecting stage did not finish all its runs")
                sys.exit(1)
            selection[stage["name"]] = {
                "selected": best["name"],
                "candidates": {n: r["best_val"]["val_absrel_all"] for n, r in results.items()},
            }
            selection_path.write_text(json.dumps(selection, indent=2) + "\n")
            log(f"{stage['name']}: selected {best['name']} (val absrel_all "
                f"{results[best['name']]['best_val']['val_absrel_all']:.4f})")
    log("queue end")


if __name__ == "__main__":
    main()
