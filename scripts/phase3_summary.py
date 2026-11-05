"""Summarises the Phase 3 runs into results/phase3/summary.md and summary.csv (the hand-back file).

    python scripts/phase3_summary.py

One row per run, finished or not: settings, best epoch, val metrics at the best epoch (val only;
the test sequence is never touched in Phase 3), the selections each stage made, and the untrained
DA3-S on the same validator as a reference.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = ["absrel_all", "absrel_grass", "absrel_vegetation", "absrel_edge_depth", "absrel_grass_far",
           "conf_floor", "conf_median"]


def read_log(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def summarise(runs_root: Path) -> tuple[list[dict], dict | None]:
    rows, reference = [], None
    for run in sorted(p for p in runs_root.iterdir() if (p / "log.csv").exists()):
        log = read_log(run / "log.csv")
        done = json.loads((run / "done.json").read_text()) if (run / "done.json").exists() else None
        trained = [r for r in log if int(r["epoch"]) > 0]
        if reference is None and log:
            reference = {m: float(log[0][f"val_{m}"]) for m in METRICS}
        if done:
            best_epoch, status = done["best_epoch"], done["stopped"]
            spec = done["spec"]
            hours = done["train_seconds"] / 3600
        else:
            best_row = min(trained, key=lambda r: float(r["val_absrel_all"])) if trained else None
            best_epoch, status, spec = (int(best_row["epoch"]) if best_row else None), "unfinished", {}
            hours = sum(float(r["time_s"]) for r in trained) / 3600
        best = next((r for r in log if best_epoch is not None and int(r["epoch"]) == best_epoch), None)
        row = {"run": run.name, "mode": spec.get("mode", ""), "lr": spec.get("lr", ""),
               "lambda_d": spec.get("lambda_d", ""), "seed": spec.get("seed", ""), "status": status,
               "best_epoch": best_epoch, "epochs_run": len(trained), "hours": round(hours, 2)}
        for m in METRICS:
            row[m] = round(float(best[f"val_{m}"]), 4) if best else None
        rows.append(row)
    return rows, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase3", default=str(ROOT / "results" / "phase3"), help="debug: another phase3 folder")
    args = parser.parse_args()
    phase3 = Path(args.phase3)
    rows, reference = summarise(phase3 / "runs")
    if not rows:
        sys.exit(f"no runs under {phase3 / 'runs'}")
    selection = json.loads((phase3 / "selection.json").read_text()) if (phase3 / "selection.json").exists() else {}

    with open(phase3 / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cols = ["run", "status", "best_epoch", "epochs_run", "hours"] + METRICS
    lines = ["# Phase 3 runs (val 00001 only)", "",
             "Val AbsRel at each run's best epoch, median alignment, official GT. `absrel_all` selects;",
             "the rest are logged only. `conf_floor` is the share of valid pixels with confidence below 1.05.", "",
             "| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    if reference:
        lines.append("| zero-shot DA3-S (epoch 0) | - | - | - | - | "
                     + " | ".join(f"{reference[m]:.4f}" for m in METRICS) + " |")
    for r in rows:
        lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    lines += ["", "## Selections", ""]
    for stage, s in selection.items():
        cands = ", ".join(f"{n} {v:.4f}" for n, v in s["candidates"].items())
        lines.append(f"- **{stage}**: {s['selected']} (candidates: {cands})")
    if not selection:
        lines.append("- none yet")
    total = sum(r["hours"] for r in rows)
    lines += ["", f"Total training time: {total:.1f} h over {len(rows)} runs."]
    (phase3 / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
