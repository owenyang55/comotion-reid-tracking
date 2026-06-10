"""Run FastTracker-style occlusion ablations through phase0_baseline.py."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "fasttracker_occlusion_ablation"


@dataclass(frozen=True)
class Experiment:
    name: str
    demo_args: tuple[str, ...]
    note: str


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "baseline_off",
        (
            "--memory-mode",
            "off",
            "--disable-reid-revival",
            "--postprocess-mode",
            "off",
        ),
        "No lost-track revive and no offline postprocess.",
    ),
    Experiment(
        "long_reid",
        (
            "--memory-mode",
            "long",
            "--postprocess-mode",
            "off",
            "--revive-accept-thr",
            "0.3",
        ),
        "Current long ReID revive, but interpolation/stitching disabled.",
    ),
    Experiment(
        "occ_gate",
        (
            "--memory-mode",
            "off",
            "--disable-reid-revival",
            "--enable-occlusion-gate",
            "--postprocess-mode",
            "off",
        ),
        "Only occlusion-gated active hold.",
    ),
    Experiment(
        "occ_gate_damped",
        (
            "--memory-mode",
            "off",
            "--disable-reid-revival",
            "--enable-occlusion-gate",
            "--enable-motion-damping",
            "--postprocess-mode",
            "off",
        ),
        "Occlusion hold plus damped motion.",
    ),
    Experiment(
        "occ_reid",
        (
            "--memory-mode",
            "occlusion",
            "--enable-occlusion-gate",
            "--postprocess-mode",
            "off",
            "--revive-accept-thr",
            "0.65",
            "--revive-occluded-app-thr",
            "0.6",
            "--revive-unoccluded-app-thr",
            "0.8",
        ),
        "Only recently occluded tracks may use ReID revive.",
    ),
    Experiment(
        "occ_reid_damped",
        (
            "--memory-mode",
            "occlusion",
            "--enable-occlusion-gate",
            "--enable-motion-damping",
            "--postprocess-mode",
            "off",
            "--revive-accept-thr",
            "0.65",
            "--revive-occluded-app-thr",
            "0.6",
            "--revive-unoccluded-app-thr",
            "0.8",
        ),
        "Occlusion-limited ReID revive plus damped motion.",
    ),
    Experiment(
        "occ_reid_damped_init_suppress",
        (
            "--memory-mode",
            "occlusion",
            "--enable-occlusion-gate",
            "--enable-motion-damping",
            "--enable-init-iou-suppress",
            "--postprocess-mode",
            "off",
            "--revive-accept-thr",
            "0.65",
            "--revive-occluded-app-thr",
            "0.6",
            "--revive-unoccluded-app-thr",
            "0.8",
        ),
        "Adds duplicate initialization suppression.",
    ),
    Experiment(
        "combined_stitch",
        (
            "--memory-mode",
            "occlusion",
            "--enable-occlusion-gate",
            "--enable-motion-damping",
            "--enable-init-iou-suppress",
            "--postprocess-mode",
            "stitch",
            "--revive-accept-thr",
            "0.65",
            "--revive-occluded-app-thr",
            "0.6",
            "--revive-unoccluded-app-thr",
            "0.8",
        ),
        "Combined online changes plus offline stitching.",
    ),
    Experiment(
        "combined_interp",
        (
            "--memory-mode",
            "occlusion",
            "--enable-occlusion-gate",
            "--enable-motion-damping",
            "--enable-init-iou-suppress",
            "--postprocess-mode",
            "interpolate",
            "--revive-accept-thr",
            "0.65",
            "--revive-occluded-app-thr",
            "0.6",
            "--revive-unoccluded-app-thr",
            "0.8",
        ),
        "Combined online changes plus interpolation only.",
    ),
    Experiment(
        "combined_stitch_interp",
        (
            "--memory-mode",
            "occlusion",
            "--enable-occlusion-gate",
            "--enable-motion-damping",
            "--enable-init-iou-suppress",
            "--postprocess-mode",
            "stitch-interpolate",
            "--revive-accept-thr",
            "0.65",
            "--revive-occluded-app-thr",
            "0.6",
            "--revive-unoccluded-app-thr",
            "0.8",
        ),
        "Combined online changes plus stitching and interpolation.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--sequence-name", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--require-reid", action="store_true")
    parser.add_argument("--run-trackeval", action="store_true")
    parser.add_argument("--gt-folder", type=Path, default=None)
    parser.add_argument("--benchmark", default="MOT20")
    parser.add_argument("--split", default="train")
    parser.add_argument("--skip-split-folder", action="store_true")
    parser.add_argument("--seqmap-from-predictions", action="store_true")
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--extra-demo-arg", action="append", default=[])
    return parser.parse_args()


def selected_experiments(names: list[str]) -> list[Experiment]:
    if not names:
        return list(EXPERIMENTS)
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(
            "Unknown experiment(s): "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(by_name)
        )
    return [by_name[name] for name in names]


def append_repeated(cmd: list[str], flag: str, values: list[object]) -> None:
    for value in values:
        if flag == "--extra-demo-arg":
            cmd.append(f"{flag}={value}")
            continue
        cmd.extend([flag, str(value)])


def run_experiment(args: argparse.Namespace, experiment: Experiment) -> Path:
    run_dir = args.output_dir / experiment.name
    prediction_dir = run_dir / "predictions"
    revive_log_path = run_dir / "revive_log.csv"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "evaluation" / "phase0_baseline.py"),
        "--output-dir",
        str(run_dir),
        "--prediction-dir",
        str(prediction_dir),
        "--run-name",
        experiment.name,
        "--demo-root",
        str(args.demo_root),
        "--start-frame",
        str(args.start_frame),
        "--num-frames",
        str(args.num_frames),
        "--frameskip",
        str(args.frameskip),
    ]
    append_repeated(cmd, "--input", args.input)
    append_repeated(cmd, "--sequence-name", args.sequence_name)

    if args.require_reid:
        cmd.append("--require-reid")
    if args.run_trackeval:
        cmd.append("--run-trackeval")
    if args.gt_folder is not None:
        cmd.extend(["--gt-folder", str(args.gt_folder)])
    if args.benchmark:
        cmd.extend(["--benchmark", args.benchmark])
    if args.split:
        cmd.extend(["--split", args.split])
    if args.skip_split_folder:
        cmd.append("--skip-split-folder")
    if args.seqmap_from_predictions:
        cmd.append("--seqmap-from-predictions")

    demo_args = list(experiment.demo_args) + ["--revive-log-path", str(revive_log_path)]
    append_repeated(cmd, "--extra-demo-arg", demo_args)
    append_repeated(cmd, "--extra-demo-arg", args.extra_demo_arg)

    print(f"\n=== Running {experiment.name}: {experiment.note} ===")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return run_dir


def read_key_value_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def read_tracker_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    keys = [
        "num_rows",
        "num_ids",
        "track_fragments_internal",
        "max_internal_gap",
        "mean_track_length",
        "short_tracks_lt5",
    ]
    return {key: rows[0].get(key, "") for key in keys}


def write_summary(output_dir: Path, experiments: list[Experiment], run_dirs: list[Path]) -> None:
    rows = []
    for experiment, run_dir in zip(experiments, run_dirs):
        row = {"experiment": experiment.name, "note": experiment.note}
        row.update(read_tracker_summary(run_dir / "phase0_tracker_summary.csv"))
        row.update(read_key_value_csv(run_dir / "phase0_trackeval_metrics.csv"))
        rows.append(row)

    columns = sorted({key for row in rows for key in row})
    preferred = [
        "experiment",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
        "IDSW",
        "MOTA",
        "Frag",
        "num_rows",
        "num_ids",
        "track_fragments_internal",
        "max_internal_gap",
        "mean_track_length",
        "short_tracks_lt5",
        "note",
    ]
    columns = [col for col in preferred if col in columns] + [
        col for col in columns if col not in preferred
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "ablation_summary.md"
    lines = [
        "# FastTracker-Style Occlusion Ablation",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {csv_path}")
    print(f"Markdown summary written to {md_path}")


def main() -> None:
    args = parse_args()
    experiments = selected_experiments(args.experiment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [run_experiment(args, experiment) for experiment in experiments]
    write_summary(args.output_dir, experiments, run_dirs)


if __name__ == "__main__":
    main()
