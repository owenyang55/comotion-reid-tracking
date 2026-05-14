"""Phase 0 基线运行与汇总工具。

本脚本刻意独立于 ``run_eval.py``，用于为后续消融实验生成可复现的
baseline 结果包：

* 可选地对一个或多个视频/帧目录运行 ``demo.py``；
* 生成不依赖 ground truth 的 MOT tracker 自身统计；
* 当存在 MOT 格式 ground truth 时，可选运行 TrackEval 指标；
* 在同一输出目录下生成 JSON/CSV/Markdown 汇总。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "phase0_baseline"
KEY_TRACKER_METRICS = (
    "HOTA",
    "DetA",
    "AssA",
    "IDF1",
    "IDSW",
    "MOTA",
    "Frag",
)


@dataclass(frozen=True)
class MotRow:
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行并汇总可复现的 Phase 0 CoMotion baseline。"
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="输入视频、图片目录或单张图片。多个序列可重复传入。",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help="已有 MOT txt 预测目录。若省略，则使用 demo.py 输出目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="预测结果和汇总文件的输出目录。",
    )
    parser.add_argument(
        "--run-name",
        default="baseline",
        help="报告名称，也会用于可选 TrackEval tracker 目录。",
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="只汇总已有预测结果，不调用 demo.py。",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument(
        "--require-reid",
        action="store_true",
        help="透传给 demo.py：如果 DINOv2 ReID 不可用则直接失败。",
    )
    parser.add_argument(
        "--extra-demo-arg",
        action="append",
        default=[],
        help="额外透传给 demo.py 的原始参数。多个参数可重复传入。",
    )
    parser.add_argument(
        "--run-trackeval",
        action="store_true",
        help="同时运行 TrackEval。需要 MOT 格式 GT 和 seqmap。",
    )
    parser.add_argument(
        "--gt-folder",
        type=Path,
        default=EVAL_ROOT / "data" / "gt" / "mot_challenge",
        help="TrackEval ground truth 根目录。",
    )
    parser.add_argument(
        "--benchmark",
        default="3DPW",
        help="TrackEval benchmark 前缀，例如 3DPW 或 mot。",
    )
    parser.add_argument(
        "--split",
        default="Test",
        help="TrackEval split 后缀，例如 Test 或 test。",
    )
    parser.add_argument(
        "--seqmap-file",
        type=Path,
        default=None,
        help="可选：显式指定 TrackEval seqmap 文件。",
    )
    parser.add_argument(
        "--skip-split-folder",
        action="store_true",
        help="自定义扁平 MOT 目录布局时，使用 TrackEval SKIP_SPLIT_FOL=True。",
    )
    parser.add_argument(
        "--seqmap-from-predictions",
        action="store_true",
        help="根据预测目录中的 txt 文件自动生成临时 seqmap，只评估已有预测的序列。",
    )
    parser.add_argument(
        "--plot-timelines",
        action="store_true",
        help="为每个 MOT txt 文件输出简单的轨迹时间线 PNG。",
    )
    return parser.parse_args()


def run_demo(args: argparse.Namespace, prediction_dir: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    if args.skip_demo:
        return commands
    if not args.input:
        raise ValueError("请至少提供一个 --input，或使用 --skip-demo 搭配 --prediction-dir。")

    prediction_dir.mkdir(parents=True, exist_ok=True)
    for input_path in args.input:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "demo.py"),
            "-i",
            str(input_path),
            "-o",
            str(prediction_dir),
            "--skip-visualization",
            "--start-frame",
            str(args.start_frame),
            "--num-frames",
            str(args.num_frames),
            "--frameskip",
            str(args.frameskip),
        ]
        if args.require_reid:
            cmd.append("--require-reid")
        cmd.extend(args.extra_demo_arg)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        commands.append(cmd)
    return commands


def read_mot_file(path: Path) -> list[MotRow]:
    rows: list[MotRow] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no} 不是有效的 MOT 格式: {stripped}")
            rows.append(
                MotRow(
                    frame=int(float(parts[0])),
                    track_id=int(float(parts[1])),
                    left=float(parts[2]),
                    top=float(parts[3]),
                    width=float(parts[4]),
                    height=float(parts[5]),
                    conf=float(parts[6]) if len(parts) > 6 else 1.0,
                )
            )
    return rows


def contiguous_segments(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    sorted_frames = sorted(set(frames))
    segments: list[tuple[int, int]] = []
    start = sorted_frames[0]
    prev = sorted_frames[0]
    for frame in sorted_frames[1:]:
        if frame > prev + 1:
            segments.append((start, prev))
            start = frame
        prev = frame
    segments.append((start, prev))
    return segments


def summarize_mot_file(path: Path) -> dict[str, Any]:
    rows = read_mot_file(path)
    by_id: dict[int, list[MotRow]] = {}
    for row in rows:
        by_id.setdefault(row.track_id, []).append(row)

    frame_values = [row.frame for row in rows]
    track_lengths = [len(track_rows) for track_rows in by_id.values()]
    fragment_count = 0
    max_gap = 0
    for track_rows in by_id.values():
        frames = [row.frame for row in track_rows]
        segments = contiguous_segments(frames)
        fragment_count += max(0, len(segments) - 1)
        for first, second in zip(segments, segments[1:]):
            max_gap = max(max_gap, second[0] - first[1] - 1)

    avg_area = mean([row.width * row.height for row in rows]) if rows else 0.0
    return {
        "sequence": path.stem,
        "mot_file": str(path),
        "num_rows": len(rows),
        "frame_start": min(frame_values) if frame_values else None,
        "frame_end": max(frame_values) if frame_values else None,
        "num_frames_with_detections": len(set(frame_values)),
        "num_ids": len(by_id),
        "track_fragments_internal": fragment_count,
        "max_internal_gap": max_gap,
        "mean_track_length": round(mean(track_lengths), 3) if track_lengths else 0.0,
        "short_tracks_lt5": sum(1 for length in track_lengths if length < 5),
        "avg_box_area": round(avg_area, 3),
    }


def summarize_predictions(prediction_dir: Path) -> list[dict[str, Any]]:
    mot_files = sorted(prediction_dir.glob("*.txt"))
    return [summarize_mot_file(path) for path in mot_files]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "无数据行。\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def prepare_trackeval_tracker(prediction_dir: Path, args: argparse.Namespace) -> Path:
    tracker_root = args.output_dir / "trackeval" / "trackers" / "mot_challenge"
    if args.skip_split_folder:
        tracker_data_dir = tracker_root / args.run_name / "data"
    else:
        tracker_data_dir = tracker_root / f"{args.benchmark}-{args.split}" / args.run_name / "data"
    tracker_data_dir.mkdir(parents=True, exist_ok=True)
    for mot_file in sorted(prediction_dir.glob("*.txt")):
        shutil.copy2(mot_file, tracker_data_dir / mot_file.name)
    return tracker_root


def trackeval_gt_file(args: argparse.Namespace, seq_name: str) -> Path:
    if args.skip_split_folder:
        gt_root = args.gt_folder
    else:
        gt_root = args.gt_folder / f"{args.benchmark}-{args.split}"
    return gt_root / seq_name / "gt" / "gt.txt"


def write_seqmap_from_predictions(prediction_dir: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    seq_names = []
    skipped = []
    for path in sorted(prediction_dir.glob("*.txt")):
        if trackeval_gt_file(args, path.stem).exists():
            seq_names.append(path.stem)
        else:
            skipped.append(path.stem)
    if not seq_names:
        raise ValueError("预测目录中的 txt 没有匹配到可用的 TrackEval GT。")
    if skipped:
        print(f"以下预测文件没有匹配 GT，已从临时 seqmap 跳过: {', '.join(skipped)}")
    seqmap_path = output_dir / "phase0_seqmap_from_predictions.txt"
    seqmap_path.write_text("name\n" + "\n".join(seq_names) + "\n", encoding="utf-8")
    return seqmap_path


def scalar(value: Any) -> float:
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        return float(value.item())
    if hasattr(value, "mean"):
        return float(value.mean())
    return float(value)


def run_trackeval(prediction_dir: Path, args: argparse.Namespace) -> dict[str, float]:
    trackeval_root = EVAL_ROOT / "TrackEval"
    if str(trackeval_root) not in sys.path:
        sys.path.insert(0, str(trackeval_root))

    import numpy as np

    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int

    import trackeval.metrics as mod_metrics
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from trackeval.eval import Evaluator

    tracker_root = prepare_trackeval_tracker(prediction_dir, args)
    seqmap_file = args.seqmap_file
    if args.seqmap_from_predictions:
        seqmap_file = write_seqmap_from_predictions(prediction_dir, args.output_dir, args)

    eval_config = Evaluator.get_default_eval_config()
    dataset_config = MotChallenge2DBox.get_default_dataset_config()

    eval_config.update(
        {
            "PRINT_ONLY_COMBINED": True,
            "DISPLAY_LESS_PROGRESS": True,
            "OUTPUT_SUMMARY": True,
            "PRINT_RESULTS": False,
        }
    )
    dataset_config.update(
        {
            "GT_FOLDER": str(args.gt_folder),
            "TRACKERS_FOLDER": str(tracker_root),
            "OUTPUT_FOLDER": str(args.output_dir / "trackeval" / "results"),
            "TRACKERS_TO_EVAL": [args.run_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": args.benchmark,
            "SPLIT_TO_EVAL": args.split,
            "INPUT_AS_ZIP": False,
            "DO_PREPROC": False,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SEQMAP_FILE": str(seqmap_file) if seqmap_file else None,
            "SKIP_SPLIT_FOL": args.skip_split_folder,
            "PRINT_CONFIG": False,
        }
    )

    evaluator = Evaluator(eval_config)
    dataset_list = [MotChallenge2DBox(dataset_config)]
    metrics_list = [getattr(mod_metrics, name)() for name in ("HOTA", "CLEAR", "Identity")]
    raw_results, _ = evaluator.evaluate(dataset_list, metrics_list)

    dataset_name = dataset_list[0].get_name()
    combined = raw_results[dataset_name][args.run_name]["COMBINED_SEQ"]["pedestrian"]
    return {
        "HOTA": scalar(combined["HOTA"]["HOTA"]),
        "DetA": scalar(combined["HOTA"]["DetA"]),
        "AssA": scalar(combined["HOTA"]["AssA"]),
        "IDF1": scalar(combined["Identity"]["IDF1"]),
        "IDSW": scalar(combined["CLEAR"]["IDSW"]),
        "MOTA": scalar(combined["CLEAR"]["MOTA"]),
        "Frag": scalar(combined["CLEAR"]["Frag"]),
    }


def plot_timelines(prediction_dir: Path, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths: list[str] = []
    for mot_file in sorted(prediction_dir.glob("*.txt")):
        rows = read_mot_file(mot_file)
        by_id: dict[int, list[int]] = {}
        for row in rows:
            by_id.setdefault(row.track_id, []).append(row.frame)
        if not by_id:
            continue

        sorted_ids = sorted(by_id, key=lambda track_id: min(by_id[track_id]))
        y_map = {track_id: idx for idx, track_id in enumerate(sorted_ids)}
        fig_height = max(4, min(18, 1 + len(sorted_ids) * 0.35))
        fig, ax = plt.subplots(figsize=(14, fig_height))
        cmap = plt.get_cmap("tab20")
        for track_id in sorted_ids:
            y = y_map[track_id]
            for start, end in contiguous_segments(by_id[track_id]):
                ax.broken_barh([(start, end - start + 1)], (y - 0.35, 0.7), facecolors=cmap(track_id % 20))
            ax.text(min(by_id[track_id]), y, f"ID {track_id}", va="center", ha="right", fontsize=7)
        ax.set_title(f"{mot_file.stem} 轨迹时间线")
        ax.set_xlabel("帧号")
        ax.set_ylabel("轨迹 ID 顺序")
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        out_path = output_dir / f"timeline_{mot_file.stem}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths.append(str(out_path))
    return paths


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    commands: list[list[str]],
    tracker_rows: list[dict[str, Any]],
    trackeval_metrics: dict[str, float] | None,
    timeline_paths: list[str],
) -> None:
    lines = [
        f"# Phase 0 基线报告：{args.run_name}",
        "",
        "## 复现命令",
        "",
    ]
    if commands:
        lines.extend(["已执行命令：", ""])
        for command in commands:
            lines.extend(["```powershell", " ".join(command), "```", ""])
    else:
        lines.append("未执行 demo 命令；本报告由已有预测结果生成。")
        lines.append("")

    lines.extend(
        [
            "## Tracker 自身统计",
            "",
            markdown_table(
                tracker_rows,
                [
                    "sequence",
                    "num_rows",
                    "num_frames_with_detections",
                    "num_ids",
                    "track_fragments_internal",
                    "max_internal_gap",
                    "mean_track_length",
                    "short_tracks_lt5",
                ],
            ),
        ]
    )

    lines.extend(["", "## TrackEval 指标", ""])
    if trackeval_metrics:
        metric_rows = [{"metric": key, "value": trackeval_metrics[key]} for key in KEY_TRACKER_METRICS]
        lines.append(markdown_table(metric_rows, ["metric", "value"]))
    else:
        lines.append("未运行 TrackEval。提供 MOT 格式 ground truth 并添加 `--run-trackeval` 后，可生成 HOTA/IDF1/IDSW/MOTA。")

    if timeline_paths:
        lines.extend(["", "## 轨迹时间线可视化", ""])
        for timeline_path in timeline_paths:
            lines.append(f"- `{timeline_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_dir = args.prediction_dir or (args.output_dir / "predictions")
    commands = run_demo(args, prediction_dir)

    tracker_rows = summarize_predictions(prediction_dir)
    trackeval_metrics = run_trackeval(prediction_dir, args) if args.run_trackeval else None
    timeline_paths = plot_timelines(prediction_dir, args.output_dir) if args.plot_timelines else []

    payload = {
        "run_name": args.run_name,
        "prediction_dir": str(prediction_dir),
        "commands": commands,
        "tracker_summary": tracker_rows,
        "trackeval_metrics": trackeval_metrics,
        "timeline_paths": timeline_paths,
    }
    write_json(args.output_dir / "phase0_summary.json", payload)
    write_csv(args.output_dir / "phase0_tracker_summary.csv", tracker_rows)
    if trackeval_metrics:
        write_csv(
            args.output_dir / "phase0_trackeval_metrics.csv",
            [{"metric": key, "value": trackeval_metrics[key]} for key in KEY_TRACKER_METRICS],
        )
    write_markdown_report(
        args.output_dir / "phase0_report.md",
        args,
        commands,
        tracker_rows,
        trackeval_metrics,
        timeline_paths,
    )
    print(f"Phase 0 报告已写入: {args.output_dir / 'phase0_report.md'}")


if __name__ == "__main__":
    main()
