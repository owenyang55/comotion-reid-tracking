from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".flv", ".webm"}


def discover_videos(input_root: Path) -> list[Path]:
	videos = [
		path
		for path in input_root.rglob("*")
		if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
	]
	videos.sort()
	return videos


def build_env(local_src: Path) -> dict[str, str]:
	env = os.environ.copy()
	existing = env.get("PYTHONPATH", "")
	if existing:
		env["PYTHONPATH"] = f"{local_src}{os.pathsep}{existing}"
	else:
		env["PYTHONPATH"] = str(local_src)
	return env


def run_demo(
	python_exec: str,
	demo_script: Path,
	video_path: Path,
	output_dir: Path,
	start_frame: int,
	num_frames: int,
	frameskip: int,
	skip_visualization: bool,
	env: dict[str, str],
) -> int:
	cmd = [
		python_exec,
		str(demo_script),
		"-i",
		str(video_path),
		"-o",
		str(output_dir),
		"-s",
		str(start_frame),
		"-n",
		str(num_frames),
		"--frameskip",
		str(frameskip),
	]
	if skip_visualization:
		cmd.append("--skip-visualization")

	print(f"\n[RUN] {' '.join(cmd)}")
	return subprocess.run(cmd, env=env).returncode


def main() -> int:
	script_dir = Path(__file__).resolve().parent
	workspace_root = script_dir.parent
	demo_script = script_dir / "demo.py"
	local_src = script_dir / "src"

	parser = argparse.ArgumentParser(
		description="Batch run demo.py for video/jinshe and write outputs to out/jinshe"
	)
	parser.add_argument(
		"--input-root",
		type=Path,
		default=workspace_root / "video" / "jinshe",
		help="Input root containing subfolders of videos",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=workspace_root / "out" / "jinshe",
		help="Output root",
	)
	parser.add_argument("--start-frame", type=int, default=0)
	parser.add_argument("--num-frames", type=int, default=1_000_000_000)
	parser.add_argument("--frameskip", type=int, default=1)
	parser.add_argument("--skip-visualization", action="store_true")
	parser.add_argument("--continue-on-error", action="store_true")
	parser.add_argument(
		"--python",
		default=sys.executable,
		help="Python executable used to run demo.py",
	)

	args = parser.parse_args()

	input_root = args.input_root.resolve()
	output_root = args.output_root.resolve()

	if not demo_script.exists():
		print(f"[ERROR] demo.py not found: {demo_script}")
		return 2
	if not local_src.exists():
		print(f"[ERROR] local src not found: {local_src}")
		return 2
	if not input_root.exists():
		print(f"[ERROR] input root not found: {input_root}")
		return 2

	videos = discover_videos(input_root)
	if not videos:
		print(f"[WARN] no videos found under: {input_root}")
		return 0

	output_root.mkdir(parents=True, exist_ok=True)
	env = build_env(local_src)

	print(f"[INFO] Found {len(videos)} videos")
	print(f"[INFO] Input root:  {input_root}")
	print(f"[INFO] Output root: {output_root}")
	print(f"[INFO] PYTHONPATH prefix: {local_src}")

	processed = 0
	failed: list[Path] = []

	for index, video in enumerate(videos, start=1):
		relative_folder = video.parent.relative_to(input_root)
		target_dir = output_root / relative_folder
		target_dir.mkdir(parents=True, exist_ok=True)

		print(f"\n[{index}/{len(videos)}] Processing: {video}")
		code = run_demo(
			python_exec=args.python,
			demo_script=demo_script,
			video_path=video,
			output_dir=target_dir,
			start_frame=args.start_frame,
			num_frames=args.num_frames,
			frameskip=args.frameskip,
			skip_visualization=args.skip_visualization,
			env=env,
		)
		processed += 1

		if code != 0:
			failed.append(video)
			print(f"[ERROR] Failed ({code}): {video}")
			if not args.continue_on_error:
				print("[INFO] stop on first error (use --continue-on-error to keep going)")
				break

	print("\n========== Summary ==========")
	print(f"Total: {len(videos)}")
	print(f"Processed: {processed}")
	print(f"Failed: {len(failed)}")
	print(f"Succeeded: {processed - len(failed)}")

	if failed:
		print("Failed files:")
		for path in failed:
			print(f" - {path}")
		return 1

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
