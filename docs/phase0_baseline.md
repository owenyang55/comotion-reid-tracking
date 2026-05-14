# Phase 0 基线评估流程

本流程用于在 A/B/C 三条算法线开始前固定 baseline。后续每次改动都应使用同一组验证输入、同一帧范围和同一份统计口径，避免改进结果无法比较。

默认输出目录为 `results/phase0_baseline/`。

## Phase 0 产物

- `predictions/*.pt` 和 `predictions/*.txt`：由 `demo.py` 生成的推理结果。
- `phase0_tracker_summary.csv`：不依赖 ground truth 的 tracker 自身统计。
- `phase0_summary.json`：记录命令、路径、统计和可选 TrackEval 指标。
- `phase0_report.md`：便于快速查看的阶段报告。
- 可选 `timeline_*.png`：每个 MOT txt 的轨迹时间线图。
- 可选 `phase0_trackeval_metrics.csv`：包含 HOTA、IDF1、IDSW、MOTA、Frag 等指标。

tracker 自身统计不需要 ground truth。TrackEval 指标需要 MOT 格式 ground truth 和 seqmap 文件。

运行 TrackEval 前需要确保当前 Python 环境安装了 `scipy`。如果报错 `No module named 'scipy'`，先在项目环境中安装：

```powershell
python -m pip install scipy
```

## 当前统计指标

`phase0_tracker_summary.csv` 是不依赖标注的 tracker 自身统计，适合快速检查输出质量：

| 字段 | 含义 | 趋势 |
| --- | --- | --- |
| `sequence` | 序列名，来自 MOT txt 文件名。 | 用于定位样例 |
| `mot_file` | 被统计的 MOT txt 路径。 | 用于复现 |
| `num_rows` | MOT 输出行数，即总检测/轨迹框数量。 | 不是越高越好，需结合漏检和误检判断 |
| `frame_start` / `frame_end` | 输出中最早和最晚帧号。 | 应覆盖设定评估帧段 |
| `num_frames_with_detections` | 有预测框的帧数。 | 过低通常说明漏检或推理失败 |
| `num_ids` | 输出中出现过的不同轨迹 ID 数。 | 拥挤场景过高可能表示碎片化 |
| `track_fragments_internal` | 同一个 ID 内部非连续片段数量。 | 越低越好 |
| `max_internal_gap` | 同一个 ID 内最大断裂帧数。 | 越低越好 |
| `mean_track_length` | 平均轨迹长度。 | 一般越长越稳定，但需结合场景人数 |
| `short_tracks_lt5` | 长度小于 5 帧的短轨迹数量。 | 越低越好 |
| `avg_box_area` | 平均 bbox 面积。 | 用于排查尺度异常，不直接代表性能 |

`phase0_trackeval_metrics.csv` 是有 ground truth 时的正式跟踪评估指标：

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| `HOTA` | 综合检测、关联和定位质量的高阶跟踪指标。 | 越高越好 |
| `DetA` | HOTA 中的检测准确性。 | 越高越好 |
| `AssA` | HOTA 中的关联准确性，更关注身份连续性。 | 越高越好 |
| `IDF1` | 身份保持 F1 分数，常用于衡量 ID switch 和重识别效果。 | 越高越好 |
| `IDSW` | ID Switch 次数。 | 越低越好 |
| `MOTA` | 综合漏检、误检和 IDSW 的 CLEAR MOT 指标。 | 越高越好 |
| `Frag` | 轨迹碎片化次数。 | 越低越好 |

论文实验中建议优先报告 `HOTA`、`IDF1`、`IDSW`、`MOTA`、`Frag`，同时保留 tracker 自身统计用于解释失败案例。

## 当前仓库内可用数据

仓库中已经存在评估用的 MOT 格式标注和部分预测结果：

| 数据 | 路径 | 当前状态 | 用途 |
| --- | --- | --- | --- |
| 3DPW-Test ground truth | `evaluation\data\gt\mot_challenge\3DPW-Test` | 24 个序列，每个序列包含 `gt\gt.txt` 和 `seqinfo.ini`。 | 推荐作为 Phase 0 正式评估 GT |
| 3DPW seqmap | `evaluation\data\gt\mot_challenge\seqmaps\3DPW-Test.txt` | 列出 24 个 3DPW-Test 序列。 | 全量 3DPW 评估 |
| 3DPW 已有 tracker 结果 | `evaluation\data\trackers\mot_challenge\3DPW-Test\original\data` 和 `...\improved\data` | 只有部分序列有 `.txt`，另有 `jinshe.txt`、`tl.txt` 等无对应 GT 的本地结果。 | 可做已有结果复盘，需用自动 seqmap 跳过无 GT 文件 |
| MOT ground truth | `evaluation\data\gt\mot_challenge\mot` | `MOT20-02`、`MOT20-03` 有 `gt\gt.txt`；`MOT20-06`、`MOT20-07` 当前只有 `det\det.txt`，没有 GT。 | 可用于少量 MOT 风格评估 |
| MOT seqmap | `evaluation\data\gt\mot_challenge\seqmaps\MOT-Test.txt` | 当前只列出 `MOT20-02`。 | 可直接评估 `MOT20-02` |
| MOT 已有 tracker 结果 | `evaluation\data\trackers\mot_challenge\mot\original\data` 和 `...\improved\data` | 有 `MOT20-02/03/06/07.txt`。 | 与 `MOT-Test.txt` 搭配时只评估 `MOT20-02` |

当前仓库没有发现 `datasets\3DPW\imageFiles` 这类原始图片目录，也没有发现可直接推理的 3DPW 原始视频目录。因此，现有仓库可以直接评估“已有 tracker txt”，但若要重新跑 `demo.py` 生成 baseline，需要额外提供原始视频或图片帧目录。

## 最小基线运行

选择一个或多个短视频、图片目录或单张图片作为固定验证集。后续消融实验应复用这些输入。

```powershell
python evaluation\phase0_baseline.py `
  --input path\to\clip_or_frames `
  --output-dir results\phase0_baseline `
  --run-name baseline `
  --num-frames 300 `
  --frameskip 1 `
  --plot-timelines
```

脚本内部会调用：

```powershell
python demo.py -i <input> -o results\phase0_baseline\predictions --skip-visualization --start-frame 0 --num-frames 300 --frameskip 1
```

## 汇总已有预测结果

如果 baseline 的 MOT txt 已经存在，可以跳过推理，只汇总已有预测：

```powershell
python evaluation\phase0_baseline.py `
  --skip-demo `
  --prediction-dir results\phase0_baseline\predictions `
  --output-dir results\phase0_baseline `
  --run-name baseline `
  --plot-timelines
```

## 加入 TrackEval 指标

如果已经有预测文件，并且预测序列与 3DPW ground truth 对齐，推荐用以下命令。`--seqmap-from-predictions` 会根据预测目录中的 `.txt` 自动生成临时 seqmap，并自动跳过没有对应 GT 的本地文件：

```powershell
python evaluation\phase0_baseline.py `
  --skip-demo `
  --prediction-dir evaluation\data\trackers\mot_challenge\3DPW-Test\improved\data `
  --output-dir results\phase0_3dpw_improved `
  --run-name improved `
  --run-trackeval `
  --benchmark 3DPW `
  --split Test `
  --seqmap-from-predictions
```

同理，复盘已有 `original` 结果时只需要替换预测目录和输出目录：

```powershell
python evaluation\phase0_baseline.py `
  --skip-demo `
  --prediction-dir evaluation\data\trackers\mot_challenge\3DPW-Test\original\data `
  --output-dir results\phase0_3dpw_original `
  --run-name original `
  --run-trackeval `
  --benchmark 3DPW `
  --split Test `
  --seqmap-from-predictions
```

如果当前只想先看 tracker 自身统计，不运行 TrackEval，去掉 `--run-trackeval` 即可：

```powershell
python evaluation\phase0_baseline.py `
  --skip-demo `
  --prediction-dir evaluation\data\trackers\mot_challenge\3DPW-Test\improved\data `
  --output-dir results\phase0_3dpw_improved_summary `
  --run-name improved
```

如果你已经补齐了 24 个 3DPW-Test 序列的预测文件，也可以使用完整 seqmap：

```powershell
python evaluation\phase0_baseline.py `
  --skip-demo `
  --prediction-dir results\phase0_baseline\predictions `
  --output-dir results\phase0_baseline `
  --run-name baseline `
  --run-trackeval `
  --benchmark 3DPW `
  --split Test `
  --seqmap-file evaluation\data\gt\mot_challenge\seqmaps\3DPW-Test.txt
```

MOT20 本地样例可以这样评估 `MOT20-02`：

```powershell
python evaluation\phase0_baseline.py `
  --skip-demo `
  --prediction-dir evaluation\data\trackers\mot_challenge\mot\improved\data `
  --output-dir results\phase0_mot20_improved `
  --run-name improved `
  --run-trackeval `
  --benchmark mot `
  --split test `
  --skip-split-folder `
  --seqmap-file evaluation\data\gt\mot_challenge\seqmaps\MOT-Test.txt
```

如果要把重新运行 `demo.py` 得到的结果接入 TrackEval，关键是保证输出文件名和 GT 序列名一致。例如 3DPW 的 `downtown_runForBus_00` 应生成：

```text
results\phase0_baseline\predictions\downtown_runForBus_00.txt
```

然后使用 `--prediction-dir results\phase0_baseline\predictions` 搭配 `--run-trackeval` 进行评估。

## 进入 Phase 1 前的验收条件

- `phase0_report.md` 中记录了固定输入列表和复现命令。
- 每个选定验证序列都有对应的 `.txt` MOT 输出。
- `phase0_tracker_summary.csv` 非空。
- 如果存在 ground truth，`phase0_trackeval_metrics.csv` 包含 HOTA、IDF1、IDSW、MOTA、Frag。
- 后续 A/B/C 实验必须复用相同输入、帧范围和 `frameskip`；如果更换验证集，需要在新报告中明确记录。
