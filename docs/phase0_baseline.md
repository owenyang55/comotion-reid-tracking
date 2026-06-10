# Phase 0：MOT20 基线与改进算法评估

本文档记录当前阶段的标准测试流程。默认在 **Anaconda Prompt/cmd** 中运行，先激活 `comotion` 环境。

```bat
conda activate comotion
cd /d D:\github\ml_motion_12-18\ml-comotion
```

当前只保留已下载的 MOT20 数据集作为公开数据集评估入口：

```text
D:\github\data\datasets\MOT\MOT20
```

本地可计算指标的是 `train` 子集；`test` 子集没有公开 GT，只能生成预测用于官方提交。

## 路径约定

| 内容 | 路径 |
| --- | --- |
| 原始基线算法仓库 | `D:\github\ml_motion\ml-comotion` |
| 当前改进算法仓库 | `D:\github\ml_motion_12-18\ml-comotion` |
| MOT20 数据集 | `D:\github\data\datasets\MOT\MOT20` |
| 本次测试序列 | `MOT20-02` |
| 输入帧目录 | `D:\github\data\datasets\MOT\MOT20\train\MOT20-02\img1` |
| GT 根目录 | `D:\github\data\datasets\MOT\MOT20\train` |

`phase0_baseline.py` 会根据 `--demo-root` 自动切换要调用的算法仓库，并把对应仓库的 `src` 加入子进程 `PYTHONPATH`。因此使用本文档命令时，不需要反复执行 `pip uninstall comotion-demo` 和 `pip install -e .`。

## 1. 运行原始基线算法

不传 `--num-frames` 时默认处理完整序列。

```bat
python evaluation\phase0_baseline.py --input D:\github\data\datasets\MOT\MOT20\train\MOT20-02\img1 --sequence-name MOT20-02 --demo-root D:\github\ml_motion\ml-comotion --output-dir results\mot20_02_baseline_full --run-name baseline --frameskip 1 --plot-timelines
```

输出位置：

```text
results\mot20_02_baseline_full\predictions\MOT20-02.txt
results\mot20_02_baseline_full\phase0_report.md
results\mot20_02_baseline_full\phase0_tracker_summary.csv
```

如果没有生成 `results\mot20_02_baseline_full\predictions\MOT20-02.txt`，不要继续执行第 2 步。先检查第 1 步是否运行成功，或者实际输出目录是否写成了其它名字。

## 2. 评估原始基线算法

```bat
scripts\evaluate_mot20_phase0.cmd results\mot20_02_baseline_full\predictions results\mot20_02_baseline_full baseline D:\github\data\datasets\MOT\MOT20\train
```

评估输出：

```text
results\mot20_02_baseline_full\phase0_trackeval_metrics.csv
results\mot20_02_baseline_full\phase0_summary.json
results\mot20_02_baseline_full\phase0_report.md
```

注意：评估命令的第一个参数必须是实际含有 `MOT20-02.txt` 的预测目录。例如，如果你之前把完整结果跑到了 `results\phase0_mot20_02_full`，则应使用：

```bat
scripts\evaluate_mot20_phase0.cmd results\phase0_mot20_02_full\predictions results\phase0_mot20_02_full baseline D:\github\data\datasets\MOT\MOT20\train
```

## 3. 运行当前改进算法

```bat
python evaluation\phase0_baseline.py --input D:\github\data\datasets\MOT\MOT20\train\MOT20-02\img1 --sequence-name MOT20-02 --demo-root D:\github\ml_motion_12-18\ml-comotion --output-dir results\mot20_02_improved_full --run-name improved --frameskip 1 --plot-timelines
```

输出位置：

```text
results\mot20_02_improved_full\predictions\MOT20-02.txt
results\mot20_02_improved_full\phase0_report.md
results\mot20_02_improved_full\phase0_tracker_summary.csv
```

如果没有生成 `results\mot20_02_improved_full\predictions\MOT20-02.txt`，不要继续执行第 4 步。

## 4. 评估当前改进算法

```bat
scripts\evaluate_mot20_phase0.cmd results\mot20_02_improved_full\predictions results\mot20_02_improved_full improved D:\github\data\datasets\MOT\MOT20\train
```

评估输出：

```text
results\mot20_02_improved_full\phase0_trackeval_metrics.csv
results\mot20_02_improved_full\phase0_summary.json
results\mot20_02_improved_full\phase0_report.md
```

## 5. 指标查看

重点查看两个目录下的 `phase0_trackeval_metrics.csv`：

```text
results\mot20_02_baseline_full\phase0_trackeval_metrics.csv
results\mot20_02_improved_full\phase0_trackeval_metrics.csv
```

核心指标：

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| `HOTA` | 综合检测、定位和关联质量 | 越高越好 |
| `IDF1` | 身份保持能力 | 越高越好 |
| `IDSW` | ID switch 次数 | 越低越好 |
| `MOTA` | CLEAR MOT 综合指标 | 越高越好 |
| `Frag` | 轨迹碎片化次数 | 越低越好 |

`phase0_tracker_summary.csv` 不依赖 GT，用于辅助排查结果是否异常，例如输出 ID 数量、短轨迹数量、轨迹断裂情况等。

## 6. 快速烟测

如果只是确认命令是否能跑通，可以在运行命令中临时添加：

```bat
--num-frames 100
```

烟测通过后再去掉 `--num-frames` 跑完整序列。

## 7. 常见问题

如果 TrackEval 报错缺少 `scipy`：

```bat
python -m pip install scipy
```

如果直接手动运行两个仓库的 `demo.py`，同一个 conda 环境会受已安装的 `comotion-demo` 指向影响。推荐统一使用本文档的 `phase0_baseline.py --demo-root ...` 命令切换算法仓库。
