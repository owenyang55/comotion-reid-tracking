# 项目说明

## 基本定位

本仓库是 Apple 开源项目 **CoMotion: Concurrent Multi-person 3D Motion** 的本地改造版本，用于从单目视频、图片目录或单张图片中检测并跟踪多人 3D SMPL 姿态。原始项目论文为 ICLR 2025 的 CoMotion，仓库主要提供推理、可视化和跟踪评估脚本。

当前代码不只是原版 CoMotion：本地版本已经在检测、跟踪和后处理里加入了 DINOv2 ReID 外观特征、丢失轨迹复活、离线轨迹缝合、插值补帧、批量视频处理和 3DPW/MOT 评估相关逻辑。

## 技术栈与依赖

- Python 项目，包名为 `comotion_demo`，源码位于 `src/`。
- Python 版本要求：`>=3.10`。
- 主要依赖见 `pyproject.toml`：
  - `torch==2.5.1`
  - `timm==1.0.13`
  - `transformers==4.48.0`
  - `opencv-python`
  - `ffmpeg-python`
  - `scenedetect`
  - `tensordict`
  - `pypose`
  - `coremltools`
  - `PyQt6`
  - `einops`
  - `chumpy`
- 可选依赖：
  - `.[all]` 包含 `aitviewer`，用于 SMPL 渲染可视化。
  - `.[colab]` 包含 `pyrender`、`smplx[all]`、`trimesh` 等。

推荐安装方式：

```bash
conda create -n comotion -y python=3.10
conda activate comotion
pip install -e '.[all]'
```

## 必需模型与数据

### CoMotion 预训练权重

运行：

```bash
bash get_pretrained_models.sh
```

预训练权重会下载到：

```text
src/comotion_demo/data/
```

关键文件包括：

- `comotion_detection_checkpoint.pt`
- `comotion_refine_checkpoint.pt`
- macOS CoreML 检测模型 `comotion_detection.mlpackage`

该目录已在 `.gitignore` 中忽略，不应提交权重文件。

### SMPL 模型

可视化需要 neutral SMPL body model。需要从 SMPL 官网下载 `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`，并复制/重命名为：

```text
src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl
```

### DINOv2 ReID

本地版本的 ReID 逻辑会通过 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` 加载 DINOv2 ViT-Small。首次运行需要联网下载或已有 torch hub 缓存。

可通过环境变量指定微调后的 ReID checkpoint：

```text
COMOTION_REID_CKPT=/path/to/reid_dinov2_best.pt
```

## 主要目录

```text
src/comotion_demo/
```

项目主包。

- `models/comotion.py`：CoMotion 总模型，组合检测模型、姿态更新模型、SMPL 解码器和镜头切换检测。
- `models/detect.py`：检测网络、NMS 后处理、SMPL 姿态解码。本地已加入 `hidden` 和 `appearance` 输出，并支持用 DINOv2 ReID 覆盖外观特征。
- `models/refine.py`：跨帧 pose refinement/update 模块，使用 image feature、SMPL 状态、2D/3D 关键点和 hidden state 更新当前轨迹。
- `models/layers.py`、`models/backbones/`：网络层、ConvNeXtV2 backbone 等基础组件。
- `utils/track.py`：轨迹管理核心。本地已加入外观特征维度、lost tracks、轨迹复活、feature gallery 和健康监控逻辑。
- `utils/reid.py`：DINOv2 ReID 特征提取器，输出 384 维 L2 归一化外观向量。
- `utils/dataloading.py`：图片/视频读取、默认相机内参、512x512 裁剪和网络输入预处理。
- `utils/helper.py`：投影、坐标转换、NMS、OKS/相似度、bbox 等工具函数。
- `utils/smpl_kinematics.py`：SMPL 解码和运动学工具。

```text
evaluation/
```

评估与可视化脚本。

- `run_eval.py`：基于 TrackEval 对 `original` 与 `improved` tracker 结果做 HOTA、CLEAR、Identity 指标比较，并输出对比图。
- `convert_3dpw_gt.py`：将 3DPW ground truth 转为 MOT/TrackEval 目录格式。
- `plot_tracking.py`：读取 MOT txt 并绘制轨迹时间线。
- `TrackEval/`：内置 TrackEval 代码副本。

```text
posetrack21_eval/
```

PoseTrack21 MOT 评估代码副本，包含本项目所需的若干修改。

```text
samples/
```

README 使用的 teaser GIF 和样例说明。

```text
temp_slice_process/
```

本地切片处理生成的临时/中间结果目录，里面已有 `tl/tr/bl/br` 的 `.pt` 和 `.txt` 结果。

## 根目录关键脚本

### `demo.py`

主推理入口。支持：

- 单张图片检测。
- 视频跟踪。
- 图片目录跟踪。
- 可选跳过可视化。
- 可选强制要求 ReID 可用。
- 输出 `.pt` 结果。
- 视频/目录模式还会输出 MOT 格式 `.txt`。
- 可视化可输出渲染后的 `.mp4` 或单图 `.png`。

常用命令：

```bash
python demo.py -i path/to/video.mp4 -o results/
python demo.py -i path/to/image.jpg -o results/
python demo.py -i path/to/frames_dir -o results/ --skip-visualization
python demo.py -i path/to/video.mp4 -o results/ --start-frame 0 --num-frames 300 --frameskip 2
python demo.py -i path/to/video.mp4 -o results/ --require-reid
```

注意：`dataloading.py` 原始视频扩展名只把 `.mp4` 视为视频；`batch_demo_jinshe.py` 自己支持更多视频扩展名，但最终仍调用 `demo.py`。

### `batch_demo_jinshe.py`

批量扫描视频目录并逐个调用 `demo.py`。默认输入根目录为上级工作区的：

```text
video/jinshe
```

默认输出根目录为：

```text
out/jinshe
```

该脚本会设置 `PYTHONPATH` 指向本仓库 `src`，适合未正式安装包时运行。

### `batch_inference.py`

面向 3DPW 图片序列的批量推理脚本。默认读取：

```text
datasets/3DPW/imageFiles
```

默认把 MOT txt 结果移动到：

```text
evaluation/data/trackers/mot_challenge/3DPW-Test/<EXP_NAME>/data
```

当前 `EXP_NAME = "improved"`。

### `run_slice_1_12.py`

视频四宫格切片推理和合并脚本：

1. 将大视频切成 `tl/tr/bl/br` 四个重叠区域。
2. 分别调用 `demo.py` 推理。
3. 修正局部切片坐标到全局视角。
4. 合并轨迹、生成 MOT txt，并可渲染可视化。

适合处理高分辨率、人员较多或局部遮挡严重的视频。

### `train_reid_dinov2.py`

DINOv2 ReID 微调脚本。期望数据目录结构：

```text
data_root/<person_id>/*.jpg
```

训练目标包含分类交叉熵和 batch-hard triplet loss。默认输出：

```text
checkpoints/reid_dinov2_best.pt
```

输出 payload 包含 `backbone`、`classifier`、`num_classes`、`label_map`、训练参数和最佳 epoch。

### `test.py`

当前只是一个很小的递归函数测试文件，而且函数名存在拼写不一致问题：定义为 `calculate_factoial`，递归调用为 `calculate_factorial`。不要把它当作正式测试套件。

## 推理流程摘要

1. `dataloading` 读取图片/视频帧，构造默认相机内参 `K`。
2. 输入图像被裁剪/缩放到 512x512，并进行 ImageNet normalization。
3. `CoMotionDetect` 用 ConvNeXtV2 backbone 提取图像特征。
4. detection head 生成候选 SMPL 参数、置信度、pose embedding、hidden 和 appearance。
5. `decode_network_outputs` 解码 SMPL 姿态和 2D/3D 关键点，进行 NMS。
6. 如果传入 ReID 模型，则根据 2D 关键点 bbox 裁剪人物区域，用 DINOv2 生成 384 维 appearance。
7. `TrackHandler` 维护当前轨迹：
   - 用 2D 关键点相似度匹配检测与已有轨迹。
   - 用 refinement 模块更新轨迹姿态。
   - 用健康监控剔除坏轨迹/重复轨迹。
   - 用 appearance gallery 尝试复活 lost tracks。
8. 视频结束后，`demo.py` 做离线轨迹缝合、tensor 插值和 MOT txt 写入。
9. 如果启用可视化，`aitviewer` 将 SMPL mesh 渲染回视频或图片。

## 本地改造点

### 外观特征

- `TrackTensorState` 增加 `appearance`，默认维度是 384。
- `detect.py` 的 `DetectionOutput` 增加 `hidden` 和 `appearance`。
- `decode_network_outputs` 支持通过 ReID 模型从原图裁剪人物并提取 DINOv2 特征。
- 未启用 ReID 时，会填充零向量以保持 tracker 张量维度一致。

### Lost track 复活

`track.py` 中 `TrackHandler` 增加：

- `lost_tracks`
- `max_lost_patience`
- `revive_lost_tracks`
- 基于外观相似度、位置距离和速度预测的复活逻辑。

`TrackHealthMonitor` 增加：

- `feature_gallery`
- `appearance_emb`
- `revive_age`
- `compute_max_similarity`

### 离线后处理

`demo.py` 增加：

- `perform_stitching`：根据时间间隔、3D 距离和 appearance 相似度合并短 tracklet。
- `apply_interpolation_to_tensor`：对 `trans`、`pose`、`betas` 做线性插值补帧。
- `write_interpolated_mot`：写出包含 bbox 插值结果的 MOT txt。

## 当前阶段实验结论与后续顺序

最近在 `MOT20-02` 全量实验上已经确认：

- 长期复活是主要风险源。`memory_mode=long` 会明显抬高 `IDSW` 和 `Frag`，而 `memory_mode=off` 与 `basic` 基本一致，说明基础跟踪本身不是主因。
- 线性插值是负贡献。开启插值后，`IDSW`、`MOTA` 和 `Frag` 变差；离线 stitching 的影响较小，不是主要矛盾。

后续默认顺序：

1. MOT 风格评估先默认关闭插值，stitching 保持可选。
2. 先做保守复活门控，再做高质量外观记忆过滤。
3. 所有改动先在 `MOT20-02` 全量上做消融，再决定是否扩到 `MOT20-01/03/05`。
4. 复现和对比时统一使用 `evaluation/phase0_baseline.py --demo-root ...` 切换原始仓库与当前仓库，不再依赖反复 `pip uninstall comotion-demo` / `pip install -e .`。

当前可作为参考的结果目录：

- `results/mot20_02_baseline_full`
- `results/mot20_02_improved_full`
- `evaluation/result/`

## 输出文件格式

`demo.py` 默认以输入文件/目录 stem 命名输出：

```text
<output_dir>/<input_name>.pt
<output_dir>/<input_name>.txt
<output_dir>/<input_name>.mp4
```

`.pt` 常见字段：

- `id`
- `frame_idx`
- `pose`
- `trans`
- `betas`
- `appearance`

单图检测模式还可能包含：

- `pred_2d`
- `pred_3d`
- `conf`
- `hidden`

`.txt` 为 MOT 格式：

```text
frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y
```

代码中实际写入形如：

```text
<frame+1>,<id>,<x>,<y>,<w>,<h>,1,1,1
```

## 评估流程

3DPW/MOT 风格评估大致流程：

1. 用 `evaluation/convert_3dpw_gt.py` 把 3DPW 标注转为 TrackEval 目录结构。
2. 用 `batch_inference.py` 或其他脚本生成 tracker txt。
3. 按 TrackEval 期望目录放入：

```text
evaluation/data/trackers/mot_challenge/3DPW-Test/<tracker_name>/data
```

4. 运行 `evaluation/run_eval.py` 比较 `original` 和 `improved`。
5. 输出指标和 `evaluation/comparison_result_optimized.png`。

## 编码与维护注意事项

- 当前多个 `.py` 文件中存在中文注释乱码，典型表现为 `鍒濆鍖`、`璋冪敤` 等。这通常是 GBK/UTF-8 编码不一致造成的。编辑这些文件时要谨慎，避免无意中大面积重写整文件。
- 如果需要修复乱码，建议只修复正在修改的局部代码块，并确认文件编码和 diff。
- 不要随意改动张量字段维度，尤其是：
  - `hidden`: 512
  - `appearance`: 384
  - `pose`: 72
  - `betas`: 10
  - `trans`: 3
- `TrackTensorState`、`default_dims`、`detect.decode_network_outputs` 和结果堆叠逻辑必须保持一致，否则容易在 `torch.stack` 或 `torch.cat` 时报维度错误。
- ReID 依赖 `torch.hub`，离线环境可能失败。`demo.py` 默认失败时降级为零 appearance；加 `--require-reid` 时会直接报错。
- 可视化依赖 `aitviewer`、SMPL 模型、PyQt6/渲染环境。远程服务器可能需要虚拟显示。
- `.gitignore` 已忽略 `*.mp4`、`src/comotion_demo/data`、`data/`、`tmp/`、`results/` 等，不要把大模型、视频和批量结果提交进仓库。

## 安全操作约束

本仓库工作时禁止批量删除文件或目录。不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件，例如：

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果确实需要批量删除文件，应停止操作并请求用户手动删除。

## 常用检查命令

```bash
python demo.py --help
python -m pip install -e '.[all]'
python demo.py -i path/to/video.mp4 -o results/ --skip-visualization
python train_reid_dinov2.py --data-root path/to/reid_data --output checkpoints/reid_dinov2_best.pt
```

本仓库没有明确的正式测试入口。修改核心跟踪逻辑后，建议至少用一段短视频或少量帧目录跑通 `demo.py --skip-visualization`，并检查 `.pt` 与 `.txt` 是否生成。
