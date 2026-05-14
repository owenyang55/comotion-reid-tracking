import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import time

# ==================== 配置区域 ====================
# 1. 3DPW 图片序列根目录
DATASET_ROOT = Path("datasets/3DPW/imageFiles")

# 2. 实验名称 (每次跑不同版本时记得改这个名字)
# 第一次跑建议叫 "improved"，改完 demo.py 关闭 ReID 后改成 "original"
EXP_NAME = "improved" 

# 3. 最终结果存放位置 (TrackEval 标准路径)
TRACKER_DEST_DIR = Path(f"evaluation/data/trackers/mot_challenge/3DPW-Test/{EXP_NAME}/data")

# 4. 临时目录 (demo.py 默认生成位置)
TEMP_OUT_DIR = Path("temp_batch_results")
# =================================================

def run_batch_inference():
    # 检查输入
    if not DATASET_ROOT.exists():
        print(f"❌ 错误: 数据集路径不存在: {DATASET_ROOT}")
        return

    # 创建目录
    TRACKER_DEST_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 扫描所有序列 (排除隐藏文件和非目录)
    sequences = sorted([
        p for p in DATASET_ROOT.iterdir() 
        if p.is_dir() and not p.name.startswith('.')
    ])
    
    print(f"🚀 开始批量推理 [实验: {EXP_NAME}]")
    print(f"📂 数据集: {DATASET_ROOT}")
    print(f"🎯 目标路径: {TRACKER_DEST_DIR}")
    print(f"🎬 共发现 {len(sequences)} 个序列")
    print("-" * 50)

    success_count = 0
    start_time = time.time()

    for seq_dir in tqdm(sequences, desc="Processing Sequences"):
        seq_name = seq_dir.name
        output_txt = TRACKER_DEST_DIR / f"{seq_name}.txt"

        # 如果结果已存在，可以选择跳过 (断点续传)
        # if output_txt.exists():
        #     continue

        cmd = [
            "python", "demo.py",
            "-i", str(seq_dir),
            "-o", str(TEMP_OUT_DIR),
            "--skip-visualization" # 关键：必须跳过渲染，否则极慢且占空间
        ]

        try:
            # 运行 demo.py
            # capture_output=True 可以让进度条不被打断，但如果报错可能看不到
            subprocess.run(cmd, check=True)
            
            # 移动并重命名结果
            # demo.py 默认输出在 temp 目录下，文件名是 {seq_name}.txt
            src_txt = TEMP_OUT_DIR / f"{seq_name}.txt"
            
            if src_txt.exists():
                shutil.move(str(src_txt), str(output_txt))
                success_count += 1
            else:
                print(f"\n⚠️ 警告: 序列 {seq_name} 运行完成但未生成 .txt 文件")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ 序列 {seq_name} 运行崩溃!")
        except Exception as e:
            print(f"\n❌ 未知错误: {e}")

    total_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ 批量推理完成!")
    print(f"📊 成功: {success_count} / {len(sequences)}")
    print(f"⏱️ 总耗时: {total_time/60:.1f} 分钟")
    print(f"📁 结果已保存在: {TRACKER_DEST_DIR}")

if __name__ == "__main__":
    run_batch_inference()