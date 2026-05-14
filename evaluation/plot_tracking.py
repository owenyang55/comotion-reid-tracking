import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# 配置：选择一个困难序列的 specific result txt path
SEQ_NAME = "MOT20-02" # 或者 courtyard_basketball_00
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIG_FILE = os.path.join(ROOT_DIR, f"data/trackers/mot_challenge/mot/original/data/{SEQ_NAME}.txt")
IMPR_FILE = os.path.join(ROOT_DIR, f"data/trackers/mot_challenge/mot/improved/data/{SEQ_NAME}.txt")

def read_tracks(file_path):
    # data: {id: [frame1, frame2, ...]}
    tracks = {}
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return tracks
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            tid = int(parts[1])
            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append(frame)
    return tracks

def plot_timeline(tracks, title, ax, color_map='tab20'):
    # 将轨迹按开始时间排序
    sorted_ids = sorted(tracks.keys(), key=lambda k: min(tracks[k]))
    
    # 为了紧凑显示，重新映射 ID 到 Y 轴索引 (0, 1, 2...)
    y_map = {old_id: i for i, old_id in enumerate(sorted_ids)}
    
    cmap = plt.get_cmap(color_map)
    
    for tid in sorted_ids:
        frames = sorted(tracks[tid])
        if not frames: continue
        
        # 寻找连续片段
        segments = []
        if len(frames) > 0:
            start = frames[0]
            prev = frames[0]
            for f in frames[1:]:
                if f > prev + 1: # 断开了
                    segments.append((start, prev - start + 1))
                    start = f
                prev = f
            segments.append((start, prev - start + 1))
            
        y = y_map[tid]
        color = cmap(tid % 20)
        
        for (start, duration) in segments:
            # 绘制长条
            ax.broken_barh([(start, duration)], (y - 0.4, 0.8), facecolors=color)
            
        # 标注 ID
        ax.text(min(frames), y, f"ID {tid}", va='center', ha='right', fontsize=8)

    ax.set_title(f"{title}\n(Total IDs: {len(tracks)})")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Track Identity (Remapped)")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

def main():
    tracks_orig = read_tracks(ORIG_FILE)
    tracks_impr = read_tracks(IMPR_FILE)
    
    if not tracks_orig or not tracks_impr:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    plot_timeline(tracks_orig, "Original Method (Baseline)", ax1)
    plot_timeline(tracks_impr, "Improved Method (Ours)", ax2)
    
    plt.tight_layout()
    save_path = f"timeline_{SEQ_NAME}.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ 可视化时间线已保存: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()