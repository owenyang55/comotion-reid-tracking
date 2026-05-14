import os
import pickle
import numpy as np
import glob
import argparse
import configparser
from tqdm import tqdm

def bbox_from_keypoints(keypoints, h, w, pad=20):
    """
    根据2D关键点计算外接矩形框 (Bounding Box)
    keypoints: shape (18, 3) -> (x, y, conf)
    """
    # 过滤掉置信度为0的点
    valid = keypoints[:, 2] > 0
    if not np.any(valid):
        return None
    
    kps = keypoints[valid, :2]
    x1, y1 = np.min(kps, axis=0)
    x2, y2 = np.max(kps, axis=0)
    
    # 添加 Padding 并进行边界检查
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    return [x1, y1, x2 - x1, y2 - y1] # 返回 [x, y, w, h]

def convert_3dpw(dataset_path, output_root):
    # 1. 确定输入路径
    # 3DPW 的真值通常在 sequenceFiles/test 目录下
    pkl_dir = os.path.join(dataset_path, 'sequenceFiles', 'test')
    pkl_files = glob.glob(os.path.join(pkl_dir, '*.pkl'))
    
    if not pkl_files:
        print(f"❌ 错误: 在路径下没找到 .pkl 文件: {pkl_dir}")
        print("请检查你的 --path 参数是否指向了包含 sequenceFiles 的 3DPW 根目录。")
        return

    print(f"🔍 发现 {len(pkl_files)} 个测试序列，准备转换...")

    # 2. 准备输出目录
    # TrackEval 要求：data/gt/mot_challenge/3DPW-Test
    gt_out_dir = os.path.join(output_root, 'data', 'gt', 'mot_challenge', '3DPW-Test')
    
    # 3. 准备 seqmaps 列表
    seq_names = []

    for pkl_file in tqdm(pkl_files, desc="Converting"):
        # 读取 PKL (3DPW 很多是 Python2 保存的，需要 latin1 编码)
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        seq_name = os.path.basename(pkl_file).replace('.pkl', '')
        seq_names.append(seq_name)
        
        # 3DPW 默认分辨率 (大部分是 1920x1080)
        # 如果你的图片不是这个分辨率，请修改这里
        H, W = 1080, 1920 

        # 提取姿态数据
        # data['poses2d'] 是一个列表，长度 = 人数
        # list[i] 是一个数组 (Num_Frames, 18, 3)
        poses2d = data['poses2d']
        num_actors = len(poses2d)
        num_frames = len(poses2d[0]) # 假设所有人帧数一致
        
        # 创建单个序列的文件夹结构
        seq_dir = os.path.join(gt_out_dir, seq_name)
        gt_txt_dir = os.path.join(seq_dir, 'gt')
        os.makedirs(gt_txt_dir, exist_ok=True)
        
        # --- A. 生成 gt.txt ---
        gt_file_path = os.path.join(gt_txt_dir, 'gt.txt')
        with open(gt_file_path, 'w') as f_gt:
            for frame_idx in range(num_frames):
                for actor_id in range(num_actors):
                    # 获取 (18, 3) 关键点
                    kps = poses2d[actor_id][frame_idx]
                    
                    # 简单过滤：如果可见关键点太少，视作被遮挡或不可见
                    if np.sum(kps[:, 2] > 0) < 3: 
                        continue
                        
                    bbox = bbox_from_keypoints(kps, H, W)
                    if bbox is None:
                        continue
                    
                    # 写入一行
                    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
                    # TrackEval 默认行人 Class ID 为 1
                    line = f"{frame_idx+1},{actor_id+1},{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f},1,1,1\n"
                    f_gt.write(line)
        
        # --- B. 生成 seqinfo.ini ---
        # 这是 TrackEval 必须的文件，用于读取视频长度等信息
        seq_ini_path = os.path.join(seq_dir, 'seqinfo.ini')
        with open(seq_ini_path, 'w') as f_ini:
            f_ini.write("[Sequence]\n")
            f_ini.write(f"name={seq_name}\n")
            f_ini.write(f"imDir=imageFiles/{seq_name}\n") # 假定路径
            f_ini.write(f"frameRate=30\n")
            f_ini.write(f"seqLength={num_frames}\n")
            f_ini.write(f"imWidth={W}\n")
            f_ini.write(f"imHeight={H}\n")
            f_ini.write(f"imExt=.jpg\n")

    # --- C. 生成 seqmaps 文件 ---
    # 文件位置：data/gt/mot_challenge/seqmaps/3DPW-Test.txt
    seqmap_dir = os.path.join(output_root, 'data', 'gt', 'mot_challenge', 'seqmaps')
    os.makedirs(seqmap_dir, exist_ok=True)
    
    seqmap_path = os.path.join(seqmap_dir, '3DPW-Test.txt')
    with open(seqmap_path, 'w') as f:
        f.write("name\n") # Header
        # 排序后写入，保证整洁
        for s in sorted(seq_names):
            f.write(f"{s}\n")
            
    print(f"✅ 转换完成！")
    print(f"   真值位置: {gt_out_dir}")
    print(f"   Seqmap位置: {seqmap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 3DPW Ground Truth to MOT Format")
    parser.add_argument('--path', type=str, required=True, help='3DPW 数据集根目录 (包含 sequenceFiles 文件夹)')
    args = parser.parse_args()
    
    # 默认输出到当前目录下的 evaluation 文件夹
    current_dir = os.getcwd()
    # 如果脚本在 evaluation 目录下运行，output_root 就是当前目录
    # 如果在 ml-comotion 根目录运行，需要调整。这里假设脚本在 evaluation 下运行。
    
    convert_3dpw(args.path, current_dir)