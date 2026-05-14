import cv2
import os
import argparse
import torch
import shutil
import numpy as np
import subprocess
import tempfile
import sys
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# ================= 配置 =================
DEMO_SCRIPT = "demo.py" 
# =======================================

try:
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.billboard import Billboard
    from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
    from aitviewer.scene.camera import OpenCVCamera
    
    from comotion_demo.models import comotion
    from comotion_demo.utils import dataloading, helper, track as track_utils
    
    comotion_model_dir = Path(comotion.__file__).parent
    CONFIG.smplx_models = os.path.join(comotion_model_dir, "../data")
    CONFIG.window_type = "pyqt6" 
    AITVIEWER_AVAILABLE = True
except ImportError:
    print("[WARNING] aitviewer or comotion_demo not found. SMPL rendering might fail.")
    AITVIEWER_AVAILABLE = False

# ============================
# Step 1: 切片 (Slicing)
# ============================
def slice_video(input_path, temp_dir, overlap):
    print(f"\n=== Step 1: Slicing video {input_path} ===")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mid_x, mid_y = W // 2, H // 2
    if mid_x % 2 != 0: mid_x -= 1
    if mid_y % 2 != 0: mid_y -= 1

    crops = [
        (0, mid_y + overlap, 0, mid_x + overlap),           # TL
        (0, mid_y + overlap, mid_x - overlap, W),           # TR
        (mid_y - overlap, H, 0, mid_x + overlap),           # BL
        (mid_y - overlap, H, mid_x - overlap, W)            # BR
    ]
    
    slice_dims = [(x2-x1, y2-y1) for y1,y2,x1,x2 in crops]
    offsets = [(y1, x1) for y1,y2,x1,x2 in crops]
    
    names = ["tl", "tr", "bl", "br"]
    slice_paths = []
    
    metadata = {
        "original_path": str(input_path), "W": W, "H": H, "fps": fps, 
        "total_frames": total_frames, "names": names,
        "offsets": offsets, "slice_dims": slice_dims
    }

    if all([os.path.exists(os.path.join(temp_dir, f"{n}.mp4")) for n in names]):
        print("✅ Slices already exist. Using cached files.")
        metadata["slice_paths"] = [os.path.join(temp_dir, f"{n}.mp4") for n in names]
        return metadata

    writers = []
    for i, name in enumerate(names):
        w, h = slice_dims[i]
        out_path = os.path.join(temp_dir, f"{name}.mp4")
        slice_paths.append(out_path)
        writers.append(cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)))

    with tqdm(total=total_frames, desc="Slicing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            for i, (y1, y2, x1, x2) in enumerate(crops):
                writers[i].write(frame[y1:y2, x1:x2])
            pbar.update(1)
            
    cap.release()
    for w in writers: w.release()
    metadata["slice_paths"] = slice_paths
    return metadata

# ============================
# Step 2: 推理 (Inference)
# ============================
def run_inference(metadata, temp_dir):
    print(f"\n=== Step 2: Running Inference with {DEMO_SCRIPT} ===")
    names = metadata["names"]
    slice_paths = metadata["slice_paths"]
    
    for name, vid_path in zip(names, slice_paths):
        output_folder = os.path.join(temp_dir, f"result_{name}")
        os.makedirs(output_folder, exist_ok=True)
        target_pt = os.path.join(output_folder, f"{name}.pt")
        
        if os.path.exists(target_pt):
            print(f"Skipping slice '{name}', result already exists.")
            continue

        print(f"🦖 Processing slice: {name} (Running CoMotion + DINOv2)...")
        
        cmd = [
            sys.executable, "-u", DEMO_SCRIPT, 
            "-i", vid_path, 
            "-o", output_folder, 
            "--skip-visualization"
        ]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                line_clean = line.strip()
                if "REVIVED" in line: 
                    print(f"\n   ✨ [Revival] {line_clean}") 
                elif "dead" in line or "died" in line:
                    print(f"\n   💀 [Death]   {line_clean}")
                elif "Running" in line:
                    print(f"   {line_clean}", end='\r')
            process.wait()
            print("")
            if process.returncode != 0:
                print(f"\n❌ Error processing slice {name}")
        except Exception as e:
            print(f"Failed to run demo.py: {e}")

# ============================
# Step 3: 合并与修正 (Merge)
# ============================
def fix_trans_for_global_view(trans, slice_idx, metadata):
    W, H = metadata["W"], metadata["H"]
    slice_w, slice_h = metadata["slice_dims"][slice_idx]
    f = 2 * max(slice_w, slice_h)
    
    cx_global, cy_global = W / 2, H / 2
    off_y, off_x = metadata["offsets"][slice_idx]
    cx_slice_in_global = off_x + slice_w / 2
    cy_slice_in_global = off_y + slice_h / 2
    
    delta_cx = cx_global - cx_slice_in_global
    delta_cy = cy_global - cy_slice_in_global
    
    z = trans[..., 2]
    trans_fixed = trans.clone()
    trans_fixed[..., 0] -= (delta_cx * z / f)
    trans_fixed[..., 1] -= (delta_cy * z / f)
    
    return trans_fixed, f

def merge_track_data(metadata, temp_dir):
    print(f"\n=== Step 3: Merging Tracks ===")
    id_shifts = [0, 10000, 20000, 30000]
    merged_data = {"id": [], "pose": [], "trans": [], "betas": [], "frame_idx": []}
    inferred_focals = []
    names = metadata["names"]
    
    for i, name in enumerate(names):
        pt_path = os.path.join(temp_dir, f"result_{name}", f"{name}.pt")
        if not os.path.exists(pt_path): 
            print(f"Warning: Result missing for {name}")
            continue
        try:
            data = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            continue
        if data['id'].numel() == 0: continue
        
        trans_fixed, focal = fix_trans_for_global_view(data["trans"], i, metadata)
        inferred_focals.append(focal)
        data['id'] = data['id'] + id_shifts[i]
        
        merged_data["id"].append(data['id'])
        merged_data["pose"].append(data['pose'])
        merged_data["trans"].append(trans_fixed)
        merged_data["betas"].append(data['betas'])
        merged_data["frame_idx"].append(data['frame_idx'])

    if not merged_data["id"]:
        return None, None

    final_preds = {}
    for k in merged_data:
        final_preds[k] = torch.cat(merged_data[k], dim=0)
    avg_focal = sum(inferred_focals) / len(inferred_focals)
    return final_preds, avg_focal

# ============================
# Step 3.5: 保存 TXT (Batch Processing)
# ============================
def save_merged_txt(metadata, preds, render_focal, output_path):
    print(f"\n=== Generating MOT txt file (Batch Processing) ===")
    
    W, H = metadata["W"], metadata["H"]
    K_global = torch.tensor([
        [render_focal, 0, 0.5 * W],
        [0, render_focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=torch.float32)

    try:
        from comotion_demo.utils import smpl_kinematics
        decoder = smpl_kinematics.SMPLKinematics()
    except ImportError:
        print("❌ Error: Could not import SMPLKinematics.")
        return

    txt_path = Path(output_path).with_suffix('.txt')
    total_instances = preds["id"].shape[0]
    batch_size = 500 # 每次处理 500 个以防止内存溢出
    
    raw_ids = preds["id"].long().squeeze().numpy()
    raw_frames = preds["frame_idx"].long().squeeze().numpy()
    if raw_ids.ndim == 0: raw_ids = np.array([raw_ids])
    if raw_frames.ndim == 0: raw_frames = np.array([raw_frames])
    
    written_keys = set()
    print(f"Total instances: {total_instances} | Batch size: {batch_size}")
    print(f"Writing to {txt_path}...")
    
    with open(txt_path, "w") as f:
        for start_idx in tqdm(range(0, total_instances, batch_size), desc="Generating BBoxes"):
            end_idx = min(start_idx + batch_size, total_instances)
            
            # 1. 提取当前批次 (移回 CPU)
            batch_inputs = {
                "betas": preds["betas"][start_idx:end_idx].cpu(),
                "pose": preds["pose"][start_idx:end_idx].cpu(),
                "trans": preds["trans"][start_idx:end_idx].cpu()
            }
            
            # 2. 计算 BBox (SMPL Mesh Generation)
            batch_bboxes = track_utils.bboxes_from_smpl(
                decoder,
                batch_inputs,
                (H, W),
                K_global,
                pad_x=0.15,
                pad_y=0.15,
                subsample_rate=1
            ).float().numpy()
            
            # 3. 写入
            batch_ids = raw_ids[start_idx:end_idx]
            batch_frames = raw_frames[start_idx:end_idx]
            lines = []
            
            for i in range(len(batch_ids)):
                fid = int(batch_frames[i]) + 1
                tid = int(batch_ids[i])
                
                if (fid, tid) in written_keys: continue
                written_keys.add((fid, tid))
                
                box = batch_bboxes[i]
                if box.ndim == 1 and box.shape[0] == 4:
                    bb_left, bb_top, x2, y2 = box
                    bb_width = x2 - bb_left
                    bb_height = y2 - bb_top
                else:
                    bb_left = box[0, 0]
                    bb_top = box[0, 1]
                    bb_width = box[1, 0] - bb_left
                    bb_height = box[1, 1] - bb_top

                lines.append(f"{fid},{tid},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},1,1,1\n")
            
            f.writelines(lines)
            
            # 释放内存
            del batch_inputs
            del batch_bboxes

    print(f"✅ Done.")

# ============================
# Step 4: 渲染 (Rendering)
# ============================
def render_output(metadata, preds, render_focal, output_path):
    print(f"\n=== Step 4: Rendering High-Res Output ===")
    if not AITVIEWER_AVAILABLE: return

    W, H = metadata["W"], metadata["H"]
    fps = metadata["fps"]
    total_frames = metadata["total_frames"]

    print("Preparing background...")
    tmp_vis_dir = Path(tempfile.mkdtemp())
    cap = cv2.VideoCapture(metadata["original_path"])
    bg_paths = []
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        save_p = tmp_vis_dir / f"{i:06d}.jpg"
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(save_p)
        bg_paths.append(str(save_p))
    cap.release()

    viewer = HeadlessRenderer(size=(W, H))
    K_global = np.array([[render_focal, 0, 0.5 * W], [0, render_focal, 0.5 * H], [0, 0, 1]], dtype=np.float32)
    viewer.reset()
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    cam = OpenCVCamera(K_global, np.eye(4)[:3], cols=W, rows=H, viewer=viewer)
    viewer.scene.add(cam)
    viewer.set_temp_camera(cam)
    viewer.playback_fps = fps
    viewer.scene.add(Billboard.from_camera_and_distance(cam, 100.0, cols=W, rows=H, textures=bg_paths))

    print("Adding SMPL tracks...")
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    max_f = int(preds["frame_idx"].max().item())
    track_subset = track_utils.query_range(preds, 0, max_f + 1)
    
    unique_ids = track_subset["id"].max(dim=0)[0].squeeze()
    if unique_ids.ndim == 0: unique_ids = unique_ids.unsqueeze(0)
    color_lib = helper.color_ref
    
    for i in tqdm(range(track_subset["pose"].shape[1]), desc="Adding Actors"):
        tid = int(unique_ids[i].item())
        if tid == 0: continue
        betas = track_subset["betas"][:, i]
        pose = track_subset["pose"][:, i]
        trans = track_subset["trans"][:, i]
        valid_mask = (betas != 0).any(dim=-1)
        if not valid_mask.any(): continue
        
        color = [c * 0.6 + 0.2 for c in color_lib[tid % len(color_lib)]] + [1.0]
        curr_trans = trans.clone()
        curr_trans[~valid_mask, 2] = -10000.0
        
        viewer.scene.add(SMPLSequence(
            smpl_layer=smpl_layer, betas=betas, poses_root=pose[..., :3],
            poses_body=pose[..., 3:], trans=curr_trans, color=color, name=f"ID_{tid}"
        ))

    print(f"Rendering to {output_path}...")
    viewer.save_video(video_dir=str(output_path), output_fps=int(fps), ensure_no_overwrite=False)
    try: shutil.rmtree(tmp_vis_dir)
    except: pass

# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input high-res video path")
    parser.add_argument("-o", "--output", required=True, help="Output mp4 path (e.g. output.mp4)")
    parser.add_argument("--temp", default="./temp_slice_process", help="Temporary folder")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap pixels between slices")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip 3D Mesh rendering")
    args = parser.parse_args()
    
    os.makedirs(args.temp, exist_ok=True)
    
    try:
        meta = slice_video(args.input, args.temp, args.overlap)
        run_inference(meta, args.temp)
        preds, focal = merge_track_data(meta, args.temp)
        
        if preds is None:
            print("No people detected in any slice.")
            return

        final_pt_path = Path(args.output).with_suffix('.pt')
        print(f"Saving merged final tracks to: {final_pt_path}")
        torch.save(preds, final_pt_path)

        # 核心：调用分批写入函数
        save_merged_txt(meta, preds, focal, args.output)

        if not args.skip_visualization:
            render_output(meta, preds, focal, args.output)
        else:
            print("⏩ Skipping visualization as requested.")
        
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()