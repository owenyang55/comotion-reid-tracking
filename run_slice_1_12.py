import cv2
import os
import argparse
import torch
import shutil
import numpy as np
import subprocess
import tempfile
import sys
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# ================= 配置 =================
DEMO_SCRIPT = "demo.py" 
# =======================================


def configure_skvideo_ffmpeg():
    ffmpeg_dir = os.environ.get("COMOTION_FFMPEG_PATH")
    if ffmpeg_dir is None:
        candidate = Path(sys.prefix) / "Library" / "bin"
        if (candidate / "ffmpeg.exe").exists() and (candidate / "ffprobe.exe").exists():
            ffmpeg_dir = str(candidate)
    if ffmpeg_dir:
        try:
            import skvideo

            skvideo.setFFmpegPath(ffmpeg_dir)
        except Exception as exc:
            print(f"[WARNING] Could not configure skvideo FFmpeg path: {exc}")


configure_skvideo_ffmpeg()

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
        "offsets": offsets, "slice_dims": slice_dims,
        "overlap": overlap, "mid_x": mid_x, "mid_y": mid_y, "crops": crops,
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
def run_inference(metadata, temp_dir, demo_args):
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
        cmd.extend(demo_args)
        
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


def make_camera_matrix(focal, width, height):
    return torch.tensor([
        [focal, 0, 0.5 * width],
        [0, focal, 0.5 * height],
        [0, 0, 1],
    ], dtype=torch.float32)


def compute_slice_bboxes(decoder, data, slice_idx, metadata, batch_size=500):
    slice_w, slice_h = metadata["slice_dims"][slice_idx]
    focal = 2 * max(slice_w, slice_h)
    K_slice = make_camera_matrix(focal, slice_w, slice_h)

    boxes = []
    total = data["id"].reshape(-1).shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_inputs = {
            "betas": data["betas"][start:end].cpu(),
            "pose": data["pose"][start:end].cpu(),
            "trans": data["trans"][start:end].cpu(),
        }
        batch_boxes = track_utils.bboxes_from_smpl(
            decoder,
            batch_inputs,
            (slice_h, slice_w),
            K_slice,
            pad_x=0.15,
            pad_y=0.15,
            subsample_rate=20,
        ).float()
        boxes.append(batch_boxes)

    local_boxes = torch.cat(boxes, dim=0)
    off_y, off_x = metadata["offsets"][slice_idx]
    global_boxes = local_boxes.clone()
    global_boxes[..., 0] += off_x
    global_boxes[..., 1] += off_y
    global_boxes[..., 0].clamp_(0.01, metadata["W"] - 1)
    global_boxes[..., 1].clamp_(0.01, metadata["H"] - 1)
    return local_boxes, global_boxes


def owner_slice_for_center(center_x, center_y, metadata):
    if center_y < metadata["mid_y"]:
        return 0 if center_x < metadata["mid_x"] else 1
    return 2 if center_x < metadata["mid_x"] else 3


def pair_overlap_band_ok(slice_a, slice_b, center_a, center_b, metadata):
    overlap = max(float(metadata.get("overlap", 0)), 1.0)
    mid_x = float(metadata["mid_x"])
    mid_y = float(metadata["mid_y"])
    margin = 0.25 * overlap

    pair = tuple(sorted((int(slice_a), int(slice_b))))
    if pair in {(0, 1), (2, 3)}:
        lo, hi = mid_x - overlap - margin, mid_x + overlap + margin
        return lo <= center_a[0] <= hi and lo <= center_b[0] <= hi
    if pair in {(0, 2), (1, 3)}:
        lo, hi = mid_y - overlap - margin, mid_y + overlap + margin
        return lo <= center_a[1] <= hi and lo <= center_b[1] <= hi

    # Diagonal slices only share the central square. Keep this stricter than
    # side-adjacent matching so it only catches center-cross duplicates.
    lo_x, hi_x = mid_x - overlap, mid_x + overlap
    lo_y, hi_y = mid_y - overlap, mid_y + overlap
    return (
        lo_x <= center_a[0] <= hi_x and lo_x <= center_b[0] <= hi_x
        and lo_y <= center_a[1] <= hi_y and lo_y <= center_b[1] <= hi_y
    )


def bbox_xyxy(box):
    return np.array([box[0, 0], box[0, 1], box[1, 0], box[1, 1]], dtype=np.float32)


def bbox_iou(box_a, box_b):
    a = bbox_xyxy(box_a)
    b = bbox_xyxy(box_b)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    if denom <= 1e-6:
        return 0.0
    return inter / denom


def longest_consecutive_run(frames):
    if not frames:
        return 0
    frames = sorted(set(int(f) for f in frames))
    best = 1
    current = 1
    for prev, curr in zip(frames, frames[1:]):
        if curr == prev + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def compute_instance_quality(preds, metadata):
    boxes_local = preds["bbox_local"].float()
    boxes_global = preds["bbox_global"].float()
    slice_idx = preds["slice_idx"].long().reshape(-1)
    ids = preds["id"].long().reshape(-1)
    apps = preds.get("appearance")

    heights = (boxes_global[:, 1, 1] - boxes_global[:, 0, 1]).clamp_min(1.0)
    left = boxes_local[:, 0, 0]
    top = boxes_local[:, 0, 1]
    right = boxes_local[:, 1, 0]
    bottom = boxes_local[:, 1, 1]

    dims = torch.tensor(
        [metadata["slice_dims"][int(i)] for i in slice_idx.tolist()],
        dtype=torch.float32,
    )
    slice_w = dims[:, 0]
    slice_h = dims[:, 1]
    edge_margin = torch.minimum(
        torch.minimum(left, top),
        torch.minimum(slice_w - right, slice_h - bottom),
    )
    edge_score = (edge_margin / heights).clamp(0.0, 1.0)
    unclipped = (
        (left > 2.0) & (top > 2.0)
        & (right < slice_w - 2.0) & (bottom < slice_h - 2.0)
    ).float()

    centers = (boxes_global[:, 0] + boxes_global[:, 1]) * 0.5
    owner = torch.tensor(
        [
            owner_slice_for_center(float(c[0]), float(c[1]), metadata)
            for c in centers.tolist()
        ],
        dtype=torch.long,
    )
    core_bonus = (owner == slice_idx).float()

    counts = defaultdict(int)
    for tid in ids.tolist():
        counts[int(tid)] += 1
    length_score = torch.tensor(
        [min(counts[int(tid)] / 30.0, 1.0) for tid in ids.tolist()],
        dtype=torch.float32,
    )

    if apps is not None and apps.shape[0] == ids.shape[0]:
        app_score = (apps.float().norm(dim=-1) > 0.01).float()
    else:
        app_score = torch.zeros_like(length_score)

    trans = preds["trans"].float()
    depth_score = torch.isfinite(trans).all(dim=-1).float() * (trans[:, 2] > 0).float()

    return (
        1.4 * edge_score
        + 0.8 * unclipped
        + 0.6 * core_bonus
        + 0.3 * length_score
        + 0.2 * app_score
        + 0.1 * depth_score
    )


class UnionFind:
    def __init__(self, values):
        self.parent = {int(v): int(v) for v in values}

    def find(self, value):
        value = int(value)
        parent = self.parent.setdefault(value, value)
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        root_l = self.find(left)
        root_r = self.find(right)
        if root_l == root_r:
            return False
        self.parent[root_r] = root_l
        return True


def select_hysteresis_indices(
    preds,
    qualities,
    metadata,
    matched_frame_pairs=None,
    switch_frames=3,
):
    ids = preds["id"].long().reshape(-1).tolist()
    frames = preds["frame_idx"].long().reshape(-1).tolist()
    slices = preds["slice_idx"].long().reshape(-1).tolist()
    boxes = preds["bbox_global"].float()
    centers = ((boxes[:, 0] + boxes[:, 1]) * 0.5).tolist()

    by_id_frame = defaultdict(lambda: defaultdict(list))
    for idx, (tid, fid) in enumerate(zip(ids, frames)):
        by_id_frame[int(tid)][int(fid)].append(idx)

    keep = []
    for tid in sorted(by_id_frame):
        active_slice = None
        pending_slice = None
        pending_count = 0

        for fid in sorted(by_id_frame[tid]):
            candidates = by_id_frame[tid][fid]
            best_idx = max(candidates, key=lambda i: float(qualities[i]))
            best_slice = int(slices[best_idx])

            if active_slice is None:
                active_slice = best_slice
                keep.append(best_idx)
                continue

            active_candidates = [i for i in candidates if int(slices[i]) == active_slice]
            if not active_candidates:
                active_slice = best_slice
                pending_slice = None
                pending_count = 0
                keep.append(best_idx)
                continue

            active_idx = max(active_candidates, key=lambda i: float(qualities[i]))
            if best_slice == active_slice:
                pending_slice = None
                pending_count = 0
                keep.append(active_idx)
                continue

            if float(qualities[best_idx]) <= float(qualities[active_idx]) + 0.05:
                pending_slice = None
                pending_count = 0
                keep.append(active_idx)
                continue

            if pending_slice == best_slice:
                pending_count += 1
            else:
                pending_slice = best_slice
                pending_count = 1

            if pending_count >= switch_frames:
                active_slice = best_slice
                pending_slice = None
                pending_count = 0
                keep.append(best_idx)
            else:
                keep.append(active_idx)

    keep_set = set(keep)
    if matched_frame_pairs:
        for idx_a, idx_b in matched_frame_pairs:
            if idx_a not in keep_set or idx_b not in keep_set:
                continue
            if int(ids[idx_a]) == int(ids[idx_b]):
                continue

            owner_a = owner_slice_for_center(centers[idx_a][0], centers[idx_a][1], metadata)
            owner_b = owner_slice_for_center(centers[idx_b][0], centers[idx_b][1], metadata)
            core_a = int(slices[idx_a]) == owner_a
            core_b = int(slices[idx_b]) == owner_b
            if core_a and not core_b:
                keep_set.discard(idx_b)
            elif core_b and not core_a:
                keep_set.discard(idx_a)
            elif float(qualities[idx_a]) >= float(qualities[idx_b]):
                keep_set.discard(idx_b)
            else:
                keep_set.discard(idx_a)

    keep = torch.tensor(sorted(keep_set), dtype=torch.long)
    return keep


def merge_cross_slice_duplicates(preds, metadata):
    total = preds["id"].reshape(-1).shape[0]
    if total == 0 or "bbox_global" not in preds:
        return preds

    ids = preds["id"].long().reshape(-1)
    frames = preds["frame_idx"].long().reshape(-1)
    slices = preds["slice_idx"].long().reshape(-1)
    boxes = preds["bbox_global"].float()
    trans = preds["trans"].float()
    apps = preds.get("appearance", torch.zeros(total, 384)).float()
    app_norms = apps.norm(dim=-1)
    centers = ((boxes[:, 0] + boxes[:, 1]) * 0.5).numpy()
    heights = (boxes[:, 1, 1] - boxes[:, 0, 1]).clamp_min(1.0).numpy()

    by_frame_slice = defaultdict(lambda: defaultdict(list))
    for idx, (fid, sid) in enumerate(zip(frames.tolist(), slices.tolist())):
        by_frame_slice[int(fid)][int(sid)].append(idx)

    adjacent_pairs = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]
    match_stats = {}
    matched_frame_pairs = []

    for fid in sorted(by_frame_slice):
        frame_groups = by_frame_slice[fid]
        for slice_a, slice_b in adjacent_pairs:
            if slice_a not in frame_groups or slice_b not in frame_groups:
                continue

            candidates = []
            for idx_a in frame_groups[slice_a]:
                for idx_b in frame_groups[slice_b]:
                    if ids[idx_a].item() == ids[idx_b].item():
                        continue
                    if not pair_overlap_band_ok(
                        slice_a, slice_b, centers[idx_a], centers[idx_b], metadata
                    ):
                        continue

                    iou = bbox_iou(boxes[idx_a].numpy(), boxes[idx_b].numpy())
                    if iou < 0.35:
                        continue

                    center_dist = float(np.linalg.norm(centers[idx_a] - centers[idx_b]))
                    height_ref = max(float(heights[idx_a]), float(heights[idx_b]), 1.0)
                    if center_dist > 0.35 * height_ref:
                        continue

                    trans_dist = float(torch.norm(trans[idx_a] - trans[idx_b]).item())
                    if trans_dist > 2.5:
                        continue

                    app_sim = 0.0
                    app_valid = bool(app_norms[idx_a] > 0.01 and app_norms[idx_b] > 0.01)
                    if app_valid:
                        app_sim = float(torch.nn.functional.cosine_similarity(
                            apps[idx_a].unsqueeze(0),
                            apps[idx_b].unsqueeze(0),
                        ).item())
                        if app_sim < 0.68:
                            continue

                    score = (
                        2.0 * iou
                        + max(0.0, 1.0 - center_dist / (0.35 * height_ref))
                        + max(0.0, 1.0 - trans_dist / 2.5)
                        + (app_sim if app_valid else 0.15)
                    )
                    candidates.append((score, idx_a, idx_b, iou, app_sim, app_valid))

            used_a = set()
            used_b = set()
            for score, idx_a, idx_b, iou, app_sim, app_valid in sorted(candidates, reverse=True):
                if idx_a in used_a or idx_b in used_b:
                    continue
                used_a.add(idx_a)
                used_b.add(idx_b)
                matched_frame_pairs.append((idx_a, idx_b))

                pair_key = tuple(sorted((int(ids[idx_a].item()), int(ids[idx_b].item()))))
                stats = match_stats.setdefault(pair_key, {
                    "frames": [],
                    "iou_sum": 0.0,
                    "app_sum": 0.0,
                    "app_count": 0,
                })
                stats["frames"].append(fid)
                stats["iou_sum"] += iou
                if app_valid:
                    stats["app_sum"] += app_sim
                    stats["app_count"] += 1

    uf = UnionFind(ids.unique().tolist())
    union_count = 0
    for pair_key, stats in match_stats.items():
        frames_unique = sorted(set(stats["frames"]))
        count = len(frames_unique)
        run = longest_consecutive_run(frames_unique)
        avg_iou = stats["iou_sum"] / max(len(stats["frames"]), 1)
        avg_app = stats["app_sum"] / stats["app_count"] if stats["app_count"] else 0.0

        strong_short = count >= 2 and avg_iou >= 0.65 and (stats["app_count"] == 0 or avg_app >= 0.75)
        if run >= 3 or count >= 5 or strong_short:
            if uf.union(pair_key[0], pair_key[1]):
                union_count += 1

    qualities = compute_instance_quality(preds, metadata)
    root_members = defaultdict(list)
    for tid in ids.unique().tolist():
        root_members[uf.find(int(tid))].append(int(tid))

    id_to_primary = {}
    for members in root_members.values():
        best_tid = min(members)
        best_score = -1e9
        for tid in members:
            idxs = (ids == tid).nonzero(as_tuple=True)[0]
            score = float(qualities[idxs].mean().item()) * 10.0 + float(len(idxs))
            if score > best_score or (score == best_score and tid < best_tid):
                best_score = score
                best_tid = tid
        for tid in members:
            id_to_primary[tid] = best_tid

    merged_ids = torch.tensor(
        [id_to_primary.get(int(tid), int(tid)) for tid in ids.tolist()],
        dtype=torch.long,
    )
    preds["id"] = merged_ids

    keep_idx = select_hysteresis_indices(
        preds,
        qualities,
        metadata,
        matched_frame_pairs=matched_frame_pairs,
        switch_frames=3,
    )
    filtered = {k: v[keep_idx] for k, v in preds.items()}

    sort_key = filtered["frame_idx"].long() * (int(filtered["id"].max().item()) + 1) + filtered["id"].long()
    order = torch.argsort(sort_key)
    filtered = {k: v[order] for k, v in filtered.items()}

    removed = total - filtered["id"].shape[0]
    print(
        f"Cross-slice merge: {len(match_stats)} candidate tracklet pairs, "
        f"{union_count} unions, removed {removed} duplicate instances."
    )
    return filtered


def merge_track_data(metadata, temp_dir):
    print(f"\n=== Step 3: Merging Tracks ===")
    id_shifts = [0, 10000, 20000, 30000]
    merged_data = {
        "id": [], "source_id": [], "slice_idx": [],
        "pose": [], "trans": [], "betas": [], "appearance": [],
        "frame_idx": [], "bbox_local": [], "bbox_global": [],
    }
    inferred_focals = []
    names = metadata["names"]

    try:
        from comotion_demo.utils import smpl_kinematics
        decoder = smpl_kinematics.SMPLKinematics()
    except ImportError:
        print("Error: Could not import SMPLKinematics for cross-slice bbox merge.")
        return None, None
    
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
        
        local_bboxes, global_bboxes = compute_slice_bboxes(decoder, data, i, metadata)
        trans_fixed, focal = fix_trans_for_global_view(data["trans"], i, metadata)
        inferred_focals.append(focal)
        source_ids = data['id'].long().reshape(-1)
        shifted_ids = source_ids + id_shifts[i]
        n = shifted_ids.shape[0]
        appearance = data.get("appearance")
        if appearance is None or appearance.shape[0] != n:
            appearance = torch.zeros(n, 384, dtype=torch.float32)
        
        merged_data["id"].append(shifted_ids)
        merged_data["source_id"].append(source_ids)
        merged_data["slice_idx"].append(torch.full((n,), i, dtype=torch.long))
        merged_data["pose"].append(data['pose'])
        merged_data["trans"].append(trans_fixed)
        merged_data["betas"].append(data['betas'])
        merged_data["appearance"].append(appearance.float())
        merged_data["frame_idx"].append(data['frame_idx'])
        merged_data["bbox_local"].append(local_bboxes)
        merged_data["bbox_global"].append(global_bboxes)

    if not merged_data["id"]:
        return None, None

    final_preds = {}
    for k in merged_data:
        final_preds[k] = torch.cat(merged_data[k], dim=0)
    final_preds = merge_cross_slice_duplicates(final_preds, metadata)
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
    print(f"Keeping temporary render frames at {tmp_vis_dir}")

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
    parser.add_argument("--require-reid", action="store_true", help="Fail if DINOv2 ReID cannot be loaded")
    parser.add_argument("--memory-mode", choices=["off", "basic", "long", "occlusion"], default=None)
    parser.add_argument("--enable-occlusion-gate", action="store_true")
    parser.add_argument("--enable-motion-damping", action="store_true")
    parser.add_argument("--enable-init-iou-suppress", action="store_true")
    parser.add_argument("--disable-reid-revival", action="store_true")
    parser.add_argument("--postprocess-mode", choices=["current", "off", "stitch", "interpolate", "stitch-interpolate"], default=None)
    args = parser.parse_args()

    demo_args = []
    if args.require_reid:
        demo_args.append("--require-reid")
    if args.memory_mode is not None:
        demo_args.extend(["--memory-mode", args.memory_mode])
    if args.enable_occlusion_gate:
        demo_args.append("--enable-occlusion-gate")
    if args.enable_motion_damping:
        demo_args.append("--enable-motion-damping")
    if args.enable_init_iou_suppress:
        demo_args.append("--enable-init-iou-suppress")
    if args.disable_reid_revival:
        demo_args.append("--disable-reid-revival")
    if args.postprocess_mode is not None:
        demo_args.extend(["--postprocess-mode", args.postprocess_mode])
    
    os.makedirs(args.temp, exist_ok=True)
    
    try:
        meta = slice_video(args.input, args.temp, args.overlap)
        run_inference(meta, args.temp, demo_args)
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
