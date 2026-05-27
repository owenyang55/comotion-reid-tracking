# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Demo CoMotion with a video file or a directory of images."""

import logging
import csv
import os
import shutil
import tempfile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from comotion_demo.models import comotion
from comotion_demo.utils import dataloading, helper
from comotion_demo.utils import track as track_utils

#导入接缝必要库
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# [修正] 导入 ReID 模块 (确保文件名匹配，如果是 reid_module.py 就改一下)
try:
    from comotion_demo.utils.reid import ReIDExtractor
except ImportError:
    try:
        from comotion_demo.utils.reid_module import ReIDExtractor
    except ImportError:
        print("WARNING: Could not import ReIDExtractor.")
        ReIDExtractor = None

# [修正] 导入必要模块
from comotion_demo.models import refine
from scenedetect.detectors import ContentDetector

try:
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.billboard import Billboard
    from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
    from aitviewer.scene.camera import OpenCVCamera

    comotion_model_dir = Path(comotion.__file__).parent
    CONFIG.smplx_models = os.path.join(comotion_model_dir, "../data")
    CONFIG.window_type = "pyqt6"
    aitviewer_available = True

except ModuleNotFoundError:
    print(
        "WARNING: Skipped aitviewer import, ensure it is installed to run visualization."
    )
    aitviewer_available = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_mps = torch.mps.is_available()


def init_reid_model(mode_name: str, require_reid: bool = False):
    """Initialize DINOv2 ReID model; fail fast when require_reid=True."""
    if ReIDExtractor is None:
        msg = "ReIDExtractor import failed; DINOv2 appearance branch is unavailable."
        if require_reid:
            raise RuntimeError(msg)
        print(f"WARNING: {msg} Falling back to zero appearance features.")
        return None

    try:
        print(f"Initializing DINOv2 ({mode_name})...")
        return ReIDExtractor(device=str(device))
    except Exception as e:
        msg = f"Failed to initialize DINOv2 ReID model: {e}"
        if require_reid:
            raise RuntimeError(msg) from e
        print(f"WARNING: {msg}. Falling back to zero appearance features.")
        return None


def prepare_scene(viewer, width, height, K, image_paths, fps=30):
    """Prepare the scene for AITViewer rendering."""
    viewer.reset()
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    extrinsics = np.eye(4)[:3]

    # Initialize camera
    cam = OpenCVCamera(K, extrinsics, cols=width, rows=height, viewer=viewer)
    viewer.scene.add(cam)
    viewer.scene.camera.position = [0, 0, -5]
    viewer.scene.camera.target = [0, 0, 10]
    viewer.auto_set_camera_target = False
    viewer.set_temp_camera(cam)
    viewer.playback_fps = fps

    # "billboard" display for video frames
    billboard = Billboard.from_camera_and_distance(
        cam, 100.0, cols=width, rows=height, textures=image_paths
    )
    viewer.scene.add(billboard)


def add_pose_to_scene(
    viewer,
    smpl_layer,
    betas,
    pose,
    trans,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    color_ref=None,
):
    """Add estimated poses to the rendered scene."""
    if betas.ndim == 2:
        betas = betas[None]
        pose = pose[None]
        trans = trans[None]

    poses_root = pose[..., :3]
    poses_body = pose[..., 3:]
    max_people = pose.shape[1]

    if (betas != 0).any():
        for person_idx in range(max_people):
            if color_ref is None:
                person_color = color
            else:
                person_color = color_ref[person_idx % len(color_ref)] * 0.4 + 0.3
            person_color = [c_ for c_ in person_color] + [alpha]

            valid_vals = (betas[:, person_idx] != 0).any(-1)
            idx_range = valid_vals.nonzero()
            if len(idx_range) > 0:
                trans[~valid_vals][..., 2] = -10000
                viewer.scene.add(
                    SMPLSequence(
                        smpl_layer=smpl_layer,
                        betas=betas[:, person_idx],
                        poses_root=poses_root[:, person_idx],
                        poses_body=poses_body[:, person_idx],
                        trans=trans[:, person_idx],
                        color=person_color,
                    )
                )


def visualize_poses(
    input_path,
    cache_path,
    video_path,
    start_frame,
    num_frames,
    frameskip=1,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    fps=30,
):
    """Visualize SMPL poses."""
    logging.info(f"Rendering SMPL video: {input_path}")

    # Prepare temporary directory with saved images
    tmp_vis_dir = Path(tempfile.mkdtemp())

    frame_idx = 0
    image_paths = []
    for image, K in dataloading.yield_image_and_K(
        input_path, start_frame, num_frames, frameskip
    ):
        image_height, image_width = image.shape[-2:]
        image = dataloading.convert_tensor_to_image(image)
        image_paths.append(f"{tmp_vis_dir}/{frame_idx:06d}.jpg")
        Image.fromarray(image).save(image_paths[-1])
        frame_idx += 1

    # Initialize viewer
    viewer = HeadlessRenderer(size=(image_width, image_height))

    if dataloading.is_a_video(input_path):
        fps = int(dataloading.get_input_video_fps(input_path))

    prepare_scene(viewer, image_width, image_height, K.cpu().numpy(), image_paths, fps)

    # Prepare SMPL poses
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    if not cache_path.exists():
        logging.warning("No detections found.")
    else:
        preds = torch.load(cache_path, weights_only=False, map_location="cpu")
        track_subset = track_utils.query_range(preds, 0, frame_idx - 1)
        id_lookup = track_subset["id"].max(0)[0]
        color_ref = helper.color_ref[id_lookup % len(helper.color_ref)]
        if len(id_lookup) == 1:
            color_ref = [color_ref]

        betas = track_subset["betas"]
        pose = track_subset["pose"]
        trans = track_subset["trans"]

        add_pose_to_scene(
            viewer, smpl_layer, betas, pose, trans, color, alpha, color_ref
        )

    # Save rendered scene
    viewer.save_video(
        video_dir=str(video_path),
        output_fps=fps,
        ensure_no_overwrite=False,
    )

    # Remove temporary directory
    shutil.rmtree(tmp_vis_dir)


def write_revive_log_csv(path: Path, rows, fieldnames) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_detection(
    input_path,
    cache_path,
    skip_visualization=False,
    model=None,
    require_reid=False,
):
    """Run model and visualize detections on single image."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    # 初始化 ReID 模型 (单图模式)
    reid_model = init_reid_model("Single Image", require_reid=require_reid)

    # Load image
    image = np.array(Image.open(input_path))
    image = dataloading.convert_image_to_tensor(image)
    K = dataloading.get_default_K(image)
    cropped_image, cropped_K = dataloading.prepare_network_inputs(image, K, device)

    # Get detections
    detections = model.detection_model(cropped_image, cropped_K)
    detections = comotion.detect.decode_network_outputs(
        K.to(device),
        model.smpl_decoder,
        detections,
        img_tensor=image.to(device),
        reid_model=reid_model,
        std=0.15,  # Adjust NMS sensitivity
        conf_thr=0.25,  # Adjust confidence cutoff
    )

    detections = {k: v[0].cpu() for k, v in detections.items()}
    torch.save(detections, cache_path)

    if not skip_visualization:
        # Initialize viewer
        image_height, image_width = image.shape[-2:]
        viewer = HeadlessRenderer(size=(image_width, image_height))
        prepare_scene(
            viewer, image_width, image_height, K.cpu().numpy(), [str(input_path)]
        )

        # Prepare SMPL poses
        smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
        add_pose_to_scene(
            viewer,
            smpl_layer,
            detections["betas"],
            detections["pose"],
            detections["trans"],
        )

        # Save rendered scene
        viewer.save_frame(str(cache_path).replace(".pt", ".png"))


def track_poses(
    input_path,
    cache_path,
    start_frame,
    num_frames,
    frameskip=1,
    model=None,
    require_reid=False,
):
    """Track poses over a video or a directory of images."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    # 初始化 ReID 模型
    reid_model = init_reid_model("Video Tracking", require_reid=require_reid)

    detections_list = []
    tracks_list = []

    K_cpu = None
    initialized = False

    # 镜头检测器
    if hasattr(model, 'shot_detector'):
        shot_detector = model.shot_detector
        model.frame_count = 0 
    else:
        shot_detector = ContentDetector(threshold=50.0, min_scene_len=3)
    
    current_frame_count = 0 

    with torch.inference_mode():
        for image, K in tqdm(
            dataloading.yield_image_and_K(input_path, start_frame, num_frames, frameskip),
            desc="Running CoMotion + DINOv2",
            ):
            # 保持 image, K 在 CPU
            K_cpu = K

            if not initialized:
                image_res = image.shape[-2:]
                model.init_tracks(image_res)
                initialized = True

            # 1. 输入图像预处理 (在 CPU 上准备)
            cropped_images, cropped_K = dataloading.prepare_network_inputs(image, K, device)

            # 2. 运行检测网络 (得到 GPU 结果)
            # 【注意】这里变量名必须叫 detect_out，因为下面的 call_update 闭包里用的是这个名字
            detect_out = model.detection_model(cropped_images, cropped_K)

            # 3. 提取 ReID 特征
            detection = comotion.detect.decode_network_outputs(
                K.to(device),
                model.smpl_decoder,
                detect_out,
                img_tensor=image.to(device), # 传入 GPU 原图用于裁剪
                reid_model=reid_model,
                std=0.15, 
                conf_thr=0.25
            )    

            # 4. 定义更新回调函数 (闭包，访问外部的 detect_out)
            def call_update(s):
                # s 是当前轨迹的状态 (TrackTensorState)
                update_args = [
                    detect_out.image_features, # 引用上面的变量
                    cropped_K,
                    s.betas,
                    s.pose,
                    s.trans,
                    s.pred_3d,
                    s.hidden, # 上一帧的 motion hidden state
                ]
                
                if use_mps:
                    update_args = [arg.to("mps") for arg in update_args]

                updated_params = model.update_step(*update_args)
                
                # 解包结果
                updated_params = refine.RefineOutput(
                    **{k: v.to(device) for k, v in updated_params._asdict().items()}
                )

                s.pose = comotion.detect.get_smpl_pose(
                    updated_params.delta_root_orient,
                    updated_params.delta_body_pose,
                )
                s.trans = updated_params.trans
                s.hidden = updated_params.hidden
                
                s.pred_3d = model.smpl_decoder(
                    s.betas, s.pose, s.trans, output_format="joints_face"
                )
                s.pred_2d = helper.project_to_2d(K.to(device), s.pred_3d)

            # 5. 镜头切换检测
            image_np = image.detach().cpu().permute(1, 2, 0).numpy()
            image_np = image_np[:, :, ::-1]  # RGB2BGR
            
            shots = shot_detector.process_frame(current_frame_count, image_np)
            current_frame_count += 1
            is_new_shot = len(shots) > 1 if current_frame_count > 1 else False

            # 6. 驱动追踪器
            # 【注意】这里必须用 model.handler，而不是 model.tracker
            track = model.handler.update(
                detection, 
                call_update, 
                shot_reset=is_new_shot
            )

            # 7. 存储结果
            detection = {k: v.cpu() for k, v in detection.items()}
            track = track.cpu()
            detections_list.append(detection)
            tracks_list.append(track)
            
            # 【重要】删除了这里原有的导致报错的重复代码

    if not detections_list:
        print("No frames processed.")
        return

    # 结果堆叠
    try:
        final_tracks = {
            "id": torch.stack([t.id for t in tracks_list], 1),
            "pose": torch.stack([t.pose for t in tracks_list], 1),
            "trans": torch.stack([t.trans for t in tracks_list], 1),
            "betas": torch.stack([t.betas for t in tracks_list], 1),
            "appearance": torch.stack([t.appearance for t in tracks_list], 1),
        }
    except Exception as e:
        print(f"Error during stacking: {e}")
        return

    # 处理 detections (过滤掉 appearance)
    detections_dict = {}
    first_det = detections_list[0]
    for k in first_det.keys():
        # 跳过导致冲突的字段，或者不需要保存的字段
        if k in ["appearance", "hidden"]: 
            continue
        detections_dict[k] = [d[k] for d in detections_list]

    # 后处理清理
    if K_cpu is not None:
        track_ref = track_utils.cleanup_tracks(
            {"detections": detections_dict, "tracks": final_tracks},
            K_cpu,
            model.smpl_decoder.cpu(),
            min_matched_frames=1,
        )
        if track_ref:
            frame_idxs, track_idxs = track_utils.convert_to_idxs(
                track_ref, final_tracks["id"][0].squeeze(-1).long()
            )
            preds = {k: v[0, frame_idxs, track_idxs] for k, v in final_tracks.items()}
            preds["id"] = preds["id"].squeeze(-1).long()
            preds["frame_idx"] = frame_idxs

            #插入缝合函数(参数含义暂时未知)
            preds = perform_stitching(preds, time_thr=15, dist_thr=1.0, app_thr=0.75)
            preds = apply_interpolation_to_tensor(preds, max_gap=30)
            torch.save(preds, cache_path)

            # Save bounding box tracks in MOT format
            #未来可能还需要传入修正ID后的pose等参数（ID缝合）
            image_res = image.shape[-2:]
            bboxes = track_utils.bboxes_from_smpl(
                model.smpl_decoder,
                {k: preds[k] for k in ["betas", "pose", "trans"]},
                image_res,
                K_cpu,
                pad_x = 0.2,
                pad_y = 0.2, #增大padding，提高IOU分数
            )
            txt_path = str(cache_path).replace(".pt", ".txt")

            write_interpolated_mot(
                txt_path,
                preds,
                bboxes,
                max_gap=30,
            )
        
            # # 准备数据
            # raw_ids = preds["id"].long().numpy()
            # raw_frames = preds["frame_idx"].long().numpy()
            # raw_bboxes = bboxes.float().numpy()
            
            # # 用于记录已写入的 (frame, id)，防止重复
            # written_keys = set()
            # unique_lines = []
            
            # # 遍历所有预测结果
            # for i in range(len(raw_ids)):
            #     fid = int(raw_frames[i]) + 1 # MOT格式帧号从1开始
            #     tid = int(raw_ids[i])
                
            #     # 如果这一帧已经有过这个 ID 了，直接跳过（去重）
            #     if (fid, tid) in written_keys:
            #         continue
                    
            #     written_keys.add((fid, tid))
                
            #     # 获取 bbox
            #     box = raw_bboxes[i] # [x1, y1, x2, y2]
            #     bb_left = box[0, 0] # 左上角 x
            #     bb_top = box[0, 1]  # 左上角 y
            #     bb_right = box[1, 0] # 右下角 x
            #     bb_bottom = box[1, 1] # 右下角 y


            #     bb_width = bb_right - bb_left
            #     bb_height = bb_bottom - bb_top
            #     # 格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            #     line = f"{fid},{tid},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},1,1,1\n"
            #     unique_lines.append(line)
                
            # # 一次性写入文件
            # with open(txt_path, "w") as f:
            #     f.writelines(unique_lines)
                
            # print(f"✅ 已保存结果到 {txt_path} (原始: {len(raw_ids)} -> 去重后: {len(unique_lines)})")



# [新增] 后处理工具函数
    if hasattr(model, "handler") and hasattr(model.handler, "revive_decision_log"):
        revive_log_path = cache_path.with_name(f"{cache_path.stem}_revive_log.csv")
        write_revive_log_csv(
            revive_log_path,
            model.handler.revive_decision_log,
            getattr(
                model.handler,
                "revive_log_fields",
                list(model.handler.revive_decision_log[0].keys())
                if model.handler.revive_decision_log
                else [],
            ),
        )
        print(f"Saved revive log to {revive_log_path}")


def perform_stitching(preds, time_thr=30, dist_thr=2.0, app_thr=0.6):
    """
    [调试版] 轨迹缝合：包含强制重叠检查
    """
    print(">>> 正在执行离线轨迹缝合 (Stitching) - 严格模式...")
    
    ids = preds["id"].long().numpy()
    frames = preds["frame_idx"].long().numpy()
    trans = preds["trans"].float().numpy()
    if "appearance" not in preds:
        return preds
    apps = preds["appearance"].float().numpy()

    # 1. 整理 Tracklets
    track_stats = {}
    for i in range(len(ids)):
        tid = ids[i]
        fid = frames[i]
        pos = trans[i]
        app = apps[i]
        
        if tid not in track_stats:
            track_stats[tid] = {
                'start_frame': fid, 'end_frame': fid,
                'start_pos': pos, 'end_pos': pos,
                'app_sum': app, 'count': 1, 'id': tid
            }
        else:
            if fid < track_stats[tid]['start_frame']:
                track_stats[tid]['start_frame'] = fid
                track_stats[tid]['start_pos'] = pos
            if fid > track_stats[tid]['end_frame']:
                track_stats[tid]['end_frame'] = fid
                track_stats[tid]['end_pos'] = pos
            
            track_stats[tid]['app_sum'] += app
            track_stats[tid]['count'] += 1

    tracks_list = []
    for tid, info in track_stats.items():
        feat = info['app_sum'] / info['count']
        norm = np.linalg.norm(feat)
        info['avg_app'] = feat / (norm + 1e-6)
        del info['app_sum']
        tracks_list.append(info)

    # 按开始时间排序
    tracks_list.sort(key=lambda x: x['start_frame'])
    
    # === 全局结束时间追踪 ===
    id_map = {t['id']: t['id'] for t in tracks_list}
    # 记录每个真实 ID (合并后的) 的所有占据时间段
    # 格式: {real_id: [(start, end), (start, end), ...]}
    cluster_intervals = {t['id']: [(t['start_frame'], t['end_frame'])] for t in tracks_list}
    
    merge_count = 0

    for i in range(len(tracks_list)):
        curr = tracks_list[i]
        
        # 寻找最佳前驱
        best_prev_idx = -1
        best_score = -100.0
        
        # 向前搜索
        for j in range(i-1, -1, -1):
            prev = tracks_list[j]
            target_real_id = id_map[prev['id']]
            
            if target_real_id == curr['id']: continue 

            # === [强力重叠检查] ===
            # 检查 current 是否与 target_real_id 已有的任何时间段重叠
            has_overlap = False
            existing_intervals = cluster_intervals[target_real_id]
            
            # curr 的时间段
            c_start, c_end = curr['start_frame'], curr['end_frame']
            
            # 遍历目标 ID 的所有片段，检查是否冲突
            # 允许 1 帧的容错 (gap >= 1)
            for (i_start, i_end) in existing_intervals:
                # 如果 curr 在 interval 之前结束，或在 interval 之后开始，则不重叠
                # 反之，则重叠
                if not (c_end < i_start or c_start > i_end):
                    has_overlap = True
                    break
            
            if has_overlap:
                continue # 绝对禁止重叠合并

            # 计算 Gap (基于最近的一个结束时间，或者直接用 prev 的结束时间)
            # 这里我们只关心和 prev 的连接紧密程度，因为全局重叠已经check过了
            gap = curr['start_frame'] - prev['end_frame']
            
            if gap <= 0: continue # 局部也不能倒流
            if gap > time_thr: continue
            
            # 空间与外貌检查
            dist = np.linalg.norm(prev['end_pos'] - curr['start_pos'])
            dynamic_dist_thr = dist_thr + (0.05 * gap)
            if dist > dynamic_dist_thr: continue
            
            sim = np.dot(prev['avg_app'], curr['avg_app'])
            if sim < app_thr: continue
            
            score = sim * 10 - dist
            if score > best_score:
                best_score = score
                best_prev_idx = j
        
        # === 执行合并 ===
        if best_prev_idx != -1:
            prev = tracks_list[best_prev_idx]
            target_id = id_map[prev['id']]
            old_id = curr['id']
            
            # double check (双重检查)
            final_gap = curr['start_frame'] - prev['end_frame']
            if final_gap <= 0:
                print(f"!!! 警告: 逻辑漏洞，跳过异常合并 {old_id}->{target_id} (Gap {final_gap})")
                continue

            # 更新映射
            id_map[old_id] = target_id
            
            # 更新该 Cluster 的时间段记录
            cluster_intervals[target_id].append((curr['start_frame'], curr['end_frame']))
            
            # 更新前驱节点的局部信息 (为了方便链式缝合)
            prev['end_frame'] = curr['end_frame']
            prev['end_pos'] = curr['end_pos']
            
            print(f"  [Stitch] ID {old_id} -> ID {target_id} (Gap: {final_gap} frames)")
            merge_count += 1

    # 应用结果
    new_ids = []
    for i in range(len(ids)):
        old = ids[i]
        new_ids.append(id_map.get(old, old))
            
    preds["id"] = torch.tensor(new_ids).long()
    print(f">>> 缝合完成，共执行 {merge_count} 次合并。")
    return preds



def apply_interpolation_to_tensor(preds, max_gap=30):
    """
    直接对 preds 张量数据进行线性插值，填补帧空缺。
    让 3D 渲染也能看到连贯的动作。
    """
    print(">>> 正在对 3D 数据进行插值 (Interpolating Tensors)...")
    
    # 1. 提取原始数据 (全部转为 list 方便操作)
    ids = preds["id"].long().flatten().tolist()
    frames = preds["frame_idx"].long().flatten().tolist()
    
    # 找出哪些 Tensor 需要插值
    # 通常我们需要: trans, pose, betas
    # appearance, pred_2d, pred_3d 可以不插值，或者简单插值
    keys_to_interp = ["trans", "pose", "betas"]
    
    # 构建数据索引: {id: {frame: index_in_tensor}}
    track_map = defaultdict(dict)
    for i, (tid, fid) in enumerate(zip(ids, frames)):
        track_map[tid][fid] = i

    new_data = defaultdict(list) # 用于存放插值生成的新数据
    new_data["id"] = []
    new_data["frame_idx"] = []
    
    # 2. 遍历轨迹寻找空缺
    for tid, frames_dict in track_map.items():
        sorted_frames = sorted(frames_dict.keys())
        if len(sorted_frames) < 2: continue
        
        for i in range(len(sorted_frames) - 1):
            f1 = sorted_frames[i]
            f2 = sorted_frames[i+1]
            gap = f2 - f1
            
            # 发现空缺
            if 1 < gap <= max_gap:
                idx1 = frames_dict[f1]
                idx2 = frames_dict[f2]
                
                # 对每个步骤进行生成
                for step in range(1, gap):
                    alpha = step / gap
                    interp_frame = f1 + step
                    
                    # 记录元数据
                    new_data["id"].append(tid)
                    new_data["frame_idx"].append(interp_frame)
                    
                    # 对关键属性进行线性插值
                    for key in keys_to_interp:
                        if key in preds:
                            val1 = preds[key][idx1]
                            val2 = preds[key][idx2]
                            # 线性插值: val = (1-a)*v1 + a*v2
                            interp_val = (1 - alpha) * val1 + alpha * val2
                            new_data[key].append(interp_val)
                            
    # 3. 如果没有新数据，直接返回
    if not new_data["id"]:
        print("没有发现需要插值的空缺。")
        return preds

    # 4. 将新数据拼接到原始 Tensor 中
    print(f"  生成了 {len(new_data['id'])} 帧补全数据。")
    
    # 转换 list 为 tensor
    device = preds["id"].device
    new_ids_tensor = torch.tensor(new_data["id"], device=device).long() # (N, 1)
    new_frames_tensor = torch.tensor(new_data["frame_idx"], device=device).long()     # (N,)
    
    # 拼接 ID 和 Frame
    preds["id"] = torch.cat([preds["id"], new_ids_tensor], dim=0)
    preds["frame_idx"] = torch.cat([preds["frame_idx"], new_frames_tensor], dim=0)
    
    # 拼接 3D 属性
    for key in keys_to_interp:
        if key in preds and key in new_data:
            new_tensor_list = new_data[key] # list of tensors
            if len(new_tensor_list) > 0:
                new_block = torch.stack(new_tensor_list).to(device)
                preds[key] = torch.cat([preds[key], new_block], dim=0)

    # 对于没有插值的键 (比如 appearance, pred_2d)，需要填充 0 或最后已知值，
    # 否则长度不一致会报错。
    # 简单起见，我们只拼接那些 visualize 需要的键。
    # 注意：AITViewer 渲染通常只需要 id, frame_idx, pose, trans, betas。
    # 如果 preds 里还有其他键 (如 appearance)，长度没跟上可能会出问题。
    # 策略：对于不在插值列表里的键，根据 new_ids 的长度补零。
    current_len = preds["id"].shape[0]
    for key in preds.keys():
        if key not in keys_to_interp and key not in ["id", "frame_idx"]:
            if preds[key].shape[0] < current_len:
                diff = current_len - preds[key].shape[0]
                # 创建零张量补齐
                pad_shape = (diff, *preds[key].shape[1:])
                zero_pad = torch.zeros(pad_shape, dtype=preds[key].dtype, device=device)
                preds[key] = torch.cat([preds[key], zero_pad], dim=0)

    # 5. (可选) 按帧号排序，保证数据整洁
    # visualize 函数通常使用 query_range，它是按 mask 提取的，只要 frame_idx 对就行，顺序不严格要求。
    # 但为了稳健，还是排个序。
    sort_idx = torch.argsort(preds["frame_idx"])
    for key in preds.keys():
        preds[key] = preds[key][sort_idx]

    return preds



def write_interpolated_mot(txt_path, preds, bboxes, max_gap=30):
    """
    带插值的 MOT 文件写入
    """
    raw_ids = preds["id"].long().numpy()
    raw_frames = preds["frame_idx"].long().numpy()
    raw_bboxes = bboxes.float().numpy() # [N, 2, 2] -> xyxy
    
    # 1. 整理数据结构: {id: {frame: bbox}}
    data_by_id = defaultdict(dict)
    for i in range(len(raw_ids)):
        tid = int(raw_ids[i])
        fid = int(raw_frames[i])
        box = raw_bboxes[i] # [[x1,y1], [x2,y2]]
        
        # 转为 [x1, y1, w, h] 用于插值计算更方便 (或者保持 xyxy 插值也可以)
        # 这里保持 xyxy 插值，写入时再转 wh
        x1, y1 = box[0, 0], box[0, 1]
        x2, y2 = box[1, 0], box[1, 1]
        data_by_id[tid][fid] = np.array([x1, y1, x2, y2])
        
    # 2. 执行插值并收集所有行
    final_lines = []
    
    for tid, frames_dict in data_by_id.items():
        sorted_frames = sorted(frames_dict.keys())
        if not sorted_frames: continue
        
        # 遍历该 ID 的每一帧
        for i in range(len(sorted_frames)):
            curr_f = sorted_frames[i]
            curr_box = frames_dict[curr_f]
            
            # 先添加当前帧
            final_lines.append((curr_f, tid, curr_box))
            
            # 检查下一帧，看是否有空缺
            if i < len(sorted_frames) - 1:
                next_f = sorted_frames[i+1]
                gap = next_f - curr_f
                
                # 如果有空缺且小于阈值，进行插值
                if 1 < gap <= max_gap:
                    next_box = frames_dict[next_f]
                    for step in range(1, gap):
                        alpha = step / gap
                        interp_f = curr_f + step
                        interp_box = (1 - alpha) * curr_box + alpha * next_box
                        final_lines.append((interp_f, tid, interp_box))

    # 3. 排序并写入
    final_lines.sort(key=lambda x: x[0]) # 按帧号排序
    
    with open(txt_path, "w") as f:
        for fid, tid, box in final_lines:
            # 转为 MOT 格式: frame, id, left, top, w, h, ...
            # 注意: MOT 帧号从 1 开始
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            line = f"{fid + 1},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1\n"
            f.write(line)
            
    print(f"Saved results to {txt_path} (with interpolation)")

@click.command()
@click.option(
    "-i",
    "--input-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the input video, a directory of images, or a single input image.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help="Path to the output directory.",
)
@click.option(
    "-s",
    "--start-frame",
    default=0,
    type=int,
    help="Frame to start with.",
)
@click.option(
    "-n",
    "--num-frames",
    default=1_000_000_000,
    type=int,
    help="Number of frames to process.",
)
@click.option(
    "--skip-visualization",
    is_flag=True,
    help="Whether to skip rendering the output SMPL meshes.",
)
@click.option(
    "--frameskip",
    default=1,
    type=int,
    help="Subsample video frames (e.g. frameskip=2 processes every other frame).",
)
@click.option(
    "--require-reid",
    is_flag=True,
    help="Fail fast if DINOv2 ReID cannot be loaded (prevents silent fallback).",
)
def main(
    input_path,
    output_dir,
    start_frame,
    num_frames,
    skip_visualization,
    frameskip,
    require_reid,
):
    """Demo entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = input_path.stem
    skip_visualization = skip_visualization | (not aitviewer_available)

    cache_path = output_dir / f"{input_name}.pt"
    if input_path.suffix.lower() in dataloading.IMAGE_EXTENSIONS:
        # Run and visualize detections for a single image
        run_detection(
            input_path,
            cache_path,
            skip_visualization,
            require_reid=require_reid,
        )
    else:
        # Run unrolled tracking on a full video
        track_poses(
            input_path,
            cache_path,
            start_frame,
            num_frames,
            frameskip,
            require_reid=require_reid,
        )
        if not skip_visualization:
            video_path = output_dir / f"{input_name}.mp4"
            visualize_poses(
                input_path, cache_path, video_path, start_frame, num_frames, frameskip
            )


if __name__ == "__main__":
    main()
