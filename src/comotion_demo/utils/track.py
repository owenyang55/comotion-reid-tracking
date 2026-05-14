# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from __future__ import annotations
from scipy.optimize import linear_sum_assignment

from typing import Dict

import tensordict
import torch
from tqdm import tqdm

from . import helper, smpl_kinematics

default_dims = {
    "betas": (smpl_kinematics.BETA_DOF,),
    "pose": (smpl_kinematics.POSE_DOF,),
    "trans": (smpl_kinematics.TRANS_DOF,),
    "pred_3d": (smpl_kinematics.NUM_PARTS_AUX, 3),
    "pred_2d": (smpl_kinematics.NUM_PARTS_AUX, 2),
    "hidden": (512,),
    "id": (1,),
    "appearance": (384,),
}


@tensordict.tensorclass
class TrackTensorState:
    betas: torch.Tensor
    pose: torch.Tensor
    trans: torch.Tensor
    pred_3d: torch.Tensor
    pred_2d: torch.Tensor
    hidden: torch.Tensor
    id: torch.Tensor
    appearance: torch.Tensor # 新增外貌

class MonitorAttribute:
    def __init__(self, ema_coef=[0.75, 0.9]):
        self.log = []
        self.ema_log = {c: [] for c in ema_coef}

    def update(self, x):
        self.log.append(x)
        for c in self.ema_log:
            if len(self.ema_log[c]) == 0:
                self.ema_log[c].append(x)
            else:
                x_ = c * self.ema_log[c][-1] + (1 - c) * x
                self.ema_log[c].append(x_)


class TrackHealthMonitor:
    """Keep track of various statistics about tracks to determine when to end them.
    升级功能：记住外貌特征"""

    def __init__(
        self,
        init_timestep,
        match_discard_thr=0.15,
        inbounds_discard_thr=0.15,
        ema_coef=0.8,
        init_hidden=None,   #加入外貌特征
        memory_size=10,     #外貌特征队列长度
    ):
        self.match_discard_thr = match_discard_thr
        self.inbounds_discard_thr = inbounds_discard_thr
        self.ema_coef = ema_coef

        self.init_timestep = init_timestep
        self.age = 0
        self.data_keys = ["root pos", "root vel", "match", "overlap", "inbounds"]
        self.attributes = {
            k: MonitorAttribute(ema_coef=[ema_coef]) for k in self.data_keys
        }
        self.is_redundant = False

        self.memory_size = memory_size
        self.feature_gallery = []   #增加特征队列及队列大小
        self.revive_age = 0 #增加复活后年龄
        self.appearance_emb = None

        if init_hidden is not None: #存入第一帧
            norm_feat = torch.nn.functional.normalize(init_hidden.float().cpu(), dim=-1)
            self.feature_gallery.append(norm_feat)
            self.appearance_emb = norm_feat

    def update(self, root_pos, match_scores, track_overlap, inbounds_pct, current_hidden=None):
        self.age += 1
        self.revive_age += 1
        if self.age > 1:
            diff = root_pos - self.attributes["root pos"].log[-1]
            diff[..., 2] /= 3  # Rescale z down
            root_vel = diff.norm()
        else:
            root_vel = torch.zeros([])
        new_vals = [root_pos, root_vel, match_scores, track_overlap, inbounds_pct]
        for k, v in zip(self.data_keys, new_vals):
            self.attributes[k].update(v)

        #加入关键帧存入操作
        #1、EMA更新
        if current_hidden is not None and match_scores >0.4:
            new_emb =torch.nn.functional.normalize(current_hidden.float().cpu(), dim=-1)
            if self.appearance_emb is None:
                self.appearance_emb = new_emb
            else:
                #保留大部分历史数据，而小部分进行学习
                self.appearance_emb = self.ema_coef * self.appearance_emb + (1 - self.ema_coef) * new_emb   
                self.appearance_emb = torch.nn.functional.normalize(self.appearance_emb, dim=-1)
        
            #2、关键帧存入(匹配度高+每隔10帧提取)
            if match_scores > 0.6:
                if len(self.feature_gallery) == 0:
                    self.feature_gallery.append(new_emb)
                else:
                    last_sim = torch.sum(self.feature_gallery[-1] * new_emb)    #计算余弦相似度
                    if last_sim < 0.95:
                        self.feature_gallery.append(new_emb)

            #3、淘汰特征队列（FIFO）
            if len(self.feature_gallery) > self.memory_size:
                self.feature_gallery.pop(0) 


        # if current_hidden is not None:
        #     # 1. 验证 EMA 是否保持在单位球面上 (应该总是接近 1.0)
        #     ema_norm = self.appearance_emb.norm().item()
            
        #     # 2. 验证画廊是否在增长 (应该从 1 慢慢涨到 10)
        #     gallery_size = len(self.feature_gallery)
            
        #     # 3. 验证是否真的发生了更新 (计算当前特征与记忆的相似度)
        #     # 如果是同一物体，相似度应该很高 (0.9 ~ 0.99)
        #     # 如果这里报错 device mismatch，说明 hidden 没转 cpu
        #     cur_feat_norm = torch.nn.functional.normalize(current_hidden.float(), dim=-1)
        #     sim = torch.sum(self.appearance_emb * cur_feat_norm).item()

        #     # 打印日志 (只打印 ID 1 或者每隔 10 帧打印一次，防止刷屏太快)
        #     # 由于 Monitor 内部暂无 ID 属性，我们直接打印
        #     if gallery_size <= self.memory_size: 
        #         print(f"[Memory Check] Gallery Size: {gallery_size}/10 | Sim: {sim:.4f} | EMA Norm: {ema_norm:.4f}")
        #         if gallery_size == self.memory_size:
        #             print("--> [Memory Full] Feature Gallery is ready!")                  

    def get(self, key, ema_coef=None, use_ema=False):
        if use_ema:
            ema_coef = self.ema_coef
        if ema_coef is None:
            return self.attributes[key].log[-1]
        else:
            return self.attributes[key].ema_log[ema_coef][-1]

    def get_current_health(self):

        if hasattr(self, 'revive_age') and self.revive_age < 5: #当健康时间不足时，强制其为健康
            return 1

        if self.is_redundant:
            # Marked redundant
            return 0
        elif self.age <= 3 and self.get("match") < self.match_discard_thr:
            # Bad new track
            return 0
        elif self.get("match", self.ema_coef) < self.match_discard_thr:
            # Track not matching well
            return 0
        elif self.get("inbounds", self.ema_coef) < self.inbounds_discard_thr:
            # Track out-of-bounds
            return 0
        else:
            return 1
        
    def compute_max_similarity(self, query_hidden):
        """
        计算查询特征与记忆队列的相似度，并返回最大值
        """
        if not self.feature_gallery:
            return 0.0
        query_norm = torch.nn.functional.normalize(query_hidden.float(), dim=-1)

        gallery_stack = torch.stack(self.feature_gallery)

        scores = torch.matmul(gallery_stack, query_norm).squeeze()

        return scores.max().item()


class TrackHandler:
    def __init__(
        self,
        ref_dims=None,
        res=None,
        init_conf_thr=0.2,
        init_match_thr=0.75,
        missing_match_thr=0.2,  #重叠度测试，方便复活
        max_output_tracks=48,
        overlap_thr=0.6,
        overlap_time_thr=20,
        vel_outlier_thr=0.25,
    ):
        if ref_dims is None:
            ref_dims = default_dims
        if res is None:
            res = (512, 512)
        self.ref_dims = ref_dims
        self.res = res

        self.current_step = 0
        self.last_track_id = 0
        self.current_tracks: TrackTensorState | None = None
        self.health_monitors: Dict[int, TrackHealthMonitor] = {}
        self.cached_tracks = []
        self.cached_detections = []
        self.collapse_count = {}
        self.init_conf_thr = init_conf_thr
        self.init_match_thr = init_match_thr
        self.missing_match_thr = missing_match_thr
        self.max_output_tracks = max_output_tracks
        self.overlap_thr = overlap_thr
        self.overlap_time_thr = overlap_time_thr
        self.vel_outlier_thr = vel_outlier_thr

        self.lost_tracks = []   #增加丢失轨迹列表
        self.max_lost_patience = 60 #允许消失最大时间

    def clear_tracks(self):
        """Clear existing tracks, preserve last_track_id."""
        self.current_tracks = None

    def convert_to_outputs(self, device="cpu"):
        if self.current_tracks is None:
            empty_state = {
                k: torch.zeros(1, 1, *ref_shape, device=device)
                for k, ref_shape in self.ref_dims.items()
            }
            tracks = TrackTensorState(**empty_state, batch_size=(1, 1))
        else:
            tracks = self.current_tracks

        num_tracks = tracks.shape[1]
        if num_tracks > self.max_output_tracks:
            tracks = tracks[:, : self.max_output_tracks]
        else:
            to_pad = self.max_output_tracks - num_tracks
            tracks = tensordict.pad(tracks, [0, 0, 0, to_pad])

        # Reset so that padded and truncated versions are compatible
        tracks = TrackTensorState(**tracks.to_dict(), batch_size=tracks.batch_size)
        return tracks

    def get_state_from_detections(self, detections):
        to_init = ["id"]
        if "hidden" not in detections:
            to_init += ["hidden"]
        if "appearance" not in detections:
            to_init += ["appearance"]
        

        to_ignore = ["scale", "conf"]
        track_state = {k: v for k, v in detections.items() if k not in to_ignore}
        conf = detections["conf"]
        batch_size = track_state["betas"].shape[:-1]
        device = track_state["betas"].device
        for k in to_init:
            track_state[k] = torch.zeros(*batch_size, *self.ref_dims[k], device=device)
        track_state["id"][:] = torch.arange(batch_size[1]).unsqueeze(-1)
        return TrackTensorState(**track_state, batch_size=batch_size), conf

    def compare_detections(self, p0, p1, normalize_factor=1024):
        """Get match scores between all pairs of predictions."""
        p0 = p0 / normalize_factor
        p1 = p1 / normalize_factor
        c0 = torch.ones_like(p0[..., 0])
        c1 = torch.ones_like(p1[..., 0])

        return helper.normalized_weighted_score(p0, c0, p1, c1)

    def initialize_tracks(self, detections, update_fn, init_state=None, conf=None):
        """Get bounding boxes from input detections, and initialize new tracks."""
        if init_state is None:
            init_state, conf = self.get_state_from_detections(detections)
        ref_2d = init_state.pred_2d.clone()
        num_detections = ref_2d.shape[1]
        if num_detections > 0:
            update_fn(init_state)

            # Check for self-consistency in detections (only keep samples that match)
            match_scores = self.compare_detections(init_state.pred_2d[0], ref_2d[0])
            to_keep = match_scores.diagonal() > self.init_match_thr
            if conf is not None:
                # We have a higher bar for initialization than for matching
                conf_filter = torch.sigmoid(conf.squeeze()) > self.init_conf_thr
                to_keep = to_keep & conf_filter
            match_scores = match_scores.diagonal()[to_keep]

            num_to_keep = to_keep.sum()
            if num_to_keep > 0:
                init_state = init_state[:, to_keep]
                new_ids = (
                    torch.arange(num_to_keep, device=init_state.id.device)
                    + self.last_track_id
                    + 1
                )
                self.last_track_id += num_to_keep
                init_state.id[:] = new_ids.float().unsqueeze(-1)

                for track in init_state[0]:
                    self.health_monitors[track.id.item()] = TrackHealthMonitor(
                        self.current_step,
                        init_hidden=track.appearance,   #改用检测头数据
                    )

                return init_state, match_scores

        return None, None
    
    def revive_lost_tracks(self, detections):
        """
        尝试用当前帧的未匹配 detections 复活 lost tracks。
        改进采用匈牙利算法算出全局最高匹配
        
        Args:
            detections: TrackTensorState (包含当前帧所有的检测结果，未经过滤)
        
        Returns:
            used_det_indices: 列表，包含所有被复活消耗掉的 detection 索引
        """
        # 1. 基础检查
        if not self.lost_tracks:
            return []
        
        # 检查 detections 里有没有 hidden (虽然我们在第一步里已经确保有了)
        if "appearance" not in detections.keys(): # 注意：tensordict用keys()判断
            return []

        # 2. 准备数据 (转到 CPU 进行计算),使用字典方式使用
        # detections.hidden 形状: (1, Num_Dets, 512)


        det_feats = detections["appearance"][0].cpu()   #记录外貌特征
        det_pos = detections["trans"][0].cpu()  #记录detection对应位置
        num_dets = det_feats.shape[0]
        num_lost = len(self.lost_tracks)
        if num_dets == 0:
            return []
        
        # 3. 构建算法代价矩阵
        # i 从 len-1 递减到 0

        cost_matrix = torch.ones((num_lost, num_dets)) * 100.0

        for i in range(num_lost):
            lost_track_data, monitor = self.lost_tracks[i]
            last_pos = lost_track_data.trans[0].cpu()   #记录Lost Track的距离
            
            #利用倒数两帧来求取矢量速度
            pos_history  = monitor.attributes["root pos"].log
            velocity_vector = torch.zeros_like(last_pos)

            if(len(pos_history) > 2):
                p_last = pos_history[-1].cpu()
                p_prev = pos_history[-2].cpu()
                velocity_vector = p_last - p_prev
            
            #线性运动预测位置
            predicted_pos = last_pos + velocity_vector * monitor.lost_age

            for j in range(num_dets):
                
                
                # [DEVA 核心] 
                # 让 Monitor 去看一眼这个 detection 的特征，算出相似度
                app_score = monitor.compute_max_similarity(det_feats[j])  #外貌特征分数
                # 计算每一个的位置距离

                #使用预测后位置判断
                dist = torch.norm(predicted_pos - det_pos[j]).item()

                #加入门控机制，同时依据消失时间放大控制范围（好坏未决）
                predicted_dist = 2.0 + (0.15 * monitor.lost_age)
                if dist > predicted_dist or app_score < 0.2:
                    continue

                pos_score = max(0, 1 - dist / predicted_dist)  #距离特征分数

                
                #动态权重，时间短位置长外貌
                if monitor.lost_age < 5:
                     final_score = 0.4 * app_score + 0.6 * pos_score
                else:
                     final_score = 0.8 * app_score + 0.2 * pos_score

                #填入代价矩阵中
                cost_matrix[i, j] = 1.0 - final_score


       

        #！！！匈牙利全局匹配
        row_indices, col_indices = linear_sum_assignment(cost_matrix.numpy())

        used_det_indices = [] # 记录哪些 detection 被用掉了
        indices_to_remove = []
        # 5. 执行复活操作   
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= 0.7:
                continue

            lost_track_data, monitor = self.lost_tracks[r]
            best_det_idx = c
            score = 1.0 - cost_matrix[r, c]
            print(f" >>> REVIVED ID {lost_track_data.id.item()}! Score: {score:.3f}")

        
            #复活逻辑构建
            revived_data_dict = {}
            for k, v in detections.items():
                # 保持 (1, 1, ...) 的维度
                if k in self.ref_dims:
                    if not isinstance(v, torch.Tensor): continue
                    revived_data_dict[k] = v[:, best_det_idx:best_det_idx+1]

            if 'betas' not in revived_data_dict:
                continue # 缺少核心数据，跳过
            
            ref_tensor = revived_data_dict['betas'] # Shape: (B, N, D)
            current_batch_size = ref_tensor.shape[:2] # (B, N)
                
            # 如果切片出来是空的，直接跳过
            if current_batch_size[1] == 0:
                continue
            
            if "id" not in revived_data_dict:
                # 创建一个临时的 id tensor
                # 形状: (Batch, 1, 1) -> (1, 1, 1)
                # device 要和其他 tensor 保持一致
                ref_tensor = revived_data_dict['betas'] # 找一个参考tensor
                temp_id = torch.zeros(
                    (*current_batch_size, 1), 
                    device=ref_tensor.device
                )
                revived_data_dict["id"] = temp_id  
            
            # [关键] 将字典封装回 TrackTensorState 对象
            # 这样后续才能用 .id, .trans 等属性访问
            revived_track = TrackTensorState(
                **revived_data_dict,
                batch_size=current_batch_size,
            )
            
            # 必须 clone，否则修改 ID 会影响原始引用（虽然这里是新切片，但 clone 是好习惯）
            revived_track = revived_track.clone()

            #将ID修改为之前的ID
            revived_track.id[:] = lost_track_data.id

            #加入到current_tracks中
            if self.current_tracks is None:
                self.current_tracks = revived_track
            else:
                self.current_tracks = torch.cat([self.current_tracks, revived_track], 1)

            #将丢失时间复原
            monitor.lost_age = 0
            monitor.revive_age = 0

            self.health_monitors[lost_track_data.id.item()] = monitor

            #更新monitor的信息
            new_pos = revived_track.trans[0].cpu()
            new_hidden = revived_track.appearance[0].squeeze(0).cpu()
            monitor.update(new_pos, 1.0, 0.0, 1.0, current_hidden=new_hidden)
            #将其标记删除，而不是直接POP
            indices_to_remove.append(r)
            used_det_indices.append(best_det_idx)

        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            self.lost_tracks.pop(idx)

                
        return used_det_indices
            


    def initialize_missing_tracks(self, detections, update_fn):
        """Check for missing detections that should be initialized.
            改进：使用未匹配到的帧来复活Lost tracks
        """
        # Check overlap of tracks with input detections
        new_state, conf = self.get_state_from_detections(detections)
        num_tracks = self.current_tracks.pred_2d.shape[1]
        num_detections = new_state.pred_2d.shape[1]

        if num_detections > 0:
            match_scores = self.compare_detections(
                self.current_tracks.pred_2d[0], new_state.pred_2d[0]
            )
            track_max_match_scores = match_scores.max(1)[0]
            detection_max_match_score = match_scores.max(0)[0]
            unmatched = (
                (detection_max_match_score < self.missing_match_thr).nonzero().flatten()
            )

            if len(unmatched) > 0:
                missing = new_state[:, unmatched]
                missing_conf = conf[:, unmatched]
                new_tracks, new_match_scores = self.initialize_tracks(
                    None, update_fn, missing, missing_conf
                )
                if new_tracks is not None and len(new_match_scores) > 0:
                    self.current_tracks = torch.cat(
                        [self.current_tracks, new_tracks], 1
                    )
                    track_max_match_scores = torch.cat(
                        [track_max_match_scores, new_match_scores], -1
                    )

            return track_max_match_scores

        else:
            return torch.zeros(num_tracks, device=self.current_tracks.pred_2d.device)

    def update_track_health(self, match_scores):
        """Report various properties of current tracks."""
        root_pos = self.current_tracks[0].trans.cpu()
        pred_2d = self.current_tracks[0].pred_2d.cpu()

        #获取hidden特征并转给cpu
        current_feats = self.current_tracks[0].appearance.cpu()

        # Check overlap with other tracks
        self_similarity = self.compare_detections(pred_2d, pred_2d)
        self_similarity.diagonal().fill_(0)
        track_overlap = self_similarity.max(1)[0]

        # Calculate percentage of keypoints that are inbounds
        inbounds = helper.check_inbounds(pred_2d, self.res)
        inbounds_pct = inbounds.float().mean(-1)

        #将hidden特征加入传递参数列表中

        ids = self.current_tracks[0].id
        update_args = [root_pos, match_scores, track_overlap, inbounds_pct, current_feats]
        for id, *args in zip(ids, *update_args):
            id = id.item()
            self.health_monitors[id].update(*args)

        # Look for duplicate/collapsed tracks
        num_tracks = len(root_pos)
        triu = torch.triu_indices(num_tracks, num_tracks, offset=1)
        collapse_candidates = triu.T[
            self_similarity[triu[0], triu[1]] > self.overlap_thr
        ]
        for c0, c1 in collapse_candidates:
            i0, i1 = ids[c0].item(), ids[c1].item()
            if (i0, i1) not in self.collapse_count:
                self.collapse_count[(i0, i1)] = []
            self.collapse_count[(i0, i1)] += [self.current_step]
            a0 = self.health_monitors[i0].age
            a1 = self.health_monitors[i1].age
            m0 = self.health_monitors[i0].get("match", use_ema=True)
            m1 = self.health_monitors[i1].get("match", use_ema=True)

            # Clear spurious early detection
            if a0 < 5 and a1 < 5:
                # Similar age, look at match score
                if m1 > m0:
                    self.health_monitors[i0].is_redundant = True
                else:
                    self.health_monitors[i1].is_redundant = True
            elif a0 < 3:
                self.health_monitors[i0].is_redundant = True
            elif a1 < 3:
                self.health_monitors[i0].is_redundant = True
            else:
                # Count how many overlap samples we have in the last second
                recent_overlap_count = torch.tensor(self.collapse_count[(i0, i1)])
                recent_overlap_count = (recent_overlap_count - self.current_step).abs()
                recent_overlap_count = (recent_overlap_count < 30).sum()

                v0 = torch.tensor(
                    self.health_monitors[i0].attributes["root vel"].log[-30:]
                ).max()
                v1 = torch.tensor(
                    self.health_monitors[i1].attributes["root vel"].log[-30:]
                ).max()

                if recent_overlap_count > self.overlap_time_thr:
                    if v0 > self.vel_outlier_thr and v1 < self.vel_outlier_thr:
                        self.health_monitors[i0].is_redundant = True
                    elif v0 < self.vel_outlier_thr and v1 > self.vel_outlier_thr:
                        self.health_monitors[i1].is_redundant = True
                    elif m1 > m0:
                        self.health_monitors[i0].is_redundant = True
                    else:
                        self.health_monitors[i1].is_redundant = True

    def clear_invalid_tracks(self):
        """Delete any tracks whose health score is below threshold.
            改进为不直接丢弃，而是先存入失踪列表
        """
        # track_health = []
        # for track in self.current_tracks[0]:
        #     track_health.append(
        #         self.health_monitors[track.id.item()].get_current_health()
        #     )
        # track_health = torch.tensor(track_health)
        # self.current_tracks = self.current_tracks[:, track_health > 0]
        keep_mask = []

        for i in range(self.current_tracks.shape[1]):   #遍历该帧画面中的每个人,决定是否存入到lost_tracks中
            track_id = self.current_tracks.id[0, i].item()  #得到该人的ID
            health = self.health_monitors[track_id].get_current_health()    #得到健康值（0or1)

            #依据健康值决定是否keep
            if health > 0:
                keep_mask.append(True)
            else:
                keep_mask.append(False)
                monitor = self.health_monitors[track_id]
                if monitor.age > 5: #存在时间超过5帧
                    already_in_lost = False
                    for existing_track, _ in self.lost_tracks:
                        if existing_track.id.item() == track_id:
                            already_in_lost = True
                            break
                    
                    if not already_in_lost:
                        lost_track_data = self.current_tracks[:,i:i+1].clone()  #直接硬拷贝，避免current_tracks释放后消失
                        monitor.lost_age = 0
                        self.lost_tracks.append((lost_track_data, monitor)) #将数据和监视器对象一起存入
                    else:
                        pass
                else:
                    del self.health_monitors[track_id]
        
        #依据keep_mask执行清理，将不健康的轨迹去除
        if len(keep_mask) > 0:
            keep_mask = torch.tensor(keep_mask, device=self.current_tracks.device)
            self.current_tracks = self.current_tracks[:,keep_mask]
        else:
            self.current_tracks = None

        #维护Lost Tracks列表,将超过最大时间的弹出
        active_lost_tracks = []
        for track_data, monitor in self.lost_tracks:
            monitor.lost_age += 1 #将Lost的时间加1
            if monitor.lost_age <= self.max_lost_patience:
                active_lost_tracks.append((track_data,monitor))
            else:
                print(f"ID{track_data.id.item()} is dead")
                pass

        self.lost_tracks = active_lost_tracks   #保留未超时轨迹

        

    def update(self, curr_detections, update_fn, shot_reset=False):
        """Run tracking update step."""
        self.cached_detections.append(curr_detections["pred_2d"])
        self.current_step += 1
        match_scores = None
        
        if shot_reset:
            self.clear_tracks()
            self.lost_tracks = [] # [新增] 换镜头了，清空长期记忆

        # --- Phase 1: 更新活着的人 (Active Tracks) ---
        if self.current_tracks is not None:
            # 1.1 让 GRU/Kalman 更新当前轨迹的位置
            update_fn(self.current_tracks)
            
            # 1.2 [关键] 尝试匹配：Active Tracks <-> Detections
            # 1. 计算 Active Tracks 和 Detections 的相似度矩阵
            curr_state, _ = self.get_state_from_detections(curr_detections)
            # 形状: (Num_Active, Num_Dets)
            sim_matrix = self.compare_detections(self.current_tracks.pred_2d[0], curr_state.pred_2d[0])
            
            # 2. 找出哪些 Detections 已经被 Active Tracks 认领了
            # 规则: 只要和任意一个 Active Track 的相似度 > 0.5，就算被认领
            max_sim_per_det = sim_matrix.max(dim=0)[0] # (Num_Dets,)
            unmatched_det_mask = max_sim_per_det < self.missing_match_thr # e.g. < 0.2
            
            # 3. 提取出“无主”的 Detections
            # 注意：这里需要构造一个新的字典只包含无主数据
            unmatched_indices = torch.nonzero(unmatched_det_mask).flatten()
            
            if len(unmatched_indices) > 0:
                # 构造无主数据子集
                candidate_detections = {}
                for k, v in curr_detections.items():
                    candidate_detections[k] = v[:, unmatched_indices]
                
                # 4. [新增] 尝试复活！
                # revive_lost_tracks 会自动处理 ID 并在内部加回 current_tracks
                revived_indices_local = self.revive_lost_tracks(candidate_detections)
                
                # 5. 剩下的才是真正的新人
                # 我们需要标记哪些 detection 既没被 active 匹配，也没被复活
                # 这有点绕，为了简单，其实可以直接让 initialize_missing_tracks 去处理
                # 因为 initialize_missing_tracks 内部会再算一遍匹配。
                # 只要我们把复活的 track 加回了 current_tracks，
                # initialize_missing_tracks 就会发现："咦，这个 detection 现在有主人了（刚复活的）"，于是就不会新建 ID。
                
                pass # 这一步其实不用显式做，上面的 revive 已经修改了 self.current_tracks
                
            # 6. 调用原有的初始化逻辑
            # 此时 self.current_tracks 可能已经包含了刚复活的人
            # initialize_missing_tracks 会重新计算矩阵，这次刚复活的人会认领掉对应的 detection
            match_scores = self.initialize_missing_tracks(curr_detections, update_fn)

        else:
            # --- Phase 2: 如果当前没人 (全空) ---
            # 1. 尝试直接复活
            # 以前是直接 initialize_tracks 全变新人，现在先看看能不能找回老人
            
            # revive_lost_tracks 会把复活的人加到 self.current_tracks
            self.revive_lost_tracks(curr_detections)
            
            if self.current_tracks is not None:
                # 如果复活成功了部分人，剩下的人再走初始化流程
                # 这里有点 tricky，因为 initialize_tracks 是用来初始化 "current_tracks 为空" 的情况的
                # 如果我们复活了人，current_tracks 就不为空了，应该走 initialize_missing_tracks
                
                # 更新一下状态 (因为 revive 只是加了数据，没跑 update_fn)
                update_fn(self.current_tracks)
                match_scores = self.initialize_missing_tracks(curr_detections, update_fn)
            else:
                # 一个都没复活，全员新人
                self.current_tracks, match_scores = self.initialize_tracks(
                    curr_detections, update_fn
                )

        # --- Phase 3: 收尾工作 ---
        if self.current_tracks is not None:
            if self.current_tracks.pred_2d.shape[1] == 0:
                self.current_tracks = None
            else:
                # 更新健康状态 (包含 EMA 记忆更新)
                self.update_track_health(match_scores.cpu())
                
                # 清理不健康轨迹 (存入停尸房)
                self.clear_invalid_tracks()
                
                if self.current_tracks is not None and self.current_tracks.pred_2d.shape[1] == 0:
                    self.current_tracks = None

        return self.convert_to_outputs(device=curr_detections["pred_2d"].device)


def rearrange_preds(preds, track_ids):
    num_timesteps = preds.shape[0]
    dim_ref = preds.shape[2:]

    unique_ids = track_ids.unique()
    num_tracks = len(unique_ids)
    id_lookup = {
        id_idx.item(): id.item()
        for id_idx, id in zip(torch.arange(num_tracks), unique_ids)
    }

    full_pred = torch.zeros(
        num_timesteps, num_tracks, *dim_ref, dtype=preds.dtype, device=preds.device
    )

    for id_idx, id in id_lookup.items():
        if id != 0:
            src_i0, src_i1 = (track_ids == id).nonzero(as_tuple=True)
            dst_i1 = torch.ones_like(src_i1).fill_(id_idx)
            full_pred[src_i0, dst_i1] = preds[src_i0, src_i1]

    return full_pred, id_lookup


def padded_reshaped(arr, frame_idxs, ids, num_frames):
    unique_ids, remapped_ids = ids.unique(return_inverse=True)
    max_tracks = len(unique_ids)
    dim_ref = [d for d in arr.shape]
    dim_ref = [num_frames, max_tracks] + dim_ref[1:]
    dst_arr = torch.zeros(*dim_ref, device=arr.device, dtype=arr.dtype)
    dst_arr[frame_idxs, remapped_ids] = arr

    return dst_arr


def query_range(tracks, i0, i1):
    to_use = (tracks["frame_idx"] >= i0) & (tracks["frame_idx"] < i1)
    filtered = {k: v[to_use] for k, v in tracks.items()}
    reshaped = {
        k: padded_reshaped(v, filtered["frame_idx"] - i0, filtered["id"], i1 - i0)
        for k, v in filtered.items()
    }
    return reshaped


def calculate_cleanup_ranges(ids, sub_window_size=100, overlap=20, unique_id_thr=100):
    """Breakdown very long videos into subsections.

    For videos with thousands of frames and lots of people it gets unwieldy to
    perform track clean up operation on everything simultaneously, so we break
    the video down into subclips.
    """
    num_frames = len(ids)

    rngs = []
    r0 = 0
    for i in range(0, num_frames, sub_window_size):
        r1 = min(i + sub_window_size + overlap, num_frames)
        unique_count = ids[r0:r1].unique().numel()
        if unique_count > unique_id_thr:
            rngs += [(r0, r1)]
            r0 = r1 - overlap

    if r0 != r1:
        rngs += [(r0, r1)]

    return rngs


def combine_refs(all_refs, rngs):
    combined = {}
    for track_ref, rng in zip(all_refs, rngs):
        offset = rng[0]
        for track_id, (i0, i1) in track_ref.items():
            i0, i1 = i0 + offset, i1 + offset
            if track_id not in combined:
                combined[track_id] = [i0, i1]
            else:
                i0_, i1_ = combined[track_id]
                combined[track_id] = [min(i0, i0_), max(i1, i1_)]
    return combined


def convert_to_idxs(track_ref, ids):
    all_idxs = []
    for curr_id, (i0, i1) in track_ref.items():
        sample_idxs = (ids[i0:i1] == curr_id).nonzero()
        sample_idxs[:, 0] += i0
        all_idxs.append(sample_idxs)

    all_idxs = torch.cat(all_idxs, 0)
    frame_idxs, track_idxs = all_idxs.unbind(1)

    return frame_idxs, track_idxs


def cleanup_tracks(
    preds,
    K,
    smpl_decoder,
    min_matched_frames=4,
    min_match_score=0.6,
    detect_conf_thr=0.2,
    return_ious=False,
    rng=None,
    use_tqdm=False,
    unique_id_thr=100,
):
    if rng is None:
        ids = preds["tracks"]["id"][0]
        rngs = calculate_cleanup_ranges(ids, unique_id_thr=unique_id_thr)
        rngs_ = tqdm(rngs) if use_tqdm else rngs
        all_refs = [
            cleanup_tracks(
                preds,
                K,
                smpl_decoder,
                min_matched_frames,
                min_match_score,
                detect_conf_thr,
                return_ious,
                rng=rng,
            )
            for rng in rngs_
        ]

        return combine_refs(all_refs, rngs)

    else:
        i0, i1 = rng
        detect_2d = torch.nn.utils.rnn.pad_sequence(
            [p[0] for p in preds["detections"]["pred_2d"][i0:i1]], batch_first=True
        )
        detect_conf = torch.nn.utils.rnn.pad_sequence(
            [p[0] for p in preds["detections"]["conf"][i0:i1]], batch_first=True
        )
        ids, betas, pose, trans = [
            preds["tracks"][k][0][i0:i1] for k in ["id", "betas", "pose", "trans"]
        ]

    # Zero out low confidence detections
    detect_conf = torch.sigmoid(detect_conf).squeeze(-1)
    detect_2d[detect_conf < detect_conf_thr] = 0

    pred_3d = smpl_decoder(betas, pose, trans, output_format="joints_face")
    track_2d = helper.project_to_2d(K, pred_3d)
    ids = ids.squeeze(-1).long()

    track_2d, id_lookup = rearrange_preds(track_2d, ids)
    smpl_params = {
        "betas": rearrange_preds(betas, ids)[0],
        "pose": rearrange_preds(pose, ids)[0],
        "trans": rearrange_preds(trans, ids)[0],
    }

    valid_tracks = (smpl_params["betas"] != 0).any(-1)
    track_2d[~valid_tracks] = 0

    p0 = track_2d.clone() / 1024
    c0 = (p0 != 0).any(-1).float().clone()
    p1 = detect_2d.clone() / 1024
    c1 = (p1 != 0).any(-1).float().clone()

    # Compare tracks with detections
    ious = helper.normalized_weighted_score(p0, c0, p1, c1)

    if return_ious:
        return ious

    track_ref = {}

    iou_sums = ious.max(-1)[0].sum(0).int()
    for i in range(ious.shape[1]):
        iou_sums = ious.max(-1)[0].sum(0).int()
        best_matched = iou_sums.argmax().item()
        max_iou, max_idxs = ious[:, best_matched].max(-1)
        matched_idxs = (max_iou > min_match_score).nonzero()

        if len(matched_idxs) < min_matched_frames:
            break

        i0, i1 = matched_idxs.min().item(), matched_idxs.max().item() + 1
        track_ref[id_lookup[best_matched]] = [i0, i1]

        idx_range = torch.arange(i0, i1)
        ious[idx_range, :, max_idxs[i0:i1]] = 0
        ious[idx_range, best_matched] = 0

    return track_ref


def bboxes_from_smpl(
    smpl_decoder,
    smpl_params,
    res,
    K,
    pad_x=0.05,
    pad_y=0.05,
    subsample_rate=20,
):
    """Use SMPL mesh to more precisely define bounding box boundary.

    Output bounding box format is x1, y1, x2, y2.
    """
    vertices = smpl_decoder(
        **smpl_params, output_format="mesh", subsample_rate=subsample_rate
    )
    vertices_2d = helper.project_to_2d(K, vertices)
    bboxes = helper.points_to_bbox2d(vertices_2d, pad_dims=[pad_x, pad_y])

    # Clamp bounding box to image boundaries
    image_height, image_width = res
    bboxes[..., 0, :].clamp_min_(0.01)
    bboxes[..., 1, 0].clamp_max_(image_width - 1)
    bboxes[..., 1, 1].clamp_max_(image_height - 1)

    return bboxes


def convert_to_mot(
    track_ids,
    frame_idxs,
    bboxes,
    valid_frames=None,
    frame_idx_ref=None,
    include_dict=False,
):
    mot_txt = ""
    mot_dict = {}

    if isinstance(track_ids, torch.Tensor):
        track_ids = track_ids.long().numpy()
    if isinstance(frame_idxs, torch.Tensor):
        frame_idxs = frame_idxs.long().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.float().numpy()

    for track_id, frame_idx, bbox in zip(track_ids, frame_idxs, bboxes):
        img_frame_idx = frame_idx
        if frame_idx_ref is not None:
            img_frame_idx = frame_idx_ref[frame_idx]

        if valid_frames is None or img_frame_idx in valid_frames:
            bbox = list(bbox.flatten())

            if include_dict:
                if track_id not in mot_dict:
                    mot_dict[track_id] = {}
                mot_dict[track_id][img_frame_idx] = bbox

            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
            str_bbox = [d for d in bbox]
            str_bbox[2] = str_bbox[2] - str_bbox[0]
            str_bbox[3] = str_bbox[3] - str_bbox[1]
            str_bbox = ",".join([f"{d + 1:.2f}" for d in str_bbox])
            mot_txt += f"{img_frame_idx + 1},{track_id},{str_bbox},1,0,0,0\n"

    if include_dict:
        return mot_txt, mot_dict
    else:
        return mot_txt
