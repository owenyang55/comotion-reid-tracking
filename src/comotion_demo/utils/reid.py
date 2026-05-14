# comotion/reid_module.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as T

class ReIDExtractor(nn.Module):
    def __init__(self, device='cuda', checkpoint_path=None):
        super().__init__()
        self.device = device
        print(" [ReID] Loading DINOv2 (ViT-Small) from PyTorch Hub...")
        
        # 1. 加载模型
        # 无需安装任何库，直接利用 torch.hub 自动下载缓存
        # dinov2_vits14: 21M 参数，速度极快，特征维度 384
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # 可选：加载微调后的 checkpoint（优先函数参数，其次环境变量）
        ckpt = checkpoint_path or os.getenv("COMOTION_REID_CKPT")
        if ckpt:
            if os.path.exists(ckpt):
                payload = torch.load(ckpt, map_location="cpu")
                if isinstance(payload, dict) and "backbone" in payload:
                    state_dict = payload["backbone"]
                elif isinstance(payload, dict) and "state_dict" in payload:
                    state_dict = payload["state_dict"]
                else:
                    state_dict = payload

                incompatible = self.model.load_state_dict(state_dict, strict=False)
                print(
                    f" [ReID] Loaded checkpoint: {ckpt} | "
                    f"missing={len(incompatible.missing_keys)}, "
                    f"unexpected={len(incompatible.unexpected_keys)}"
                )
            else:
                print(f" [ReID] Checkpoint not found, fallback to pretrained only: {ckpt}")

        self.model.to(device)
        self.model.eval()

        # 2. 预处理
        # DINOv2 使用 patch_size=14，输入尺寸最好是 14 的倍数
        # 252 = 14 * 18, 126 = 14 * 9 (接近标准的 256x128)
        self.preprocess = T.Compose([
            T.Resize((252, 126)), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, crops):
        """
        输入: List[Tensor] (N个裁剪图)
        输出: (N, 384) 归一化特征
        """
        if len(crops) == 0:
            return torch.empty((0, 384), device=self.device)

        batch = []
        for img in crops:
            # 确保在 GPU
            if img.device != torch.device(self.device):
                img = img.to(self.device)
            # 确保是 float 且归一化 [0, 1]
            if img.dtype == torch.uint8 or img.max() > 1.0:
                 img = img.float() / 255.0
            
            img = self.preprocess(img)
            batch.append(img)
        
        # 堆叠成 batch (N, 3, 252, 126)
        batch = torch.stack(batch)

        with torch.no_grad():
            # DINOv2 forward 直接输出 [CLS] token 特征
            features = self.model(batch) 
            # ReID 必须做 L2 归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
        return features

    @staticmethod
    def crop_persons(image_tensor, bboxes):
        """
        image_tensor: (3, H, W) 原图
        bboxes: (N, 4) [x1, y1, x2, y2]
        """
        crops = []
        _, H, W = image_tensor.shape
        for box in bboxes:
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x2 > x1 and y2 > y1:
                crops.append(image_tensor[:, y1:y2, x1:x2])
            else:
                crops.append(torch.zeros((3, 252, 126), device=image_tensor.device))
        return crops