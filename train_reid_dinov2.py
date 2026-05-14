import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReIDFolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    @staticmethod
    def from_identity_folders(root: Path, transform, min_images_per_id=2, val_ratio=0.1, seed=42):
        id_dirs = [d for d in root.iterdir() if d.is_dir()]
        id_dirs = sorted(id_dirs)

        label_map = {}
        all_by_id = {}
        label_idx = 0
        for d in id_dirs:
            imgs = [p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS]
            if len(imgs) < min_images_per_id:
                continue
            label_map[d.name] = label_idx
            all_by_id[label_idx] = sorted(imgs)
            label_idx += 1

        if len(all_by_id) < 2:
            raise RuntimeError("有效身份数不足（至少需要 2 个 ID 且每个 ID 至少 2 张图）")

        rng = random.Random(seed)
        train_samples = []
        val_samples = []
        for pid, paths in all_by_id.items():
            paths = paths.copy()
            rng.shuffle(paths)
            n_val = max(1, int(len(paths) * val_ratio))
            val_part = paths[:n_val]
            train_part = paths[n_val:]
            if len(train_part) == 0:
                train_part = val_part[:1]
                val_part = val_part[1:] if len(val_part) > 1 else val_part

            train_samples.extend([(p, pid) for p in train_part])
            val_samples.extend([(p, pid) for p in val_part])

        return (
            ReIDFolderDataset(train_samples, transform),
            ReIDFolderDataset(val_samples, transform),
            len(all_by_id),
            label_map,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(pid, dtype=torch.long)


class DinoReID(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.classifier = nn.Linear(384, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        logits = self.classifier(feat)
        return feat, logits


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    if embeddings.size(0) < 2:
        return embeddings.new_zeros([])

    dist = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.unsqueeze(1)
    same = labels.eq(labels.t())
    diff = ~same

    eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
    same = same & ~eye

    if same.sum() == 0 or diff.sum() == 0:
        return embeddings.new_zeros([])

    hardest_pos = torch.where(same, dist, dist.new_full(dist.shape, -1e6)).max(dim=1).values
    hardest_neg = torch.where(diff, dist, dist.new_full(dist.shape, 1e6)).min(dim=1).values
    loss = torch.relu(hardest_pos - hardest_neg + margin).mean()
    return loss


def run_epoch(model, loader, optimizer, ce_loss, device, triplet_weight, triplet_margin, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_tri = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            emb, logits = model(images)
            loss_ce = ce_loss(logits, labels)
            loss_tri = batch_hard_triplet_loss(emb, labels, margin=triplet_margin)
            loss = loss_ce + triplet_weight * loss_tri

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_ce += loss_ce.item() * images.size(0)
        total_tri += loss_tri.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)

    if total_count == 0:
        return {"loss": 0.0, "ce": 0.0, "tri": 0.0, "acc": 0.0}

    return {
        "loss": total_loss / total_count,
        "ce": total_ce / total_count,
        "tri": total_tri / total_count,
        "acc": total_correct / total_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for ReID (folder-based IDs)")
    parser.add_argument("--data-root", type=Path, required=True, help="目录结构: data_root/<person_id>/*.jpg")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/reid_dinov2_best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--triplet-weight", type=float, default=0.5)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = T.Compose([
        T.Resize((252, 126)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize((252, 126)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set, val_set, num_classes, label_map = ReIDFolderDataset.from_identity_folders(
        root=args.data_root,
        transform=train_tf,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    val_set.transform = val_tf

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = DinoReID(num_classes=num_classes, freeze_backbone=args.freeze_backbone).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            ce_loss,
            device,
            args.triplet_weight,
            args.triplet_margin,
            train=True,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer,
            ce_loss,
            device,
            args.triplet_weight,
            args.triplet_margin,
            train=False,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            payload = {
                "backbone": model.backbone.state_dict(),
                "classifier": model.classifier.state_dict(),
                "num_classes": num_classes,
                "label_map": label_map,
                "args": vars(args),
                "epoch": epoch,
                "val_loss": best_val,
            }
            torch.save(payload, args.output)
            print(f"[Save] best checkpoint -> {args.output}")

    print("Training done.")


if __name__ == "__main__":
    main()