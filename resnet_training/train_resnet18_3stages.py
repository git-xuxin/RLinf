#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ResNet-18 多分类训练脚本 (Stage Classifier)

用于训练一个基于ResNet-18的 3分类器 (stage0, stage1, stage2)。
训练结束后，会自动加载最佳模型，将验证集的预测结果输出到 eval 文件夹。

使用方法:
    python resnet_training/train_resnet18_3stages.py --data_dir ~/Workspace/Beijing/data/pnp/3stage

数据集结构:
    data_dir/
    ├── stage0/ ...
    ├── stage1/ ...
    └── stage2/ ...

输出结构:
    data_dir/
    ├── eval/  <-- 新增预测结果文件夹
    │   ├── pred_stage0/
    │   ├── pred_stage1/
    │   └── pred_stage2/
"""

import argparse
import os
import random
import shutil  # 新增引用
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# 默认保存目录
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR

# 定义类别映射
CLASS_NAMES = ["stage0", "stage1", "stage2"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StageDataset(Dataset):
    """
    Stage 分类图像数据集
    """

    def __init__(self, images: list, labels: list, transform=None):
        self.images = images  # 存储完整路径
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 加载图片
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (128, 128))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def load_and_split_data(
    data_dir: str,
    train_split: float = 0.9,
    seed: int = 42,
):
    """加载 stage0/1/2 数据并进行分层分割"""
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    data_path = Path(data_dir)
    
    all_data = {cls_name: [] for cls_name in CLASS_NAMES}
    
    print(f"正在扫描数据目录: {data_path}")

    # 加载每个类别的数据
    for cls_name in CLASS_NAMES:
        cls_dir = data_path / cls_name
        if cls_dir.exists():
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    all_data[cls_name].append(str(img_path))
        else:
            print(f"警告: 目录 {cls_dir} 不存在!")

    # 打印统计
    total_images = sum(len(imgs) for imgs in all_data.values())
    print(f"原始数据总计: {total_images} 张")
    for cls_name in CLASS_NAMES:
        print(f"  - {cls_name}: {len(all_data[cls_name])} 张")

    rng = random.Random(seed)
    
    train_images = []
    train_labels = []
    eval_images = []
    eval_labels = []

    # 对每个类别分别进行切分 (Stratified Split)
    for cls_name in CLASS_NAMES:
        images = all_data[cls_name]
        label_idx = CLASS_TO_IDX[cls_name]
        
        # 打乱
        rng.shuffle(images)
        
        # 切分
        n_train = int(len(images) * train_split)
        
        cls_train_imgs = images[:n_train]
        cls_eval_imgs = images[n_train:]
        
        # 添加到总列表
        train_images.extend(cls_train_imgs)
        train_labels.extend([label_idx] * len(cls_train_imgs))
        
        eval_images.extend(cls_eval_imgs)
        eval_labels.extend([label_idx] * len(cls_eval_imgs))

    print(f"\n数据集划分完成 (Train: {len(train_images)}, Eval: {len(eval_images)})")
    
    # 再次整体打乱训练集
    combined_train = list(zip(train_images, train_labels))
    rng.shuffle(combined_train)
    if combined_train:
        train_images, train_labels = zip(*combined_train)
        train_images, train_labels = list(train_images), list(train_labels)

    return train_images, train_labels, eval_images, eval_labels


def get_transforms(image_size: int = 128, is_train: bool = True):
    """获取数据变换"""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class ResNetClassifier(nn.Module):
    """基于ResNet的多分类模型"""

    SUPPORTED_ARCHS = ["resnet18", "resnet34", "resnet50"]

    def __init__(
        self,
        num_classes: int = 3,
        arch: str = "resnet18",
        hidden_dim: int = None,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"不支持的架构: {arch}. 支持: {self.SUPPORTED_ARCHS}")

        self.arch = arch
        self.num_classes = num_classes

        # 加载预训练ResNet
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = getattr(models, arch)(weights=weights)

        num_features = self.backbone.fc.in_features

        if hidden_dim is not None and hidden_dim > 0:
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, labels=None):
        logits = self.backbone(x)
        probabilities = F.softmax(logits, dim=1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().float()
            accuracy = correct / labels.size(0)
            
            result["loss"] = loss
            result["accuracy"] = accuracy

        return result


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        predictions = torch.argmax(outputs["logits"], dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, labels)
            loss = outputs["loss"]

            total_loss += loss.item()
            predictions = torch.argmax(outputs["logits"], dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def save_evaluation_results(model, dataset, output_root, device):
    """
    对验证集进行预测并将结果保存到文件系统中
    """
    print(f"\n[Post-Processing] 正在生成验证集预测结果 -> {output_root}")
    
    # 确保输出目录存在
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 创建每个预测类别的子文件夹
    pred_folder_names = [f"pred_{name}" for name in CLASS_NAMES]
    for name in pred_folder_names:
        (output_root / name).mkdir(parents=True, exist_ok=True)

    model.eval()
    
    # 使用 batch_size=1, shuffle=False 确保索引与 dataset.images 严格对应
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, desc="Saving Results")):
            images = images.to(device)
            
            # 模型推理
            outputs = model(images)
            logits = outputs["logits"]  # Shape: (1, 3)
            
            # 规则：取最大概率。
            # 如果概率相同(例如 [0.3, 0.3, 0.4] -> index 2; [0.5, 0.5, 0.0] -> index 0)
            # torch.argmax 默认返回最大值的第一个索引。
            # 由于我们的索引是 [0, 1, 2] 对应 stage0, stage1, stage2
            # 返回第一个索引即意味着返回了更小的 stage，符合需求。
            pred_idx = torch.argmax(logits, dim=1).item()
            
            # 获取预测的类别名
            pred_folder = pred_folder_names[pred_idx]
            
            # 获取原始文件路径
            original_path = Path(dataset.images[i])
            
            # 获取真实的类别名 (Ground Truth)，用于文件名标记
            # 假设原始路径类似 .../stage0/abc.png，父目录名即为 GT
            gt_class_name = original_path.parent.name 
            
            # 构造新文件名: GT_{真值}_NAME_{原名}
            # 这样做可以避免不同文件夹下同名文件的冲突，并方便人工检查错误
            new_filename = f"GT_{gt_class_name}_{original_path.name}"
            
            # 目标路径
            dest_path = output_root / pred_folder / new_filename
            
            # 复制文件
            try:
                shutil.copy2(original_path, dest_path)
            except Exception as e:
                print(f"复制文件失败: {original_path} -> {e}")

    print(f"验证集预测图片已保存至: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="训练ResNet Stage Classifier (stage0/1/2)"
    )

    # 数据参数
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据目录路径，应包含 stage0, stage1, stage2 子目录",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出模型路径 (默认: ./stage_classifier.pt)",
    )
    parser.add_argument("--image_size", type=int, default=128, help="输入图像尺寸")

    # 模型参数
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="ResNet架构",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="MLP头隐藏层维度 (设为0表示直接线性层)",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比率")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument(
        "--train_split", 
        type=float, 
        default=0.9, 
        help="训练集比例"
    )

    # 早停参数
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="早停耐心值",
    )

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="使用ImageNet预训练权重"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="数据加载器worker数量"
    )

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = str(DEFAULT_OUTPUT_DIR / f"stage_classifier_{timestamp}.pt")

    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else None

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据变换
    train_transform = get_transforms(args.image_size, is_train=True)
    val_transform = get_transforms(args.image_size, is_train=False)

    # 加载数据
    train_images, train_labels, eval_images, eval_labels = load_and_split_data(
        data_dir=args.data_dir,
        train_split=args.train_split,
        seed=args.seed,
    )

    if len(train_images) == 0:
        print("错误: 训练数据集为空，请检查数据目录")
        return

    # 创建数据集
    train_dataset = StageDataset(train_images, train_labels, transform=train_transform)
    val_dataset = StageDataset(eval_images, eval_labels, transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 创建模型 (3分类)
    model = ResNetClassifier(
        num_classes=len(CLASS_NAMES),
        arch=args.arch,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
    )
    model = model.to(device)
    print(f"模型架构: {args.arch} (Classes: {len(CLASS_NAMES)})")

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"* 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            if args.patience > 0:
                print(f"  (无改进: {epochs_without_improvement}/{args.patience})")

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n早停触发")
            break

    # 保存模型
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "arch": args.arch,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "hidden_dim": hidden_dim,
        "image_size": [3, args.image_size, args.image_size],
    }

    checkpoint = {
        "state_dict": best_model_state,
        "config": config,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    torch.save(checkpoint, args.output_path)
    print(f"\n模型已保存到: {args.output_path}")

    # ==========================================
    # 预测并保存验证集结果到 eval 文件夹
    # ==========================================
    if best_model_state is not None:
        # 加载最佳权重
        model.load_state_dict(best_model_state)
        
        # 定义 eval 文件夹位置: data_dir/eval
        eval_output_dir = Path(args.data_dir) / "eval"
        
        # 执行保存
        save_evaluation_results(model, val_dataset, eval_output_dir, device)

    # 简单的测试演示
    print("\n映射关系:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()