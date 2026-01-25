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
ResNet-18 二分类训练脚本 (兼容 RLinf ResNetRewardModel 格式)

用于训练一个基于ResNet-18的success/fail图像分类器。
输出格式与 rlinf/models/embodiment/reward/resnet_reward_model.py 兼容。

使用方法:
    python resnet_training/train_resnet18_classifier.py --data_dir ~/Workspace/Beijing/data/pnp/classified
    python train_resnet18_classifier.py --data_dir /path/to/data

    # 指定输出路径
    python train_resnet18_classifier.py --data_dir /path/to/data --output_path ./my_model.pt

    # 自定义数据比例 (fail:success = 2:1) 和训练集比例 (8:2)
    python train_resnet18_classifier.py --data_dir /path/to/data --ratio 2.0 --train_split 0.8

数据集结构:
    data_dir/
    ├── success/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── fail/
        ├── img1.png
        ├── img2.png
        └── ...

数据平衡:
    - --ratio: 控制 fail:success 的比例，默认 4.0 (即 fail 数量是 success 的 4 倍)
              设为 0 表示使用全部数据不做采样
    - --train_split: 训练集比例，默认 0.9 (即 train:eval = 9:1)
    - 使用分层抽样 (Stratified Split)，确保 train 和 eval 中类别分布一致

输出格式:
    checkpoint 包含:
    - state_dict: 完整模型权重 (兼容 ResNetRewardModel.load_checkpoint)
    - config: 训练配置
    - arch: 架构名称 (resnet18)
    - best_val_acc: 最佳验证准确率
"""

import argparse
import os
import random
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


def set_seed(seed: int):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SuccessFailDataset(Dataset):
    """
    Success/Fail 图像数据集

    Args:
        images: 图片路径列表
        labels: 标签列表
        transform: 图像变换
    """

    def __init__(self, images: list, labels: list, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 加载图片
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def load_and_split_data(
    data_dir: str,
    ratio: float = 10.0,
    train_split: float = 0.9,
    seed: int = 42,
):
    """
    加载数据并进行分层分割 (Stratified Split)

    Args:
        data_dir: 数据目录路径，包含 success/ 和 fail/ 子目录
        ratio: fail:success 的比例，默认 4.0 表示 fail 数量是 success 的 4 倍
               设为 None 或 0 表示使用全部数据不做采样
        train_split: 训练集比例，默认 0.9 (即 train:eval = 9:1)
        seed: 随机种子

    Returns:
        train_images, train_labels, eval_images, eval_labels
    """
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    data_path = Path(data_dir)
    
    success_images = []
    fail_images = []

    # 加载 success 图片
    success_dir = data_path / "success"
    if success_dir.exists():
        for img_path in success_dir.iterdir():
            if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                success_images.append(str(img_path))

    # 加载 fail 图片
    fail_dir = data_path / "fail"
    if fail_dir.exists():
        for img_path in fail_dir.iterdir():
            if img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                fail_images.append(str(img_path))

    print(f"原始数据: Success {len(success_images)} 张, Fail {len(fail_images)} 张")

    # 设置随机种子确保可复现
    rng = random.Random(seed)
    
    # 打乱数据（用于后续采样和分割）
    rng.shuffle(success_images)
    rng.shuffle(fail_images)

    # 根据 ratio 采样数据
    if ratio is not None and ratio > 0:
        n_success = len(success_images)
        n_fail = len(fail_images)
        
        # 目标 fail 数量 = success 数量 * ratio
        target_fail = int(n_success * ratio)
        
        if target_fail < n_fail:
            # fail 过多，需要下采样
            fail_images = fail_images[:target_fail]
            print(f"按比例 {ratio}:1 下采样 Fail: {n_fail} -> {len(fail_images)}")
        elif target_fail > n_fail:
            # fail 不足，可以上采样或保持原样
            # 这里选择保持原样并警告
            actual_ratio = n_fail / n_success if n_success > 0 else 0
            print(f"警告: Fail 数据不足以达到 {ratio}:1 比例，实际比例为 {actual_ratio:.2f}:1")
        
        # 也可以反向调整：如果 success 过多
        target_success = int(len(fail_images) / ratio) if ratio > 0 else n_success
        if len(success_images) > target_success and target_success > 0:
            # success 过多，需要下采样
            original_success = len(success_images)
            success_images = success_images[:target_success]
            print(f"按比例 {ratio}:1 下采样 Success: {original_success} -> {len(success_images)}")

    # 分层分割：对 success 和 fail 分别按 train_split 比例分割
    n_train_success = int(len(success_images) * train_split)
    n_train_fail = int(len(fail_images) * train_split)
    
    train_success = success_images[:n_train_success]
    eval_success = success_images[n_train_success:]
    
    train_fail = fail_images[:n_train_fail]
    eval_fail = fail_images[n_train_fail:]

    # 合并训练数据
    train_images = train_success + train_fail
    train_labels = [1.0] * len(train_success) + [0.0] * len(train_fail)
    
    # 合并验证数据
    eval_images = eval_success + eval_fail
    eval_labels = [1.0] * len(eval_success) + [0.0] * len(eval_fail)

    # 打印统计信息
    final_success = len(train_success) + len(eval_success)
    final_fail = len(train_fail) + len(eval_fail)
    final_ratio = final_fail / final_success if final_success > 0 else 0
    
    print(f"\n最终数据集统计:")
    print(f"  - 总计: {final_success + final_fail} 张图片")
    print(f"  - Success: {final_success} 张, Fail: {final_fail} 张")
    print(f"  - 实际比例 (fail:success): {final_ratio:.2f}:1")
    print(f"\n分层分割 (train:eval = {train_split:.0%}:{1-train_split:.0%}):")
    print(f"  - 训练集: {len(train_images)} 张 (Success: {len(train_success)}, Fail: {len(train_fail)})")
    print(f"  - 验证集: {len(eval_images)} 张 (Success: {len(eval_success)}, Fail: {len(eval_fail)})")
    
    # 验证 train 和 eval 没有重叠
    train_set = set(train_images)
    eval_set = set(eval_images)
    overlap = train_set & eval_set
    if overlap:
        print(f"\n警告: 发现 {len(overlap)} 张图片在 train 和 eval 中重复!")
        for path in list(overlap)[:5]:
            print(f"  - {path}")
    else:
        print(f"\n数据验证: train 和 eval 无重叠 ✓")

    return train_images, train_labels, eval_images, eval_labels


def get_transforms(image_size: int = 128, is_train: bool = True):
    """
    获取数据变换

    Args:
        image_size: 目标图像尺寸
        is_train: 是否为训练集（训练集使用数据增强）
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
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


class ResNetRewardModel(nn.Module):
    """
    基于ResNet的Reward模型 (兼容 RLinf ResNetRewardModel 格式)

    输出单个值，使用 sigmoid 得到 [0, 1] 概率
    使用 Binary Cross Entropy Loss 训练

    Args:
        arch: ResNet架构 (resnet18, resnet34, resnet50)
        hidden_dim: MLP头的隐藏层维度 (None表示直接线性层)
        dropout: Dropout比率
        pretrained: 是否使用ImageNet预训练权重
    """

    SUPPORTED_ARCHS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    def __init__(
        self,
        arch: str = "resnet18",
        hidden_dim: int = None,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        if arch not in self.SUPPORTED_ARCHS:
            raise ValueError(f"不支持的架构: {arch}. 支持: {self.SUPPORTED_ARCHS}")

        self.arch = arch
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # 加载预训练ResNet
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = getattr(models, arch)(weights=weights)

        # 获取原始fc层的输入特征数
        num_features = self.backbone.fc.in_features

        # 替换fc层为reward head (输出单个值)
        if hidden_dim is not None:
            # MLP头
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # 简单线性层
            self.backbone.fc = nn.Linear(num_features, 1)

        # 初始化权重
        self._init_head_weights()

    def _init_head_weights(self):
        """初始化reward head权重"""
        for module in self.backbone.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, labels=None):
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)
            labels: 可选的标签 (B,)，用于计算loss

        Returns:
            dict: 包含 logits, probabilities, loss (如果提供labels), accuracy
        """
        logits = self.backbone(x).squeeze(-1)  # (B,)
        probabilities = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            labels = labels.float().to(logits.device)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            predictions = (probabilities > 0.5).float()
            accuracy = (predictions == labels).float().mean()
            result["loss"] = loss
            result["accuracy"] = accuracy

        return result

    def predict_prob(self, x):
        """返回success概率 (用于推理)"""
        with torch.no_grad():
            logits = self.backbone(x).squeeze(-1)
            return torch.sigmoid(logits)


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
        predictions = (outputs["probabilities"] > 0.5).float()
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
            predictions = (outputs["probabilities"] > 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(
        description="训练ResNet Reward Model (兼容RLinf格式)"
    )

    # 数据参数
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据目录路径，包含success/和fail/子目录",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="输出模型路径 (默认: resnet_training/reward_model_YYYYMMDD_HHMMSS.pt)",
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
        help="训练集比例，默认 0.9 (即 train:eval = 9:1)"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=4.0,
        help="fail:success 数据比例，默认 4:1。设为 0 表示使用全部数据不做采样",
    )

    # 早停参数
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="早停耐心值，连续多少个 epoch 验证集无改进则停止训练。设为 0 禁用早停",
    )

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="使用ImageNet预训练权重"
    )
    parser.add_argument(
        "--no_pretrained", dest="pretrained", action="store_false", help="不使用预训练权重"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="数据加载器worker数量"
    )

    args = parser.parse_args()

    # 设置默认输出路径
    if args.output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = str(DEFAULT_OUTPUT_DIR / f"reward_model_{timestamp}.pt")

    # hidden_dim 为 0 表示不使用隐藏层
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else None

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建数据变换
    train_transform = get_transforms(args.image_size, is_train=True)
    val_transform = get_transforms(args.image_size, is_train=False)

    # ratio 为 0 表示不做采样
    ratio = args.ratio if args.ratio > 0 else None
    
    # 加载数据并进行分层分割
    train_images, train_labels, eval_images, eval_labels = load_and_split_data(
        data_dir=args.data_dir,
        ratio=ratio,
        train_split=args.train_split,
        seed=args.seed,
    )

    if len(train_images) == 0:
        print("错误: 训练数据集为空，请检查数据目录")
        return

    # 创建数据集
    train_dataset = SuccessFailDataset(train_images, train_labels, transform=train_transform)
    val_dataset = SuccessFailDataset(eval_images, eval_labels, transform=val_transform)

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

    # 创建模型
    model = ResNetRewardModel(
        arch=args.arch,
        hidden_dim=hidden_dim,
        dropout=args.dropout,
        pretrained=args.pretrained,
    )
    model = model.to(device)
    print(f"模型架构: {args.arch}")
    print(f"隐藏层维度: {hidden_dim}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和学习率调度器
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

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, device)

        # 更新学习率
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"* 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            if args.patience > 0:
                print(f"  (无改进: {epochs_without_improvement}/{args.patience})")

        # 早停检查
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\n早停: 连续 {args.patience} 个 epoch 验证集无改进，停止训练")
            break

    # 保存最终模型 (兼容 ResNetRewardModel.load_checkpoint 格式)
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存checkpoint (兼容格式)
    config = {
        "arch": args.arch,
        "hidden_dim": hidden_dim,
        "dropout": args.dropout,
        "image_size": [3, args.image_size, args.image_size],
        "normalize": True,
    }

    checkpoint = {
        "state_dict": best_model_state,  # ResNetRewardModel.load_checkpoint 支持此key
        "model_state_dict": best_model_state,  # 备用key
        "config": config,
        "arch": args.arch,
        "best_val_acc": best_val_acc,
        "args": vars(args),
    }
    torch.save(checkpoint, args.output_path)
    print(f"\n模型已保存到: {args.output_path}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    # 打印使用说明
    print("\n" + "=" * 50)
    print("使用方法:")
    print("=" * 50)
    print(f"在配置文件中设置 checkpoint_path:")
    print(f"  reward:")
    print(f"    model:")
    print(f"      checkpoint_path: {args.output_path}")
    print(f"      arch: {args.arch}")
    print(f"      hidden_dim: {hidden_dim if hidden_dim else 0}")
    print(f"      dropout: {args.dropout}")


if __name__ == "__main__":
    main()
