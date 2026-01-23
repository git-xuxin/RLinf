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
ResNet-18 二分类训练脚本

用于训练一个基于ResNet-18的success/fail图像分类器。

使用方法:
    python train_resnet18_classifier.py --data_dir /path/to/data --output_path ./reward_model.pt

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
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm


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
        data_dir: 数据目录路径，包含 success/ 和 fail/ 子目录
        transform: 图像变换
    """
    
    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        # 加载 success 图片 (label = 1)
        success_dir = self.data_dir / "success"
        if success_dir.exists():
            for img_path in success_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.images.append(str(img_path))
                    self.labels.append(1)
        
        # 加载 fail 图片 (label = 0)
        fail_dir = self.data_dir / "fail"
        if fail_dir.exists():
            for img_path in fail_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.images.append(str(img_path))
                    self.labels.append(0)
        
        print(f"加载数据集: {len(self.images)} 张图片")
        print(f"  - Success: {self.labels.count(1)} 张")
        print(f"  - Fail: {self.labels.count(0)} 张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size: int = 128, is_train: bool = True):
    """
    获取数据变换
    
    Args:
        image_size: 目标图像尺寸
        is_train: 是否为训练集（训练集使用数据增强）
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])


class ResNet18Classifier(nn.Module):
    """
    基于ResNet-18的二分类器
    
    Args:
        num_classes: 分类数量（默认为2：success/fail）
        pretrained: 是否使用ImageNet预训练权重
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # 加载ResNet-18
        self.backbone = resnet18(pretrained=pretrained)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict_proba(self, x):
        """返回softmax概率"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict_success_prob(self, x):
        """返回success的概率（可用作reward）"""
        probs = self.predict_proba(x)
        return probs[:, 1]  # success类别的概率


def train_epoch(model, dataloader, criterion, optimizer, device):
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description="训练ResNet-18 Success/Fail分类器")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录路径，包含success/和fail/子目录")
    parser.add_argument("--output_path", type=str, default="./reward_model.pt",
                        help="输出模型路径")
    parser.add_argument("--image_size", type=int, default=128,
                        help="输入图像尺寸")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="验证集比例")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="使用ImageNet预训练权重")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                        help="不使用预训练权重")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器worker数量")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_transform = get_transforms(args.image_size, is_train=True)
    val_transform = get_transforms(args.image_size, is_train=False)
    
    full_dataset = SuccessFailDataset(args.data_dir, transform=train_transform)
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 为验证集设置不同的transform
    val_dataset.dataset = SuccessFailDataset(args.data_dir, transform=val_transform)
    
    print(f"训练集: {train_size} 张图片")
    print(f"验证集: {val_size} 张图片")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = ResNet18Classifier(num_classes=2, pretrained=args.pretrained)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
    
    # 保存最终模型
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整的checkpoint
    checkpoint = {
        'model_state_dict': best_model_state,
        'args': vars(args),
        'best_val_acc': best_val_acc,
    }
    torch.save(checkpoint, args.output_path)
    print(f"\n模型已保存到: {args.output_path}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 同时保存纯backbone权重（兼容ResNetEncoder格式）
    backbone_path = args.output_path.replace('.pt', '_backbone.pt')
    torch.save(best_model_state, backbone_path)
    print(f"Backbone权重已保存到: {backbone_path}")


if __name__ == "__main__":
    main()
