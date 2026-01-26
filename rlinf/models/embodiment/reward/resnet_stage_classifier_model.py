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

"""ResNet-based multi-stage classifier model for embodied RL reward inference."""

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_image_reward_model import BaseImageRewardModel

logger = logging.getLogger(__name__)


class ResNetStageClassifierModel(BaseImageRewardModel):
    """ResNet-based classifier for multi-stage reward prediction."""

    SUPPORTED_ARCHS = ["resnet18", "resnet34", "resnet50"]

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.arch = cfg.get("arch", "resnet18")
        if self.arch not in self.SUPPORTED_ARCHS:
            raise ValueError(
                f"Unsupported architecture: {self.arch}. "
                f"Supported: {self.SUPPORTED_ARCHS}"
            )

        self.num_classes = int(cfg.get("num_classes", 3))
        self.pretrained = cfg.get("pretrained", True)
        self.hidden_dim = cfg.get("hidden_dim", None)
        self.dropout_rate = cfg.get("dropout", 0.1)

        self._build_model()

        checkpoint_path = cfg.get("checkpoint_path")
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        elif checkpoint_path:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}, using random weights"
            )

    def _build_model(self) -> None:
        weights = "IMAGENET1K_V1" if self.pretrained else None
        self.backbone = getattr(models, self.arch)(weights=weights)

        num_features = self.backbone.fc.in_features
        if self.hidden_dim is not None and self.hidden_dim > 0:
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(num_features, self.num_classes)

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        images = self.preprocess_images(input_data)
        logits = self.backbone(images)  # (B, num_classes)
        probabilities = F.softmax(logits, dim=1)

        outputs: dict[str, Any] = {
            "logits": logits,
            "probabilities": probabilities,
        }

        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()
            outputs["loss"] = loss
            outputs["accuracy"] = accuracy

        return outputs

    def compute_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        images = observations.get("images")
        if images is None:
            images = observations.get("main_images")
        if images is None:
            raise ValueError("Observations must contain 'images' or 'main_images' key")

        images = self.preprocess_images(images)
        with torch.no_grad():
            logits = self.backbone(images)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["module.", "_orig_mod.", "model."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            if new_key in ["mean", "std", "_mean", "_std"]:
                continue
            new_state_dict[new_key] = v

        self.load_state_dict(new_state_dict, strict=False)

    @property
    def model_type(self) -> str:
        return "resnet_stage_classifier"



