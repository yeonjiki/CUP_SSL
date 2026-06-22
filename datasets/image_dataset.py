# image_dataset.py

import os
import random
from typing import Tuple, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import TRAIN_RATIO, SEED, IMG_SIZE


# ---------------------------------------------------------
# 🔥 Novel SSL Augmentation: Intensity-only perturbation
# ---------------------------------------------------------
class MethylationIntensityJitter:
    """
    Probe ordering과 spatial topology는 유지한 채,
    methylation intensity 값만 약하게 perturb하는 transform.

    - Spatial transform ❌
    - Crop / Flip / Rotation ❌
    - Value-level noise & scaling만 허용
    """

    def __init__(self, noise_std: float = 0.03, scale_range=(0.95, 1.05)):
        self.noise_std = noise_std
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor):
        # x: (C, H, W), ToTensor 이후 [0, 1] 범위
        scale = random.uniform(*self.scale_range)
        noise = torch.randn_like(x) * self.noise_std
        x = x * scale + noise
        return torch.clamp(x, 0.0, 1.0)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class MultiCancerDataset(Dataset):
    """
    img_root/
      ├── CancerTypeA/
      │     ├── sample1.png
      │     └── ...
      └── CancerTypeB/
            └── ...

    SSL 모드:
      - (xi, xj) 두 개의 augmented view 반환
    Supervised 모드:
      - (img, label) 반환

    split: 'train', 'val', 'all'
      - 'all': 모든 샘플 반환 (테스트 / meta-eval 용)
    """

    def __init__(
        self,
        img_root: str,
        transform=None,
        split: str = "train",
        train_ratio: float = TRAIN_RATIO,
        seed: int = SEED,
        ssl: bool = False,
    ):
        assert split in ("train", "val", "all"), "split must be 'train', 'val' or 'all'"
        self.transform = transform
        self.ssl = ssl
        self.samples: List[Tuple[str, int]] = []

        # class 목록 구성
        self.classes = sorted(
            [d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))]
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 모든 이미지 로드
        for cancer in self.classes:
            cancer_path = os.path.join(img_root, cancer)
            for f in os.listdir(cancer_path):
                if f.lower().endswith(".png"):
                    path = os.path.join(cancer_path, f)
                    label = self.class_to_idx[cancer]
                    self.samples.append((path, label))

        # split 처리
        random.seed(seed)
        random.shuffle(self.samples)

        if split != "all":
            split_idx = int(len(self.samples) * train_ratio)
            if split == "train":
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # 원본 methylation 이미지는 grayscale
        img = Image.open(path).convert("L")

        if self.ssl:
            if self.transform is None:
                raise ValueError("SSL 모드에서는 transform을 반드시 지정해야 합니다.")
            xi = self.transform(img)
            xj = self.transform(img)
            return xi, xj
        else:
            if self.transform:
                img = self.transform(img)
            return img, label


# ---------------------------------------------------------
# SSL Transform (🔥 Probe-aware Novelty)
# ---------------------------------------------------------
def get_ssl_transform(img_size: int = IMG_SIZE):
    """
    SSL용 augmentation:
    - probe ordering 유지
    - spatial topology 유지
    - methylation intensity만 perturb
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # ViT 입력을 위해 3채널로 확장 (공간 구조는 동일)
        transforms.Grayscale(num_output_channels=3),

        transforms.ToTensor(),

        # 🔥 methylation-aware intensity perturbation
        MethylationIntensityJitter(
            noise_std=0.03,
            scale_range=(0.95, 1.05),
        ),

        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


# ---------------------------------------------------------
# Supervised Transform (No spatial augmentation)
# ---------------------------------------------------------
def get_sup_transform(img_size: int = IMG_SIZE):
    """
    Supervised fine-tuning:
    - augmentation 최소화
    - alignment 완전 보존
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
