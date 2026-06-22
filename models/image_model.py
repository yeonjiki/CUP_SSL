# image_model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm  # pip install timm


class ViTEncoder(nn.Module):
    """
    timm의 Vision Transformer를 encoder로 사용하는 래퍼.
    num_classes=0 으로 생성하면 feature vector만 반환.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = False):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # classification head 제거
        )
        self.feature_dim = self.vit.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D) feature
        return self.vit(x)


class ProjectionHead(nn.Module):
    """
    SimCLR projection head: f: D -> 128 (기본)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (SimCLR) loss.
    z_i, z_j: (N, D)
    """
    N = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    # 자기 자신과의 similarity 제외
    mask = (~torch.eye(2 * N, dtype=torch.bool, device=z.device)).float()
    exp_sim = torch.exp(sim) * mask

    # positive pair: (i, i+N), (i+N, i)
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
    numerator = torch.exp(pos)
    denominator = exp_sim.sum(dim=1)

    loss = -torch.log(numerator / denominator)
    return loss.mean()


class ViTClassifier(nn.Module):
    """
    SSL로 pretrain된 ViT encoder 위에 Linear classifier를 올린 모델.
    """

    def __init__(self, encoder: ViTEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)       # (B, D)
        logits = self.classifier(feats)
        return logits
