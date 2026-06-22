# train_image.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    IMG_ROOT,
    SSL_BATCH_SIZE,
    SSL_EPOCHS,
    SSL_LR,
    SSL_TEMPERATURE,
    SSL_CHECKPOINT,
    SEED,
    NUM_WORKERS,
)
from image_dataset import MultiCancerDataset, get_ssl_transform
from image_model import ViTEncoder, ProjectionHead, nt_xent_loss


def get_device() -> torch.device:
    """
    CUDA → MPS → CPU 순서로 가능한 디바이스를 선택합니다.
    - 맥북(Apple Silicon)에서는 주로 MPS가 선택됩니다.
    - Linux/Windows + NVIDIA GPU에서는 CUDA가 선택됩니다.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[Device] Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA 환경일 때만 개별 시드 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_ssl_vit():
    # ---------------------------------------------------
    # 0) Seed & Device
    # ---------------------------------------------------
    set_seed(SEED)
    device = get_device()
    print(f"[SSL] Using device: {device}")

    # ---------------------------------------------------
    # 1) Dataset & DataLoader
    #    - CUP methylation 이미지를 대상으로 두 뷰(xi, xj)를 만들어
    #      self-supervised 학습을 수행합니다.
    # ---------------------------------------------------
    transform_ssl = get_ssl_transform()
    dataset = MultiCancerDataset(
        IMG_ROOT,
        transform=transform_ssl,
        ssl=True,
        split="train",  # SSL에서는 보통 train 부분 전체 사용
    )

    # CUDA에서만 pin_memory=True 로 설정 (MPS/CPU는 False 권장)
    pin_memory = device.type == "cuda"

    dataloader = DataLoader(
        dataset,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # ---------------------------------------------------
    # 2) 모델 정의 (ViT + Projection Head)
    #    - ViTEncoder: methylation 이미지(단일 채널)를 입력으로 받아
    #                  sequence로 보고 long-range dependency를 학습
    #    - ImageNet pretrain을 사용해 더 좋은 초기 표현에서 시작
    # ---------------------------------------------------
    encoder = ViTEncoder(pretrained=True).to(device)
    projection_head = ProjectionHead(in_dim=encoder.feature_dim).to(device)

    # AdamW: weight decay가 잘 작동해서 안정적으로 representation 학습
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=SSL_LR,
        weight_decay=1e-4,
    )

    # Cosine Annealing LR: 초반에는 큰 lr로 탐색, 후반으로 갈수록 부드럽게 줄여줌
    total_steps = SSL_EPOCHS * max(1, len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    global_step = 0

    # ---------------------------------------------------
    # 3) SSL 학습 루프 (NT-Xent / SimCLR 스타일)
    #    - 같은 샘플의 두 뷰(xi, xj)는 서로 가깝게,
    #      다른 샘플들은 멀어지도록 embedding 공간을 정렬
    # ---------------------------------------------------
    for epoch in range(1, SSL_EPOCHS + 1):
        encoder.train()
        projection_head.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[SSL] Epoch {epoch}/{SSL_EPOCHS}")
        for xi, xj in pbar:
            xi = xi.to(device, non_blocking=pin_memory)
            xj = xj.to(device, non_blocking=pin_memory)

            # ViT encoder로 feature 추출
            hi = encoder(xi)  # (B, D)
            hj = encoder(xj)  # (B, D)

            # Projection head를 통과시켜 contrastive 공간으로 매핑
            zi = projection_head(hi)  # (B, proj_dim)
            zj = projection_head(hj)  # (B, proj_dim)

            loss = nt_xent_loss(zi, zj, temperature=SSL_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()

            # gradient clip (폭주 방지)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projection_head.parameters()),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        avg_loss = total_loss / len(dataloader)
        print(f"[SSL] Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

    # ---------------------------------------------------
    # 4) Encoder 저장 (downstream에서 classifier head만 붙여 fine-tuning)
    # ---------------------------------------------------
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
        },
        SSL_CHECKPOINT,
    )
    print(f"[SSL] Saved encoder to {SSL_CHECKPOINT}")


if __name__ == "__main__":
    pretrain_ssl_vit()
