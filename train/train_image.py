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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretrain_ssl_vit():
    set_seed(SEED)
    device = get_device()
    print(f"[SSL] Using device: {device}")
    transform_ssl = get_ssl_transform()
    dataset = MultiCancerDataset(
        IMG_ROOT,
        transform=transform_ssl,
        ssl=True,
        split="train", 
    )

    pin_memory = device.type == "cuda"

    dataloader = DataLoader(
        dataset,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    encoder = ViTEncoder(pretrained=True).to(device)
    projection_head = ProjectionHead(in_dim=encoder.feature_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=SSL_LR,
        weight_decay=1e-4,
    )

    total_steps = SSL_EPOCHS * max(1, len(dataloader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )

    global_step = 0

    for epoch in range(1, SSL_EPOCHS + 1):
        encoder.train()
        projection_head.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"[SSL] Epoch {epoch}/{SSL_EPOCHS}")
        for xi, xj in pbar:
            xi = xi.to(device, non_blocking=pin_memory)
            xj = xj.to(device, non_blocking=pin_memory)

            hi = encoder(xi)  # (B, D)
            hj = encoder(xj)  # (B, D)

            zi = projection_head(hi)  # (B, proj_dim)
            zj = projection_head(hj)  # (B, proj_dim)

            loss = nt_xent_loss(zi, zj, temperature=SSL_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()

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

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
        },
        SSL_CHECKPOINT,
    )
    print(f"[SSL] Saved encoder to {SSL_CHECKPOINT}")


if __name__ == "__main__":
    pretrain_ssl_vit()
