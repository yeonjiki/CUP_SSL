import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from config import SEED

torch.manual_seed(SEED)


# --------------------------------------------------
# Dataset with structured masking
# --------------------------------------------------
class MirnaSSLDataset(Dataset):
    def __init__(self, feature_path):
        self.features = np.load(feature_path)
        self.features = torch.tensor(self.features, dtype=torch.float32)

        # variance 기반 feature importance
        self.var = torch.var(self.features, dim=0)
        self.high_var_idx = torch.topk(self.var, k=int(0.3 * len(self.var))).indices

    def __len__(self):
        return len(self.features)

    def structured_mask(self, x):
        x_masked = x.clone()

        mask_ratio = 0.3
        num_mask = int(len(self.high_var_idx) * mask_ratio)

        selected = self.high_var_idx[
            torch.randperm(len(self.high_var_idx))[:num_mask]
        ]
        x_masked[selected] = 0

        return x_masked

    def __getitem__(self, idx):
        x = self.features[idx]

        x_view1 = self.structured_mask(x)
        x_view2 = self.structured_mask(x)

        return x_view1, x_view2, x


# --------------------------------------------------
# Encoder + Projection Head
# --------------------------------------------------
class MirnaSSLModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        recon = self.decoder(z)
        return z, p, recon


# --------------------------------------------------
# NT-Xent contrastive loss
# --------------------------------------------------
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    batch_size = z1.size(0)
    similarity = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size).to(z1.device)

    loss = nn.CrossEntropyLoss()(similarity, labels)
    return loss


# --------------------------------------------------
# Generic SSL Trainer
# --------------------------------------------------
def train_ssl(feature_path, save_name):

    print(f"\n🚀 Starting SSL pretrain for: {feature_path}")

    dataset = MirnaSSLDataset(feature_path)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = MirnaSSLModel(dataset.features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    recon_loss_fn = nn.MSELoss()

    for epoch in range(100):

        total_loss = 0

        for x1, x2, x_gt in tqdm(loader):

            z1, p1, recon1 = model(x1)
            z2, p2, recon2 = model(x2)

            loss_recon = recon_loss_fn(recon1, x_gt) + recon_loss_fn(recon2, x_gt)
            loss_contrast = contrastive_loss(p1, p2)

            loss = loss_recon + 0.5 * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")

    torch.save(model.encoder.state_dict(), save_name)
    print(f"✅ Saved encoder to {save_name}")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    # TCGA pretrain
    train_ssl(
        feature_path="processed/processed_mirna_all.npy",
        save_name="mirna_ssl_encoder_tcga.pt"
    )

    # META pretrain
    train_ssl(
        feature_path="processed/processed_meta_mirna_all.npy",
        save_name="mirna_ssl_encoder_meta.pt"
    )

    print("\n🎉 ALL SSL PRETRAIN DONE")
