import os
import heapq
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
from torchvision import models

from multimodal_model import MultimodalModel
from image_dataset import get_sup_transform

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
SEED                  = 42
MULTIMODAL_BATCH_SIZE = 32
NUM_WORKERS           = 2
MULTIMODAL_EPOCHS     = 30
META_INDEX            = "multimodal_index_meta.pkl"

IMAGE_ENCODER_PATH    = "image_encoder_fold{}.pth"
MIRNA_ENCODER_PATH    = "mirna_encoder_fold{}.pth"

IMG_FEAT_DIM      = 256
MIRNA_FEAT_DIM    = 256

# SSL (변경 없음)
SSL_PROJ_DIM      = 128
SSL_TEMPERATURE   = 0.07
SSL_WEIGHT        = 0.3

# Stability — 정규화 강화
WARMUP_EPOCHS     = 5    # 3 → 5
MIXUP_ALPHA       = 0.4  # 0.2 → 0.4
LABEL_SMOOTHING   = 0.2  # 0.1 → 0.2
TOP_K_CHECKPOINTS = 5    # 3 → 5


# -----------------------------------------------------------------------
# CrossModalWrapper
# -----------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class CrossModalWrapper(nn.Module):
    def __init__(self, base: MultimodalModel, proj_dim: int = SSL_PROJ_DIM):
        super().__init__()
        self.base           = base
        self.ssl_img_proj   = ProjectionHead(IMG_FEAT_DIM,   proj_dim)
        self.ssl_mirna_proj = ProjectionHead(MIRNA_FEAT_DIM, proj_dim)

    def forward(self, img: torch.Tensor, mirna: torch.Tensor):
        img_feat   = self.base.image_encoder(img)       # (B, 768)
        img_feat   = self.base.img_proj(img_feat)       # (B, 256)
        mirna_feat = self.base.mirna_encoder(mirna)     # (B, 256)

        fusion_cat = torch.cat([img_feat, mirna_feat], dim=1)
        gate       = self.base.gate(fusion_cat)
        fusion     = img_feat + gate * mirna_feat
        logits     = self.base.classifier(fusion)

        z_img   = self.ssl_img_proj(img_feat)
        z_mirna = self.ssl_mirna_proj(mirna_feat)

        return logits, z_img, z_mirna


# -----------------------------------------------------------------------
# Cross-modal InfoNCE
# -----------------------------------------------------------------------
def cross_modal_infonce(z_img, z_mirna, temperature=SSL_TEMPERATURE):
    B       = z_img.size(0)
    sim_i2m = torch.mm(z_img,   z_mirna.T) / temperature
    sim_m2i = torch.mm(z_mirna, z_img.T)   / temperature
    labels  = torch.arange(B, device=z_img.device)
    return (F.cross_entropy(sim_i2m, labels) +
            F.cross_entropy(sim_m2i, labels)) * 0.5


# -----------------------------------------------------------------------
# Mixup
# -----------------------------------------------------------------------
def mixup_batch(img, mirna, label, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return img, mirna, label, label, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(img.size(0), device=img.device)
    return (lam * img   + (1 - lam) * img[idx],
            lam * mirna + (1 - lam) * mirna[idx],
            label, label[idx], lam)


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# -----------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------
def build_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1)
    )
    return warmup, cosine


# -----------------------------------------------------------------------
# Misc utils
# -----------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_mauc(y_true, y_prob):
    unique = np.unique(y_true)
    if len(unique) < 2:
        return 0.0
    y_prob = y_prob[:, unique]
    y_prob = y_prob / (y_prob.sum(axis=1, keepdims=True) + 1e-8)
    try:
        return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        return 0.0


def compute_accuracy(y_true, y_prob):
    """Top-1 accuracy: argmax of predicted probabilities vs ground truth."""
    preds = np.argmax(y_prob, axis=1)
    return float(np.mean(preds == y_true))


def load_mirna_txt(path):
    df     = pd.read_csv(path, sep="\t")
    values = (df["read_count"].values.astype(np.float32)
              if "read_count" in df.columns
              else df.iloc[:, -1].values.astype(np.float32))
    values = np.log2(values + 1.0)
    log_x  = np.log(values + 1e-6)
    return log_x - np.mean(log_x)


def majority_label(indices, samples):
    lbls = [samples[i]["label"] for i in indices]
    return max(set(lbls), key=lbls.count)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
class MetaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["methylation_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (img,
                torch.tensor(s["mirna_norm"], dtype=torch.float32),
                torch.tensor(s["label"],      dtype=torch.long))


# -----------------------------------------------------------------------
# Model loader (ViT-B/16)
# -----------------------------------------------------------------------
def load_model(num_classes, mirna_dim, device, fold) -> CrossModalWrapper:
    image_encoder = models.vit_b_16(weights=None)
    image_encoder.heads = nn.Identity()  # output: (B, 768)

    img_path   = IMAGE_ENCODER_PATH.format(fold)
    mirna_path = MIRNA_ENCODER_PATH.format(fold)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing image encoder: {img_path}")
    if not os.path.exists(mirna_path):
        raise FileNotFoundError(f"Missing miRNA encoder: {mirna_path}")

    image_encoder.load_state_dict(
        torch.load(img_path, map_location=device), strict=False
    )
    image_encoder.to(device)

    base = MultimodalModel(
        num_classes     = num_classes,
        mirna_input_dim = mirna_dim,
        image_encoder   = image_encoder,
    )
    base.mirna_encoder.load_state_dict(
        torch.load(mirna_path, map_location=device)
    )

    model = CrossModalWrapper(base, proj_dim=SSL_PROJ_DIM)
    model.to(device)
    return model


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def train_meta():
    set_seed()
    device = get_device()

    with open(META_INDEX, "rb") as f:
        data = pickle.load(f)
    samples = data["samples"]

    valid = []
    for s in samples:
        try:
            s["mirna_feat"] = load_mirna_txt(s["mirna_path"])
            valid.append(s)
        except Exception:
            continue
    samples = valid
    print(f"Loaded samples: {len(samples)}")

    unique_labels = sorted(set(s["label"] for s in samples))
    label_map     = {l: i for i, l in enumerate(unique_labels)}
    num_classes   = len(unique_labels)
    for s in samples:
        s["label"] = label_map[s["label"]]

    patient_to_indices = defaultdict(list)
    for i, s in enumerate(samples):
        patient_to_indices[s["patient_id"]].append(i)

    patients       = list(patient_to_indices.keys())
    patient_labels = [majority_label(patient_to_indices[p], samples) for p in patients]

    skf         = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    criterion   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    fold_scores = []  # (mauc, acc) per fold

    for fold, (train_p_idx, val_p_idx) in enumerate(skf.split(patients, patient_labels)):
        print(f"\n========== Fold {fold + 1} ==========")

        train_idx = [i for p in train_p_idx for i in patient_to_indices[patients[p]]]
        val_idx   = [i for p in val_p_idx   for i in patient_to_indices[patients[p]]]

        train_feat = np.stack([samples[i]["mirna_feat"] for i in train_idx])
        mean, std  = train_feat.mean(0), train_feat.std(0) + 1e-6
        for i in train_idx:
            samples[i]["mirna_norm"] = (samples[i]["mirna_feat"] - mean) / std
        for i in val_idx:
            samples[i]["mirna_norm"] = (samples[i]["mirna_feat"] - mean) / std

        transform    = get_sup_transform()
        train_loader = DataLoader(
            Subset(MetaDataset(samples, transform), train_idx),
            batch_size=MULTIMODAL_BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(MetaDataset(samples, transform), val_idx),
            batch_size=MULTIMODAL_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )

        mirna_dim = samples[train_idx[0]]["mirna_norm"].shape[0]
        model     = load_model(num_classes, mirna_dim, device, fold)

        optimizer = torch.optim.AdamW([
            {"params": model.base.image_encoder.parameters(), "lr": 5e-6},  # 1e-5 → 5e-6
            {"params": model.base.mirna_encoder.parameters(), "lr": 1e-4},
            {"params": model.base.img_proj.parameters(),      "lr": 1e-3},
            {"params": model.base.gate.parameters(),          "lr": 1e-3},
            {"params": model.base.classifier.parameters(),    "lr": 1e-3},
            {"params": model.ssl_img_proj.parameters(),       "lr": 1e-3},
            {"params": model.ssl_mirna_proj.parameters(),     "lr": 1e-3},
        ], weight_decay=3e-4)  # 1e-4 → 3e-4

        warmup_sched, cosine_sched = build_scheduler(
            optimizer, WARMUP_EPOCHS, MULTIMODAL_EPOCHS
        )

        top_k_pool = []

        for epoch in range(1, MULTIMODAL_EPOCHS + 1):
            model.train()
            for img, mirna, label in tqdm(train_loader,
                                          desc=f"Fold {fold+1} Ep {epoch:02d}",
                                          leave=False):
                img, mirna, label = img.to(device), mirna.to(device), label.to(device)

                m_img, m_mirna, y_a, y_b, lam = mixup_batch(img, mirna, label)
                logits, _, _ = model(m_img, m_mirna)
                loss_sup = mixup_criterion(criterion, logits, y_a, y_b, lam)

                _, z_img, z_mirna = model(img, mirna)
                loss_ssl = cross_modal_infonce(z_img, z_mirna, SSL_TEMPERATURE)

                loss = loss_sup + SSL_WEIGHT * loss_ssl

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            if epoch <= WARMUP_EPOCHS:
                warmup_sched.step()
            else:
                cosine_sched.step()

            # ── Validation ──────────────────────────────────────────────
            model.eval()
            all_probs, all_labels_list = [], []
            with torch.no_grad():
                for img, mirna, label in val_loader:
                    img, mirna = img.to(device), mirna.to(device)
                    logits, _, _ = model(img, mirna)
                    all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                    all_labels_list.append(label.numpy())

            all_probs      = np.concatenate(all_probs)
            all_labels_arr = np.concatenate(all_labels_list)
            val_mauc = compute_mauc(all_labels_arr, all_probs)
            val_acc  = compute_accuracy(all_labels_arr, all_probs)

            print(f"  Ep {epoch:02d} | mAUC {val_mauc:.4f} | Acc {val_acc:.4f} "
                  f"| L_sup {loss_sup.item():.4f} | L_ssl {loss_ssl.item():.4f}")

            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if len(top_k_pool) < TOP_K_CHECKPOINTS:
                heapq.heappush(top_k_pool, (val_mauc, epoch, state))
            elif val_mauc > top_k_pool[0][0]:
                heapq.heapreplace(top_k_pool, (val_mauc, epoch, state))

        top_epochs = sorted(ep for _, ep, _ in top_k_pool)
        print(f"\nEnsembling top-{TOP_K_CHECKPOINTS} checkpoints (epochs: {top_epochs})")

        # ── Ensemble evaluation ─────────────────────────────────────────
        ensemble_probs = []
        for _, _, state in top_k_pool:
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            model.eval()
            ep_probs, ep_labels = [], []
            with torch.no_grad():
                for img, mirna, label in val_loader:
                    img, mirna = img.to(device), mirna.to(device)
                    logits, _, _ = model(img, mirna)
                    ep_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                    ep_labels.append(label.numpy())
            ensemble_probs.append(np.concatenate(ep_probs))

        avg_probs    = np.mean(ensemble_probs, axis=0)
        final_labels = np.concatenate(ep_labels)
        fold_mauc = compute_mauc(final_labels, avg_probs)
        fold_acc  = compute_accuracy(final_labels, avg_probs)

        best_state = max(top_k_pool, key=lambda x: x[0])[2]
        torch.save(best_state, f"meta_best_fold{fold}.pt")

        print(f"Fold {fold + 1} Ensemble mAUC: {fold_mauc:.4f} | Acc: {fold_acc:.4f}")
        fold_scores.append((fold_mauc, fold_acc))

    mauc_list = [s[0] for s in fold_scores]
    acc_list  = [s[1] for s in fold_scores]

    print("\n===================================")
    print(f"Final mAUC : {np.mean(mauc_list):.4f} ± {np.std(mauc_list):.4f}")
    print(f"Final Acc  : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"SSL_WEIGHT={SSL_WEIGHT} | TEMPERATURE={SSL_TEMPERATURE}")
    print(f"TOP_K={TOP_K_CHECKPOINTS} | WARMUP={WARMUP_EPOCHS} epochs")
    print("===================================")


if __name__ == "__main__":
    train_meta()
