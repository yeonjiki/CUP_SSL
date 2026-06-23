import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import models

from config import *
from multimodal_dataset import MultimodalDataset
from multimodal_model import MultimodalModel
from image_dataset import get_sup_transform


def set_seed(seed):
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
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


def normalize_features(train_feat, val_feat):
    mean = train_feat.mean(axis=0)
    std = train_feat.std(axis=0) + 1e-6
    return (train_feat - mean) / std, (val_feat - mean) / std


def train_multimodal():
    set_seed(SEED)
    device = get_device()
    BASE_DIR = "/Users/annkim/PycharmProjects/CUP_Project_Journal"

    dataset = MultimodalDataset(
        index_path=os.path.join(BASE_DIR, "multimodal_index.pkl"),
        mirna_feature_path=os.path.join(BASE_DIR, "processed/processed_mirna_all.npy"),
        mirna_id_path=os.path.join(BASE_DIR, "processed/processed_mirna_ids.npy"),
        transform=get_sup_transform(),
    )

    # 유효 샘플만
    valid_indices = [i for i, s in enumerate(dataset.samples) if s["label"] >= 0]
    print(f"✅ Valid samples: {len(valid_indices)}")

    # label remap
    raw_labels = [dataset.samples[i]["label"] for i in valid_indices]
    unique_labels = sorted(list(set(raw_labels)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    for i in valid_indices:
        dataset.samples[i]["label"] = label_map[dataset.samples[i]["label"]]

    labels = np.array([dataset.samples[i]["label"] for i in valid_indices])
    NUM_CLASSES = len(unique_labels)
    print(f"✅ Num classes: {NUM_CLASSES}")

    # class weights
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # patient grouping
    patient_to_indices = defaultdict(list)
    for idx, s in enumerate(dataset.samples):
        if idx in valid_indices:
            patient_to_indices[s["patient_id"]].append(idx)

    patients = list(patient_to_indices.keys())
    patient_labels = [dataset.samples[patient_to_indices[p][0]]["label"] for p in patients]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = []
    original_mirna = dataset.mirna_features.copy()

    for fold, (train_p, val_p) in enumerate(skf.split(patients, patient_labels)):
        print(f"\n========== Fold {fold+1} ==========")
        dataset.mirna_features = original_mirna.copy()

        train_idx = [i for p in train_p for i in patient_to_indices[patients[p]]]
        val_idx = [i for p in val_p for i in patient_to_indices[patients[p]]]

        train_feat, val_feat = normalize_features(
            dataset.mirna_features[train_idx],
            dataset.mirna_features[val_idx]
        )

        dataset.mirna_features[train_idx] = train_feat
        dataset.mirna_features[val_idx] = val_feat

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=MULTIMODAL_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=MULTIMODAL_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        # -----------------------
        # Image encoder
        # -----------------------
        image_encoder = models.vit_b_16(weights=None)
        image_encoder.heads = nn.Identity()
        if os.path.exists(SSL_CHECKPOINT):
            ckpt = torch.load(SSL_CHECKPOINT, map_location=device)
            image_encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        image_encoder = image_encoder.to(device)

        # -----------------------
        # Multimodal model
        # -----------------------
        model = MultimodalModel(
            num_classes=NUM_CLASSES,
            mirna_input_dim=dataset.mirna_features.shape[1],
            image_encoder=image_encoder
        ).to(device)

        if os.path.exists("mirna_ssl_encoder_tcga.pt"):
            mirna_ssl = torch.load("mirna_ssl_encoder_tcga.pt", map_location=device)
            model.mirna_encoder.load_state_dict(mirna_ssl)

        optimizer = torch.optim.AdamW([
            {"params": model.image_encoder.parameters(), "lr": 1e-5},
            {"params": model.mirna_encoder.parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MULTIMODAL_EPOCHS
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_mauc = 0.0
        best_model_state = None

        # -----------------------
        # Epoch loop
        # -----------------------
        for epoch in range(1, MULTIMODAL_EPOCHS + 1):
            model.train()
            for img, mirna, label in tqdm(train_loader):
                img, mirna, label = img.to(device), mirna.to(device), label.to(device)
                logits = model(img, mirna)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for img, mirna, label in val_loader:
                    img, mirna = img.to(device), mirna.to(device)
                    logits = model(img, mirna)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(label.cpu().numpy())

            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            val_mauc = compute_mauc(all_labels, all_probs)
            print(f"Epoch {epoch} | Val mAUC: {val_mauc:.4f}")

            if val_mauc > best_mauc:
                best_mauc = val_mauc
                best_model_state = model.state_dict()  # best 모델 저장

        # -----------------------
        # Fold 최종 저장
        # -----------------------
        fold_model_path = f"best_fold{fold}.pt"
        torch.save(best_model_state, fold_model_path)
        print(f"Fold {fold+1} Best mAUC: {best_mauc:.4f} | Saved: {fold_model_path}")
        fold_scores.append(best_mauc)

        # -----------------------
        # 인코더만 따로 저장 (final_eval용)
        # -----------------------
        image_encoder_path = f"image_encoder_fold{fold}.pth"
        mirna_encoder_path = f"mirna_encoder_fold{fold}.pth"
        torch.save(model.image_encoder.state_dict(), image_encoder_path)
        torch.save(model.mirna_encoder.state_dict(), mirna_encoder_path)
        print(f"Saved encoders: {image_encoder_path}, {mirna_encoder_path}")

    print("\n===================================")
    print(f"Final mAUC: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print("===================================")


if __name__ == "__main__":
    train_multimodal()
