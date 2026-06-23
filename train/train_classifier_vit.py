import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from PIL import Image
import numpy as np
import csv
import random
import math

from config import (
    IMG_ROOT,
    TEST_IMG_ROOT,
    SUP_BATCH_SIZE,
    SUP_EPOCHS,
    SUP_LR,
    SUP_CHECKPOINT,
    SEED,
    NUM_WORKERS,
    TRAIN_RATIO,
    SSL_CHECKPOINT,
)
from image_dataset import MultiCancerDataset, get_sup_transform
from image_model import ViTEncoder, ViTClassifier

def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[Sup] Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Sup] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Sup] Using CPU")
    return device


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label

TARGET_TCGA = [
    "TCGA-CHOL",
    "TCGA-DLBC",
    "TCGA-HNSC",
    "TCGA-PCPG",
    "TCGA-READ",
    "TCGA-SKCM",
    "TCGA-THYM",
]

def compute_multiclass_mauc(probs_arr: np.ndarray, labels_arr: np.ndarray, class_indices: list = None):
    n_classes = probs_arr.shape[1]
    if class_indices is None:
        class_indices = list(range(n_classes))
    else:
        class_indices = sorted(set(class_indices))

    pairwise_vals = {}
    valid_pair_count = 0
    sum_pairs = 0.0

    for i_idx in range(len(class_indices)):
        i = class_indices[i_idx]
        for j_idx in range(i_idx + 1, len(class_indices)):
            j = class_indices[j_idx]
            # filter samples belonging to class i or j
            mask = np.logical_or(labels_arr == i, labels_arr == j)
            if mask.sum() == 0:
                continue
            labels_pair = labels_arr[mask]

            n_i = (labels_pair == i).sum()
            n_j = (labels_pair == j).sum()
            if n_i == 0 or n_j == 0:
                continue

            # A_i|j: positive = i, negative = j, scores = probs[:, i] for masked samples
            try:
                A_i_j = roc_auc_score((labels_pair == i).astype(int), probs_arr[mask, i])
            except Exception:
                A_i_j = float("nan")
            try:
                A_j_i = roc_auc_score((labels_pair == j).astype(int), probs_arr[mask, j])
            except Exception:
                A_j_i = float("nan")

            if np.isnan(A_i_j) or np.isnan(A_j_i):
                continue

            A_ij = 0.5 * (A_i_j + A_j_i)
            pairwise_vals[(i, j)] = A_ij
            sum_pairs += A_ij
            valid_pair_count += 1

    if valid_pair_count == 0:
        return float("nan"), pairwise_vals

    mAUC = float(sum_pairs / valid_pair_count)
    return mAUC, pairwise_vals

def train_classifier():
    set_seed(SEED)
    device = get_device()

    transform = get_sup_transform()
    base_ds = MultiCancerDataset(
        IMG_ROOT,
        transform=transform,
        split="all",  
        ssl=False,
    )
    ds_labels = [base_ds[i][1] for i in range(len(base_ds))]

    skf_init = StratifiedKFold(n_splits=int(1 / (1.0 - TRAIN_RATIO)), shuffle=True, random_state=SEED)
    # 첫 번째 fold를 Train/Val로 사용 (가장 간단한 Stratified Holdout)
    train_idx, val_idx = next(skf_init.split(range(len(base_ds)), ds_labels))

    train_subset = Subset(base_ds, train_idx)
    val_subset = Subset(base_ds, val_idx)

    num_classes = len(base_ds.classes)
    train_classes = base_ds.classes
    train_class_to_idx = base_ds.class_to_idx

    print(f"[Sup] Number of cancer types: {num_classes}")
    print(f"[Sup] Total labeled samples: {len(base_ds)}")
    print(f"[Sup] Train samples (80%): {len(train_subset)}")
    print(f"[Sup] Validation samples (20%): {len(val_subset)}")

    train_labels = [ds_labels[i] for i in train_idx]
    class_counts = Counter(train_labels)

    class_weights = []
    for c in range(num_classes):
        cnt = class_counts.get(c, 0)
        if cnt > 0:
            class_weights.append(1.0 / cnt)
        else:
            class_weights.append(0.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    print(f"[Main] Class counts (Train): {class_counts}")
    print(f"[Main] Class weights (Train): {class_weights.cpu().numpy()}")

    dl_pin_memory = (device.type == "cuda")

    train_dl = DataLoader(
        train_subset,
        batch_size=SUP_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=dl_pin_memory,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=SUP_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=dl_pin_memory,
    )

    mapped_test_samples = []
    unmapped_dirs = []
    if os.path.exists(TEST_IMG_ROOT):
        # (원래의 매핑 로직은 유지합니다)
        for candidate_dir in sorted(os.listdir(TEST_IMG_ROOT)):
            full_dir = os.path.join(TEST_IMG_ROOT, candidate_dir)
            if not os.path.isdir(full_dir):
                continue

            mapped_idx = None
            # 1) exact match
            if candidate_dir in train_class_to_idx:
                mapped_idx = train_class_to_idx[candidate_dir]
            else:
                # 2) remove '-meta' suffix and try common conventions
                if candidate_dir.endswith("-meta"):
                    base_name = candidate_dir[:-5]
                    candidate1 = "TCGA-" + base_name
                    if candidate1 in train_class_to_idx:
                        mapped_idx = train_class_to_idx[candidate1]
                    elif base_name in train_class_to_idx:
                        mapped_idx = train_class_to_idx[base_name]
                # 3) partial case-insensitive match
                if mapped_idx is None:
                    for tname in train_classes:
                        if candidate_dir.lower() in tname.lower() or tname.lower() in candidate_dir.lower():
                            mapped_idx = train_class_to_idx[tname]
                            break

            if mapped_idx is None:
                unmapped_dirs.append(candidate_dir)
                continue

            for fn in os.listdir(full_dir):
                if fn.lower().endswith(".png"):
                    mapped_test_samples.append((os.path.join(full_dir, fn), mapped_idx))
           
    print(f"[Sup] Mapped Meta test sample count: {len(mapped_test_samples)}")
    if len(mapped_test_samples) == 0:
        print("[Warn] No Meta test samples were mapped to train classes. Final evaluation will be skipped.")
        # Meta 평가를 건너뛸 수 있도록 오류 대신 경고 처리
        filtered_test_ds = None
    else:
        filtered_test_ds = SimpleImageDataset(mapped_test_samples, transform=transform)
        test_dl = DataLoader(
            filtered_test_ds,
            batch_size=SUP_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=dl_pin_memory,
        )
        print("[Sup] Unmapped Meta test dirs (if any):", unmapped_dirs)

    encoder = ViTEncoder(pretrained=False)
    ckpt_ssl = torch.load(SSL_CHECKPOINT, map_location="cpu")
    encoder.load_state_dict(ckpt_ssl["encoder_state_dict"])
    print(f"[Main] Loaded SSL encoder from {SSL_CHECKPOINT}")

    model = ViTClassifier(encoder, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=SUP_LR)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n========== [Training Start] (Patience: {EARLY_STOPPING_PATIENCE}) ==========")

    for epoch in range(1, SUP_EPOCHS + 1):
        # --------- Train ---------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        pbar = tqdm(
            train_dl,
            desc=f"[Sup] Epoch {epoch}/{SUP_EPOCHS} [Train]",
        )
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(
                    val_dl,
                    desc=f"[Sup] Epoch {epoch}/{SUP_EPOCHS} [Val]",
            ):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[Sup][Epoch {epoch}] "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        # 조기 종료 (Early Stopping) 및 최적 모델 저장 (TCGA Val Acc 기준)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": base_ds.classes,
                },
                SUP_CHECKPOINT,
            )
            print(f"[Sup] Model saved to {SUP_CHECKPOINT} (Best Val Acc: {best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[Sup] Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
                break

    print(f"\n========== [Final Evaluation on Meta Data ({TEST_IMG_ROOT})] ==========")

    if filtered_test_ds is None:
        print("[Final Eval] Skipping Meta evaluation due to no mapped samples.")
        return

    if not os.path.exists(SUP_CHECKPOINT):
        # 조기 종료가 없었으면 현재 모델 사용, 있었으면 저장된 최적 모델 로드
        print("[Warn] Best model checkpoint not found. Using current model state (if training finished).")
        eval_model = model
    else:
        ck = torch.load(SUP_CHECKPOINT, map_location="cpu")
        eval_encoder = ViTEncoder(pretrained=False)
        eval_encoder.load_state_dict(ckpt_ssl["encoder_state_dict"])  # SSL CKPT 재사용
        eval_model = ViTClassifier(eval_encoder, num_classes=num_classes).to(device)
        eval_model.load_state_dict(ck["model_state_dict"])

    eval_model.eval()

    target_train_indices = []
    for name in TARGET_TCGA:
        if name in train_class_to_idx:
            target_train_indices.append(train_class_to_idx[name])
        else:
            base_name_try = name
            if name.startswith("TCGA-"):
                base_name_try = name[5:]
            matched = None
            for tname, idx in train_class_to_idx.items():
                if base_name_try.lower() == tname.lower() or base_name_try.lower() in tname.lower() or tname.lower() in base_name_try.lower():
                    matched = idx
                    break
            if matched is not None:
                target_train_indices.append(matched)
            else:
                print(f"[Warn] Target TCGA class '{name}' not found in train classes. Skipping.")

    if len(target_train_indices) == 0:
        print("[Warn] No target TCGA classes mapped to training classes — skipping final TCGA evaluation.")
        return

    tcga_probs = []
    tcga_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_dl, desc=f"[Meta Eval]"):
            labels_np = labels.numpy()
            # check if any sample in this batch belongs to target indices
            mask = np.isin(labels_np, target_train_indices)
            if not mask.any():
                continue
            imgs = imgs.to(device)
            outputs = eval_model(imgs)  # logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            for i in range(len(labels_np)):
                if mask[i]:
                    tcga_probs.append(probs[i])
                    tcga_labels.append(labels_np[i])

    if len(tcga_labels) == 0:
        print(f"[Final Eval] No TCGA target samples found in Meta dataset.")
        return

    probs_arr = np.vstack(tcga_probs)  # (N, C)
    labels_arr = np.array(tcga_labels)  # (N,)

    preds = probs_arr.argmax(axis=1)
    acc_tcga = float((preds == labels_arr).mean())

    tcga_mAUC, pairwise = compute_multiclass_mauc(probs_arr, labels_arr,
                                                  class_indices=sorted(set(target_train_indices)))

    valid_aucs = []
    per_class_auc = {}
    for tidx in sorted(set(target_train_indices)):
        cls_name = base_ds.classes[tidx]
        y_true_bin = (labels_arr == tidx).astype(int)
        y_score = probs_arr[:, tidx]
        if y_true_bin.sum() > 0 and y_true_bin.sum() < len(y_true_bin):
            try:
                auc = roc_auc_score(y_true_bin, y_score)
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")
        per_class_auc[cls_name] = auc
        if not np.isnan(auc):
            valid_aucs.append(auc)
    macro_auc_tcga = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else float("nan")

    print(f"\n[Final Meta Eval] Summary (N={len(labels_arr)})")
    print(f"  Accuracy (targets only): {acc_tcga:.4f}")
    print(f"  Macro AUC (targets only): {macro_auc_tcga:.4f}")
    print(f"  **Hand & Till mAUC (targets only): {tcga_mAUC:.4f}**")

    print("\n  Per-class One-vs-Rest AUC:")
    for k, v in per_class_auc.items():
        print(f"    {k}: {v:.4f}")

    if not np.isnan(tcga_mAUC):
        print("\n  Pairwise mAUC:")
        for (i, j), val in pairwise.items():
            print(f"    {base_ds.classes[i]} vs {base_ds.classes[j]}: {val:.4f}")

    csv_path = "final_meta_evaluation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["n_samples", len(labels_arr)])
        writer.writerow(["Accuracy", f"{acc_tcga:.4f}"])
        writer.writerow(["Macro_AUC", f"{macro_auc_tcga:.4f}"])
        writer.writerow(["mAUC", f"{tcga_mAUC:.4f}"])
        for k, v in per_class_auc.items():
            writer.writerow([f"AUC_{k}", f"{v:.4f}"])
    print(f"[Info] Final Meta evaluation summary saved to {csv_path}")


if __name__ == "__main__":
    # NOTE: config.py에 EARLY_STOPPING_PATIENCE = 10 (예시) 와 같은 설정이 추가되어야 합니다.
    # 또한 MultiCancerDataset의 'all' split 로직이 전체 데이터를 반환하도록 구현되어야 합니다.
    # 안전을 위해 config에 PATIENCE가 없으면 기본값 설정
    if 'EARLY_STOPPING_PATIENCE' not in globals():
        EARLY_STOPPING_PATIENCE = 10
        print("[Warn] EARLY_STOPPING_PATIENCE not found in config, using default: 10")

    train_classifier()
