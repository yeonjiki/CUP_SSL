# multimodal_dataset.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def normalize_id(pid: str):
    """
    TCGA ID 정규화
    필요 시 여기서 12자리 barcode까지만 자를 수 있음.
    """
    pid = str(pid)
    pid = pid.replace(".txt", "")
    pid = pid.replace(".mirnas.quantification", "")
    pid = pid.strip()
    return pid


# =========================================================
# 🔹 Train / Validation Dataset
# =========================================================
class MultimodalDataset(Dataset):

    def __init__(
        self,
        index_path,
        mirna_feature_path,
        mirna_id_path,
        transform=None,
    ):
        self.transform = transform

        # ----------------------------------
        # 1️⃣ Load image index
        # ----------------------------------
        with open(index_path, "rb") as f:
            all_samples = pickle.load(f)

        # ----------------------------------
        # 2️⃣ Load miRNA features
        # ----------------------------------
        self.mirna_features = np.load(mirna_feature_path)
        self.mirna_ids = np.load(mirna_id_path)

        self.mirna_ids = np.array(
            [normalize_id(i) for i in self.mirna_ids]
        )

        id_to_idx = {
            pid: i for i, pid in enumerate(self.mirna_ids)
        }

        # ----------------------------------
        # 3️⃣ Match patients
        # ----------------------------------
        matched_samples = []

        for sample in all_samples:

            raw_pid = sample.get("patient_id", None)
            if raw_pid is None:
                continue

            pid = normalize_id(raw_pid)

            if pid in id_to_idx:

                new_sample = {
                    "patient_id": pid,
                    "label": sample["label"],
                    "methylation_path": sample["methylation_path"],
                    "mirna_idx": id_to_idx[pid],
                }

                matched_samples.append(new_sample)

        self.samples = matched_samples

        # ----------------------------------
        # Logging
        # ----------------------------------
        print("✅ MultimodalDataset Loaded")
        print(f"   Total index samples : {len(all_samples)}")
        print(f"   Total miRNA samples : {len(self.mirna_ids)}")
        print(f"   Matched samples     : {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError(
                "❌ No matched samples found.\n"
                "Check patient_id format"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(
            sample["methylation_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mirna = torch.tensor(
            self.mirna_features[sample["mirna_idx"]],
            dtype=torch.float32,
        )

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return image, mirna, label


# =========================================================
# 🔥 CUP Dataset (JSON 기반, 최종)
# =========================================================
class CUPMultimodalDataset(Dataset):
    def __init__(
        self,
        json_root,
        img_root,
        mirna_root,
        transform=None,
        class_to_idx=None
    ):

        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        # ----------------------------------
        # 1️⃣ Load miRNA
        # ----------------------------------
        mirna_dict = {}

        for fn in os.listdir(mirna_root):
            if fn.endswith(".npy"):
                pid = normalize_id(fn.replace(".npy", ""))
                mirna_dict[pid] = np.load(os.path.join(mirna_root, fn))

        print(f"[CUP] Loaded miRNA: {len(mirna_dict)}")

        # ----------------------------------
        # 2️⃣ Build image index (🔥 속도 개선)
        # ----------------------------------
        image_index = {}

        for cancer_dir in os.listdir(img_root):
            full_dir = os.path.join(img_root, cancer_dir)
            if not os.path.isdir(full_dir):
                continue

            for img_fn in os.listdir(full_dir):
                if not img_fn.endswith(".png"):
                    continue

                # pid 포함 여부 기반 인덱싱
                for pid in mirna_dict.keys():
                    if pid in img_fn:
                        if pid not in image_index:
                            image_index[pid] = []
                        image_index[pid].append(
                            os.path.join(full_dir, img_fn)
                        )

        print(f"[CUP] Image index built: {len(image_index)} patients")

        # ----------------------------------
        # 3️⃣ Load JSON metadata
        # ----------------------------------
        import json

        json_files = [
            f for f in os.listdir(json_root)
            if f.endswith(".json")
        ]

        print(f"[CUP] Found JSON files: {len(json_files)}")

        # ----------------------------------
        # 4️⃣ Match all modalities
        # ----------------------------------
        for jf in json_files:

            json_path = os.path.join(json_root, jf)

            with open(json_path, "r") as f:
                data = json.load(f)

            pid = normalize_id(data.get("patient_id", None))
            if pid is None:
                continue

            cancer_type = data.get("project_id", None)
            if cancer_type is None:
                continue

            if class_to_idx is None or cancer_type not in class_to_idx:
                continue

            if pid not in mirna_dict:
                continue

            if pid not in image_index:
                continue

            label = class_to_idx[cancer_type]
            mirna_feat = mirna_dict[pid]

            for img_path in image_index[pid]:

                self.samples.append({
                    "img_path": img_path,
                    "mirna": mirna_feat,
                    "label": label,
                    "patient_id": pid,
                })

        # ----------------------------------
        # Logging
        # ----------------------------------
        print("✅ CUPMultimodalDataset Loaded (FINAL)")
        print(f"   Total samples: {len(self.samples)}")

        if len(self.samples) == 0:
            print("⚠️ WARNING: No matched CUP samples found")
            print("Check:")
            print("- JSON patient_id")
            print("- image naming")
            print("- miRNA naming")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]

        img = Image.open(s["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        mirna = torch.tensor(
            s["mirna"],
            dtype=torch.float32
        )

        label = torch.tensor(
            s["label"],
            dtype=torch.long
        )

        return img, mirna, label