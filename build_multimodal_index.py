import os
import json
import pickle
from collections import defaultdict

METH_DIR = "/Volumes/T7/CUP Project Data/Methylation array Datasets"
MIRNA_DIR = "/Volumes/T7/CUP Project Data/miRNA-seq Datasets"
METH_PNG_DIR = "/Users/annkim/PycharmProjects/CUP_Project_Journal/methylation_images"

# META
METH_META_DIR = "/Volumes/T7/CUP Project Data/TCGA-meta"
MIRNA_META_DIR = "/Volumes/T7/CUP Project Data/TCGA-meta-miRNA"
METH_META_PNG_DIR = "/Users/annkim/PycharmProjects/CUP_Project_Journal/methylation_images_tcga"

TRAIN_OUTPUT = "multimodal_index_train.pkl"
META_OUTPUT = "multimodal_index_meta.pkl"


def to_patient_id(entity_submitter_id):
    return "-".join(entity_submitter_id.split("-")[:3])


def normalize_cancer_name(cancer):
    if cancer.endswith("-meta"):
        return cancer.replace("-meta", "")
    return cancer


def build_patient_file_map(base_dir, file_suffix):

    cancer_map = defaultdict(dict)

    for cancer_type in os.listdir(base_dir):

        cancer_path = os.path.join(base_dir, cancer_type)
        if not os.path.isdir(cancer_path):
            continue

        metadata_file = None
        for f in os.listdir(cancer_path):
            if f.startswith("metadata.repository") and f.endswith(".json"):
                metadata_file = os.path.join(cancer_path, f)
                break

        if metadata_file is None:
            continue

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        for entry in metadata:

            file_name = entry.get("file_name")
            file_id = entry.get("file_id")

            if not file_name or not file_id:
                continue

            if not file_name.endswith(file_suffix):
                continue

            full_path = os.path.join(cancer_path, file_id, file_name)

            if not os.path.exists(full_path):
                continue

            for ent in entry.get("associated_entities", []):
                esid = ent.get("entity_submitter_id")

                if esid and esid.startswith("TCGA"):
                    patient_id = to_patient_id(esid)
                    cancer_map[cancer_type][patient_id] = full_path

    return cancer_map


# ======================================================
# 1️⃣ MAP 생성
# ======================================================

print("🔍 Building TRAIN maps...")
meth_txt_map = build_patient_file_map(METH_DIR, ".txt")
mirna_map = build_patient_file_map(MIRNA_DIR, "mirnas.quantification.txt")

print("🔍 Building META maps...")
meth_meta_map = build_patient_file_map(METH_META_DIR, ".txt")
mirna_meta_map = build_patient_file_map(MIRNA_META_DIR, "mirnas.quantification.txt")


# ======================================================
# 2️⃣ 🔥 전체 cancer set으로 label_dict 생성
# ======================================================

train_cancers = set(meth_txt_map.keys()) & set(mirna_map.keys())
meta_cancers = set(meth_meta_map.keys()) & set(mirna_meta_map.keys())

meta_cancers_norm = set(normalize_cancer_name(c) for c in meta_cancers)

all_cancers = sorted(train_cancers | meta_cancers_norm)

label_dict = {cancer: i for i, cancer in enumerate(all_cancers)}

print("🔥 ALL cancers:", all_cancers)


# ======================================================
# 3️⃣ TRAIN samples
# ======================================================

train_samples = []

for cancer in train_cancers:

    common_patients = set(meth_txt_map[cancer]) & set(mirna_map[cancer])

    for patient_id in common_patients:

        txt_path = meth_txt_map[cancer][patient_id]
        file_id = os.path.basename(txt_path).split(".")[0]

        png_path = os.path.join(METH_PNG_DIR, cancer, file_id + ".png")
        if not os.path.exists(png_path):
            continue

        train_samples.append({
            "patient_id": patient_id,
            "cancer_type": cancer,
            "label": label_dict[cancer],
            "methylation_path": png_path,
            "mirna_path": mirna_map[cancer][patient_id]
        })

print(f"✅ TRAIN samples: {len(train_samples)}")

with open(TRAIN_OUTPUT, "wb") as f:
    pickle.dump({
        "samples": train_samples,
        "label_dict": label_dict
    }, f)


# ======================================================
# 4️⃣ META samples (🔥 이제 skip 없음)
# ======================================================

meta_samples = []

for cancer in meta_cancers:

    base_cancer = normalize_cancer_name(cancer)

    common_patients = set(meth_meta_map[cancer]) & set(mirna_meta_map[cancer])

    print(f"{cancer} → {base_cancer} patients:", len(common_patients))

    for patient_id in common_patients:

        txt_path = meth_meta_map[cancer][patient_id]
        file_id = os.path.basename(txt_path).split(".")[0]

        png_path = os.path.join(METH_META_PNG_DIR, cancer, file_id + ".png")
        if not os.path.exists(png_path):
            continue

        meta_samples.append({
            "patient_id": patient_id,
            "cancer_type": base_cancer,
            "label": label_dict[base_cancer],  # 🔥 항상 존재
            "methylation_path": png_path,
            "mirna_path": mirna_meta_map[cancer][patient_id]
        })

print(f"\n✅ META samples: {len(meta_samples)}")

with open(META_OUTPUT, "wb") as f:
    pickle.dump({
        "samples": meta_samples,
        "label_dict": label_dict
    }, f)

print(f"💾 Saved META index → {META_OUTPUT}")