import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# 경로 설정
# =========================
METH_DIR = "/Volumes/T7/CUP Project Data/Methylation array Datasets"
MIRNA_DIR = "/Volumes/T7/CUP Project Data/miRNA-seq Datasets"
META_MIRNA_DIR = "/Volumes/T7/CUP Project Data/TCGA-meta-miRNA"

SAVE_DIR = "./processed"
os.makedirs(SAVE_DIR, exist_ok=True)

def to_patient_id(barcode):
    return "-".join(barcode.split("-")[:3])

def clr_transform(x):
    x = x + 1e-6
    log_x = np.log(x)
    return log_x - np.mean(log_x)

# =========================
# 1️⃣ Methylation 환자 추출
# =========================
def extract_patients(base_dir):
    patients = set()

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
            data = json.load(f)

        for entry in data:
            for ent in entry.get("associated_entities", []):
                esid = ent.get("entity_submitter_id")
                if esid and esid.startswith("TCGA"):
                    patients.add(to_patient_id(esid))

    return patients


print("🔍 Extracting methylation patients...")
meth_patients = extract_patients(METH_DIR)

# =========================
# 2️⃣ miRNA metadata 매핑
# =========================
print("🔍 Mapping miRNA file_id to patients...")

fileid_to_patient = {}
mirna_patients = set()

for cancer_type in os.listdir(MIRNA_DIR):
    cancer_path = os.path.join(MIRNA_DIR, cancer_type)
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
        data = json.load(f)

    for entry in data:
        file_id = entry.get("file_id")

        for ent in entry.get("associated_entities", []):
            esid = ent.get("entity_submitter_id")
            if esid and esid.startswith("TCGA"):
                pid = to_patient_id(esid)
                fileid_to_patient[file_id] = pid
                mirna_patients.add(pid)

# =========================
# 3️⃣ 공통 환자 계산
# =========================
common_patients = meth_patients & mirna_patients

print(f"Total methylation patients: {len(meth_patients)}")
print(f"Total miRNA patients: {len(mirna_patients)}")
print(f"Common patients: {len(common_patients)}")

if len(common_patients) == 0:
    raise RuntimeError("❌ No overlapping patients found.")

# =========================
# 4️⃣ TCGA miRNA expression 로딩
# =========================
print("📦 Loading TCGA miRNA expression...")

patient_to_vector = {}

for cancer_type in os.listdir(MIRNA_DIR):
    cancer_path = os.path.join(MIRNA_DIR, cancer_type)
    if not os.path.isdir(cancer_path):
        continue

    for root, _, files in os.walk(cancer_path):

        file_id = os.path.basename(root)

        if file_id not in fileid_to_patient:
            continue

        patient_id = fileid_to_patient[file_id]

        if patient_id not in common_patients:
            continue

        for f in files:
            if not f.endswith("mirnas.quantification.txt"):
                continue

            file_path = os.path.join(root, f)

            try:
                df = pd.read_csv(file_path, sep="\t")

                if "read_count" in df.columns:
                    values = df["read_count"].values.astype(np.float32)
                else:
                    values = df.iloc[:, -1].values.astype(np.float32)

                values = np.log2(values + 1.0)
                values = clr_transform(values)

                if patient_id not in patient_to_vector:
                    patient_to_vector[patient_id] = values

            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue

print("🧮 Converting TCGA to numpy...")

all_patient_ids = sorted(patient_to_vector.keys())
all_vectors = [patient_to_vector[pid] for pid in all_patient_ids]
X = np.stack(all_vectors, axis=0)

np.save(os.path.join(SAVE_DIR, "processed_mirna_all.npy"), X)
np.save(os.path.join(SAVE_DIR, "processed_mirna_ids.npy"), np.array(all_patient_ids))

print(f"✅ TCGA saved: {X.shape}")

# =========================
# 5️⃣ META miRNA 동일 방식 처리
# =========================
print("\n🔵 Loading META miRNA expression...")

meta_patient_to_vector = {}

for root, _, files in os.walk(META_MIRNA_DIR):

    for f in files:
        if not f.endswith("mirnas.quantification.txt"):
            continue

        file_path = os.path.join(root, f)

        try:
            df = pd.read_csv(file_path, sep="\t")

            if "read_count" in df.columns:
                values = df["read_count"].values.astype(np.float32)
            else:
                values = df.iloc[:, -1].values.astype(np.float32)

            values = np.log2(values + 1.0)
            values = clr_transform(values)

            # 파일명에서 patient id 추출
            patient_id = to_patient_id(f)

            if patient_id not in meta_patient_to_vector:
                meta_patient_to_vector[patient_id] = values

        except Exception as e:
            print(f"⚠️ META Error reading {file_path}: {e}")
            continue

print("🧮 Converting META to numpy...")

meta_ids = sorted(meta_patient_to_vector.keys())
meta_vectors = [meta_patient_to_vector[pid] for pid in meta_ids]
X_meta = np.stack(meta_vectors, axis=0)

np.save(os.path.join(SAVE_DIR, "processed_meta_mirna_all.npy"), X_meta)
np.save(os.path.join(SAVE_DIR, "processed_meta_mirna_ids.npy"), np.array(meta_ids))

print(f"✅ META saved: {X_meta.shape}")

print("\n🎉 ALL DONE")
