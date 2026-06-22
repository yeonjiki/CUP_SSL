# config.py

import os

# ==========================================================
# 1️⃣ Methylation (기존 - 절대 수정 X)
# ==========================================================

BASE_DIR = "/Volumes/T7/CUP Project Data/Methylation array Datasets"
TEST_BASE_DIR = "/Volumes/T7/CUP Project Data/TCGA-meta"

IMG_ROOT = "./methylation_images"
TEST_IMG_ROOT = "./methylation_images_tcga"

os.makedirs(IMG_ROOT, exist_ok=True)
os.makedirs(TEST_IMG_ROOT, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.8
NUM_WORKERS = 4
IMG_SIZE = 224

SSL_BATCH_SIZE = 32
SSL_EPOCHS = 20
SSL_LR = 3e-4
SSL_TEMPERATURE = 0.5
SSL_CHECKPOINT = "vit_ssl_cup_encoder.pth"

SUP_BATCH_SIZE = 32
SUP_EPOCHS = 20
SUP_LR = 3e-4
SUP_CHECKPOINT = "vit_ssl_cup_classifier.pth"


# ==========================================================
# 2️⃣ miRNA 설정
# ==========================================================

# 🔥 파일 패턴
MIRNA_FILE_SUFFIX = "mirnas.quantification.txt"

# 🔥 실제 RPM 컬럼명 (TCGA 기준)
MIRNA_RPM_COLUMN = "reads_per_million_miRNA_mapped"

# CUP miRNA-seq 데이터 경로
MIRNA_BASE_DIR = "/Volumes/T7/CUP Project Data/miRNA-seq Datasets"

# TCGA-meta miRNA 데이터 경로
MIRNA_TEST_BASE_DIR = "/Volumes/T7/CUP Project Data/TCGA-meta-miRNA"

# 전처리 결과 저장 경로
MIRNA_PROCESSED_TRAIN = "./processed_mirna_train.npy"
MIRNA_PROCESSED_VAL   = "./processed_mirna_val.npy"
MIRNA_PROCESSED_TEST  = "./processed_mirna_test.npy"

# StandardScaler 저장 경로
MIRNA_SCALER_PATH = "./mirna_scaler.pkl"


# ==========================================================
# 3️⃣ Multimodal 학습 설정
# ==========================================================

MM_BATCH_SIZE = 32
MM_EPOCHS = 20
MM_LR = 3e-4
MM_CHECKPOINT = "vit_multimodal_classifier.pth"

# ------------------------
# Multimodal Training
# ------------------------

MIRNA_SSL_CHECKPOINT = "checkpoints/mirna_ssl.pt"  # 없으면 None으로 둬도 됨

MULTIMODAL_EPOCHS = 30
MULTIMODAL_BATCH_SIZE = 16
MULTIMODAL_LR = 1e-4

NUM_CLASSES = 5   # ← 네 cancer class 개수로 수정

MULTIMODAL_INDEX_PATH = "path/to/multimodal_index.pkl"

MIRNA_FEATURE_TRAIN_PATH = "path/to/processed_mirna_train.npy"
MIRNA_ID_TRAIN_PATH = "path/to/processed_mirna_train_ids.npy"
