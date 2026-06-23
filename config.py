import os

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

MIRNA_FILE_SUFFIX = "mirnas.quantification.txt"

MIRNA_RPM_COLUMN = "reads_per_million_miRNA_mapped"

MIRNA_BASE_DIR = "/Volumes/T7/CUP Project Data/miRNA-seq Datasets"

MIRNA_TEST_BASE_DIR = "/Volumes/T7/CUP Project Data/TCGA-meta-miRNA"

MIRNA_PROCESSED_TRAIN = "./processed_mirna_train.npy"
MIRNA_PROCESSED_VAL   = "./processed_mirna_val.npy"
MIRNA_PROCESSED_TEST  = "./processed_mirna_test.npy"

MIRNA_SCALER_PATH = "./mirna_scaler.pkl"

MM_BATCH_SIZE = 32
MM_EPOCHS = 20
MM_LR = 3e-4
MM_CHECKPOINT = "vit_multimodal_classifier.pth"

MIRNA_SSL_CHECKPOINT = "checkpoints/mirna_ssl.pt" 

MULTIMODAL_EPOCHS = 30
MULTIMODAL_BATCH_SIZE = 16
MULTIMODAL_LR = 1e-4

NUM_CLASSES = 5  

MULTIMODAL_INDEX_PATH = "path/to/multimodal_index.pkl"

MIRNA_FEATURE_TRAIN_PATH = "path/to/processed_mirna_train.npy"
MIRNA_ID_TRAIN_PATH = "path/to/processed_mirna_train_ids.npy"
