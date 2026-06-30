import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import csv

from multimodal_dataset import MultimodalDataset
from multimodal_model import MultimodalModel
from image_dataset import get_sup_transform
from config import NUM_WORKERS

def compute_multiclass_mauc(probs, labels):

    classes = np.unique(labels)
    pairwise = []
    scores = []

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):

            c1 = classes[i]
            c2 = classes[j]

            mask = (labels == c1) | (labels == c2)

            y = labels[mask]
            p = probs[mask]

            if len(np.unique(y)) < 2:
                continue

            auc1 = roc_auc_score((y == c1).astype(int), p[:, c1])
            auc2 = roc_auc_score((y == c2).astype(int), p[:, c2])

            scores.append((auc1 + auc2) / 2)

    if len(scores) == 0:
        return float("nan")

    return float(np.mean(scores))

def get_device():

    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")

    else:
        print("Using CPU")
        return torch.device("cpu")

def evaluate_meta():

    device = get_device()

    BASE_DIR = "/Users/annkim/PycharmProjects/CUP_Project_Journal"

    dataset = MultimodalDataset(

        index_path=os.path.join(BASE_DIR, "processed/meta_multimodal_index.pkl"),

        mirna_feature_path=os.path.join(
            BASE_DIR,
            "processed/processed_meta_mirna_all.npy"
        ),

        mirna_id_path=os.path.join(
            BASE_DIR,
            "processed/processed_meta_mirna_ids.npy"
        ),

        transform=get_sup_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    NUM_CLASSES = len(dataset.classes)

    print("META samples:", len(dataset))
    print("NUM_CLASSES:", NUM_CLASSES)

    all_fold_probs = []

    for fold in range(1, 6):

        print(f"\nLoading fold {fold} model")

        model = MultimodalModel(
            num_classes=NUM_CLASSES,
            mirna_input_dim=dataset.mirna_features.shape[1],
        ).to(device)

        ckpt = torch.load(
            f"multimodal_best_fold{fold}.pt",
            map_location=device
        )

        model.load_state_dict(ckpt)

        mirna_ssl = torch.load(
            "mirna_ssl_encoder_meta.pt",
            map_location=device
        )

        model.mirna_encoder.load_state_dict(
            mirna_ssl,
            strict=False
        )

        model.eval()

        probs_list = []

        with torch.no_grad():

            for img, mirna, _ in tqdm(loader):

                img = img.to(device)
                mirna = mirna.to(device)

                logits = model(img, mirna)

                probs = torch.softmax(logits, dim=1)

                probs_list.append(
                    probs.cpu().numpy()
                )

        probs = np.concatenate(probs_list)

        all_fold_probs.append(probs)

    final_probs = np.mean(all_fold_probs, axis=0)

    labels = np.array([s["label"] for s in dataset.samples])

    preds = final_probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)

    mauc = compute_multiclass_mauc(final_probs, labels)

    print("\n==============================")
    print("META FINAL RESULT")
    print("==============================")
    print("Accuracy:", acc)
    print("mAUC:", mauc)

    with open("meta_multimodal_results.csv", "w") as f:

        writer = csv.writer(f)

        writer.writerow(["sample", "label", "prediction"])

        for i in range(len(labels)):

            writer.writerow([
                dataset.samples[i]["patient_id"],
                labels[i],
                preds[i],
            ])

    print("Saved meta_multimodal_results.csv")


if __name__ == "__main__":

    evaluate_meta()
