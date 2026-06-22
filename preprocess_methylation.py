# preprocess_methylation.py

import os
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import BASE_DIR, IMG_ROOT, TEST_BASE_DIR, TEST_IMG_ROOT  # config에서 경로를 가져옵니다


def txt_to_image(txt_path: str) -> Image.Image | None:
    """
    하나의 methylation array txt를 (sqrt(N) x sqrt(N)) 그레이스케일 이미지로 변환.
    값은 0~255로 정규화.
    """
    try:
        betas = pd.read_csv(
            txt_path,
            sep="\t",
            index_col=0,
            header=None,
            names=["probe", "beta"],
        )
        betas_array = np.nan_to_num(
            betas["beta"].values.astype(float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    except Exception as e:
        print(f"[Error] {txt_path}: {e}")
        return None

    side = int(math.ceil(math.sqrt(len(betas_array))))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[: len(betas_array)] = betas_array
    img_array = padded.reshape((side, side))

    # 0~255 정규화
    minv, maxv = img_array.min(), img_array.max()
    if maxv - minv > 0:
        img_norm = (img_array - minv) / (maxv - minv)
    else:
        img_norm = img_array
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def preprocess_all(base_dir: str, output_dir: str, filename_filter: str = ".methylation_array.sesame.level3betas.txt") -> int:
    """
    base_dir 아래 각 cancer type 폴더를 순회하면서
    filename_filter로 끝나는 txt 파일들을 png로 변환.
    반환값: 저장한 이미지 개수
    """
    if not os.path.exists(base_dir):
        print(f"[Preprocess] Warning: base_dir does not exist: {base_dir}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    processed = 0

    cancer_types = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if len(cancer_types) == 0:
        print(f"[Preprocess] No subdirectories found in {base_dir}.")
        return 0

    for cancer_type in cancer_types:
        cancer_dir = os.path.join(base_dir, cancer_type)
        out_sub = os.path.join(output_dir, cancer_type)
        os.makedirs(out_sub, exist_ok=True)
        print(f"[Preprocess] Processing {cancer_type}")

        # os.walk로 하위 디렉토리까지 모두 탐색
        for root, _, files in os.walk(cancer_dir):
            for fname in tqdm(files, desc=f"{cancer_type}", leave=False):
                if fname.startswith("._"):
                    continue
                if not fname.endswith(filename_filter):
                    continue

                txt_path = os.path.join(root, fname)
                img = txt_to_image(txt_path)
                if img is None:
                    continue

                out_name = fname.replace(
                    filename_filter,
                    ".png",
                )
                save_path = os.path.join(out_sub, out_name)
                if not os.path.exists(save_path):
                    try:
                        img.save(save_path)
                        processed += 1
                    except Exception as e:
                        print(f"[Error] saving {save_path}: {e}")

    print(f"[Preprocess] Done for {base_dir}. Saved {processed} images to {output_dir}.")
    return processed


def count_pngs(root_dir: str) -> int:
    total = 0
    if not os.path.exists(root_dir):
        return 0
    for _, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".png"):
                total += 1
    return total


if __name__ == "__main__":
    # 1) 먼저 TRAIN (CUP) dataset 처리
    print("=== Start preprocessing TRAIN (CUP) dataset ===")
    saved_train = preprocess_all(BASE_DIR, IMG_ROOT)

    # 2) 다음으로 TEST (TCGA-meta) dataset 처리
    print("\n=== Start preprocessing TEST (TCGA-meta) dataset ===")
    saved_test = preprocess_all(TEST_BASE_DIR, TEST_IMG_ROOT)

    # 3) 요약 출력: 디스크에 실제로 저장된 png 수 확인
    print("\n=== Summary ===")
    print(f"Requested saved images (train): {saved_train}")
    print(f"Actual PNG files in {IMG_ROOT}: {count_pngs(IMG_ROOT)}")
    print(f"Requested saved images (test): {saved_test}")
    print(f"Actual PNG files in {TEST_IMG_ROOT}: {count_pngs(TEST_IMG_ROOT)}")
