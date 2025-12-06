import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

TRAIN_IMG = "train_img"
TRAIN_MASK = "train_lab"
TEST_IMG = "test_img"
TEST_MASK = "test_lab"

OUT = "yolo_dataset_fixed"

(Path(OUT) / "images/train").mkdir(parents=True, exist_ok=True)
(Path(OUT) / "images/val").mkdir(parents=True, exist_ok=True)
(Path(OUT) / "labels/train").mkdir(parents=True, exist_ok=True)
(Path(OUT) / "labels/val").mkdir(parents=True, exist_ok=True)

def process_split(img_dir, mask_dir, split):
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    for img_name in tqdm(img_files, desc=f"Converting {split}"):

        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.rsplit('.',1)[0] + ".png")

        out_img = os.path.join(OUT, f"images/{split}", img_name)
        out_label = os.path.join(OUT, f"labels/{split}", img_name.rsplit('.',1)[0] + ".txt")

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        cv2.imwrite(out_img, img)

        if not os.path.isfile(mask_path):
            open(out_label, "w").close()
            continue

        mask = cv2.imread(mask_path, 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Thicken thin cracks
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with open(out_label, "w") as f:
            for cnt in contours:
                if cv2.contourArea(cnt) < 20:
                    continue

                poly = cnt.reshape(-1, 2)
                poly_norm = []

                for x, y in poly:
                    poly_norm.append(x / w)
                    poly_norm.append(y / h)

                if len(poly_norm) >= 6:
                    f.write("0 " + " ".join([str(round(p, 6)) for p in poly_norm]) + "\n")


process_split(TRAIN_IMG, TRAIN_MASK, "train")
process_split(TEST_IMG,  TEST_MASK,  "val")

print("\nDONE! YOLO-ready dataset saved to:", OUT)
