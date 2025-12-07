import os
import random
import shutil
from pathlib import Path

random.seed(42)

ROOT = Path("data")
SRC_CRACKED = ROOT / "cracked"
SRC_NON = ROOT / "non_cracked"

IMG_ROOT = ROOT / "images"
LBL_ROOT = ROOT / "labels"

TRAIN_IMG = IMG_ROOT / "train"
VAL_IMG = IMG_ROOT / "val"
TRAIN_LBL = LBL_ROOT / "train"
VAL_LBL = LBL_ROOT / "val"

for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

exts = [".jpg", ".jpeg", ".png", ".bmp"]
def list_images(folder):
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]

def process_folder(src_folder, label_for_noncrack=False):
    imgs = list_images(src_folder)
    random.shuffle(imgs)
    split = int(len(imgs) * 0.8)
    train = imgs[:split]
    val = imgs[split:]
    return train, val

train_cracked, val_cracked = process_folder(SRC_CRACKED)
train_non, val_non = process_folder(SRC_NON)

def move_and_create_labels(img_paths, dest_img_dir, dest_lbl_dir, is_cracked_group):
    for p in img_paths:
        dest_img = dest_img_dir / p.name
        shutil.copy2(p, dest_img)  
        lbl_name = dest_lbl_dir / (p.stem + ".txt")
        lbl_name.write_text("") 

move_and_create_labels(train_cracked, TRAIN_IMG, TRAIN_LBL, True)
move_and_create_labels(val_cracked, VAL_IMG, VAL_LBL, True)
move_and_create_labels(train_non, TRAIN_IMG, TRAIN_LBL, False)
move_and_create_labels(val_non, VAL_IMG, VAL_LBL, False)

print("Done copying images and creating label placeholders.")
print(f"Train images: {len(list(TRAIN_IMG.iterdir()))}, Val images: {len(list(VAL_IMG.iterdir()))}")

yaml_path = ROOT / "data.yaml"
yaml_content = f"""path: {ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: crack
"""
yaml_path.write_text(yaml_content)
print("Wrote", yaml_path)
