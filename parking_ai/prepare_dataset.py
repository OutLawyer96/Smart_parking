import os
import shutil


SOURCE_DIR = r"C:\Users\savit\Downloads\Dataset"

DEST_DIR = "dataset"

train_file = os.path.join(SOURCE_DIR, "train.txt")
test_file = os.path.join(SOURCE_DIR, "test.txt")

for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "labels", split), exist_ok=True)

def move_files(file_list_path, split):
    with open(file_list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        img_path = line.strip()
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + ".txt"

        src_img = os.path.join(SOURCE_DIR, img_name)
        src_label = os.path.join(SOURCE_DIR, label_name)

        dst_img = os.path.join(DEST_DIR, "images", split, img_name)
        dst_label = os.path.join(DEST_DIR, "labels", split, label_name)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

# Move training files
move_files(train_file, "train")

# Move validation files
move_files(test_file, "val")

print("Dataset prepared successfully!")