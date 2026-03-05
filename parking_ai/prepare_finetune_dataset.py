"""
Copies images and labels from a Roboflow YOLOv8 export into dataset_finetune/.
Polygon annotations (YOLO segment format) are automatically converted to
axis-aligned bounding boxes.

Roboflow export structure:
    <SOURCE>/
        train/images/  train/labels/
        valid/images/  valid/labels/   (Roboflow uses 'valid', not 'val')
"""

import os
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DIR = r"C:\Users\savit\Downloads\My First Project.v1i.yolov8"
DEST_DIR   = "dataset_finetune"
# ─────────────────────────────────────────────────────────────────────────────


def polygon_to_bbox_line(line: str) -> str:
    """
    Convert a YOLO segmentation line to a YOLO detection bbox line.
    Bounding box  : class_id cx cy w h          (5 tokens)
    Polygon       : class_id x1 y1 x2 y2 ...   (>5 tokens)
    Returns the line unchanged if it is already a bbox.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return line  # malformed, leave as-is
    if len(parts) == 5:
        return line  # already a bounding box

    class_id = parts[0]
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]   # x values at even indices
    ys = coords[1::2]   # y values at odd  indices

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w  =  max_x - min_x
    h  =  max_y - min_y

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def convert_and_write_label(src_label: str, dst_label: str):
    with open(src_label, "r") as f:
        lines = f.readlines()
    converted = [polygon_to_bbox_line(l) for l in lines if l.strip()]
    with open(dst_label, "w") as f:
        f.writelines(converted)


# Roboflow names "valid", we store as "val"
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val":   "val",
}

copied    = {"train": 0, "val": 0}
converted = {"train": 0, "val": 0}

for src_split, dst_split in SPLIT_MAP.items():
    src_img_dir   = os.path.join(SOURCE_DIR, src_split, "images")
    src_label_dir = os.path.join(SOURCE_DIR, src_split, "labels")

    if not os.path.isdir(src_img_dir):
        continue  # this split doesn't exist in source

    dst_img_dir   = os.path.join(DEST_DIR, "images", dst_split)
    dst_label_dir = os.path.join(DEST_DIR, "labels", dst_split)

    os.makedirs(dst_img_dir,   exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    for fname in os.listdir(src_img_dir):
        src_img = os.path.join(src_img_dir, fname)
        dst_img = os.path.join(dst_img_dir, fname)
        shutil.copy2(src_img, dst_img)

        label_name = os.path.splitext(fname)[0] + ".txt"
        src_label  = os.path.join(src_label_dir, label_name)
        dst_label  = os.path.join(dst_label_dir, label_name)

        if os.path.exists(src_label):
            convert_and_write_label(src_label, dst_label)
            # Check if any polygon was converted
            with open(src_label) as f:
                raw = f.readlines()
            if any(len(l.strip().split()) > 5 for l in raw if l.strip()):
                converted[dst_split] += 1

        copied[dst_split] += 1

print(f"Copied     {copied['train']} train  |  {copied['val']} val  images")
print(f"Converted  {converted['train']} train  |  {converted['val']} val  polygon labels → bounding boxes")
