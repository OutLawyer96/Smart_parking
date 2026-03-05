"""
Fine-tunes the best model from the original training on the 100-image dataset.
Starts from runs/detect/train2/weights/best.pt so we keep what the model
already learned about vehicle shapes.
"""

from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "runs/detect/train2/weights/best.pt"   # pretrained weights
DATA_YAML   = "dataset_finetune/data.yaml"
EPOCHS      = 50      # small dataset → fewer epochs needed
IMG_SIZE    = 640
BATCH       = 8       # lower batch for small dataset
PROJECT     = "runs/detect/runs/finetune"
RUN_NAME    = "real_cars_v1"
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO(BASE_MODEL)

results = model.train(
    data      = DATA_YAML,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH,
    project   = PROJECT,
    name      = RUN_NAME,
    patience  = 15,     # stop early if no improvement for 15 epochs
    lr0       = 0.001,  # lower LR for fine-tuning
    lrf       = 0.01,
    freeze    = 10,     # freeze first 10 backbone layers, fine-tune head only
    verbose   = True,
)

print(f"\nBest weights saved to: {results.save_dir}/weights/best.pt")
