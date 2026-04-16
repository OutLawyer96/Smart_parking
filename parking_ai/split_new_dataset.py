from pathlib import Path
import random
import shutil
import yaml

SRC_ROOT = Path('/Users/hitendrasingh/Desktop/smart_parking/Dataset/Smart Parking Yolov8/train')
SRC_IMAGES = SRC_ROOT / 'images'
SRC_LABELS = SRC_ROOT / 'labels'
OUT_ROOT = Path('/Users/hitendrasingh/Desktop/smart_parking/parking_ai/dataset_smart_parking')

SPLIT = {
    'train': 0.8,
    'val': 0.1,
    'test': 0.1,
}


def main() -> None:
    if not SRC_IMAGES.exists() or not SRC_LABELS.exists():
        raise SystemExit(f'Source dataset not found under: {SRC_ROOT}')

    images = sorted([p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if not images:
        raise SystemExit('No images found in source dataset')

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for split in SPLIT:
        (OUT_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT['train'])
    n_val = int(n * SPLIT['val'])

    split_items = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:],
    }

    for split, files in split_items.items():
        for img_path in files:
            stem = img_path.stem
            lbl_path = SRC_LABELS / f'{stem}.txt'

            shutil.copy2(img_path, OUT_ROOT / split / 'images' / img_path.name)

            target_label = OUT_ROOT / split / 'labels' / f'{stem}.txt'
            if lbl_path.exists():
                shutil.copy2(lbl_path, target_label)
            else:
                target_label.write_text('')

    data_yaml = {
        'path': str(OUT_ROOT),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['Car'],
    }
    (OUT_ROOT / 'data.yaml').write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print(f'Total images: {n}')
    print(
        'Split counts: '
        f"train={len(split_items['train'])}, "
        f"val={len(split_items['val'])}, "
        f"test={len(split_items['test'])}"
    )
    print(f'Dataset ready: {OUT_ROOT}')


if __name__ == '__main__':
    main()
