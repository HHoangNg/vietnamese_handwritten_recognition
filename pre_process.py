import cv2
import numpy as np
import os
from pathlib import Path
import json

# === CẤU HÌNH ===
raw_train_dir = 'split/train'
raw_val_dir = 'split/validation'
raw_test_dir = 'split/test'

processed_train_dir = 'processed_images/train'
processed_val_dir = 'processed_images/validation'
processed_test_dir = 'processed_images/test'

labels_json_path = 'train_labels.json'

HEIGHT = 118
PADDING_WIDTH = 2167
TIME_STEPS = 512

with open("charset.txt", "r", encoding="utf-8") as f:
    CHAR_LIST = [line.rstrip("\n").replace("<space>", " ") for line in f]

CHAR_DICT = {c: i + 1 for i, c in enumerate(CHAR_LIST)}  # start from 1
print(CHAR_DICT)
# === TẠO THƯ MỤC ===
for p in [processed_train_dir, processed_val_dir, processed_test_dir]:
    Path(p).mkdir(parents=True, exist_ok=True)

# === ĐỌC labels.json ===
def load_labels(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === MÃ HÓA NHÃN ===
def encode_to_labels(text):
    return np.array([CHAR_DICT[c] for c in text if c in CHAR_DICT], dtype=np.int32)

# === XỬ LÝ DATASET ===
def process_dataset(image_paths, processed_dir, labels_dict):
    i = 0
    skipped = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        if filename not in labels_dict:
            skipped += 1
            continue

        label = labels_dict[filename]
        encoded = encode_to_labels(label)
        if len(encoded) == 0 or len(encoded) > TIME_STEPS:
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape

        new_w = int(HEIGHT / h * w)
        img = cv2.resize(img, (new_w, HEIGHT), interpolation=cv2.INTER_CUBIC)

        if new_w < PADDING_WIDTH:
            img = np.pad(img, ((0, 0), (0, PADDING_WIDTH - new_w)), mode='median')
        else:
            img = img[:, :PADDING_WIDTH]

        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 4
        )

        output_path = os.path.join(processed_dir, filename)
        cv2.imwrite(output_path, img)

        i += 1
        if i % 500 == 0:
            print(f"  → Đã xử lý {i} ảnh...")

    print(f"HOÀN TẤT {processed_dir} | Đã xử lý: {i} | Bỏ qua: {skipped}")

# === CHẠY ===
labels_dict = load_labels(labels_json_path)

train_paths = [os.path.join(raw_train_dir, f) for f in os.listdir(raw_train_dir) if f.lower().endswith(('.png', '.jpg'))]
val_paths = [os.path.join(raw_val_dir, f) for f in os.listdir(raw_val_dir) if f.lower().endswith(('.png', '.jpg'))]
test_paths = [os.path.join(raw_test_dir, f) for f in os.listdir(raw_test_dir) if f.lower().endswith(('.png', '.jpg'))]

process_dataset(train_paths, processed_train_dir, labels_dict)
process_dataset(val_paths, processed_val_dir, labels_dict)
process_dataset(test_paths, processed_test_dir, labels_dict)

print("HOÀN TẤT!")

# padding labels = TIME STEP