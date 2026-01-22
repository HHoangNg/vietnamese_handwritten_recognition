# split_data.py
import os
import shutil
import random
from pathlib import Path

# === CẤU HÌNH ĐƯỜNG DẪN ===
data_folder = 'data'     # Folder chứa toàn bộ ảnh .png
train_folder = 'train'   # 80%
val_folder = 'validation'       # 10%
test_folder = 'test'     # 10%

# Tạo folder nếu chưa có
Path(train_folder).mkdir(exist_ok=True)
Path(val_folder).mkdir(exist_ok=True)
Path(test_folder).mkdir(exist_ok=True)

# === LẤY DANH SÁCH ẢNH ===
image_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.png')]
print(f"Tìm thấy {len(image_files)} ảnh .png")

if len(image_files) == 0:
    print("Không tìm thấy ảnh .png nào!")
    exit()

# === CHIA NGẪU NHIÊN ===
random.seed(42)  # đảm bảo kết quả reproducible
random.shuffle(image_files)

n_total = len(image_files)
train_end = int(0.8 * n_total)
val_end = int(0.9 * n_total)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

print(f"Train: {len(train_files)} ảnh → {train_folder}/")
print(f"Val:   {len(val_files)} ảnh → {val_folder}/")
print(f"Test:  {len(test_files)} ảnh → {test_folder}/")

# === COPY ẢNH VÀO FOLDER MỚI ===
def copy_files(file_list, dest_folder):
    for file in file_list:
        src = os.path.join(data_folder, file)
        dst = os.path.join(dest_folder, file)
        shutil.copy2(src, dst)  # giữ metadata

copy_files(train_files, train_folder)
copy_files(val_files, val_folder)
copy_files(test_files, test_folder)

print("HOÀN TẤT! Dữ liệu đã được chia train / val / test.")
