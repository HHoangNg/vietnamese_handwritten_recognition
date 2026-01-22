# evaluate_final.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import unicodedata
import string
import editdistance
from model import build_predict_model, SqueezeLayer

# ================== CẤU HÌNH ==================
IMG_H = 118
IMG_W = 2167

CHARSET_PATH = "charset.txt"
VAL_IMG_FOLDER = "processed_images/test"
VAL_LABEL_FILE = "train_labels.json"  # JSON: {"filename": "text"}
FINAL_MODEL_WEIGHTS = "final_model.h5"
BATCH_SIZE = 8

#  LOAD CHARSET 
with open(CHARSET_PATH, "r", encoding="utf-8") as f:
    CHAR_LIST = [line.rstrip("\n").replace("<space>", " ") for line in f]
IDX2CHAR = {i+1: c for i, c in enumerate(CHAR_LIST)}  # blank=0

# BUILD PREDICT MODEL 
print("Loading predict model")
predict_model = build_predict_model()
predict_model.load_weights(FINAL_MODEL_WEIGHTS)
print("Model loaded.")

# LOAD VAL DATA 
with open(VAL_LABEL_FILE, "r", encoding="utf-8") as f:
    val_labels_dict = json.load(f)

# Đọc tên ảnh từ folder trước, giữ thứ tự alphabetically
val_filenames = sorted([f for f in os.listdir(VAL_IMG_FOLDER) if f.lower().endswith((".png",".jpg"))])
# Lấy nhãn thật từ JSON theo tên ảnh
val_texts = [val_labels_dict[f] for f in val_filenames]

# LOAD ẢNH ĐÃ PREPROCESS 
def load_processed_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # chuyển uint8 -> float32 [0,1]
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 1)
    return img

# BATCH GENERATOR
def batch_generator(file_list, folder, batch_size=BATCH_SIZE):
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        batch_imgs = [load_processed_image(os.path.join(folder, f))[0] for f in batch_files]
        yield tf.stack(batch_imgs, axis=0), batch_files

# CTC DECODE 
def ctc_greedy_decode(y_pred):
    time_steps = y_pred.shape[1]
    input_length = np.ones(y_pred.shape[0]) * time_steps
    decoded, _ = K.ctc_decode(y_pred, input_length=input_length, greedy=True)
    decoded = decoded[0].numpy()
    texts = []
    for seq in decoded:
        text = ""
        for idx in seq:
            if idx < 1:
                continue
            text += IDX2CHAR.get(idx, "")
        texts.append(text)
    return texts

# PREDICT & GROUND TRUTH 
all_predictions = []
all_ground_truths = []

for batch_imgs, batch_files in batch_generator(val_filenames, VAL_IMG_FOLDER):
    y_pred = predict_model(batch_imgs, training=False)
    texts = ctc_greedy_decode(y_pred)
    all_predictions.extend(texts)
    all_ground_truths.extend([val_labels_dict[f] for f in batch_files])

# IN 5 MẪU ĐẦU
print("\n" + "="*60)
print("5 MẪU ĐẦU TIÊN: NHÃN THẬT vs DỰ ĐOÁN")
print("="*60)
for i in range(min(5, len(all_ground_truths))):
    print(f"\nMẫu {i+1}:")
    print(f"   Ảnh: {val_filenames[i]}")
    print(f"   Thật: {all_ground_truths[i]}")
    print(f"   Dự đoán: {all_predictions[i]}")
print("="*60)

#TÍNH CER / WER / SER
def ocr_metrics(predicts, ground_truth, norm_accentuation=True, norm_punctuation=True):
    cer, wer, ser = [], [], []
    for pd, gt in zip(predicts, ground_truth):
        if norm_accentuation:
            pd = unicodedata.normalize("NFKD", pd).encode("ASCII", "ignore").decode()
            gt = unicodedata.normalize("NFKD", gt).encode("ASCII", "ignore").decode()
        if norm_punctuation:
            pd = pd.translate(str.maketrans("", "", string.punctuation))
            gt = gt.translate(str.maketrans("", "", string.punctuation))

        pd_c, gt_c = list(pd.lower()), list(gt.lower())
        cer.append(editdistance.eval(pd_c, gt_c) / max(len(gt_c), 1))

        pd_w, gt_w = pd.lower().split(), gt.lower().split()
        wer.append(editdistance.eval(pd_w, gt_w) / max(len(gt_w), 1))

        ser.append(int(pd.lower() != gt.lower()))
    return np.mean(cer), np.mean(wer), np.mean(ser)

cer, wer, ser = ocr_metrics(all_predictions, all_ground_truths)
print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
print(f"SER: {ser:.4f} ({ser*100:.2f}%)")
print(f"Số mẫu: {len(all_predictions)}")

# LƯU KẾT QUẢ 
with open('evaluation_result.txt', 'w', encoding='utf-8') as f:
    f.write(f"CER: {cer:.4f}\nWER: {wer:.4f}\nSER: {ser:.4f}\nSamples: {len(all_predictions)}")
print("\n✅ ĐÃ LƯU: evaluation_result.txt")
