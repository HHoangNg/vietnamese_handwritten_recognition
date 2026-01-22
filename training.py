import os
import json
import numpy as np
import tensorflow as tf

from model import (
    build_training_model,
    build_predict_model,
    max_label_len,
    FEATURE_MAP_WIDTH
)

from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint,
    EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# ===== GPU =====
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU DETECTED: {len(gpus)} device(s)")
else:
    print("WARNING: No GPU found → training will use CPU")

# ===== CHAR LIST (PHẢI GIỐNG MODEL) =====
with open("charset.txt", "r", encoding="utf-8") as f:
    CHAR_LIST = [line.rstrip("\n").replace("<space>", " ") for line in f]

# blank = 0
CHAR_DICT = {c: i + 1 for i, c in enumerate(CHAR_LIST)}

BLANK_INDEX = 0
NUM_CLASSES = len(CHAR_DICT) + 1
NUM_CHARS = len(CHAR_DICT)          
print(CHAR_DICT)
print("NUM CHARS:", NUM_CHARS)
print("BLANK_INDEX:", BLANK_INDEX)
print("NUM CLASSES:", NUM_CLASSES)


# ===== Encode label =====
def encode_label(text, fn=""):
    encoded = []

    for c in text:
        if c not in CHAR_DICT:
            raise ValueError(f"Ký tự không có trong charset: '{c}' | file={fn}")

        idx = CHAR_DICT[c]

        # CTC cấm BLANK xuất hiện trong label
        if idx == BLANK_INDEX:
            raise ValueError(f"label chứa BLANK: {fn} → {text}")

        encoded.append(idx)

    return np.array(encoded, dtype=np.int32)




# ===== Load labels =====
with open("train_labels.json", "r", encoding="utf-8") as f:
    labels_dict = json.load(f)


# ===== DATA GENERATOR =====
class HandwritingGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_folder, labels_dict, batch_size, shuffle=True):
        self.img_folder = img_folder
        self.labels_dict = labels_dict
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.filenames = sorted([
            f for f in os.listdir(img_folder)
            if f.lower().endswith((".png", ".jpg"))
        ])

        self.indexes = np.arange(len(self.filenames))
        self.on_epoch_end()

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_files = [self.filenames[i] for i in batch_idx]

        images, texts, label_lens = [], [], []

        for fn in batch_files:
            # ===== LOAD IMAGE =====
            img_path = os.path.join(self.img_folder, fn)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            images.append(img)

            # ===== LABEL =====
            text = self.labels_dict[fn]
            enc = encode_label(text)
            enc = enc[:max_label_len]
            
            # SAU ĐÓ MỚI LƯU
            texts.append(enc)
            label_lens.append(len(enc))


        # ===== PAD LABELS (blank = 0) =====
        padded_labels = np.zeros(
            (len(texts), max_label_len),
            dtype=np.int32
        )

        for i, lab in enumerate(texts):
            lab = lab[:max_label_len]
            padded_labels[i, :len(lab)] = lab

        images = tf.stack(images, axis=0)

        input_length = np.full(
            (len(images), 1),
            FEATURE_MAP_WIDTH,
            dtype=np.int32
        )

        label_length = np.array(label_lens, dtype=np.int32).reshape(-1, 1)

        return (
            {
                "input_img": images,
                "the_labels": padded_labels,
                "input_length": input_length,
                "label_length": label_length
            },
            {}  # CTC add_loss → không cần y_true
        )


# ===== CALLBACKS =====
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

callbacks = [
    TensorBoard(log_dir="logs/tensorboard", update_freq="epoch"),
    CSVLogger("logs/training_log.csv", append=True),

    ModelCheckpoint(
        "checkpoints/epoch_{epoch:03d}.h5",
        save_weights_only=True
    ),

    ModelCheckpoint(
        "checkpoints/best_weights.h5",
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3
    )
]

# ===== TRAIN =====
batch_size = 8

train_dataset = HandwritingGenerator(
    "processed_images/train",
    labels_dict,
    batch_size,
    shuffle=True
)

val_dataset = HandwritingGenerator(
    "processed_images/validation",
    labels_dict,
    batch_size,
    shuffle=False
)

model = build_training_model()


model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# ===== SAVE FINAL =====
model.save_weights("final_model.h5")

predict_model = build_predict_model()
predict_model.set_weights(model.get_weights())
predict_model.save("final_predict_model.h5")

print("TRAINING DONE!")
# Adapted from:
# https://github.com/TomHuynhSG/Vietnamese-Handwriting-Recognition-OCR
# Original Author: Tom Huynh
# License: MIT
# Modified by: Hoang Nguyen
