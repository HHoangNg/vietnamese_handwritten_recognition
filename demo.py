import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
from tensorflow.keras import backend as K
from model import build_predict_model, SqueezeLayer

# ================= CONFIG =================
HEIGHT = 118
PADDING_WIDTH = 2167

# ================= LOAD CHARSET =================
with open("charset.txt", "r", encoding="utf-8") as f:
    CHAR_LIST = [l.rstrip("\n").replace("<space>", " ") for l in f]
IDX2CHAR = {i + 1: c for i, c in enumerate(CHAR_LIST)}

# ================= LOAD MODEL =================
predict_model = build_predict_model()
predict_model.load_weights("final_model.h5")

# ================= PREPROCESS (GIỐNG TRAIN) =================
def preprocess_image_opencv(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    new_w = int(HEIGHT / h * w)
    img = cv2.resize(img, (new_w, HEIGHT), interpolation=cv2.INTER_CUBIC)

    if new_w < PADDING_WIDTH:
        img = np.pad(img, ((0, 0), (0, PADDING_WIDTH - new_w)), mode="median")
    else:
        img = img[:, :PADDING_WIDTH]

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )

    img_tensor = img.astype(np.float32) / 255.0
    img_tensor = img_tensor[np.newaxis, ..., np.newaxis]  # (1,118,2167,1)

    return img, img_tensor

# ================= CTC DECODE =================
def ctc_decode_prediction(y_pred):
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded, _ = K.ctc_decode(y_pred, input_length=input_len, greedy=True)
    decoded = decoded[0].numpy()

    text = ""
    for idx in decoded[0]:
        if idx > 0:
            text += IDX2CHAR[idx]
    return text

# ================= GUI =================
root = tk.Tk()
root.title("Handwriting OCR Demo")
root.geometry("900x750")

raw_img_tk = None
proc_img_tk = None

# ----- RESULT TEXT -----
result_var = tk.StringVar(value="Chọn ảnh RAW để nhận dạng")

tk.Label(
    root,
    textvariable=result_var,
    font=("Arial", 16),
    fg="blue",
    wraplength=850,
    justify="center"
).pack(pady=15)

# ----- IMAGE FRAME -----
img_frame = tk.Frame(root)
img_frame.pack(padx=10, pady=10)

tk.Label(img_frame, text="Ảnh gốc (RAW)", font=("Arial", 12)).pack()
raw_img_label = tk.Label(img_frame, bg="#eaeaea")
raw_img_label.pack(pady=5)

tk.Label(img_frame, text="Ảnh sau preprocess", font=("Arial", 12)).pack()
proc_img_label = tk.Label(img_frame, bg="#eaeaea")
proc_img_label.pack(pady=5)

# ----- FILE NAME (BOTTOM) -----
file_var = tk.StringVar(value="File: ---")
tk.Label(
    root,
    textvariable=file_var,
    font=("Arial", 11),
    fg="gray"
).pack(pady=5)

# ================= FUNCTION =================
def choose_image_and_predict():
    global raw_img_tk, proc_img_tk

    path = filedialog.askopenfilename(
        title="Chọn ảnh handwriting",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if not path:
        return

    filename = os.path.basename(path)
    file_var.set(f"File: {filename}")

    # ----- SHOW RAW IMAGE -----
    raw_pil = Image.open(path).convert("L")
    w, h = raw_pil.size
    new_w = 750
    new_h = int(h * new_w / w)
    raw_pil = raw_pil.resize((new_w, new_h))

    raw_img_tk = ImageTk.PhotoImage(raw_pil)
    raw_img_label.config(image=raw_img_tk)
    raw_img_label.image = raw_img_tk

    # ----- PREPROCESS -----
    img_proc, img_tensor = preprocess_image_opencv(path)

    proc_pil = Image.fromarray(img_proc)
    w, h = proc_pil.size
    new_w = 750
    new_h = int(h * new_w / w)
    proc_pil = proc_pil.resize((new_w, new_h))

    proc_img_tk = ImageTk.PhotoImage(proc_pil)
    proc_img_label.config(image=proc_img_tk)
    proc_img_label.image = proc_img_tk

    # ----- PREDICT -----
    y_pred = predict_model(img_tensor, training=False)
    text = ctc_decode_prediction(y_pred)

    result_var.set(f"Prediction:\n{text}")

# ----- BUTTON -----
tk.Button(
    root,
    text="📂 Chọn ảnh",
    font=("Arial", 14),
    width=20,
    command=choose_image_and_predict
).pack(pady=10)

root.mainloop()
