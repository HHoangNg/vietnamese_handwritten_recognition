import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, LSTM, Bidirectional,
    Activation, BatchNormalization, Add, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable


# ===== CONFIG =====
IMG_H = 118
IMG_W = 2167
TIME_STEPS = 512
FEATURE_MAP_WIDTH = 240  # 2167 → 723 → 240

# ===== CHAR LIST =====
with open("charset.txt", "r", encoding="utf-8") as f:
    CHAR_LIST = [line.rstrip("\n").replace("<space>", " ") for line in f]

CHAR_DICT = {c: i + 1 for i, c in enumerate(CHAR_LIST)}  # start from 1
NUM_CLASSES = len(CHAR_LIST) + 1  # + blank
max_label_len = TIME_STEPS


# ===== SQUEEZE LAYER =====
@register_keras_serializable()
class SqueezeLayer(Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


# ===== CTC LAYER (LOG METRIC) =====
@register_keras_serializable()
class CTCLayer(Layer):
    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        y_pred, labels, input_length, label_length = inputs
        loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        # add loss cho training
        self.add_loss(loss)

        # ghi metric để TensorBoard theo dõi
        self.add_metric(loss, name="ctc_loss", aggregation="mean")

        return y_pred

    def get_config(self):
        return super().get_config()


# ===== PREDICT MODEL =====
def build_predict_model():
    inputs = Input(shape=(IMG_H, IMG_W, 1), name="input_img")

    x = Conv2D(64, (3, 3), padding="same")(inputs)
    x = MaxPooling2D(pool_size=3, strides=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = MaxPooling2D(pool_size=3, strides=3)(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x3 = x

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x3])
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x5 = x

    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x5])
    x = Activation("relu")(x)

    x = Conv2D(1024, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)

    # (B, 1, T, C) → (B, T, C)
    x = SqueezeLayer(axis=1)(x)

    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2))(x)

    outputs = Dense(NUM_CLASSES+1, activation="softmax", name="softmax")(x)

    return Model(inputs, outputs, name="recognition_model")


# ===== TRAINING MODEL =====
def build_training_model():
    act_model = build_predict_model()

    image = act_model.input
    y_pred = act_model.output

    labels = Input(name="the_labels", shape=[max_label_len], dtype="int64")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    ctc_out = CTCLayer()([y_pred, labels, input_length, label_length])

    model = Model(
        inputs=[image, labels, input_length, label_length],
        outputs=ctc_out,
        name="training_model"
    )

    model.compile(optimizer="adam")
    model.save_weights("training_model.h5")

    return model


# ===== MAIN =====
if __name__ == "__main__":
    act_model = build_predict_model()
    train_model = build_training_model()

    act_model.summary()
    train_model.summary()
    m = build_predict_model()
    print(m.output_shape)
# (None, T, NUM_CLASSES)


