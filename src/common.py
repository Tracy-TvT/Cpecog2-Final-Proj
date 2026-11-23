import os, time, json, random
import numpy as np, pandas as pd
import tensorflow as tf, keras
from keras import layers
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
SEED = 42; IMG = 224
NAMES = ["Bacterial blight", "Blast", "Brown spot", "Tungro"]
os.makedirs("outputs", exist_ok=True)
pre = keras.applications.mobilenet_v2.preprocess_input
CUT = "block_14_add"

def load_arrays():
    a = np.load("outputs/arrays.npz")
    return {s: a["X_" + s] for s in ["train", "val", "test"]}, {s: a["Y_" + s] for s in ["train", "val", "test"]}

def make_ds(x, y=None, bs=32, shuffle=False):
    d = tf.data.Dataset.from_tensor_slices((x, y) if y is not None else x)
    if shuffle: d = d.shuffle(len(x), seed=SEED)
    return d.batch(bs).prefetch(tf.data.AUTOTUNE)

def evaluate(y_true, y_pred):
    return dict(acc=accuracy_score(y_true, y_pred), f1=f1_score(y_true, y_pred, average="macro"),
                recall=recall_score(y_true, y_pred, average=None), precision=precision_score(y_true, y_pred, average=None, zero_division=0),
                f1_per=f1_score(y_true, y_pred, average=None))

def callbacks(name, patience=3):
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            keras.callbacks.CSVLogger(f"outputs/{name}.csv")]

def build_backbone():
    backbone = keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet", pooling="avg")
    lower = keras.Model(backbone.input, backbone.get_layer(CUT).output)
    upper = keras.Model(backbone.get_layer(CUT).output, backbone.output)
    return backbone, lower, upper

def make_head(dropout, units=128):
    return keras.Sequential([keras.Input((1280,)), layers.Dropout(dropout), layers.Dense(units, activation="relu"),
                             layers.Dropout(dropout), layers.Dense(4, activation="softmax")])

def set_trainable_from(upper, start_block):
    names = [l.name for l in upper.layers]
    start = next(i for i, n_ in enumerate(names) if n_.startswith(start_block))
    for i, l in enumerate(upper.layers): l.trainable = i >= start
    return sum(1 for i, l in enumerate(upper.layers) if i >= start and l.weights)
