from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras import layers, regularizers


SEED = 42


def configure_cpu() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(8)


def get_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def _normalize(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def _build_ds(
    data_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )


def load_data(
    dataset_dir: str = "dataset",
    image_size: Tuple[int, int] = (160, 160),
    batch_size: int = 16,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    root = Path(dataset_dir)
    train_ds = _build_ds(root / "train", image_size, batch_size, shuffle=True)
    val_ds = _build_ds(root / "val", image_size, batch_size, shuffle=False)
    test_ds = _build_ds(root / "test", image_size, batch_size, shuffle=False)

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(_normalize, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(_normalize, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(_normalize, num_parallel_calls=autotune).prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def get_class_distribution(dataset_dir: str) -> Dict[str, Dict[str, int]]:
    root = Path(dataset_dir)
    distribution: Dict[str, Dict[str, int]] = {}

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue

        split_counts: Dict[str, int] = {}
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            file_count = sum(1 for item in class_dir.rglob("*") if item.is_file())
            split_counts[class_dir.name] = file_count

        distribution[split] = split_counts

    return distribution


def compute_binary_class_weight(
    train_distribution: Dict[str, int],
    class_names: list[str],
    imbalance_threshold: float = 1.2,
) -> Optional[Dict[int, float]]:
    if len(class_names) != 2:
        return None

    class_0_count = train_distribution.get(class_names[0], 0)
    class_1_count = train_distribution.get(class_names[1], 0)

    if class_0_count == 0 or class_1_count == 0:
        return None

    majority = max(class_0_count, class_1_count)
    minority = min(class_0_count, class_1_count)

    if (majority / minority) < imbalance_threshold:
        return None

    total = class_0_count + class_1_count
    return {
        0: total / (2.0 * class_0_count),
        1: total / (2.0 * class_1_count),
    }


def build_custom_model(
    input_shape: Tuple[int, int, int] = (160, 160, 3),
    dropout_rate: float = 0.5,
    l2_weight: float = 1e-4,
    use_augmentation: bool = False,
) -> tf.keras.Model:
    reg = regularizers.l2(l2_weight) if l2_weight > 0 else None

    inputs = layers.Input(shape=input_shape)
    x = inputs

    if use_augmentation:
        x = get_augmentation_layer()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.GlobalAveragePooling2D()(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_mobilenet_model(
    input_shape: Tuple[int, int, int] = (160, 160, 3),
    dropout_rate: float = 0.5,
    l2_weight: float = 1e-4,
    base_trainable: bool = False,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    reg = regularizers.l2(l2_weight)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model, base_model


def make_callbacks(output_dir: str, run_name: str, patience: int = 5) -> list[tf.keras.callbacks.Callback]:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    output_dir: str,
    run_name: str,
    patience: int = 5,
    class_weight: Optional[Dict[int, float]] = None,
) -> tf.keras.callbacks.History:
    callbacks = make_callbacks(output_dir=output_dir, run_name=run_name, patience=patience)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    return history


def plot_history(history: tf.keras.callbacks.History, output_dir: str, run_name: str) -> None:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("loss", []), label="Train Loss")
    plt.plot(hist.get("val_loss", []), label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("accuracy", []), label="Train Accuracy")
    plt.plot(hist.get("val_accuracy", []), label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=150)
    plt.close()


def _collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred_prob = [], []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).ravel()
        y_pred_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().ravel().tolist())

    y_true_arr = np.array(y_true, dtype=np.int32)
    y_prob_arr = np.array(y_pred_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.int32)
    return y_true_arr, y_pred_arr, y_prob_arr


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    output_dir: str,
    run_name: str,
) -> Dict[str, float]:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = _collect_predictions(model, dataset)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_fig = plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False)
    disp.ax_.set_title("Confusion Matrix")
    cm_fig.tight_layout()
    cm_fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(cm_fig)

    return metrics


def save_model(model: tf.keras.Model, output_dir: str, run_name: str, filename: str = "final_model.keras") -> str:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / filename
    model.save(model_path)
    return str(model_path)
