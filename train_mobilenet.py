from __future__ import annotations

import argparse

import tensorflow as tf

from utils import (
    build_mobilenet_model,
    configure_cpu,
    evaluate_model,
    load_data,
    plot_history,
    save_model,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 transfer learning model.")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-frozen", type=int, default=20)
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fine-tune-layers", type=int, default=20)
    return parser.parse_args()


def unfreeze_last_layers(base_model: tf.keras.Model, fine_tune_layers: int) -> None:
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False


def main() -> None:
    args = parse_args()
    configure_cpu()

    train_ds, val_ds, test_ds, class_names = load_data(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )
    print("Class mapping:", class_names)

    model, base_model = build_mobilenet_model(
        input_shape=(args.image_size, args.image_size, 3),
        dropout_rate=0.5,
        l2_weight=1e-4,
        base_trainable=False,
    )

    frozen_history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs_frozen,
        output_dir=args.output_dir,
        run_name="mobilenet_frozen",
        patience=args.patience,
    )

    plot_history(frozen_history, output_dir=args.output_dir, run_name="mobilenet_frozen")
    save_model(model, output_dir=args.output_dir, run_name="mobilenet_frozen", filename="final_model.keras")

    unfreeze_last_layers(base_model, fine_tune_layers=args.fine_tune_layers)
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

    finetune_history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs_finetune,
        output_dir=args.output_dir,
        run_name="mobilenet_finetuned",
        patience=args.patience,
    )

    plot_history(finetune_history, output_dir=args.output_dir, run_name="mobilenet_finetuned")
    save_model(model, output_dir=args.output_dir, run_name="mobilenet_finetuned", filename="final_model.keras")

    metrics = evaluate_model(
        model=model,
        dataset=test_ds,
        output_dir=args.output_dir,
        run_name="mobilenet_finetuned",
    )

    print("\n[mobilenet_finetuned] Test Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
