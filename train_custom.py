from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    build_custom_model,
    compute_binary_class_weight,
    configure_cpu,
    evaluate_model,
    get_class_distribution,
    load_data,
    plot_history,
    save_model,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom CNN for pneumonia classification.")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--imbalance-threshold", type=float, default=1.2)
    return parser.parse_args()


def run_phase(
    phase_name: str,
    output_dir: str,
    train_ds,
    val_ds,
    test_ds,
    image_size: int,
    epochs: int,
    patience: int,
    use_augmentation: bool,
    dropout_rate: float,
    l2_weight: float,
    class_weight: dict[int, float] | None = None,
) -> None:
    model = build_custom_model(
        input_shape=(image_size, image_size, 3),
        use_augmentation=use_augmentation,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight,
    )

    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        output_dir=output_dir,
        run_name=phase_name,
        patience=patience,
        class_weight=class_weight,
    )

    plot_history(history, output_dir=output_dir, run_name=phase_name)
    save_model(model, output_dir=output_dir, run_name=phase_name, filename="final_model.keras")

    metrics = evaluate_model(
        model=model,
        dataset=test_ds,
        output_dir=output_dir,
        run_name=phase_name,
    )

    print(f"\n[{phase_name}] Test Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def main() -> None:
    args = parse_args()
    configure_cpu()

    train_ds, val_ds, test_ds, class_names = load_data(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    print("Class mapping:", class_names)

    distribution = get_class_distribution(args.dataset_dir)
    for split_name in ("train", "val", "test"):
        if split_name in distribution:
            print(f"{split_name} distribution: {distribution[split_name]}")

    class_weight = compute_binary_class_weight(
        distribution.get("train", {}),
        class_names,
        imbalance_threshold=args.imbalance_threshold,
    )
    if class_weight is not None:
        print(f"Using class_weight in improved phase: {class_weight}")
    else:
        print("Class imbalance below threshold; class_weight disabled.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_phase(
        phase_name="custom_baseline",
        output_dir=args.output_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        image_size=args.image_size,
        epochs=args.epochs,
        patience=args.patience,
        use_augmentation=False,
        dropout_rate=0.0,
        l2_weight=0.0,
        class_weight=None,
    )

    run_phase(
        phase_name="custom_aug_reg",
        output_dir=args.output_dir,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        image_size=args.image_size,
        epochs=args.epochs,
        patience=args.patience,
        use_augmentation=True,
        dropout_rate=0.6,
        l2_weight=1e-4,
        class_weight=class_weight,
    )


if __name__ == "__main__":
    main()
