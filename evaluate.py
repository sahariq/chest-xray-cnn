from __future__ import annotations

import argparse

import tensorflow as tf

from utils import configure_cpu, evaluate_model, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test dataset.")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-name", type=str, default="evaluation")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu()

    _, _, test_ds, class_names = load_data(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    model = tf.keras.models.load_model(args.model_path)
    metrics = evaluate_model(
        model=model,
        dataset=test_ds,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    print("Class mapping:", class_names)
    print(f"\n[{args.run_name}] Test Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
