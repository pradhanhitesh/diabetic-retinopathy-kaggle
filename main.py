import argparse
import os
from src.train.pipeline import train_pipeline
from src.test.pipeline import test_pipeline 


def main():
    parser = argparse.ArgumentParser(description="Train or Test CNN-based OCT classification model")

    # ----------------------------
    # General arguments
    # ----------------------------
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Choose operation mode: 'train' or 'test'"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )

    # ----------------------------
    # Training-specific arguments
    # ----------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to a single dataset directory (expects train/ and valid/ subfolders)"
    )

    parser.add_argument(
        "--datasets",
        type=str,
        help="Path to a parent directory containing multiple dataset folders (each with train/ and valid/)"
    )

    # ----------------------------
    # Testing-specific arguments
    # ----------------------------
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model file (.pth) for testing (optional)"
    )

    args = parser.parse_args()

    # ----------------------------
    # Mode: TRAIN
    # ----------------------------
    if args.mode == "train":
        if args.dataset:
            if not os.path.isdir(args.dataset):
                raise ValueError(f"{args.dataset} is not a valid dataset directory")

            print(f"📦 Training on dataset: {args.dataset}")
            train_pipeline(args.config)
        else:
            raise ValueError("You must provide either --dataset or --datasets")

        print("✅ All training runs completed successfully.")
    # ----------------------------
    # Mode: TEST
    # ----------------------------
    elif args.mode == "test":
        print("🧪 Running test pipeline...")
        test_pipeline(args.config, args.model_path)
        print("✅ Test pipeline completed successfully.")


if __name__ == "__main__":
    main()
