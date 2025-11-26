# main.py
"""
Train a CNN VAE on fruit images, then test it
using reconstruction error as an anomaly score.

This script relies on:
  - dataset_loader.load_fruits360_split
  - cnn_vae.build_vae

Directory structure:

    fruits-360/
        Training/
            Apple/
                *.jpg
            Banana/
                *.jpg
            ...
        Test/
            Apple/
                *.jpg
            Banana/
                *.jpg
            ...

Make sure this file is in the same project as:
  - dataset_loader.py  (with load_fruits360_split)
  - cnn_vae.py         (with build_vae)
  - pre_processing.py  (used by dataset_loader)
"""

import os
import numpy as np

from dataset_loader import load_fruits360_split
from cnn_vae import build_vae
from pathlib import Path
from pre_processing import preprocess_image


# -------------------------
# Configuration
# -------------------------

DATASET_BASE = "fruits-360"

NUM_TRAIN_IMAGES = 6400 # total images to actually use for training
NUM_TEST_IMAGES = 1600 # total images to actually use for testing

EPOCHS = 60
BATCH_SIZE = 64
TRAIN_SPLIT_NAME = "Training"
TEST_SPLIT_NAME = "Test"


# -------------------------
# Utility: reconstruction error
# -------------------------

def reconstruction_error(model, x_batch):
    """Compute per-image MSE reconstruction error."""
    recon = model.predict(x_batch, verbose=0)
    errors = np.mean((x_batch - recon) ** 2, axis=(1, 2, 3))
    return errors


def scan_test_folders_for_anomalies(
    vae,
    base_dir: str = "fruits-360",
    split: str = "Test",
    threshold: float = 0.001,
    max_images_per_folder: int | None = None,
    max_images: int | None = None,
):
    root = Path(base_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"Test directory not found: {root}")

    anomalies = {}
    total_images_seen = 0

    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if max_images is not None and total_images_seen >= max_images:
            break

        folder_name = class_dir.name
        print(f"\nScanning folder: {folder_name}")

        folder_anomalies = []
        images_seen = 0


        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in class_dir.glob(pattern):

                # per-folder limit
                if max_images_per_folder is not None and images_seen >= max_images_per_folder:
                    break
            
                if max_images is not None and total_images_seen >= max_images:
                    break

                # Preprocess
                out = preprocess_image(str(img_path), plot=False)
                norm = out["norm"].astype("float32")[..., np.newaxis]
                x = norm[np.newaxis, ...]

                # Compute error
                err = reconstruction_error(vae, x)[0]
                images_seen += 1
                total_images_seen += 1

                # Print each image
                # print(f"  {img_path.name}: error={err:.6f}")

                # Check anomaly
                if err > threshold:
                    folder_anomalies.append((img_path.name, err))

            if max_images_per_folder is not None and images_seen >= max_images_per_folder:
                break

        anomalies[folder_name] = folder_anomalies

    # -------- Summary --------
    print("\n===== ANOMALY SUMMARY =====")
    for folder, items in anomalies.items():
        if items:
            print(f"\nFolder '{folder}' — {len(items)} anomalies:")
            # for fname, err in items:
                # print(f"    {fname} — error {err:.6f}")
        else:
            print(f"\nFolder '{folder}' — no anomalies")

    return anomalies

def main():
    # Make TensorFlow quieter
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    #  Load training data
    print("Loading training data (non-rotten only)...")
    X_train_full, y_train_full = load_fruits360_split(
        base_dir=DATASET_BASE,
        split=TRAIN_SPLIT_NAME,
        max_images_per_class=None,
        exclude_rotten=True,
        # total_limit=NUM_TRAIN_IMAGES,
    )

    n_train_full = X_train_full.shape[0]
    if n_train_full <= NUM_TRAIN_IMAGES:
        X_train = X_train_full
        print(f"Training set has only {n_train_full} images; using all of them.")
    else:
        idx = np.random.choice(n_train_full, size=NUM_TRAIN_IMAGES, replace=False)
        X_train = X_train_full[idx]
        print(f"Sampled {NUM_TRAIN_IMAGES} images out of {n_train_full} for training.")

    print(f"X_train shape: {X_train.shape}")

    # Load test data (mixed, no rotten filtering)
    print("\nLoading test data (mixed, no rotten filtering)...")
    X_test_full, y_test_full = load_fruits360_split(
        base_dir=DATASET_BASE,
        split=TEST_SPLIT_NAME,
        max_images_per_class=None,
        exclude_rotten=False,
        # total_limit=NUM_TEST_IMAGES,
    )

    n_test_full = X_test_full.shape[0]
    if n_test_full <= NUM_TEST_IMAGES:
        X_test = X_test_full
        print(f"Test set has only {n_test_full} images; using all of them.")
    else:
        idx_test = np.random.choice(n_test_full, size=NUM_TEST_IMAGES, replace=False)
        X_test = X_test_full[idx_test]
        y_test_full = y_test_full[idx_test]
        print(f"Sampled {NUM_TEST_IMAGES} images out of {n_test_full} for testing.")

    print(f"X_test shape: {X_test.shape}")

    # Build VAE from cnn_vae.py 
    print("\nBuilding VAE...")
    vae, encoder, decoder = build_vae()

    # Train VAE on the training images
    print("\nTraining VAE...")
    history = vae.fit(
        X_train,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
    )

    # Compute reconstruction errors on train and test 
    print("\nComputing reconstruction errors...")
    train_errors = reconstruction_error(vae, X_train)
    # test_errors = reconstruction_error(vae, X_test)

    # threshold from training error distribution
    threshold = np.percentile(train_errors, 95)

    # print(f"\nTrain errors: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
    # print(f"Test  errors: mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")
    # print(f"Anomaly threshold (95th percentile of train errors): {threshold:.6f}")

    # Flag anomalies in test set
    # is_anomaly = test_errors > threshold
    # num_anomalies = int(is_anomaly.sum())

    # print(f"\nOut of {len(test_errors)} test images, "
    #       f"{num_anomalies} are above threshold (flagged as anomalies).")

    # # Show details for the first 10 test samples
    # print("\nFirst 10 test reconstruction errors and flags:")
    # for i, err in enumerate(test_errors[:10]):
    #     flag = "ANOMALY" if is_anomaly[i] else "normal"
    #     label = y_test_full[i] if i < len(y_test_full) else "unknown"
    #     print(f"  Test image {i:02d} ({label}): error={err:.6f} -> {flag}")

    print("\nRunning full folder-level anomaly scan on Test split...")
    scan_test_folders_for_anomalies(
        vae,
        base_dir=DATASET_BASE,
        split=TEST_SPLIT_NAME,
        threshold=threshold,
        max_images_per_folder=None,
        # max_images=NUM_TEST_IMAGES,
    )

    print("Run complete.")


if __name__ == "__main__":
    main()
