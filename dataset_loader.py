import os
import glob
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

from pre_processing import preprocess_image


def load_fruits360_split(
    base_dir: str,
    split: str,
    max_images_per_class: Optional[int] = None,
    exclude_rotten: bool = False,
    total_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from fruits-360/<split>/<class> directories.

    Parameters
    ----------
    base_dir : str
        Base dataset directory (e.g., "fruits-360").
    split : str
        Subdirectory name ("Training" or "Test").
    max_images_per_class : int, optional
        Maximum number of images to load per class. If None, loads all from each class.
    exclude_rotten : bool
        If True, skip any class whose folder name contains 'rot'.
    total_limit : int, optional
        Maximum number of images to load in total (across all classes).
        Once reached, loading stops early.

    Returns
    -------
    (X, y) : tuple
        X -- ndarray of shape (N, H, W, 1)
        y -- ndarray of class labels (strings)
    """

    root = Path(base_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"Split directory not found: {root}")

    x_list: List[np.ndarray] = []
    y_list: List[str] = []

    # List all class directories
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    for class_dir in class_dirs:
        class_name = class_dir.name

        # Rotten filter
        if exclude_rotten and "rot" in class_name.lower():
            continue

        images_loaded_for_class = 0

        # Load image files
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in class_dir.glob(pattern):
                print("Loading image number " + str(len(x_list)))
                # If total limit reached → stop immediately
                if total_limit is not None and len(x_list) >= total_limit:
                    X = np.stack(x_list, axis=0)
                    y = np.array(y_list, dtype=object)
                    print(f"Loaded {len(X)} images (stopped early due to total_limit={total_limit}).")
                    return X, y

                # Class-level limit
                if max_images_per_class is not None and images_loaded_for_class >= max_images_per_class:
                    break

                # Preprocess using your existing function
                out = preprocess_image(str(img_path), plot=False)
                norm = out["norm"].astype("float32")

                # Add channel dimension
                norm = norm[..., np.newaxis]

                x_list.append(norm)
                y_list.append(class_name)
                images_loaded_for_class += 1

            # Stop class scan if class-level limit reached
            if max_images_per_class is not None and images_loaded_for_class >= max_images_per_class:
                break

    if not x_list:
        raise RuntimeError(f"No images loaded from {root}. Check filters/path.")

    X = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=object)
    print(f"Loaded {len(X)} images (no early stop).")
    return X, y
