import os
import glob
import random
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


def _resolve_image_path(path_or_stem: str) -> str:
    """
    Accepts either a full file path or a stem without an extension.
    If a stem/dir is provided, tries common extensions and returns the first match.
    Raises FileNotFoundError if nothing is found.
    """
    # If it's already a file, return as-is
    if os.path.isfile(path_or_stem):
        return path_or_stem

    # Try common extensions if the provided path lacks one
    base, ext = os.path.splitext(path_or_stem)
    search_roots = [path_or_stem] if ext else [base]
    candidates = []

    for root in search_roots:
        for e in (".jpg", ".jpeg", ".png"):
            p = root + e
            if os.path.isfile(p):
                candidates.append(p)

    # If path points to a directory, fall back to "any image in the folder"
    if not candidates and os.path.isdir(path_or_stem):
        for e in (".jpg", ".jpeg", ".png"):
            candidates.extend(glob.glob(os.path.join(path_or_stem, "*" + e)))

    if not candidates:
        raise FileNotFoundError(
            f"No image file found for '{path_or_stem}'. "
            "Provide a real file path or a stem; we try .jpg/.jpeg/.png and directories."
        )

    # If multiple, pick deterministically the first sorted candidate
    return sorted(candidates)[0]


def preprocess_image(
    image_path_or_stem: str,
    *,
    plot: bool = False,
    rand_pick_if_dir: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load and preprocess an image:
      - load RGB
      - split channels
      - convert to grayscale
      - min-max normalize grayscale

    Parameters
    ----------
    image_path_or_stem : str
        Full path to an image file OR a stem without extension OR a directory.
        If a directory is provided and rand_pick_if_dir=True, a random image is selected.
    plot : bool
        If True, reproduces your 1x6 visualization (Original, R, G, B, Gray, Norm).
    rand_pick_if_dir : bool
        If the input is a directory, choose a random image from it. If False, choose the first sorted.

    Returns
    -------
    Dict[str, np.ndarray]
        {
          "image_rgb": HxWx3 uint8 array,
          "r": HxW,
          "g": HxW,
          "b": HxW,
          "gray": HxW float64 in [0,1],
          "norm": HxW float64 in [0,1]
        }
    """
    path_input = image_path_or_stem

    # If it's a directory and we allow random pick, choose one file
    if os.path.isdir(path_input):
        img_files = []
        for e in (".jpg", ".jpeg", ".png"):
            img_files.extend(glob.glob(os.path.join(path_input, "*" + e)))
        if not img_files:
            raise FileNotFoundError(f"No images found in directory: {path_input}")
        image_path = random.choice(img_files) if rand_pick_if_dir else sorted(img_files)[0]
    else:
        image_path = _resolve_image_path(path_input)

    # Read image
    image = io.imread(image_path)
    # Drop alpha if present
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image. Got shape {image.shape} from '{image_path}'.")

    # Channels
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Gray (skimage returns float in [0,1])
    gray = color.rgb2gray(image)

    # Min-max normalize the gray (guard against constant images)
    gmin, gmax = float(gray.min()), float(gray.max())
    if gmax > gmin:
        norm = (gray - gmin) / (gmax - gmin)
    else:
        norm = np.zeros_like(gray)

    if plot:
        # First figure: original
        i, im1 = plt.subplots(1, 1)
        i.set_figwidth(15)
        im1.imshow(image)
        im1.set_title(os.path.basename(image_path))
        im1.axis("off")

        # Second figure: 1x6 panel (Original, R, G, B, Gray, Norm)
        i, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True)
        i.set_figwidth(20)
        ax1.imshow(image)           # Original
        ax2.imshow(r, cmap="Reds")  # Red
        ax3.imshow(g, cmap="Greens")# Green
        ax4.imshow(b, cmap="Blues") # Blue
        i.suptitle("Original & RGB image channels")

        ax5.imshow(gray, cmap="gray")   # Gray
        ax6.imshow(norm, cmap="gray")   # Normalized gray (0..1)

        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.axis("off")
        plt.show()

    return {
        "image_rgb": image,
        "r": r,
        "g": g,
        "b": b,
        "gray": gray,
        "norm": norm,
    }

