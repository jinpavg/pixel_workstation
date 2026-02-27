import os
from datetime import datetime

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(REPO_ROOT, "input")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")


def load_image(path_or_name):
    """Load an image as a numpy array (H, W, 3) uint8.

    If path_or_name is just a filename, looks in input/.
    Converts to RGB (strips alpha if present).
    """
    path = path_or_name
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(INPUT_DIR, path)
        if os.path.exists(candidate):
            path = candidate

    img = Image.open(path).convert("RGB")
    pixels = np.array(img)
    print(f"Loaded {os.path.basename(path)}: {pixels.shape[1]}x{pixels.shape[0]}")
    return pixels


def save_image(pixels, name="output"):
    """Save a numpy array as PNG to the output/ directory.

    Returns the full path to the saved file.
    Filename: {name}_{timestamp}.png
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    full_path = os.path.join(OUTPUT_DIR, filename)
    Image.fromarray(pixels.astype(np.uint8)).save(full_path)
    print(f"Saved: {full_path}")
    return full_path
