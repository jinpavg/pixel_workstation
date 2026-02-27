"""Drunk-walk hue bleed -- organic boundary pixel sort with color-matched edges.

For each row, sorts pixels [0, n) by hue into a circular gradient, then
rotates the gradient so the boundary pixel's hue matches the unsorted
image at position n.  The boundary n follows a random walk (drunk walk)
constrained between the row midpoint and the right edge.

Usage:
    python scripts/drunk_hue_bleed.py [input_image]
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_image, save_image, hue

# --- CONFIG ---
INPUT = sys.argv[1] if len(sys.argv) > 1 else None
STEP_SIZE = 50       # max drunk-walk step per row (controls boundary smoothness)
SEED = 42            # set None for non-reproducible

# --- LOAD ---
if INPUT is None:
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            INPUT = os.path.join(input_dir, f)
            break
    if INPUT is None:
        print("No input image found. Place an image in input/ or pass a path.")
        sys.exit(1)

pixels = load_image(INPUT)
h, w, _ = pixels.shape

# --- PROCESS ---
rng = np.random.default_rng(SEED)
hue_map = hue(pixels)  # (H, W) float, 0-360

result = pixels.copy()
mid = w // 2

# Initialize drunk walk starting point
n = rng.integers(mid, w)

for row in range(h):
    # Drunk walk: nudge boundary
    step = rng.integers(-STEP_SIZE, STEP_SIZE + 1)
    n = int(np.clip(n + step, mid, w - 1))

    if n < 2:
        continue

    # Sort pixels [0, n) by hue
    row_pixels = pixels[row, :n].copy()
    row_hues = hue_map[row, :n].copy()

    order = np.argsort(row_hues)
    sorted_pixels = row_pixels[order]
    sorted_hues = row_hues[order]

    # Find sorted pixel whose hue is closest to the boundary pixel's hue
    target_hue = hue_map[row, n]
    diffs = np.minimum(
        np.abs(sorted_hues - target_hue),
        360.0 - np.abs(sorted_hues - target_hue),
    )
    best_idx = int(np.argmin(diffs))

    # Rotate so best match lands at position n-1 (the boundary)
    shift = (n - 1) - best_idx
    sorted_pixels = np.roll(sorted_pixels, shift, axis=0)

    result[row, :n] = sorted_pixels

# --- SAVE ---
save_image(result, name="drunk_hue_bleed")
