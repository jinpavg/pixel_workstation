"""Mirror drunk-walk hue bleed -- clear center band with sorted bleed on both edges."""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_image, save_image, hue

# --- CONFIG ---
INPUT = sys.argv[1] if len(sys.argv) > 1 else None
STEP_SIZE = 15       # max drunk-walk step per row
MIN_GAP = 0.2        # minimum untouched middle width as fraction of image dimension
LEFT_LIMIT = 0.4     # how far inward the left boundary can reach (fraction of width)
RIGHT_LIMIT = 0.6    # how far inward the right boundary can reach (from left edge)
SEED = 43


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
hue_map = hue(pixels)


# --- PROCESS ---
def process_rows(src, src_hue, rows, cols, rng):
    """Apply mirror drunk hue bleed row-by-row."""
    result = src.copy()
    min_gap = int(cols * MIN_GAP)
    n_lo, n_hi = 2, int(cols * LEFT_LIMIT)
    m_lo, m_hi = int(cols * RIGHT_LIMIT), cols - 2

    n = rng.integers(n_lo, n_hi + 1)
    m = rng.integers(m_lo, m_hi + 1)

    for row in range(rows):
        # Drunk walk both boundaries
        n = int(np.clip(n + rng.integers(-STEP_SIZE, STEP_SIZE + 1), n_lo, n_hi))
        m = int(np.clip(m + rng.integers(-STEP_SIZE, STEP_SIZE + 1), m_lo, m_hi))

        # Enforce minimum gap
        if m - n < min_gap:
            deficit = min_gap - (m - n)
            push_l = deficit // 2
            push_r = deficit - push_l
            n = max(n_lo, n - push_l)
            m = min(m_hi, m + push_r)
            if m - n < min_gap:
                n = max(n_lo, m - min_gap)
            if m - n < min_gap:
                m = min(m_hi, n + min_gap)

        # --- Left segment [0, n) ---
        if n >= 2:
            left_px = src[row, :n].copy()
            order = np.argsort(src_hue[row, :n])
            sorted_left = left_px[order]

            target = src[row, n].astype(float)
            diffs = np.sqrt(((sorted_left.astype(float) - target) ** 2).sum(axis=1))
            best = int(np.argmin(diffs))
            sorted_left = np.roll(sorted_left, (n - 1) - best, axis=0)
            result[row, :n] = sorted_left

        # --- Right segment (m, cols) ---
        rs = m + 1
        if rs < cols - 1:
            right_px = src[row, rs:].copy()
            order = np.argsort(src_hue[row, rs:])
            sorted_right = right_px[order]

            target = src[row, m].astype(float)
            diffs = np.sqrt(((sorted_right.astype(float) - target) ** 2).sum(axis=1))
            best = int(np.argmin(diffs))
            sorted_right = np.roll(sorted_right, -best, axis=0)
            result[row, rs:] = sorted_right

    return result


# Horizontal
rng_h = np.random.default_rng(SEED)
result_h = process_rows(pixels, hue_map, h, w, rng_h)
save_image(result_h, name="mirror_drunk_hue_bleed_rows")

# Vertical (transpose, process, transpose back)
rng_v = np.random.default_rng(SEED + 1 if SEED is not None else None)
v_px = np.transpose(pixels, (1, 0, 2))
v_hue = hue_map.T
result_v = process_rows(v_px, v_hue, w, h, rng_v)
result_v = np.transpose(result_v, (1, 0, 2))
save_image(result_v, name="mirror_drunk_hue_bleed_cols")
