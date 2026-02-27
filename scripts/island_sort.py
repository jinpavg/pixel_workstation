"""Island sort -- procedural archipelago of untouched pixels in hue-sorted seas."""
import sys
import os

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_image, save_image, hue

# --- CONFIG ---
INPUT = sys.argv[1] if len(sys.argv) > 1 else None
ISLAND_COUNT = 8       # number of island blobs to place
MIN_RADIUS = 10         # smallest island radius in pixels
MAX_RADIUS = 60        # largest island radius in pixels
COAST_NOISE = 0.8       # coastline roughness (0 = smooth circles, 1 = very ragged)
AXIS = 0                # 0 = horizontal rows, 1 = vertical columns, 2 = both blended
SEED = 42               # set None for non-reproducible


def bleed_rotate(sorted_seg, left_nbr, right_nbr):
    """Rotate sorted segment so boundary pixels color-match island neighbors."""
    n = len(sorted_seg)

    if left_nbr is None and right_nbr is None:
        return sorted_seg

    if left_nbr is not None and right_nbr is None:
        target = left_nbr.astype(float)
        dists = np.sqrt(((sorted_seg.astype(float) - target) ** 2).sum(axis=1))
        best = int(np.argmin(dists))
        return np.roll(sorted_seg, -best, axis=0)

    if left_nbr is None and right_nbr is not None:
        target = right_nbr.astype(float)
        dists = np.sqrt(((sorted_seg.astype(float) - target) ** 2).sum(axis=1))
        best = int(np.argmin(dists))
        return np.roll(sorted_seg, (n - 1) - best, axis=0)

    # Both sides flanked by islands -- try forward and reversed, pick best
    left_t = left_nbr.astype(float)
    right_t = right_nbr.astype(float)
    ld = np.sqrt(((sorted_seg.astype(float) - left_t) ** 2).sum(axis=1))
    rd = np.sqrt(((sorted_seg.astype(float) - right_t) ** 2).sum(axis=1))
    li, ri = int(np.argmin(ld)), int(np.argmin(rd))

    if li == ri:
        return np.roll(sorted_seg, -li, axis=0)

    # Forward: roll so left-match lands at position 0
    fwd = np.roll(sorted_seg, -li, axis=0)
    ri_fwd = (ri - li) % n
    fwd_err = abs((n - 1) - ri_fwd)

    # Reversed: flip sort order, roll so left-match lands at position 0
    rev = sorted_seg[::-1].copy()
    li_r, ri_r = n - 1 - li, n - 1 - ri
    rev = np.roll(rev, -li_r, axis=0)
    ri_rev = (ri_r - li_r) % n
    rev_err = abs((n - 1) - ri_rev)

    return fwd if fwd_err <= rev_err else rev


def sort_pass(work_pixels, work_hue, work_mask):
    """Run one sorting pass (rows) with coastline bleed."""
    rows, cols, _ = work_pixels.shape
    result = work_pixels.copy()

    for row in range(rows):
        row_px = work_pixels[row]
        row_key = work_hue[row]
        row_mask = work_mask[row]

        changes = np.diff(row_mask.astype(np.int8), prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            if end - start < 2:
                continue

            seg = row_px[start:end].copy()
            order = np.argsort(row_key[start:end])
            sorted_seg = seg[order]

            left_nbr = row_px[start - 1] if start > 0 and not row_mask[start - 1] else None
            right_nbr = row_px[end] if end < cols and not row_mask[end] else None

            result[row, start:end] = bleed_rotate(sorted_seg, left_nbr, right_nbr)

    return result


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

# --- GENERATE ISLAND MASK ---
rng = np.random.default_rng(SEED)
heightmap = np.zeros((h, w), dtype=float)
yy, xx = np.mgrid[0:h, 0:w]

for _ in range(ISLAND_COUNT):
    cy, cx = rng.integers(0, h), rng.integers(0, w)
    radius = rng.integers(MIN_RADIUS, MAX_RADIUS + 1)
    sigma = radius / 2.5
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    heightmap += np.exp(-dist_sq / (2 * sigma ** 2))

# Add blurred noise for organic coastlines
coast_noise = rng.random((h, w)).astype(np.float32)
noise_img = Image.fromarray((coast_noise * 255).astype(np.uint8), mode="L")
blurred_noise = noise_img.filter(ImageFilter.GaussianBlur(radius=15))
heightmap += np.array(blurred_noise).astype(float) / 255.0 * COAST_NOISE

sea_mask = heightmap <= 0.5  # True = sea (participates in sorting)

# --- SAVE MASK PREVIEW ---
# White = island (preserved), black = sea (sorted)
mask_preview = np.where(sea_mask, 0, 255).astype(np.uint8)
save_image(np.stack([mask_preview] * 3, axis=-1), name="island_mask")

# --- PROCESS ---
hue_map = hue(pixels)

if AXIS == 2:
    h_result = sort_pass(pixels, hue_map, sea_mask)
    v_pixels = np.transpose(pixels, (1, 0, 2))
    v_hue = hue_map.T
    v_mask = sea_mask.T
    v_result = np.transpose(sort_pass(v_pixels, v_hue, v_mask), (1, 0, 2))
    result = ((h_result.astype(np.uint16) + v_result.astype(np.uint16)) // 2).astype(np.uint8)
elif AXIS == 1:
    v_pixels = np.transpose(pixels, (1, 0, 2))
    v_hue = hue_map.T
    v_mask = sea_mask.T
    result = np.transpose(sort_pass(v_pixels, v_hue, v_mask), (1, 0, 2))
else:
    result = sort_pass(pixels, hue_map, sea_mask)

# Restore island pixels (ensure untouched)
result[~sea_mask] = pixels[~sea_mask]

# --- SAVE ---
save_image(result, name="island_sort")
