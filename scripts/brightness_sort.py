"""Brightness-threshold pixel sort -- the classic glitch art look.

Sorts pixels horizontally within regions where brightness falls
between a low and high threshold. Bright and dark areas stay untouched.

Usage:
    python scripts/brightness_sort.py [input_image]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_image, save_image, sort_pixels, brightness, threshold_mask

# --- CONFIG ---
INPUT = sys.argv[1] if len(sys.argv) > 1 else None
LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.75
REVERSE = False
AXIS = 0  # 0 = horizontal rows, 1 = vertical columns

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

# --- PROCESS ---
key = brightness(pixels)
mask = threshold_mask(key, low=LOW_THRESHOLD, high=HIGH_THRESHOLD)
result = sort_pixels(pixels, key, mask, reverse=REVERSE, axis=AXIS)

# --- SAVE ---
save_image(result, name="brightness_sort")
