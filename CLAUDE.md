# Pixel Workstation

Personal pixel sorting and procedural image manipulation workspace. The user
works from their phone via Claude Code, describes effects conversationally, and
iterates on Python scripts that process images.

## Layout

- `core/` -- Shared Python utilities (I/O, sorting engine, color math, intervals)
- `scripts/` -- Standalone scripts, one per effect. Each can be run independently.
- `input/` -- Drop source images here. Not tracked by git.
- `output/` -- Processed images appear here with timestamps. Not tracked by git.

## Running a Script

```
pip install -r requirements.txt   # first time only
python scripts/brightness_sort.py [optional_image_path]
```

Scripts auto-detect images in `input/` or accept a file path as the first argument.

## Writing a New Script

Every script follows this structure:

```python
"""One-line description of the effect."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_image, save_image, sort_pixels, brightness, threshold_mask

# --- CONFIG ---
INPUT = sys.argv[1] if len(sys.argv) > 1 else None
# ... tweakable parameters ...

# --- LOAD ---
# (standard load block)

# --- PROCESS ---
# (the creative part)

# --- SAVE ---
save_image(result, name="effect_name")
```

## Core API

### `core.io`
- `load_image(path)` -- returns (H, W, 3) uint8 numpy array
- `save_image(pixels, name)` -- saves PNG to output/ with timestamp, prints path

### `core.color` -- Sort keys (all return (H, W) float arrays)
- `brightness(pixels)` -- perceptual luminance, 0-1
- `lightness(pixels)` -- HSL lightness, 0-1
- `hue(pixels)` -- HSV hue, 0-360
- `saturation(pixels)` -- HSL saturation, 0-1
- `intensity(pixels)` -- sum of RGB, 0-765
- `channel(pixels, 0|1|2)` -- single R/G/B channel, 0-1

### `core.intervals` -- Masks (all return (H, W) boolean arrays)
- `threshold_mask(values, low, high)` -- True where values in range
- `random_intervals(shape, min_len, max_len)` -- random on/off bands per row
- `edge_mask(values, threshold)` -- True between detected edges
- `full_row_mask(shape)` -- all True

### `core.sorting` -- Engine
- `sort_pixels(pixels, key, mask, reverse=False, axis=0)` -- axis 0=rows, 1=columns
- `sort_by_angle(pixels, key, mask, angle=0, reverse=False)` -- 0/90/180/270 degrees

## Conventions

- Images: `(H, W, 3)` uint8 numpy arrays
- Sort keys: `(H, W)` float arrays
- Masks: `(H, W)` boolean arrays (True = pixel participates in sorting)
- Sorting happens within contiguous True intervals per row

## Workflow

1. User describes desired effect
2. Write/modify a script in `scripts/`
3. Run it: `python scripts/<name>.py`
4. User views output, gives feedback ("more aggressive", "try vertical")
5. Adjust CONFIG values or processing steps
6. Repeat

## Dependencies

Only `numpy` and `Pillow`. See `requirements.txt`.
