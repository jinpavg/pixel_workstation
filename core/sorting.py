import numpy as np


def sort_row(row_pixels, row_key, row_mask, reverse=False):
    """Sort a single row of pixels within contiguous True intervals.

    row_pixels: (W, 3) pixel values.
    row_key:    (W,) float sort key.
    row_mask:   (W,) bool mask -- True = participates in sorting.
    reverse:    sort descending if True.

    Returns: (W, 3) sorted pixel row.
    """
    result = row_pixels.copy()

    # Find contiguous True intervals
    changes = np.diff(row_mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        order = np.argsort(row_key[start:end])
        if reverse:
            order = order[::-1]
        result[start:end] = row_pixels[start:end][order]

    return result


def sort_pixels(pixels, key, mask, reverse=False, axis=0):
    """Sort pixels within masked intervals.

    pixels:  (H, W, 3) image array.
    key:     (H, W) float sort key (from color functions).
    mask:    (H, W) bool mask (from interval functions).
    reverse: sort descending.
    axis:    0 = sort along rows (horizontal), 1 = columns (vertical).

    Returns: (H, W, 3) sorted image.
    """
    if axis == 1:
        pixels = np.transpose(pixels, (1, 0, 2))
        key = key.T
        mask = mask.T

    result = np.empty_like(pixels)
    for i in range(pixels.shape[0]):
        result[i] = sort_row(pixels[i], key[i], mask[i], reverse)

    if axis == 1:
        result = np.transpose(result, (1, 0, 2))

    return result


def sort_by_angle(pixels, key, mask, angle=0, reverse=False):
    """Sort pixels at 0/90/180/270 degree angles.

    Uses numpy rotation. For arbitrary angles, rotate the image
    before calling sort_pixels and rotate back after.
    """
    rotations = {0: 0, 90: 1, 180: 2, 270: 3}
    rot = rotations.get(angle % 360, 0)

    p = np.rot90(pixels, rot)
    k = np.rot90(key, rot)
    m = np.rot90(mask, rot)

    sorted_p = sort_pixels(p, k, m, reverse)
    return np.rot90(sorted_p, -rot)
