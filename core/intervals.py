import numpy as np


def threshold_mask(values, low=0.25, high=0.75):
    """Boolean mask: True where values fall between low and high.

    values: (H, W) float array (e.g. from brightness()).
    Returns: (H, W) boolean array.
    """
    return (values >= low) & (values <= high)


def random_intervals(shape, min_len=10, max_len=100, seed=None):
    """Generate a mask with alternating random-length on/off bands per row.

    shape: (H, W) tuple.
    Returns: (H, W) boolean array.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=bool)

    for row in range(h):
        pos = 0
        sorting = True
        while pos < w:
            length = rng.integers(min_len, max_len + 1)
            if sorting:
                mask[row, pos:pos + length] = True
            pos += length
            sorting = not sorting

    return mask


def edge_mask(values, threshold=0.1):
    """True between detected edges (horizontal gradient).

    values: (H, W) float array.
    Returns: (H, W) boolean array.
    """
    grad = np.abs(np.diff(values, axis=1, prepend=values[:, :1]))
    edges = grad > threshold
    return ~edges


def full_row_mask(shape):
    """All True -- sort entire rows.

    shape: (H, W) tuple.
    Returns: (H, W) boolean array.
    """
    return np.ones(shape, dtype=bool)
