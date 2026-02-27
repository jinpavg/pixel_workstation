import numpy as np


def brightness(pixels):
    """Perceptual luminance (BT.601). Returns (H, W) float array, 0-1."""
    return np.dot(pixels[..., :3].astype(float), [0.299, 0.587, 0.114]) / 255.0


def lightness(pixels):
    """HSL lightness: (max + min) / 2. Returns (H, W) float array, 0-1."""
    flt = pixels[..., :3].astype(float) / 255.0
    return (flt.max(axis=-1) + flt.min(axis=-1)) / 2.0


def hue(pixels):
    """HSV hue. Returns (H, W) float array, 0-360."""
    flt = pixels[..., :3].astype(float) / 255.0
    cmax = flt.max(axis=-1)
    cmin = flt.min(axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(delta)

    # Red is max
    mask = (delta > 0) & (cmax == flt[..., 0])
    h[mask] = 60.0 * (((flt[..., 1][mask] - flt[..., 2][mask]) / delta[mask]) % 6)

    # Green is max
    mask = (delta > 0) & (cmax == flt[..., 1])
    h[mask] = 60.0 * (((flt[..., 2][mask] - flt[..., 0][mask]) / delta[mask]) + 2)

    # Blue is max
    mask = (delta > 0) & (cmax == flt[..., 2])
    h[mask] = 60.0 * (((flt[..., 0][mask] - flt[..., 1][mask]) / delta[mask]) + 4)

    return h


def saturation(pixels):
    """HSL saturation. Returns (H, W) float array, 0-1."""
    flt = pixels[..., :3].astype(float) / 255.0
    cmax = flt.max(axis=-1)
    cmin = flt.min(axis=-1)
    delta = cmax - cmin
    light = (cmax + cmin) / 2.0

    s = np.zeros_like(delta)
    mask = delta > 0
    s[mask] = delta[mask] / (1.0 - np.abs(2.0 * light[mask] - 1.0) + 1e-10)
    return np.clip(s, 0, 1)


def intensity(pixels):
    """Sum of RGB channels. Returns (H, W) float array, 0-765."""
    return pixels[..., :3].astype(float).sum(axis=-1)


def channel(pixels, index):
    """Single R(0)/G(1)/B(2) channel. Returns (H, W) float array, 0-1."""
    return pixels[..., index].astype(float) / 255.0
