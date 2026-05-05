from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

RGB = Tuple[int, int, int]

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)


def srgb_to_linear(value: int | float | np.ndarray) -> float | np.ndarray:
    """Convert 8-bit or normalized sRGB values to linear-light values."""
    arr = np.asarray(value, dtype=np.float64)
    if np.nanmax(arr) > 1.0:
        arr = arr / 255.0
    out = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    if np.isscalar(value):
        return float(out)
    return out


def linear_rgb(rgb: RGB | np.ndarray) -> np.ndarray:
    return np.asarray(srgb_to_linear(np.asarray(rgb, dtype=np.float64)), dtype=np.float64)


def relative_luminance(rgb: RGB | np.ndarray) -> float | np.ndarray:
    linear = linear_rgb(rgb)
    lum = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )
    if np.asarray(lum).shape == ():
        return float(lum)
    return lum


def _xyz(rgb: RGB | np.ndarray) -> np.ndarray:
    linear = linear_rgb(rgb)
    return linear @ _SRGB_TO_XYZ.T


def cie1976_ucs_uv(rgb: RGB | np.ndarray) -> tuple[float, float] | np.ndarray:
    xyz = _xyz(rgb)
    denom = xyz[..., 0] + 15 * xyz[..., 1] + 3 * xyz[..., 2]
    u = np.divide(4 * xyz[..., 0], denom, out=np.zeros_like(denom), where=denom != 0)
    v = np.divide(9 * xyz[..., 1], denom, out=np.zeros_like(denom), where=denom != 0)
    uv = np.stack([u, v], axis=-1)
    if uv.shape == (2,):
        return float(uv[0]), float(uv[1])
    return uv


def cie1976_ucs_distance(a: RGB, b: RGB) -> float:
    au, av = cie1976_ucs_uv(a)
    bu, bv = cie1976_ucs_uv(b)
    return math.hypot(au - bu, av - bv)


def is_saturated_red(rgb: RGB | np.ndarray) -> bool | np.ndarray:
    linear = linear_rgb(rgb)
    total = linear[..., 0] + linear[..., 1] + linear[..., 2]
    saturated = np.divide(linear[..., 0], total, out=np.zeros_like(total), where=total > 0) >= 0.8
    if np.asarray(saturated).shape == ():
        return bool(saturated)
    return saturated


def wcag20_red_critical_value(rgb: RGB) -> float:
    r, g, b = linear_rgb(rgb)
    value = r - g - b
    return float(value * 320 if value > 0 else 0)


def is_red_transition(a: RGB, b: RGB, *, min_ucs_distance: float = 0.2) -> bool:
    a_sat = bool(is_saturated_red(a))
    b_sat = bool(is_saturated_red(b))
    if not (a_sat or b_sat):
        return False
    if a_sat and b_sat:
        return False
    return cie1976_ucs_distance(a, b) >= min_ucs_distance


def red_transition_mask(
    previous_rgb: np.ndarray, current_rgb: np.ndarray, *, min_ucs_distance: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    prev_sat = is_saturated_red(previous_rgb)
    curr_sat = is_saturated_red(current_rgb)
    one_sat = np.logical_xor(prev_sat, curr_sat)

    prev_uv = cie1976_ucs_uv(previous_rgb)
    curr_uv = cie1976_ucs_uv(current_rgb)
    distance = np.linalg.norm(curr_uv - prev_uv, axis=-1)
    mask = one_sat & (distance >= min_ucs_distance)
    direction = np.where(curr_sat & ~prev_sat, 1, -1)
    return mask, direction


def parse_rgb(text: str) -> RGB:
    parts = text.strip().strip("()").split(",")
    if len(parts) < 3:
        raise ValueError(f"expected RGB text, got {text!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])
