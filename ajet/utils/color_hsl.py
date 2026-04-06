import colorsys
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=2048)
def adjust_color_hsl(base_color, logprob):
    """
    Adjust color saturation using the HSL color space based on log probability.
    Args:
        base_color (str): Hexadecimal color string (e.g., '#ff0000').
        logprob (float): Log probability value to determine saturation.
    Returns:
        str: Adjusted hexadecimal color string.
    """
    # Map logprob to a saturation adjustment factor in the range [sat_min, sat_max]
    sat_min = 0.333
    sat_max = 1.0
    lp_min = -7
    lp_max = 0

    if logprob <= lp_min:
        saturation_factor = sat_min
    elif logprob >= 0:
        saturation_factor = sat_max
    else:
        saturation_factor = sat_min + (logprob - lp_min) / (lp_max - lp_min) * (sat_max - sat_min)

    # Convert hexadecimal color to RGB
    r = int(base_color[1:3], 16) / 255.0
    g = int(base_color[3:5], 16) / 255.0
    b = int(base_color[5:7], 16) / 255.0

    # Convert to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Adjust saturation
    s_adjusted = s * saturation_factor

    # Convert back to RGB
    r_adjusted, g_adjusted, b_adjusted = colorsys.hls_to_rgb(h, l, s_adjusted)

    # Convert back to hexadecimal
    return f"#{int(r_adjusted*255):02x}{int(g_adjusted*255):02x}{int(b_adjusted*255):02x}"


def adjust_color_hsl_batch(base_colors, logprobs):
    """
    Vectorized version of adjust_color_hsl for batch processing.
    Args:
        base_colors (list[str]): List of hexadecimal color strings.
        logprobs (list[float]): List of log probability values.
    Returns:
        list[str]: List of adjusted hexadecimal color strings.
    """
    if not base_colors or not logprobs:
        return []

    # Constants
    sat_min = 0.333
    sat_max = 1.0
    lp_min = -7
    lp_max = 0

    # Convert to numpy arrays for vectorized operations
    logprobs_arr = np.array(logprobs, dtype=np.float32)

    # Vectorized saturation factor calculation
    saturation_factors = np.where(
        logprobs_arr <= lp_min,
        sat_min,
        np.where(
            logprobs_arr >= 0,
            sat_max,
            sat_min + (logprobs_arr - lp_min) / (lp_max - lp_min) * (sat_max - sat_min)
        )
    )

    # Pre-convert unique base colors to RGB and HSL
    unique_colors = list(set(base_colors))
    color_to_hls = {}

    for color in unique_colors:
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        color_to_hls[color] = (h, l, s)

    # Process each color
    result = []
    for base_color, sat_factor in zip(base_colors, saturation_factors):
        h, l, s = color_to_hls[base_color]
        s_adjusted = s * sat_factor
        r_adj, g_adj, b_adj = colorsys.hls_to_rgb(h, l, s_adjusted)
        hex_color = f"#{int(r_adj*255):02x}{int(g_adj*255):02x}{int(b_adj*255):02x}"
        result.append(hex_color)

    return result
