import colorsys


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
