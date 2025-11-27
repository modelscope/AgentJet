import colorsys


def adjust_color_hsl(base_color, logprob):
    """
    使用HSL颜色空间根据logprob调整颜色饱和度
    """
    # 将logprob映射到[sat_min, sat_max]的饱和度调整因子
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

    # 将十六进制颜色转换为RGB
    r = int(base_color[1:3], 16) / 255.0
    g = int(base_color[3:5], 16) / 255.0
    b = int(base_color[5:7], 16) / 255.0

    # 转换为HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # 调整饱和度
    s_adjusted = s * saturation_factor

    # 转换回RGB
    r_adjusted, g_adjusted, b_adjusted = colorsys.hls_to_rgb(h, l, s_adjusted)

    # 转换回十六进制
    return f"#{int(r_adjusted*255):02x}{int(g_adjusted*255):02x}{int(b_adjusted*255):02x}"
