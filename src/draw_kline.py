"""
纯matplotlib实现的K线绘图器
- 输出精确640x640像素的纯K线图
- 无坐标轴、无网格、无标题、无成交量副图
- 阳线红色、阴线绿色，背景白色
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 全局关闭交互式后端
plt.rcParams.update({
    "font.family": "sans-serif",
    "toolbar": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.labelbottom": False,
    "ytick.labelleft": False,
    "xtick.bottom": False,
    "ytick.left": False,
})


def draw_kline_chart(
    df,
    save_path: str,
    width: int = 640,
    height: int = 640,
    up_color: str = "#EF1310",   # 阳线红色
    dn_color: str = "#0FC327",  # 阴线绿色
    bg_color: str = "white",
):
    """
    将K线DataFrame绘制为精确尺寸的纯K线图

    Args:
        df: 必须包含 open, high, low, close 列（pandas.DataFrame）
        save_path: 保存路径
        width/height: 输出像素尺寸
    """
    if df.empty or len(df) < 2:
        return False

    # 兼容大小写列名
    col_map = {}
    for col in ["open", "Open", "high", "High", "low", "Low", "close", "Close"]:
        if col in df.columns:
            col_map["_target"] = col
            break
    if not col_map:
        return False

    # 统一列名
    df = df.copy()
    rename = {}
    for std, alt in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")]:
        if alt in df.columns and std not in df.columns:
            rename[alt] = std
    df = df.rename(columns=rename)

    # 取最近20根K线
    df = df.tail(20).reset_index(drop=True)

    n = len(df)
    open_prices = df["open"].values.astype(float)
    high_prices = df["high"].values.astype(float)
    low_prices = df["low"].values.astype(float)
    close_prices = df["close"].values.astype(float)

    # 计算Y轴范围，留5%边距
    price_min = low_prices.min()
    price_max = high_prices.max()
    price_range = price_max - price_min
    y_min = price_min - price_range * 0.05
    y_max = price_max + price_range * 0.05

    # 映射K线位置到[0,1]区间
    margin = 0.05
    bar_area = 1.0 - 2 * margin  # K线可占据的宽度比例

    # 单根K线宽度（柱宽为可用的80%，间隙20%）
    bar_w = bar_area / n * 0.80

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # 填充整个figure
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()  # K线上面是高价，下面是低价

    for i in range(n):
        o, h, l, c = open_prices[i], high_prices[i], low_prices[i], close_prices[i]
        x_center = margin + (i + 0.5) * (bar_area / n)
        color = up_color if c >= o else dn_color

        # 画上下影线（从low到high的垂直线）
        ax.plot([x_center, x_center], [l, h], color=color, linewidth=1.0, zorder=2)

        # 画实体（从open到close的矩形）
        body_top = max(o, c)
        body_bot = min(o, c)
        if body_top == body_bot:
            body_top += price_range * 0.005  # 光头光脚K线加小横线

        rect = patches.FancyBboxPatch(
            (x_center - bar_w / 2, body_bot),
            bar_w,
            body_top - body_bot,
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor=color,
            linewidth=1,
            zorder=3,
        )
        ax.add_patch(rect)

    # 确保输出尺寸精确
    fig.set_size_inches(width / 100, height / 100)
    fig.savefig(
        save_path,
        dpi=100,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=False,
        facecolor=bg_color,
    )
    plt.close(fig)
    return True
