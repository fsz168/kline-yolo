#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线形态标注系统 v7.0 — 中文优化版
自动在K线图上标注形态名称、置信度、趋势方向
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import yfinance as yf
import os
import glob
from datetime import datetime, timedelta

# ===== 中文字体配置 =====
FONT_PATH = "/usr/share/fonts/truetype/NotoSansCJKsc-Regular.ttf"
fm.fontManager.addfont(FONT_PATH)
CHINESE_FONT = fm.FontProperties(fname=FONT_PATH, size=9)
CHINESE_FONT_SMALL = fm.FontProperties(fname=FONT_PATH, size=7)
CHINESE_FONT_BOLD = fm.FontProperties(fname=FONT_PATH, size=9, weight="bold")

# 形态中英文映射（用于图例和标注）
PATTERN_NAMES = {
    "engulfing_bull": "看涨吞没",
    "engulfing_bear": "看跌吞没",
    "morning_star": "晨星",
    "evening_star": "晚星",
    "piercing": "刺穿线",
    "dark_cloud": "暗云盖顶",
    "pinbar_bull": "锤子线",
       "pinbar_bear": "射击之星",
    "shaved_bottom": "光头光脚阳",
    "shaved_top": "光头光脚阴",
    "inverted_hammer": "倒锤线",
    "dragonfly": "蜻蜓线",
    "gravestone": "墓碑线",
    "doji": "十字星",
    "spinning_top": "纺锤线",
    "dragonfly_doji": "蜻蜓十字",
    "gravestone_doji": "墓碑十字",
    "flat_bottom": "平头底",
    "flat_top": "平头顶",
    "consecutive_bull": "连续阳线",
    "consecutive_bear": "连续阴线",
    "inside_bar": "孕线",
    "abc_bull": "ABC反弹",
    "abc_bear": "ABC回调",
    "w_bottom": "W底",
    "m_top": "M顶",
    "rise_retrace": "上升回踩",
    "fall_retrace": "下降反弹",
}

# 颜色配置
COLOR_UP = "#E8161B"     # 阳线红色
COLOR_DOWN = "#158607"  # 阴线绿色
COLOR_SIGNAL = "#FF6600" # 信号橙色
COLOR_NEUTRAL = "#2196F3" # 中性蓝色
COLOR_TEXT = "#1a1a1a"  # 深灰文字
COLOR_BG = "#FFFFFF"    # 白色背景

def draw_kline_chart(df: pd.DataFrame, patterns: list = None,
                     score: float = 0, direction: str = "", trend: str = "",
                     title: str = "", save_path: str = None, show_label: bool = True):
    """
    绘制带形态标注的K线图（中文版）
    patterns: 识别到的形态列表，每项为 (pattern_type, confidence, bbox)
    """
    if df is None or len(df) < 5:
        return

    df = df.tail(25).copy()
    df = df.reset_index(drop=True)

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    n = len(df)
    x = np.arange(n)

    fig_h = 3.5
    fig = plt.figure(figsize=(6.4, fig_h), facecolor=COLOR_BG)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_facecolor(COLOR_BG)

    # 绘制K线
    for i in range(n):
        color = COLOR_UP if c[i] >= o[i] else COLOR_DOWN
        # 实体
        body_bottom = min(o[i], c[i])
        body_height = abs(c[i] - o[i])
        body = plt.Rectangle((i - 0.35, body_bottom), 0.7, max(body_height, 0.0001),
                              facecolor=color, edgecolor=color, linewidth=0)
        ax.add_patch(body)
        # 上影线
        if h[i] > max(o[i], c[i]):
            ax.plot([i, i], [max(o[i], c[i]), h[i]], color=color, linewidth=0.8)
        # 下影线
        if l[i] < min(o[i], c[i]):
            ax.plot([i, i], [l[i], min(o[i], c[i])], color=color, linewidth=0.8)

    # 均线
    if len(c) >= 5:
        ma5 = pd.Series(c).rolling(5).mean().values
        ax.plot(x, ma5, color="#999999", linewidth=0.6, alpha=0.7, label="MA5")

    # ===== 形态标注 =====
    if patterns and show_label:
        for pat_type, conf, k1, k2 in patterns:
            pattern_name = PATTERN_NAMES.get(pat_type, pat_type)
            # 计算标注位置
            pat_highs = h[max(0, k1):min(n, k2+1)]
            pat_lows = l[max(0, k1):min(n, k2+1)]
            y_top = max(pat_highs) if pat_highs.size > 0 else h[k1]
            y_bottom = min(pat_lows) if pat_lows.size > 0 else l[k1]

            # 画矩形框
            x1, x2 = k1 - 0.4, k2 + 0.4
            y_pad = (y_top - y_bottom) * 0.08
            rect = plt.Rectangle((x1, y_bottom - y_pad), x2 - x1,
                                  y_top - y_bottom + y_pad * 2,
                                  linewidth=1.5, edgecolor=COLOR_SIGNAL,
                                  facecolor=COLOR_SIGNAL, alpha=0.12)
            ax.add_patch(rect)

            # 标签背景
            label_text = f"{pattern_name} {conf}%"
            label_x = (x1 + x2) / 2
            label_y = y_bottom - y_pad - 0.008
            ax.text(label_x, label_y, label_text,
                    fontproperties=CHINESE_FONT_BOLD,
                    ha="center", va="top",
                    color=COLOR_SIGNAL, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=COLOR_SIGNAL, alpha=0.85, linewidth=1))

    # ===== 方向指示 =====
    if direction:
        arrow_map = {"UP": ("▲ 上涨形态", COLOR_UP),
                     "DOWN": ("▼ 下跌形态", COLOR_DOWN),
                     "NEUTRAL": ("■ 中性形态", COLOR_NEUTRAL)}
        dir_text, dir_color = arrow_map.get(direction, ("", COLOR_NEUTRAL))
        if dir_text:
            ax.text(n - 0.5, h[:n].max(), dir_text,
                    fontproperties=CHINESE_FONT_BOLD,
                    ha="right", va="top", fontsize=8, color=dir_color)

    # 边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.5)

    ax.set_xlim(-0.5, n - 0.5)
    price_range = h.max() - l.min()
    ax.set_ylim(l.min() - price_range * 0.1, h.max() + price_range * 0.15)
    ax.set_xticks([])
    ax.set_yticks([])

    # 标题
    title_text = f"{title}" if title else os.path.basename(save_path).replace("_", " ")
    ax.set_title(title_text, fontproperties=CHINESE_FONT, fontsize=9,
                 color=COLOR_TEXT, pad=3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight",
                    facecolor=COLOR_BG, pad_inches=0.05)
        plt.close(fig)

    return save_path


# ===== 独立测试字体 =====
def test_chinese_font():
    """测试中文字体渲染"""
    fig, ax = plt.subplots(figsize=(6, 1.5), facecolor="white")
    ax.text(0.5, 0.75, "中文字体测试：看涨吞没 刺穿线 晨星 锤子线",
            fontproperties=CHINESE_FONT, fontsize=12, ha="center", va="center", color="black")
    ax.text(0.5, 0.35, "测试：W底 M顶 光头光脚 连续阳线",
            fontproperties=CHINESE_FONT, fontsize=12, ha="center", va="center", color="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.savefig("/root/kline-yolo/font_test.png", dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.1)
    plt.close()
    print("✅ 字体测试图已保存到 /root/kline-yolo/font_test.png")


if __name__ == "__main__":
    test_chinese_font()
