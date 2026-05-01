#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯K线+成交量生成器：无任何文字/标注/干扰，仅保留K线+成交量柱
输出尺寸：640x640，正方形，适合YOLO训练
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 全局样式配置，完全无干扰
plt.rcParams['font.size'] = 0
plt.rcParams['axes.linewidth'] = 0
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['figure.subplot.left'] = 0
plt.rcParams['figure.subplot.right'] = 1
plt.rcParams['figure.subplot.top'] = 1
plt.rcParams['figure.subplot.bottom'] = 0
plt.rcParams['figure.subplot.wspace'] = 0
plt.rcParams['figure.subplot.hspace'] = 0


def draw_kline_chart(df: pd.DataFrame, save_path: str) -> bool:
    """
    生成纯K线+成交量图，无任何干扰元素
    :param df: K线数据，必须包含 Date/Open/High/Low/Close/Volume 列
    :param save_path: 保存路径
    """
    try:
        df = df.copy().reset_index(drop=True)
        # 最多取最近30根K线，保证形态清晰
        df = df.tail(30).reset_index(drop=True)
        n = len(df)
        if n < 10:
            return False
        
        # 归一化价格，适配画布大小
        price_min = df[['Low', 'High']].min().min()
        price_max = df[['Low', 'High']].max().max()
        price_range = price_max - price_min if price_max > price_min else 1
        # 归一化成交量
        vol_max = df['Volume'].max()
        vol_min = df['Volume'].min()
        vol_range = vol_max - vol_min if vol_max > vol_min else 1

        # 创建画布，640x640，无白边
        fig = plt.figure(figsize=(6.4, 6.4), dpi=100, facecolor='white')
        # 布局：K线占80%高度，成交量占20%高度
        gs = GridSpec(5, 1, height_ratios=[4, 1, 0, 0, 0], hspace=0)
        ax_kline = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1])

        # 绘制K线
        for i, row in df.iterrows():
            open_p, high_p, low_p, close_p = row['Open'], row['High'], row['Low'], row['Close']
            # 颜色：阳线红，阴线绿
            color = '#ff4444' if close_p >= open_p else '#00cc66'
            # 上下影线
            ax_kline.plot([i, i], [low_p, high_p], color=color, linewidth=1.5)
            # 实体
            ax_kline.bar(i, height=abs(close_p - open_p), bottom=min(open_p, close_p), 
                        width=0.7, color=color, edgecolor=color)
        # 关闭K线轴所有刻度、边框
        ax_kline.set_xlim(-1, n)
        ax_kline.set_ylim(price_min, price_max)
        ax_kline.axis('off')

        # 绘制成交量
        for i, row in df.iterrows():
            open_p, close_p, vol = row['Open'], row['Close'], row['Volume']
            color = '#ff4444' if close_p >= open_p else '#00cc66'
            # 成交量归一化到0-1区间
            vol_norm = (vol - vol_min) / vol_range * 0.9 + 0.05
            ax_vol.bar(i, height=vol_norm, width=0.7, color=color, edgecolor=color)
        # 关闭成交量轴所有刻度、边框
        ax_vol.set_xlim(-1, n)
        ax_vol.set_ylim(0, 1)
        ax_vol.axis('off')

        # 保存图片，无任何白边/额外信息
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close(fig)
        return True
    except Exception as e:
        plt.close('all')
        return False