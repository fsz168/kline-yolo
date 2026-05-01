#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线形态识别模块，支持10种目标形态识别
识别成功后返回形态在K线窗口中的位置，用于生成YOLO标注
"""
import pandas as pd
import numpy as np

# 形态类别ID，和data.yaml一致
PATTERN_IDS = {
    "看涨吞没": 0,
    "看跌吞没": 1,
    "晨星": 2,
    "黄昏星": 3,
    "锤子线": 4,
    "射击之星": 5,
    "平底": 6,
    "平顶": 7,
    "连阳": 8,
    "连阴": 9
}

def kline_is_bullish(kline: pd.Series) -> bool:
    """判断K线是否是阳线"""
    return kline['Close'] > kline['Open']

def kline_is_bearish(kline: pd.Series) -> bool:
    """判断K线是否是阴线"""
    return kline['Close'] < kline['Open']

def kline_body_size(kline: pd.Series) -> float:
    """获取K线实体大小"""
    return abs(kline['Close'] - kline['Open'])

def kline_upper_shadow(kline: pd.Series) -> float:
    """获取上影线长度"""
    return kline['High'] - max(kline['Open'], kline['Close'])

def kline_lower_shadow(kline: pd.Series) -> float:
    """获取下影线长度"""
    return min(kline['Open'], kline['Close']) - kline['Low']

def is_bullish_engulfing(k1: pd.Series, k2: pd.Series) -> bool:
    """判断是否是看涨吞没：前阴后阳，阳实体完全包住阴实体"""
    if not kline_is_bearish(k1) or not kline_is_bullish(k2):
        return False
    k1_body_high = max(k1['Open'], k1['Close'])
    k1_body_low = min(k1['Open'], k1['Close'])
    k2_body_high = max(k2['Open'], k2['Close'])
    k2_body_low = min(k2['Open'], k2['Close'])
    return k2_body_low < k1_body_low and k2_body_high > k1_body_high and kline_body_size(k2) > kline_body_size(k1) * 1.2

def is_bearish_engulfing(k1: pd.Series, k2: pd.Series) -> bool:
    """判断是否是看跌吞没：前阳后阴，阴实体完全包住阳实体"""
    if not kline_is_bullish(k1) or not kline_is_bearish(k2):
        return False
    k1_body_high = max(k1['Open'], k1['Close'])
    k1_body_low = min(k1['Open'], k1['Close'])
    k2_body_high = max(k2['Open'], k2['Close'])
    k2_body_low = min(k2['Open'], k2['Close'])
    return k2_body_high > k1_body_high and k2_body_low < k1_body_low and kline_body_size(k2) > kline_body_size(k1) * 1.2

def is_morning_star(k1: pd.Series, k2: pd.Series, k3: pd.Series) -> bool:
    """判断是否是晨星：阴 → 小实体 → 阳，阳线实体超过阴线一半"""
    if not kline_is_bearish(k1) or not kline_is_bullish(k3):
        return False
    k2_body = kline_body_size(k2)
    if k2_body > kline_body_size(k1) * 0.3:
        return False
    k3_body_high = max(k3['Open'], k3['Close'])
    k3_body_low = min(k3['Open'], k3['Close'])
    k1_mid = (k1['Open'] + k1['Close']) / 2
    return k3_body_high > k1_mid

def is_evening_star(k1: pd.Series, k2: pd.Series, k3: pd.Series) -> bool:
    """判断是否是黄昏星：阳 → 小实体 → 阴，阴线实体超过阳线一半"""
    if not kline_is_bullish(k1) or not kline_is_bearish(k3):
        return False
    k2_body = kline_body_size(k2)
    if k2_body > kline_body_size(k1) * 0.3:
        return False
    k3_body_low = min(k3['Open'], k3['Close'])
    k1_mid = (k1['Open'] + k1['Close']) / 2
    return k3_body_low < k1_mid

def is_hammer(k: pd.Series, window: pd.DataFrame, idx: int) -> bool:
    """判断是否是锤子线：下影线≥实体2倍，上影线很短，出现在窗口低位"""
    body = kline_body_size(k)
    if body == 0:
        return False
    lower = kline_lower_shadow(k)
    upper = kline_upper_shadow(k)
    if lower < body * 2 or upper > body * 0.3:
        return False
    # 判断是否在低位：最低价是窗口前20%的低位
    window_low = window['Low'].min()
    return k['Low'] < window_low + (window['High'].max() - window_low) * 0.2

def is_shooting_star(k: pd.Series, window: pd.DataFrame, idx: int) -> bool:
    """判断是否是射击之星：上影线≥实体2倍，下影线很短，出现在窗口高位"""
    body = kline_body_size(k)
    if body == 0:
        return False
    upper = kline_upper_shadow(k)
    lower = kline_lower_shadow(k)
    if upper < body * 2 or lower > body * 0.3:
        return False
    # 判断是否在高位：最高价是窗口前20%的高位
    window_high = window['High'].max()
    return k['High'] > window_high - (window_high - window['Low'].min()) * 0.2

def is_floor(window: pd.DataFrame, idx: int) -> bool:
    """判断是否是平底：连续2根K线最低价几乎相同，出现在低位"""
    if idx < 1:
        return False
    k1 = window.iloc[idx-1]
    k2 = window.iloc[idx]
    if abs(k1['Low'] - k2['Low']) > (window['High'].max() - window['Low'].min()) * 0.01:
        return False
    # 最低价在窗口前20%低位
    window_low = window['Low'].min()
    return k1['Low'] < window_low + (window['High'].max() - window_low) * 0.25

def is_ceiling(window: pd.DataFrame, idx: int) -> bool:
    """判断是否是平顶：连续2根K线最高价几乎相同，出现在高位"""
    if idx < 1:
        return False
    k1 = window.iloc[idx-1]
    k2 = window.iloc[idx]
    if abs(k1['High'] - k2['High']) > (window['High'].max() - window['Low'].min()) * 0.01:
        return False
    # 最高价在窗口前20%高位
    window_high = window['High'].max()
    return k1['High'] > window_high - (window_high - window['Low'].min()) * 0.25

def is_consecutive_bullish(window: pd.DataFrame, idx: int) -> int | None:
    """判断是否是连阳：连续≥3根阳线，返回连续长度"""
    if idx < 2:
        return None
    count = 0
    for i in range(idx, max(-1, idx-5), -1):
        if kline_is_bullish(window.iloc[i]):
            count +=1
        else:
            break
    return count if count >=3 else None

def is_consecutive_bearish(window: pd.DataFrame, idx: int) -> int | None:
    """判断是否是连阴：连续≥3根阴线，返回连续长度"""
    if idx < 2:
        return None
    count = 0
    for i in range(idx, max(-1, idx-5), -1):
        if kline_is_bearish(window.iloc[i]):
            count +=1
        else:
            break
    return count if count >=3 else None

def recognize_patterns(window: pd.DataFrame) -> list[tuple[int, int, int]]:
    """
    识别窗口内所有形态
    返回列表：[(形态ID, 起始K线索引, 结束K线索引)]
    """
    patterns = []
    n = len(window)
    
    for i in range(n):
        # 双K线形态
        if i >= 1:
            if is_bullish_engulfing(window.iloc[i-1], window.iloc[i]):
                patterns.append((PATTERN_IDS["看涨吞没"], i-1, i))
            if is_bearish_engulfing(window.iloc[i-1], window.iloc[i]):
                patterns.append((PATTERN_IDS["看跌吞没"], i-1, i))
        # 三K线形态
        if i >= 2:
            if is_morning_star(window.iloc[i-2], window.iloc[i-1], window.iloc[i]):
                patterns.append((PATTERN_IDS["晨星"], i-2, i))
            if is_evening_star(window.iloc[i-2], window.iloc[i-1], window.iloc[i]):
                patterns.append((PATTERN_IDS["黄昏星"], i-2, i))
        # 单K线形态
        if is_hammer(window.iloc[i], window, i):
            patterns.append((PATTERN_IDS["锤子线"], i, i))
        if is_shooting_star(window.iloc[i], window, i):
            patterns.append((PATTERN_IDS["射击之星"], i, i))
        # 双K线平形态
        if is_floor(window, i):
            patterns.append((PATTERN_IDS["平底"], i-1, i))
        if is_ceiling(window, i):
            patterns.append((PATTERN_IDS["平顶"], i-1, i))
        # 连续K线形态
        bullish_len = is_consecutive_bullish(window, i)
        if bullish_len:
            patterns.append((PATTERN_IDS["连阳"], i - bullish_len + 1, i))
        bearish_len = is_consecutive_bearish(window, i)
        if bearish_len:
            patterns.append((PATTERN_IDS["连阴"], i - bearish_len + 1, i))
    
    # 去重：同一个K线范围只保留一个形态，优先级：反转形态 > 连续形态
    unique_patterns = []
    seen = set()
    for p in patterns:
        key = (p[1], p[2])
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    return unique_patterns

def calculate_yolo_annotation(window: pd.DataFrame, start_idx: int, end_idx: int, img_size: int = 640) -> tuple[float, float, float, float]:
    """
    计算形态的YOLO归一化标注坐标
    返回：(x_center, y_center, width, height)，都是0~1的浮点数
    """
    # K线区域坐标：x从10到630，共620像素，30根K线，每根宽度≈20.67像素
    # Y轴K线区域：10到470，共460像素，对应价格区间[window_low, window_high]
    # 成交量区域：470到630，共160像素
    kline_x_start = 10
    kline_x_end = 630
    kline_y_start = 10
    kline_y_end = 470
    k_count = len(window)
    k_width = (kline_x_end - kline_x_start) / k_count
    
    # 计算x坐标
    x1 = kline_x_start + start_idx * k_width
    x2 = kline_x_start + (end_idx + 1) * k_width
    x_center = ((x1 + x2) / 2) / img_size
    width = (x2 - x1) / img_size
    
    # 计算y坐标
    window_high = window.iloc[start_idx:end_idx+1]['High'].max()
    window_low = window.iloc[start_idx:end_idx+1]['Low'].min()
    total_high = window['High'].max()
    total_low = window['Low'].min()
    if total_high == total_low:
        total_high += 0.01
    
    y1 = kline_y_start + (total_high - window_high) / (total_high - total_low) * (kline_y_end - kline_y_start)
    y2 = kline_y_start + (total_high - window_low) / (total_high - total_low) * (kline_y_end - kline_y_start)
    y_center = ((y1 + y2) / 2) / img_size
    height = (y2 - y1) / img_size
    
    # 确保范围在0~1
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.01, min(0.99, width)),
        max(0.01, min(0.99, height))
    )
