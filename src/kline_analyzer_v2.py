#!/usr/bin/env python3
"""
K线形态识别系统 v2.0 — 7项重大升级
- 形态标注可视化（在图上直接画框+标注）
- 信号强度评分（多重加权排序）
- 10种新形态（三角形/旗形/楔形/头肩顶等）
- 布林带验证（二次确认过滤假信号）
- 多周期共振（日+周+月三周期）
- 全A股扫描（4500+标的）
- 结果导出（带图可视化报告）
"""

import os
import math
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import yfinance as yf
from finta import TA
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

# ============ 全局配置 ============
BASE_DIR = "/root/kline-yolo"
IMG_DIR = f"{BASE_DIR}/images_v2"
REPORT_DIR = f"{BASE_DIR}/reports"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 标注配色方案
COLORS = {
    'bullish': '#00C853',  # 强阳绿
    'bearish': '#D50000',  # 强阴红
    'neutral': '#FF6D00',  # 警示橙
    'breakout': '#2962FF', # 突破蓝
    'text': '#212121',
    'bg': '#FAFAFA',
}

# ============ 核心1: 多周期K线数据获取 ============
def get_multi_period_klines(symbol: str, periods=['1d', '1wk', '1mo']):
    """获取多周期K线数据"""
    interval_map = {'1d': '1d', '1wk': '1wk', '1mo': '1mo'}
    period_map = {'1d': '3mo', '1wk': '6mo', '1mo': '2y'}
    result = {}
    for p in periods:
        try:
            df = yf.Ticker(symbol).history(period=period_map[p], interval=interval_map[p])
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'close', 'volume']
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            result[p] = df.dropna()
        except Exception:
            result[p] = pd.DataFrame()
    return result

# ============ 核心2: 指标计算 ============
def calc_atr(high, low, close, window=14):
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high.iloc[0] - low.iloc[0]
    return pd.Series(tr).rolling(window).mean().values

def calc_bollinger_bands(close, window=20, num_std=2):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return ma + num_std * std, ma - num_std * std, ma

def calc_volume_ma(volume, window=20):
    return volume.rolling(window).mean()

# ============ 核心3: K线形态识别 (对齐PDF规则) ============
def is_bullish_engulfing(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 1: return False
    prev, cur = i - 1, i
    if c[prev] >= o[prev] or c[cur] <= o[cur]: return False
    if body_ratio[prev] < 0.6 or body_ratio[cur] < 0.6: return False
    if ranges[cur] < 1.2 * atr[cur]: return False
    if o[cur] >= c[prev] or c[cur] <= o[prev]: return False
    return True

def is_bearish_engulfing(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 1: return False
    prev, cur = i - 1, i
    if c[prev] <= o[prev] or c[cur] >= o[cur]: return False
    if body_ratio[prev] < 0.6 or body_ratio[cur] < 0.6: return False
    if ranges[cur] < 1.2 * atr[cur]: return False
    if c[cur] >= o[prev] or o[cur] <= c[prev]: return False
    return True

def is_morning_star(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 2: return False
    p1, p2, p3 = i - 2, i - 1, i
    if body_ratio[p1] < 0.6 or body_ratio[p3] < 0.6: return False
    if body_ratio[p2] > 0.65 or ranges[p2] > 0.75 * atr[p2]: return False
    if c[p3] <= (o[p1] + c[p1]) / 2: return False
    if c[p1] <= o[p1]: return False
    if c[p3] > o[p3]: return False
    return True

def is_evening_star(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 2: return False
    p1, p2, p3 = i - 2, i - 1, i
    if body_ratio[p1] < 0.6 or body_ratio[p3] < 0.6: return False
    if body_ratio[p2] > 0.65 or ranges[p2] > 0.75 * atr[p2]: return False
    if c[p3] >= (o[p1] + c[p1]) / 2: return False
    if c[p1] >= o[p1]: return False
    if c[p3] < o[p3]: return False
    return True

def is_piercing(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 1: return False
    prev, cur = i - 1, i
    if body_ratio[prev] < 0.6 or body_ratio[cur] < 0.6: return False
    if c[prev] >= o[prev] or c[cur] <= o[cur]: return False
    mid = (o[prev] + c[prev]) / 2
    if c[cur] <= mid: return False
    if ranges[cur] < 1.2 * atr[cur]: return False
    return True

def is_dark_cloud(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 1: return False
    prev, cur = i - 1, i
    if body_ratio[prev] < 0.6 or body_ratio[cur] < 0.6: return False
    if c[prev] <= o[prev] or c[cur] >= o[cur]: return False
    mid = (o[prev] + c[prev]) / 2
    if c[cur] >= mid: return False
    if ranges[cur] < 1.2 * atr[cur]: return False
    return True

def is_hammer(o, h, l, c, body_ratio, atr, ranges, i):
    body = abs(c[i] - o[i])
    upper_shadow = h[i] - max(o[i], c[i])
    lower_shadow = min(o[i], c[i]) - l[i]
    if lower_shadow < 3 * body or body == 0: return False
    if ranges[i] < 1.0 * atr[i]: return False
    if upper_shadow > body * 0.5: return False
    return True

def is_shooting_star(o, h, l, c, body_ratio, atr, ranges, i):
    body = abs(c[i] - o[i])
    upper_shadow = h[i] - max(o[i], c[i])
    lower_shadow = min(o[i], c[i]) - l[i]
    if upper_shadow < 3 * body or body == 0: return False
    if ranges[i] < 1.0 * atr[i]: return False
    if lower_shadow > body * 0.5: return False
    return True

def is_bullish_three_inside(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 2: return False
    p1, p2, p3 = i - 2, i - 1, i
    if body_ratio[p1] < 0.6 or body_ratio[p3] < 0.8: return False
    if c[p1] <= o[p1]: return False
    if c[p3] <= o[p3]: return False
    if c[p3] <= (o[p1] + c[p1]) / 2: return False
    if h[p2] > h[p1] or l[p2] < l[p1]: return False
    return True

def is_bearish_three_inside(o, h, l, c, body_ratio, atr, ranges, i):
    if i < 2: return False
    p1, p2, p3 = i - 2, i - 1, i
    if body_ratio[p1] < 0.6 or body_ratio[p3] < 0.8: return False
    if c[p1] >= o[p1]: return False
    if c[p3] >= o[p3]: return False
    if c[p3] >= (o[p1] + c[p1]) / 2: return False
    if l[p2] < l[p1] or h[p2] > h[p1]: return False
    return True

# ============ 核心4: 新增10种形态 ============
def is_double_bottom(o, h, l, c, atr, ranges, i, lookback=30):
    """双重底形态 — W底"""
    if i < lookback: return False
    window = 20
    segment = c[max(0,i-lookback):i+1]
    if len(segment) < window: return False
    lows = segment[-window:]
    min_idx = np.argmin(lows)
    if min_idx < 5 or min_idx > len(lows) - 5: return False
    left_low = np.min(lows[:min_idx])
    right_low = np.min(lows[min_idx:])
    diff_pct = abs(left_low - right_low) / left_low
    peak = np.max(lows)
    if diff_pct > 0.05: return False
    if c[i] < peak: return False
    if ranges[i] < 1.0 * atr[i]: return False
    return True

def is_double_top(o, h, l, c, atr, ranges, i, lookback=30):
    """双重顶形态 — M顶"""
    if i < lookback: return False
    window = 20
    segment = c[max(0,i-lookback):i+1]
    if len(segment) < window: return False
    highs = segment[-window:]
    max_idx = np.argmax(highs)
    if max_idx < 5 or max_idx > len(highs) - 5: return False
    left_high = np.max(highs[:max_idx])
    right_high = np.max(highs[max_idx:])
    diff_pct = abs(left_high - right_high) / left_high
    trough = np.min(highs)
    if diff_pct > 0.05: return False
    if c[i] > trough: return False
    if ranges[i] < 1.0 * atr[i]: return False
    return True

def is_head_shoulders(o, h, l, c, atr, ranges, i, lookback=60):
    """头肩顶形态"""
    if i < lookback: return False
    window = 40
    segment = c[max(0,i-lookback):i+1]
    if len(segment) < window: return False
    highs = segment[-window:]
    from scipy.signal import argrelextrema
    try:
        local_max = argrelextrema(np.array(highs), np.greater, order=3)[0]
        if len(local_max) < 3: return False
        last_3 = local_max[-3:]
        head = highs[last_3[1]]
        left_shoulder = highs[last_3[0]]
        right_shoulder = highs[last_3[2]]
        if head <= max(left_shoulder, right_shoulder): return False
        if abs(left_shoulder - right_shoulder) / head > 0.1: return False
        if c[i] < (highs[last_3[0]] + highs[last_3[2]]) / 2: return False
        return True
    except Exception:
        return False

def is_inverted_head_shoulders(o, h, l, c, atr, ranges, i, lookback=60):
    """倒头肩底形态"""
    if i < lookback: return False
    window = 40
    segment = c[max(0,i-lookback):i+1]
    if len(segment) < window: return False
    lows = segment[-window:]
    from scipy.signal import argrelextrema
    try:
        local_min = argrelextrema(np.array(lows), np.less, order=3)[0]
        if len(local_min) < 3: return False
        last_3 = local_min[-3:]
        head = lows[last_3[1]]
        left_shoulder = lows[last_3[0]]
        right_shoulder = lows[last_3[2]]
        if head >= min(left_shoulder, right_shoulder): return False
        if abs(left_shoulder - right_shoulder) / head > 0.1: return False
        if c[i] > (lows[last_3[0]] + lows[last_3[2]]) / 2: return False
        return True
    except Exception:
        return False

def is_rising_wedge(o, h, l, c, atr, ranges, i, lookback=30):
    """上升楔形 — 看跌"""
    if i < lookback: return False
    window = 20
    segment_h = h[max(0,i-lookback):i+1][-window:]
    segment_l = l[max(0,i-lookback):i+1][-window:]
    if len(segment_h) < 15: return False
    try:
        slope_h = (segment_h[-1] - segment_h[0]) / len(segment_h)
        slope_l = (segment_l[-1] - segment_l[0]) / len(segment_l)
        if slope_h <= 0 or slope_l <= 0: return False
        if slope_h < slope_l * 0.7 or slope_h > slope_l * 1.3: return False
        if c[i] < h[i] and c[i] > o[i]: return False
        return True
    except Exception:
        return False

def is_falling_wedge(o, h, l, c, atr, ranges, i, lookback=30):
    """下降楔形 — 看涨"""
    if i < lookback: return False
    window = 20
    segment_h = h[max(0,i-lookback):i+1][-window:]
    segment_l = l[max(0,i-lookback):i+1][-window:]
    if len(segment_h) < 15: return False
    try:
        slope_h = (segment_h[-1] - segment_h[0]) / len(segment_h)
        slope_l = (segment_l[-1] - segment_l[0]) / len(segment_l)
        if slope_h >= 0 or slope_l >= 0: return False
        if slope_l > slope_h * 0.7 or slope_l < slope_h * 1.3: return False
        if c[i] > l[i] and c[i] < o[i]: return False
        return True
    except Exception:
        return False

def is_bull_flag(o, h, l, c, atr, ranges, i, lookback=30):
    """牛旗形 — 看涨持续"""
    if i < lookback: return False
    window = 20
    segment = c[max(0,i-lookback):i+1][-window:]
    if len(segment) < 15: return False
    try:
        slope = (segment[-1] - segment[0]) / len(segment)
        if slope < 0.005: return False
        pole_height = segment[-1] - segment[0]
        consolidation = segment[-5:]
        consolidation_range = np.max(consolidation) - np.min(consolidation)
        if consolidation_range > pole_height * 0.3: return False
        if c[i] > o[i] and ranges[i] > 0.8 * atr[i]: return False
        return True
    except Exception:
        return False

def is_bear_flag(o, h, l, c, atr, ranges, i, lookback=30):
    """熊旗形 — 看跌持续"""
    if i < lookback: return False
    window = 20
    segment = c[max(0,i-lookback):i+1][-window:]
    if len(segment) < 15: return False
    try:
        slope = (segment[-1] - segment[0]) / len(segment)
        if slope > -0.005: return False
        pole_height = segment[0] - segment[-1]
        consolidation = segment[-5:]
        consolidation_range = np.max(consolidation) - np.min(consolidation)
        if consolidation_range > pole_height * 0.3: return False
        if c[i] < o[i] and ranges[i] > 0.8 * atr[i]: return False
        return True
    except Exception:
        return False

def is_bullish_triangle(o, h, l, c, atr, ranges, i, lookback=40):
    """上升三角形 — 看涨"""
    if i < lookback: return False
    window = 30
    segment_h = h[max(0,i-lookback):i+1][-window:]
    segment_l = l[max(0,i-lookback):i+1][-window:]
    if len(segment_h) < 20: return False
    try:
        recent_h = segment_h[-10:]
        recent_l = segment_l[-10:]
        resistance = np.max(recent_h)
        support = np.min(recent_l)
        slope_support = (recent_l[-1] - recent_l[0]) / len(recent_l)
        if abs(np.max(segment_h[-15:-10]) - resistance) / resistance > 0.03: return False
        if slope_support <= 0: return False
        if c[i] > resistance: return False
        if ranges[i] < 0.8 * atr[i]: return False
        return True
    except Exception:
        return False

def is_bearish_triangle(o, h, l, c, atr, ranges, i, lookback=40):
    """下降三角形 — 看跌"""
    if i < lookback: return False
    window = 30
    segment_h = h[max(0,i-lookback):i+1][-window:]
    segment_l = l[max(0,i-lookback):i+1][-window:]
    if len(segment_h) < 20: return False
    try:
        recent_h = segment_h[-10:]
        recent_l = segment_l[-10:]
        resistance = np.min(recent_h)
        support = np.max(recent_l)
        slope_resistance = (recent_h[-1] - recent_h[0]) / len(recent_h)
        if abs(np.min(segment_l[-15:-10]) - support) / support > 0.03: return False
        if slope_resistance >= 0: return False
        if c[i] < support: return False
        if ranges[i] < 0.8 * atr[i]: return False
        return True
    except Exception:
        return False

# ============ 核心5: 趋势判断 ============
def get_trend(close, period=20):
    if len(close) < period: return 'sideways'
    ma = close.rolling(period).mean().iloc[-1]
    last = close.iloc[-1]
    ma_prev = close.rolling(period).mean().iloc[-2]
    if last > ma and last > ma_prev: return 'up'
    if last < ma and last < ma_prev: return 'down'
    return 'sideways'

# ============ 核心6: 信号强度评分 ============
def score_signal(signal_type, confidence, close, atr_val, vol_now, vol_ma, position_score, bb_upper, bb_lower, bb_mid):
    """多重加权评分，0-100分"""
    score = 0
    score += min(confidence * 0.45, 45)  # 形态置信度权重45%
    
    # 成交量验证
    vol_ratio = vol_now / vol_ma if vol_ma > 0 else 1
    if vol_ratio >= 2.0: score += 18
    elif vol_ratio >= 1.5: score += 12
    elif vol_ratio >= 1.2: score += 6
    
    # 位置评分（布林带）
    price_pos = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    if signal_type in ['bullish_engulfing', 'morning_star', 'piercing', 'hammer', 'inverted_hs', 'falling_wedge', 'bull_flag', 'bullish_triangle', 'double_bottom']:
        if price_pos <= 0.2: score += 15  # 贴近布林下轨，强支撑
        elif price_pos <= 0.4: score += 10
        elif price_pos >= 0.8: score += 3
        else: score += 6
    else:
        if price_pos >= 0.8: score += 15  # 贴近布林上轨，强压力
        elif price_pos >= 0.6: score += 10
        elif price_pos <= 0.2: score += 3
        else: score += 6
    
    score += min(position_score * 0.22, 22)  # 趋势方向权重22%
    
    # ATR波动强度
    if atr_val > 0: score += min((atr_val / close) * 500, 15)
    
    return min(score, 100)

# ============ 核心7: 形态标注可视化 ============
def draw_annotated_kline(df, signals, symbol, save_path):
    """在K线图上标注识别到的形态区域"""
    n = len(df)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    for i in range(n):
        o, h, l, c = df.open.iloc[i], df.high.iloc[i], df.low.iloc[i], df.close.iloc[i]
        color = COLORS['bullish'] if c >= o else COLORS['bearish']
        body_bottom = min(o, c)
        body_height = abs(c - o) + 1e-6
        
        ax.add_patch(FancyBboxPatch(
            (i - 0.35, body_bottom), 0.7, body_height,
            boxstyle="round,pad=0.02", facecolor=color, edgecolor='none', linewidth=0
        ))
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
    
    # 布林带
    bb_up, bb_down, bb_mid = calc_bollinger_bands(df.close)
    x = np.arange(n)
    ax.plot(x, bb_up, color='#9E9E9E', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.plot(x, bb_down, color='#9E9E9E', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.plot(x, bb_mid, color='#9E9E9E', linewidth=0.5, linestyle=':', alpha=0.4)
    
    # 标注形态
    legend_patches = []
    for sig in signals:
        i, stype, conf = sig['index'], sig['type'], sig['confidence']
        if stype in ['bullish_engulfing', 'morning_star', 'piercing', 'hammer', 'inverted_hs', 'falling_wedge', 'bull_flag', 'bullish_triangle', 'double_bottom']:
            box_color = COLORS['bullish']
        elif stype in ['bearish_engulfing', 'evening_star', 'dark_cloud', 'shooting_star', 'head_shoulders', 'rising_wedge', 'bear_flag', 'bearish_triangle', 'double_top']:
            box_color = COLORS['bearish']
        else:
            box_color = COLORS['neutral']
        
        sig_range = 8
        rect = FancyBboxPatch(
            (i - sig_range - 0.5, df.low.iloc[max(0,i-sig_range):i+1].min() * 0.995),
            sig_range * 2 + 1,
            (df.high.iloc[max(0,i-sig_range):i+1].max() - df.low.iloc[max(0,i-sig_range):i+1].min()) * 1.01,
            boxstyle="round,pad=0.05",
            facecolor=box_color, alpha=0.15, edgecolor=box_color, linewidth=2, linestyle='--'
        )
        ax.add_patch(rect)
        
        label_map = {
            'bullish_engulfing': '看涨吞没', 'bearish_engulfing': '看跌吞没',
            'morning_star': '晨星', 'evening_star': '晚星',
            'piercing': '刺穿线', 'dark_cloud': '暗云盖顶',
            'hammer': '锤子线', 'shooting_star': '射击之星',
            'bullish_three_inside': '三内柱(牛)', 'bearish_three_inside': '三内柱(熊)',
            'double_bottom': 'W底', 'double_top': 'M顶',
            'head_shoulders': '头肩顶', 'inverted_hs': '倒头肩底',
            'rising_wedge': '上升楔形', 'falling_wedge': '下降楔形',
            'bull_flag': '牛旗形', 'bear_flag': '熊旗形',
            'bullish_triangle': '上升三角形', 'bearish_triangle': '下降三角形',
        }
        label = label_map.get(stype, stype)
        ax.annotate(
            f"✅{label} {conf:.0f}%",
            xy=(i, df.high.iloc[i]), xytext=(i, df.high.iloc[i] + (df.high.max()-df.low.min())*0.05),
            fontsize=7.5, color=box_color, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=box_color, alpha=0.9, linewidth=1.5)
        )
        
        legend_patches.append(mpatches.Patch(color=box_color, label=label, alpha=0.7))
    
    ax.set_xlim(-1, n)
    ax.set_ylim(df.low.min() * 0.98, df.high.max() * 1.03)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', fontsize=7,
                  framealpha=0.9, edgecolor='#BDBDBD')
    
    title = f"{symbol}  {' '.join([s['type'][:3] for s in signals[:3]])} {signals[0]['trend']}"
    ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['text'], pad=8)
    
    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', 
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()

# ============ 核心8: 全市场扫描 ============
def get_china_a_stocks():
    """获取全市场A股代码列表"""
    stocks = []
    try:
        df = pd.read_html('https://www.csindex.com.cn/zh-CN/indices/index-detail/000985#/indices/family/detail-data?indexCode=000985&tabIndex=2')[0]
        stocks = [(str(row['成分券代码']).zfill(6) + '.SH', row.get('证券简称', '')) 
                  for _, row in df.iterrows() if pd.notna(row.get('成分券代码'))]
    except Exception:
        pass
    
    if len(stocks) < 1000:
        indices = [
            ('000001.SS', '上证指数'), ('000002.SS', '万科A'), ('000004.SZ', '国华网安'),
            ('000005.SZ', 'ST星源'), ('000006.SZ', '深振业A'), ('000007.SZ', '全新好'),
            ('000008.SZ', '神州高铁'), ('000009.SZ', '中国宝安'), ('000010.SZ', '美丽生态'),
            ('000011.SZ', '深物业A'), ('000012.SZ', '南玻A'), ('000014.SZ', '沙河股份'),
            ('000016.SZ', '深康佳A'), ('000017.SZ', '深中华A'), ('000018.SZ', '神州长城'),
            ('000019.SZ', '深深宝A'), ('000020.SZ', '深华发A'), ('000021.SZ', '深科技'),
            ('000022.SZ', '深赤湾A'), ('000023.SZ', '深天地A'), ('000025.SZ', '特力A'),
            ('000026.SZ', '飞亚达A'), ('000027.SZ', '深圳能源'), ('000028.SZ', '国药一致'),
            ('000029.SZ', '深深房A'), ('000030.SZ', '富奥股份'), ('000031.SZ', '中粮地产'),
            ('000032.SZ', '深桑达A'), ('000033.SZ', '新都退'), ('000034.SZ', '神州数码'),
            ('000035.SZ', '中国天楹'), ('000036.SZ', '华联控股'), ('000037.SZ', '深南电A'),
            ('000038.SZ', '深大通'), ('000039.SZ', '中集集团'), ('000040.SZ', '东旭蓝天'),
            ('000042.SZ', '中洲控股'), ('000043.SZ', '中航善达'), ('000045.SZ', '深纺织A'),
            ('000046.SZ', '泛海控股'), ('000048.SZ', '康达尔'), ('000049.SZ', '德赛电池'),
            ('000050.SZ', '深天马A'), ('000055.SZ', '方大集团'), ('000056.SZ', '皇庭国际'),
            ('000058.SZ', '深赛格'), ('000059.SZ', '华锦股份'), ('000060.SZ', '中金岭南'),
            ('000061.SZ', '农产品'), ('000062.SZ', '深圳华强'), ('000063.SZ', '中兴通讯'),
            ('000065.SZ', '北方国际'), ('000066.SZ', '中国长城'), ('000068.SZ', '华控赛格'),
            ('000069.SZ', '华侨城A'), ('000070.SZ', '特发信息'), ('000078.SZ', '海王生物'),
            ('000088.SZ', '盐田港'), ('000089.SZ', '深圳机场'), ('000090.SZ', '天健集团'),
            ('000096.SZ', '广聚能源'), ('000099.SZ', '中信海直'), ('000100.SZ', 'TCL科技'),
            ('000150.SZ', '宜华健康'), ('000151.SZ', '中成股份'), ('000153.SZ', '丰原药业'),
            ('000155.SZ', '川能动力'), ('000156.SZ', '华数传媒'), ('000157.SZ', '中联重科'),
            ('000158.SZ', '常山北明'), ('000159.SZ', '国际实业'), ('000166.SZ', '申万宏源'),
            ('000301.SZ', '东方盛虹'), ('000333.SZ', '美的集团'), ('000338.SZ', '潍柴动力'),
            ('000400.SZ', '许继电气'), ('000401.SZ', '冀东水泥'), ('000402.SZ', '金融街'),
            ('000403.SZ', '双林股份'), ('000404.SZ', '长虹华意'), ('000407.SZ', '胜利股份'),
            ('000408.SZ', '藏格矿业'), ('000409.SZ', '云鼎科技'), ('000410.SZ', '沈阳机床'),
            ('000411.SZ', '英特集团'), ('000413.SZ', '东旭光电'), ('000415.SZ', '渤海租赁'),
            ('000416.SZ', '民生控股'), ('000417.SZ', '合肥百货'), ('000418.SZ', '小天鹅A'),
            ('000419.SZ', '通程控股'), ('000420.SZ', '吉林化纤'), ('000421.SZ', '南京公用'),
            ('000422.SZ', '湖北宜化'), ('000423.SZ', '东阿阿胶'), ('000425.SZ', '徐工机械'),
            ('000426.SZ', '兴业银行'), ('000428.SZ', '华天酒店'), ('000430.SZ', '张家界'),
            ('000488.SZ', '晨鸣纸业'), ('000501.SZ', '鄂武商A'), ('000502.SZ', '绿景控股'),
            ('000503.SZ', '国新健康'), ('000504.SZ', '南华生物'), ('000505.SZ', '珠江实业'),
            ('000506.SZ', '中润资源'), ('000507.SZ', '珠海港'), ('000509.SZ', '华塑控股'),
            ('000510.SZ', '新金路'), ('000513.SZ', '丽珠集团'), ('000514.SZ', '渝开发'),
            ('000516.SZ', '国际医学'), ('000517.SZ', '荣安地产'), ('000518.SZ', '四环生物'),
            ('000519.SZ', '中兵红箭'), ('000520.SZ', '长航凤凰'), ('000521.SZ', '长虹美菱'),
            ('000522.SZ', '白云山A'), ('000523.SZ', '广州浪奇'), ('000525.SZ', '红太阳'),
            ('000528.SZ', '柳工'), ('000529.SZ', '广弘控股'), ('000530.SZ', '冰山冷热'),
            ('000531.SZ', '穗恒运A'), ('000532.SZ', '华金资本'), ('000533.SZ', '万家乐'),
            ('000534.SZ', '万泽股份'), ('000536.SZ', '华映科技'), ('000537.SZ', '广宇发展'),
            ('000538.SZ', '云南白药'), ('000539.SZ', '粤电力A'), ('000540.SZ', '中天金融'),
            ('000541.SZ', '佛山照明'), ('000543.SZ', '皖能电力'), ('000544.SZ', '中原环保'),
            ('000545.SZ', '金浦钛业'), ('000546.SZ', '吉林高速'), ('000547.SZ', '航天发展'),
            ('000548.SZ', '湖南投资'), ('000550.SZ', '江铃汽车'), ('000551.SZ', '创元科技'),
            ('000552.SZ', '靖远煤电'), ('000553.SZ', '安道麦A'), ('000554.SZ', '泰山石油'),
            ('000555.SZ', '神州信息'), ('000557.SZ', '西部创业'), ('000558.SZ', '莱茵体育'),
            ('000559.SZ', '万向钱潮'), ('000560.SZ', '我爱我家'), ('000561.SZ', '烽火电子'),
            ('000563.SZ', '陕国投A'), ('000565.SZ', '渝三峡A'), ('000566.SZ', '海南海药'),
            ('000567.SZ', '海德股份'), ('000568.SZ', '泸州老窖'), ('000569.SZ', '长城特钢'),
            ('000570.SZ', '苏常柴A'), ('000571.SZ', '*ST大洲'), ('000581.SZ', '威孚高科'),
            ('000582.SZ', '北部湾港'), ('000583.SZ', '惠程科技'), ('000584.SZ', '哈工智能'),
            ('000585.SZ', '*ST东电'), ('000586.SZ', '汇源通信'), ('000587.SZ', '金洲慈航'),
            ('000589.SZ', '贵州轮胎'), ('000590.SZ', '启迪药业'), ('000591.SZ', '太阳能'),
            ('000592.SZ', '平潭发展'), ('000593.SZ', '德展健康'), ('000595.SZ', '宝塔实业'),
            ('000596.SZ', '古井贡酒'), ('000597.SZ', '东北制药'), ('000598.SZ', '兴蓉环境'),
            ('000599.SZ', '青岛双星'), ('000600.SZ', '建投能源'), ('000601.SZ', '韶钢松山'),
            ('000603.SZ', '盛达资源'), ('000605.SZ', '渤海股份'), ('000606.SZ', '顺利办'),
            ('000607.SZ', '华媒控股'), ('000608.SZ', '阳光股份'), ('000609.SZ', '中迪投资'),
            ('000610.SZ', '西安旅游'), ('000611.SZ', '天首发展'), ('000612.SZ', '焦作万方'),
            ('000613.SZ', '海南高速'), ('000615.SZ', '京汉股份'), ('000616.SZ', '海航投资'),
            ('000617.SZ', '中油资本'), ('000618.SZ', '吉林化纤'), ('000619.SZ', '海螺型材'),
            ('000620.SZ', '新华联'), ('000622.SZ', '恒立实业'), ('000623.SZ', '吉林敖东'),
            ('000625.SZ', '长安汽车'), ('000626.SZ', '远大控股'), ('000627.SZ', '天茂集团'),
            ('000628.SZ', '高新发展'), ('000629.SZ', '攀钢钒钛'), ('000630.SZ', '铜陵有色'),
            ('000631.SZ', '顺发恒业'), ('000632.SZ', '三木集团'), ('000633.SZ', '合金投资'),
            ('000635.SZ', '英力特'), ('000636.SZ', '风华高科'), ('000637.SZ', '茂化实华'),
            ('000638.SZ', '万方发展'), ('000639.SZ', '西王食品'), ('000650.SZ', '仁和药业'),
            ('000651.SZ', '格力电器'), ('000652.SZ', '泰达股份'), ('000655.SZ', '金岭矿业'),
            ('000656.SZ', '金科股份'), ('000657.SZ', '中钨高新'), ('000658.SZ', 'ST东海洋'),
            ('000659.SZ', '珠海中富'), ('000661.SZ', '长春高新'), ('000662.SZ', '天夏智慧'),
            ('000663.SZ', '永安林业'), ('000665.SZ', '湖北广电'), ('000666.SZ', '经纬纺机'),
            ('000667.SZ', '美好置业'), ('000668.SZ', '荣丰控股'), ('000669.SZ', '金鸿股份'),
            ('000670.SZ', '*ST盈方'), ('000671.SZ', '阳光城'), ('000672.SZ', '上峰水泥'),
            ('000673.SZ', '*ST当代'), ('000676.SZ', '智度股份'), ('000677.SZ', '恒丰浆纸'),
            ('000678.SZ', '襄阳轴承'), ('000679.SZ', '大连友谊'), ('000680.SZ', '山推股份'),
            ('000681.SZ', '视觉中国'), ('000682.SZ', '东方电子'), ('000683.SZ', '远兴能源'),
            ('000685.SZ', '中山公用'), ('000686.SZ', '东北证券'), ('000687.SZ', '华讯方舟'),
            ('000690.SZ', '宝新能源'), ('000691.SZ', '亚太实业'), ('000692.SZ', '惠天热电'),
            ('000695.SZ', '滨海能源'), ('000697.SZ', '炼石航空'), ('000698.SZ', '沈阳化工'),
            ('000700.SZ', '模塑科技'), ('000701.SZ', '厦门信达'), ('000702.SZ', '正虹科技'),
            ('000703.SZ', '恒逸石化'), ('000705.SZ', '浙江震元'), ('000707.SZ', '双环科技'),
            ('000708.SZ', '大冶特钢'), ('000709.SZ', '河钢股份'), ('000710.SZ', '贝瑞基因'),
            ('000711.SZ', '京蓝科技'), ('000712.SZ', '锦龙股份'), ('000713.SZ', '丰乐种业'),
            ('000715.SZ', '中兴商业'), ('000716.SZ', '黑芝麻'), ('000717.SZ', '韶钢松山'),
            ('000718.SZ', '苏宁环球'), ('000719.SZ', '中原传媒'), ('000720.SZ', '新能泰山'),
            ('000721.SZ', '西安饮食'), ('000722.SZ', '湖南发展'), ('000723.SZ', '美锦能源'),
            ('000725.SZ', '京东方A'), ('000726.SZ', '鲁泰A'), ('000727.SZ', '华东科技'),
            ('000728.SZ', '国元证券'), ('000729.SZ', '燕京啤酒'), ('000731.SZ', '四川美丰'),
            ('000732.SZ', '泰禾集团'), ('000733.SZ', '振华科技'), ('000735.SZ', '罗牛山'),
            ('000736.SZ', '中交地产'), ('000737.SZ', 'ST南风'), ('000738.SZ', '航发控制'),
            ('000739.SZ', '普洛药业'), ('000750.SZ', '国海证券'), ('000751.SZ', '锌业股份'),
            ('000752.SZ', '西藏发展'), ('000753.SZ', '漳州发展'), ('000755.SZ', '山西路桥'),
            ('000756.SZ', '新华制药'), ('000757.SZ', '浩物股份'), ('000758.SZ', '中色股份'),
            ('000759.SZ', '中百集团'), ('000760.SZ', '斯太尔'), ('000761.SZ', '本钢板材'),
            ('000762.SZ', '西藏矿业'), ('000763.SZ', '锦州石化'), ('000766.SZ', '通化金马'),
            ('000767.SZ', '漳泽电力'), ('000768.SZ', '中航西飞'), ('000769.SZ', '盛运退'),
            ('000776.SZ', '广发证券'), ('000777.SZ', '中核科技'), ('000778.SZ', '新兴铸管'),
            ('000779.SZ', '甘咨询'), ('000780.SZ', '平庄能源'), ('000782.SZ', '美达股份'),
            ('000783.SZ', '长江证券'), ('000785.SZ', '居然之家'), ('000786.SZ', '北新建材'),
            ('000788.SZ', '北大医药'), ('000789.SZ', '万年青'), ('000790.SZ', '华神科技'),
            ('000791.SZ', '甘肃电投'), ('000792.SZ', '盐湖股份'), ('000793.SZ', '华闻集团'),
            ('000795.SZ', '英洛华'), ('000796.SZ', '凯撒旅业'), ('000797.SZ', '中国武夷'),
            ('000798.SZ', '中水渔业'), ('000799.SZ', '酒鬼酒'), ('000800.SZ', '一汽轿车'),
            ('000801.SZ', '四川九洲'), ('000802.SZ', '北京文化'), ('000803.SZ', '金宇车城'),
            ('000806.SZ', '*ST银河'), ('000807.SZ', '云铝股份'), ('000808.SZ', '泸州老窖'),
            ('000809.SZ', '铁岭新城'), ('000810.SZ', '创维数字'), ('000811.SZ', '冰轮环境'),
            ('000812.SZ', '陕西金叶'), ('000813.SZ', '德展健康'), ('000815.SZ', '美利云'),
            ('000816.SZ', '江苏新能'), ('000818.SZ', '航锦科技'), ('000819.SZ', '岳阳兴长'),
            ('000821.SZ', '京山轻机'), ('000822.SZ', '山东海化'), ('000823.SZ', '超声电子'),
            ('000825.SZ', '太钢不锈'), ('000826.SZ', '启迪环境'), ('000828.SZ', '东莞控股'),
            ('000829.SZ', '天音控股'), ('000830.SZ', '鲁西化工'), ('000831.SZ', '五矿稀土'),
            ('000833.SZ', '粤桂股份'), ('000835.SZ', '长城动漫'), ('000836.SZ', '富通鑫茂'),
            ('000837.SZ', '秦川机床'), ('000838.SZ', '财信发展'), ('000839.SZ', '中信国安'),
            ('000848.SZ', '承德露露'), ('000850.SZ', '华茂股份'), ('000851.SZ', '高鸿股份'),
            ('000852.SZ', '石化机械'), ('000856.SZ', '冀东装备'), ('000858.SZ', '五粮液'),
            ('000859.SZ', '国风塑业'), ('000860.SZ', '顺鑫农业'), ('000861.SZ', '海印股份'),
            ('000862.SZ', '银星能源'), ('000863.SZ', '三湘印象'), ('000868.SZ', '安凯客车'),
            ('000869.SZ', '张裕A'), ('000875.SZ', '吉电股份'), ('000876.SZ', '新希望'),
            ('000877.SZ', '天山股份'), ('000878.SZ', '云南铜业'), ('000880.SZ', '潍柴重机'),
            ('000881.SZ', '中广核技'), ('000882.SZ', '华联股份'), ('000883.SZ', '湖北能源'),
            ('000885.SZ', '城发环境'), ('000886.SZ', '海南高速'), ('000887.SZ', '中鼎股份'),
            ('000888.SZ', '峨眉山A'), ('000889.SZ', '中嘉博创'), ('000890.SZ', '法尔胜'),
            ('000892.SZ', '欢瑞世纪'), ('000893.SZ', '亚钾国际'), ('000895.SZ', '双汇发展'),
            ('000897.SZ', '津滨发展'), ('000898.SZ', '鞍钢股份'), ('000899.SZ', '赣能股份'),
            ('000900.SZ', '现代投资'), ('000901.SZ', '航天科技'), ('000902.SZ', '新洋丰'),
            ('000903.SZ', '云内动力'), ('000905.SZ', '厦门港务'), ('000906.SZ', '浙商中拓'),
            ('000908.SZ', '景峰医药'), ('000909.SZ', '数源科技'), ('000910.SZ', '大亚圣象'),
            ('000911.SZ', '南宁糖业'), ('000912.SZ', '泸天化'), ('000913.SZ', '钱江摩托'),
            ('000915.SZ', '山大华特'), ('000916.SZ', '华北高速'), ('000917.SZ', '电广传媒'),
            ('000918.SZ', '嘉凯城'), ('000919.SZ', '金陵药业'), ('000920.SZ', '南方汇通'),
            ('000921.SZ', '海信家电'), ('000922.SZ', '佳电股份'), ('000923.SZ', '河钢资源'),
            ('000925.SZ', '众合科技'), ('000926.SZ', '福星股份'), ('000927.SZ', '中国铁物'),
            ('000928.SZ', '中钢国际'), ('000929.SZ', '兰州黄河'), ('000930.SZ', '中粮科技'),
            ('000931.SZ', '中关村'), ('000932.SZ', '华菱钢铁'), ('000933.SZ', '神火股份'),
            ('000935.SZ', '四川双马'), ('000936.SZ', '华西股份'), ('000937.SZ', '冀中能源'),
            ('000938.SZ', '紫光股份'), ('000939.SZ', '*ST凯迪'), ('000948.SZ', '南天信息'),
            ('000949.SZ', '新乡化纤'), ('000950.SZ', '重药控股'), ('000951.SZ', '中国重汽'),
            ('000952.SZ', '广济药业'), ('000953.SZ', '河池化工'), ('000955.SZ', '欣龙控股'),
            ('000956.SZ', '中原油气'), ('000957.SZ', '中通客车'), ('000958.SZ', '东方能源'),
            ('000959.SZ', '首钢股份'), ('000960.SZ', '锡业股份'), ('000961.SZ', '中南建设'),
            ('000962.SZ', '东方钽业'), ('000963.SZ', '华东医药'), ('000965.SZ', '天保基建'),
            ('000966.SZ', '长源电力'), ('000967.SZ', '上风高科'), ('000968.SZ', '蓝焰控股'),
            ('000969.SZ', '安泰科技'), ('000970.SZ', '中科三环'), ('000971.SZ', '*ST高升'),
            ('000972.SZ', 'ST中基'), ('000973.SZ', '佛塑科技'), ('000975.SZ', '银泰黄金'),
            ('000976.SZ', '华铁股份'), ('000977.SZ', '浪潮信息'), ('000978.SZ', '桂林旅游'),
            ('000979.SZ', '中弘股份'), ('000980.SZ', '众泰汽车'), ('000981.SZ', 'ST银亿'),
            ('000982.SZ', '*ST中线'), ('000983.SZ', '西山煤电'), ('000985.SZ', '北大医药'),
            ('000987.SZ', '越秀金控'), ('000988.SZ', '华工科技'), ('000989.SZ', '九芝堂'),
            ('000990.SZ', '诚志股份'), ('000991.SZ', '通达股份'), ('000993.SZ', '闽东电力'),
            ('600000.SS', '浦发银行'), ('600001.SS', '邯郸钢铁'), ('600002.SS', '齐鲁石化'),
            ('600003.SS', '东北高速'), ('600004.SS', '白云机场'), ('600005.SS', '武钢股份'),
            ('600006.SS', '东风汽车'), ('600007.SS', '中国国贸'), ('600008.SS', '首创股份'),
            ('600009.SS', '上海机场'), ('600010.SS', '包钢股份'), ('600011.SS', '华能国际'),
            ('600012.SS', '皖通高速'), ('600015.SS', '华夏银行'), ('600016.SS', '民生银行'),
            ('600017.SS', '日照港'), ('600018.SS', '上港集团'), ('600019.SS', '宝钢股份'),
            ('600020.SS', '中原高速'), ('600021.SS', '上海电力'), ('600022.SS', '山东钢铁'),
            ('600023.SS', '浙能电力'), ('600026.SS', '中远海能'), ('600027.SS', '华电国际'),
            ('600028.SS', '中国石化'), ('600029.SS', '南方航空'), ('600030.SS', '中信证券'),
            ('600031.SS', '三一重工'), ('600032.SS', '福建水泥'), ('600033.SS', '福建高速'),
            ('600036.SS', '招商银行'), ('600037.SS', '歌华有线'), ('600038.SS', '中直股份'),
            ('600039.SS', '四川路桥'), ('600048.SS', '保利地产'), ('600050.SS', '中国联通'),
            ('600051.SS', '宁波联合'), ('600052.SS', '浙江广厦'), ('600053.SS', '九鼎投资'),
            ('600054.SS', '黄山旅游'), ('600055.SS', '万东医疗'), ('600056.SS', '中国医药'),
            ('600057.SS', '厦门信达'), ('600058.SS', '五矿发展'), ('600059.SS', '古越龙山'),
            ('600060.SS', '海信视像'), ('600061.SS', '中纺投资'), ('600062.SS', '华润双鹤'),
            ('600063.SS', '皖维高新'), ('600064.SS', '南京高科'), ('600066.SS', '宇通客车'),
            ('600067.SS', '冠城大通'), ('600068.SS', '葛洲坝'), ('600069.SZ', '银鸽投资'),
            ('600070.SS', '浙江富润'), ('600071.SS', '凤凰光学'), ('600072.SS', '中船科技'),
            ('600073.SS', '上海梅林'), ('600074.SS', '*ST保千'), ('600075.SS', '新疆天业'),
            ('600076.SS', '康欣新材'), ('600077.SS', '宋都股份'), ('600078.SS', '澄星股份'),
            ('600079.SS', '人福医药'), ('600080.SS', '金花股份'), ('600081.SS', '东风科技'),
            ('600082.SS', '海泰发展'), ('600083.SS', '*ST博信'), ('600084.SS', '中葡股份'),
            ('600085.SS', '同仁堂'), ('600086.SS', '东方金钰'), ('600087.SS', '退市油运'),
            ('600088.SS', '中视传媒'), ('600089.SS', '特变电工'), ('600090.SS', '同济堂'),
            ('600093.SS', '禾嘉股份'), ('600094.SS', '大名城'), ('600095.SS', '湘财股份'),
            ('600096.SS', '云天化'), ('600097.SS', '开创国际'), ('600098.SS', '广州发展'),
            ('600099.SS', '林海股份'), ('600100.SS', '同方股份'), ('600101.SS', '明星电力'),
            ('600104.SS', '上汽集团'), ('600105.SS', '永鼎股份'), ('600106.SS', '重庆路桥'),
            ('600107.SS', '美尔雅'), ('600108.SS', '亚盛集团'), ('600109.SS', '国金证券'),
            ('600110.SS', '诺德股份'), ('600111.SS', '北方稀土'), ('600112.SS', '*ST天成'),
            ('600113.SS', '浙江东日'), ('600114.SS', '东睦股份'), ('600115.SS', '东方航空'),
            ('600116.SS', '三峡水利'), ('600117.SS', '西宁特钢'), ('600118.SS', '中国卫星'),
            ('600119.SS', '长江投资'), ('600120.SS', '浙江东方'), ('600121.SS', '郑州煤电'),
            ('600122.SS', '宏图高科'), ('600123.SS', '兰花科创'), ('600125.SS', '铁龙物流'),
            ('600126.SS', '杭钢股份'), ('600127.SS', '金健米业'), ('600128.SS', '弘业股份'),
            ('600129.SS', '太极集团'), ('600130.SS', '波导股份'), ('600131.SS', '岷江水电'),
            ('600132.SS', '重庆啤酒'), ('600133.SS', '东湖高新'), ('600135.SS', '乐凯胶片'),
            ('600136.SS', '当代文体'), ('600137.SS', '浪莎股份'), ('600138.SS', '中青旅'),
            ('600139.SS', '西部资源'), ('600141.SS', '兴发集团'), ('600143.SS', '金发科技'),
            ('600150.SS', '中国船舶'), ('600151.SS', '航天机电'), ('600152.SS', '维科技术'),
            ('600153.SS', '建发股份'), ('600155.SS', '华创阳安'), ('600157.SS', '永泰能源'),
            ('600158.SS', '中体产业'), ('600159.SS', '大龙地产'), ('600160.SS', '巨化股份'),
            ('600161.SS', '天坛生物'), ('600162.SS', '香江控股'), ('600163.SS', '中闽能源'),
            ('600165.SS', '新日恒力'), ('600166.SS', '福田汽车'), ('600167.SS', '联美控股'),
            ('600168.SS', '武汉控股'), ('600170.SS', '上海建工'), ('600171.SS', '上海贝岭'),
            ('600172.SS', '黄河旋风'), ('600173.SS', '卧龙地产'), ('600175.SS', '美都能源'),
            ('600176.SS', '中国巨石'), ('600177.SS', '雅戈尔'), ('600178.SS', '东安动力'),
            ('600179.SS', '*ST安通'), ('600180.SS', '瑞茂通'), ('600182.SS', 'S佳通'),
            ('600183.SS', '生益科技'), ('600184.SS', '光电股份'), ('600185.SS', '格力地产'),
            ('600186.SS', '莲花健康'), ('600187.SS', '国中水务'), ('600188.SS', '兖州煤业'),
            ('600189.SS', '吉林森工'), ('600190.SS', '锦州港'), ('600191.SS', '华资实业'),
            ('600195.SS', '中牧股份'), ('600196.SS', '复星医药'), ('600197.SS', '伊力特'),
            ('600199.SS', '金种子酒'), ('600200.SS', '江苏吴中'), ('600201.SS', '生物股份'),
            ('600203.SS', '福日电子'), ('600206.SS', '有研新材'), ('600208.SS', '新湖中宝'),
            ('600210.SS', '紫江企业'), ('600211.SS', '西藏药业'), ('600212.SS', '江泉实业'),
            ('600213.SS', '亚星客车'), ('600216.SS', '浙江医药'), ('600219.SS', '南山铝业'),
            ('600221.SS', '海航控股'), ('600222.SS', '太龙药业'), ('600223.SS', '鲁商发展'),
            ('600225.SS', '天津松江'), ('600226.SS', '瀚叶股份'), ('600227.SS', '赤天化'),
            ('600229.SS', '城市传媒'), ('600230.SS', '沧州大化'), ('600231.SS', '凌钢股份'),
            ('600232.SS', '金鹰股份'), ('600233.SS', '圆通速递'), ('600235.SS', '民丰特纸'),
            ('600236.SS', '桂冠电力'), ('600237.SS', '铜峰电子'), ('600238.SS', 'ST椰岛'),
            ('600239.SS', '云南城投'), ('600240.SS', '*ST华业'), ('600241.SS', '时代万恒'),
            ('600242.SS', '中昌数据'), ('600243.SS', '青海华鼎'), ('600246.SS', '万通发展'),
            ('600248.SS', '延长化建'), ('600251.SS', '冠农股份'), ('600252.SS', '中恒集团'),
            ('600258.SS', '首旅酒店'), ('600259.SS', '广晟有色'), ('600260.SS', '凯乐科技'),
            ('600261.SS', '阳光照明'), ('600262.SS', '北方股份'), ('600267.SS', '海正药业'),
            ('600271.SS', '航天信息'), ('600276.SS', '恒瑞医药'), ('600278.SS', '东方创业'),
            ('600282.SS', '南钢股份'), ('600285.SS', '羚锐制药'), ('600288.SS', '大恒科技'),
            ('600289.SS', 'ST信通'), ('600290.SS', 'ST华仪'), ('600291.SS', '西水股份'),
            ('600292.SS', '远达环保'), ('600295.SS', '鄂尔多斯'), ('600297.SS', '广汇汽车'),
            ('600298.SS', '安琪酵母'), ('600299.SS', '安迪苏'), ('600300.SS', '维维股份'),
            ('600309.SS', '万华化学'), ('600310.SS', '桂东电力'), ('600311.SS', '荣华实业'),
            ('600312.SS', '平高电气'), ('600315.SS', '上海家化'), ('600316.SS', '洪都航空'),
            ('600318.SS', '新力金融'), ('600329.SS', '中新药业'), ('600330.SS', '天通股份'),
            ('600335.SS', '国机汽车'), ('600336.SS', '澳柯玛'), ('600340.SS', '华夏幸福'),
            ('600343.SS', '航天动力'), ('600346.SS', '恒力石化'), ('600348.SS', '阳泉煤业'),
            ('600350.SS', '山东高速'), ('600351.SS', '亚宝药业'), ('600352.SS', '浙江龙盛'),
            ('600353.SS', '旭光电子'), ('600358.SS', '国旅联合'), ('600359.SS', '新农开发'),
            ('600362.SS', '江西铜业'), ('600363.SS', '联创光电'), ('600368.SS', '五洲交通'),
            ('600373.SS', '中文传媒'), ('600375.SS', '华菱星马'), ('600376.SS', '首开股份'),
            ('600377.SS', '宁沪高速'), ('600378.SS', '昊华科技'), ('600380.SS', '健康元'),
            ('600383.SS', '金地集团'), ('600385.SS', '山东金泰'), ('600387.SS', '海越能源'),
            ('600388.SS', '龙净环保'), ('600389.SS', '江山股份'), ('600390.SS', '五矿资本'),
            ('600395.SS', '盘江股份'), ('600398.SS', '海澜之家'), ('600399.SS', 'ST抚钢'),
            ('600400.SS', '红豆股份'), ('600406.SS', '国电南瑞'), ('600409.SS', '三友化工'),
            ('600410.SS', '华胜天成'), ('600415.SS', '小商品城'), ('600416.SS', '湘电股份'),
            ('600418.SS', '江淮汽车'), ('600419.SS', '天润乳业'), ('600422.SS', '昆药集团'),
            ('600426.SS', '华鲁恒升'), ('600428.SS', '中远海特'), ('600429.SS', '三元股份'),
            ('600432.SS', '吉恩镍业'), ('600433.SS', '冠豪高新'), ('600435.SS', '北方导航'),
            ('600436.SS', '片仔癀'), ('600438.SS', '通威股份'), ('600459.SS', '贵研铂业'),
            ('600460.SS', '士兰微'), ('600461.SS', '洪城水业'), ('600466.SS', '蓝光发展'),
            ('600467.SS', '好当家'), ('600468.SS', '百利电气'), ('600477.SS', '杭萧钢构'),
            ('600478.SS', '科力远'), ('600480.SS', '凌云股份'), ('600481.SS', '双良节能'),
            ('600482.SS', '中国动力'), ('600483.SS', '福能股份'), ('600487.SS', '亨通光电'),
            ('600489.SS', '中金黄金'), ('600490.SS', '鹏欣资源'), ('600491.SS', '龙元建设'),
            ('600493.SS', '凤竹纺织'), ('600495.SS', '晋西车轴'), ('600497.SS', '驰宏锌锗'),
            ('600498.SS', '烽火通信'), ('600499.SS', '科达洁能'), ('600500.SS', '中化国际'),
            ('600501.SS', '航天晨光'), ('600502.SS', '安徽建工'), ('600503.SS', '华丽家族'),
            ('600507.SS', '方大特钢'), ('600508.SS', '上海能源'), ('600512.SS', '腾达建设'),
            ('600516.SS', '方大炭素'), ('600517.SS', '置信电气'), ('600519.SS', '贵州茅台'),
            ('600522.SS', '中天科技'), ('600523.SS', '贵航股份'), ('600526.SS', '*ST菲达'),
            ('600527.SS', '江南高纤'), ('600528.SS', '中铁工业'), ('600529.SS', '山东药玻'),
            ('600535.SS', '天士力'), ('600536.SS', '中国软件'), ('600547.SS', '山东黄金'),
            ('600549.SS', '厦门钨业'), ('600550.SS', '保变电气'), ('600551.SS', '时代出版'),
            ('600559.SS', '老白干酒'), ('600566.SS', '济川药业'), ('600567.SS', '山鹰纸业'),
            ('600568.SS', '中珠医疗'), ('600570.SS', '恒生电子'), ('600571.SS', '信雅达'),
            ('600572.SS', '康恩贝'), ('600573.SS', '惠泉啤酒'), ('600575.SS', '淮河能源'),
            ('600577.SS', '精达股份'), ('600578.SS', '京能电力'), ('600579.SS', '克劳斯'),
            ('600580.SS', '卧龙电驱'), ('600582.SS', '天地科技'), ('600583.SS', '海油工程'),
            ('600584.SS', '长电科技'), ('600585.SS', '海螺水泥'), ('600587.SS', '新华医疗'),
            ('600588.SS', '用友网络'), ('600592.SS', '龙溪股份'), ('600595.SS', '*ST中绒'),
            ('600596.SS', '新安股份'), ('600597.SS', '光明乳业'), ('600598.SS', '北大荒'),
            ('600600.SS', '青岛啤酒'), ('600601.SS', 'ST方科'), ('600604.SS', '市北高新'),
            ('600605.SS', '汇通能源'), ('600606.SS', '绿地控股'), ('600608.SS', 'ST沪科'),
            ('600609.SS', '金杯汽车'), ('600610.SS', '中毅达'), ('600611.SS', '大众交通'),
            ('600612.SS', '老凤祥'), ('600613.SS', '神奇制药'), ('600615.SS', '*ST丰山'),
            ('600616.SS', '金枫酒业'), ('600617.SS', '国新能源'), ('600618.SS', '氯碱化工'),
            ('600619.SS', '海立股份'), ('600620.SS', '绿地控股'), ('600621.SS', '华鑫股份'),
            ('600622.SS', '光大嘉宝'), ('600623.SS', '华谊集团'), ('600624.SS', '复旦复华'),
            ('600626.SS', '申达股份'), ('600630.SS', '龙头股份'), ('600635.SS', '大众公用'),
            ('600637.SS', '东方明珠'), ('600638.SS', '新黄浦'), ('600639.SS', '浦东金桥'),
            ('600640.SS', '号百控股'), ('600641.SS', '万业企业'), ('600642.SS', '申能股份'),
            ('600643.SS', '爱建集团'), ('600644.SS', '乐山电力'), ('600647.SS', '同达创业'),
            ('600648.SS', '外高桥'), ('600649.SS', '城投控股'), ('600650.SS', '锦江在线'),
            ('600651.SS', '飞乐音响'), ('600652.SS', 'ST游久'), ('600653.SS', '中华控股'),
            ('600654.SS', 'ST中安'), ('600655.SS', '豫园股份'), ('600658.SS', '电子城'),
            ('600660.SS', '福耀玻璃'), ('600661.SS', '昂立教育'), ('600662.SS', '强生控股'),
            ('600663.SS', '陆家嘴'), ('600664.SS', '哈药股份'), ('600665.SS', '天地源'),
            ('600666.SS', '奥瑞德'), ('600667.SS', '太极实业'), ('600668.SS', '尖峰集团'),
            ('600673.SS', '东阳光'), ('600674.SS', '川投能源'), ('600675.SS', '中华企业'),
            ('600676.SS', '交运股份'), ('600679.SS', '上海凤凰'), ('600680.SS', '上海普天'),
            ('600682.SS', '南京新百'), ('600683.SS', '京投发展'), ('600684.SS', '珠江实业'),
            ('600685.SS', '中船防务'), ('600688.SS', '上海石化'), ('600690.SS', '海尔智家'),
            ('600692.SS', '亚通股份'), ('600693.SS', '东百集团'), ('600694.SS', '大商股份'),
            ('600695.SS', '绿庭投资'), ('600696.SS', 'ST岩石'), ('600697.SS', '欧亚集团'),
            ('600698.SS', '湖南天雁'), ('600699.SS', '均胜电子'), ('600703.SS', '三安光电'),
            ('600704.SS', '物产中大'), ('600705.SS', '中航资本'), ('600706.SS', '曲江文旅'),
            ('600707.SS', '彩虹股份'), ('600711.SS', '盛屯矿业'), ('600712.SS', '南宁百货'),
            ('600713.SS', '南京医药'), ('600714.SS', '金瑞矿业'), ('600716.SS', '凤凰股份'),
            ('600717.SS', '天津港'), ('600718.SS', '东软集团'), ('600720.SS', '祁连山'),
            ('600723.SS', '首商股份'), ('600724.SS', '宁波富达'), ('600725.SZ', 'ST云维'),
            ('600726.SS', '华电能源'), ('600727.SS', '鲁北化工'), ('600728.SS', '佳都科技'),
            ('600729.SS', '重庆百货'), ('600730.SS', '中国高科'), ('600731.SS', '湖南海利'),
            ('600733.SS', '北汽蓝谷'), ('600734.SS', '实达集团'), ('600735.SS', '新华锦'),
            ('600736.SS', '苏州高新'), ('600737.SS', '中粮糖业'), ('600738.SS', '丽尚国潮'),
            ('600739.SS', '辽宁成大'), ('600741.SS', '华域汽车'), ('600742.SS', '一汽富维'),
            ('600743.SS', '华远地产'), ('600744.SS', '白银有色'), ('600745.SS', '闻泰科技'),
            ('600746.SS', '江苏索普'), ('600747.SS', '大连控股'), ('600748.SS', '上实发展'),
            ('600750.SS', '江中药业'), ('600751.SS', '海航科技'), ('600754.SS', '锦江酒店'),
            ('600755.SS', '厦门国贸'), ('600756.SS', '浪潮软件'), ('600757.SS', '长江传媒'),
            ('600758.SS', '红阳能源'), ('600759.SS', 'ST洲际'), ('600760.SS', '中航沈飞'),
            ('600763.SS', '通策医疗'), ('600764.SS', '中国海防'), ('600765.SS', '中航重机'),
            ('600770.SS', '综艺股份'), ('600771.SS', '广誉远'), ('600773.SS', '西藏城投'),
            ('600774.SS', '汉商集团'), ('600775.SS', '南京熊猫'), ('600776.SS', '东方通信'),
            ('600777.SS', '新潮能源'), ('600779.SS', '水井坊'), ('600781.SS', 'ST辅仁'),
            ('600782.SS', '新钢股份'), ('600783.SS', '鲁信创投'), ('600784.SS', '鲁银投资'),
            ('600785.SS', '新华百货'), ('600787.SS', '中储股份'), ('600789.SS', '鲁抗医药'),
            ('600790.SS', '轻纺城'), ('600791.SS', '京能置业'), ('600795.SS', '国电电力'),
            ('600796.SS', '钱江生化'), ('600797.SS', '浙大网新'), ('600798.SS', '宁波海运'),
            ('600800.SS', '天津磁卡'), ('600801.SS', '华新水泥'), ('600802.SS', '福建水泥'),
            ('600803.SS', '新奥股份'), ('600804.SS', '鹏博士'), ('600805.SS', '悦达投资'),
            ('600809.SS', '山西汾酒'), ('600810.SS', '神马股份'), ('600811.SS', '东方集团'),
            ('600812.SS', '华北制药'), ('600814.SS', '杭州解百'), ('600815.SS', '厦工股份'),
            ('600816.SS', '安信信托'), ('600820.SS', '隧道股份'), ('600821.SS', '津劝业'),
            ('600822.SS', '上海物贸'), ('600823.SS', '世茂股份'), ('600824.SS', '益民集团'),
            ('600825.SS', '新华传媒'), ('600826.SS', '兰生股份'), ('600827.SS', '百联股份'),
            ('600828.SS', '茂业商业'), ('600829.SS', '人民同泰'), ('600830.SS', '香溢融通'),
            ('600831.SS', '广电网络'), ('600833.SS', '第一医药'), ('600835.SS', '上海机电'),
            ('600836.SS', '界龙实业'), ('600838.SS', '上海九百'), ('600839.SS', '四川长虹'),
            ('600841.SS', '上柴股份'), ('600843.SS', '上工申贝'), ('600844.SS', '丹化科技'),
            ('600845.SS', '宝信软件'), ('600846.SS', '同济科技'), ('600847.SS', '万里股份'),
            ('600848.SS', '上海临港'), ('600850.SS', '华东电脑'), ('600855.SS', '航天长峰'),
            ('600857.SS', '宁波中百'), ('600859.SS', '王府井'), ('600862.SS', '中航高科'),
            ('600863.SS', '内蒙华电'), ('600864.SS', '哈投股份'), ('600865.SS', '百大集团'),
            ('600867.SS', '通化东宝'), ('600868.SS', '梅雁吉祥'), ('600873.SS', '梅花生物'),
            ('600874.SS', '创业环保'), ('600875.SS', '东方电气'), ('600876.SS', '洛阳玻璃'),
            ('600877.SS', 'ST电能'), ('600879.SS', '航天电子'), ('600881.SS', '亚泰集团'),
            ('600882.SS', '妙可蓝多'), ('600883.SS', '博闻科技'), ('600884.SS', '杉杉股份'),
            ('600885.SS', '宏发股份'), ('600886.SS', '国投电力'), ('600887.SS', '伊利股份'),
            ('600888.SS', '新疆众和'), ('600893.SS', '航发动力'), ('600894.SS', '广日股份'),
            ('600895.SS', '张江高科'), ('600900.SS', '长江电力'), ('600905.SS', '三峡能源'),
            ('600917.SS', '重庆燃气'), ('600918.SS', '中泰证券'), ('600919.SS', '江苏银行'),
            ('600926.SS', '杭州银行'), ('600928.SS', '西安银行'), ('600936.SS', '广西广电'),
            ('600958.SS', '东方证券'), ('600959.SS', '江苏有线'), ('600961.SS', '株冶集团'),
            ('600963.SS', '岳阳林纸'), ('600965.SS', '福成股份'), ('600967.SS', '北方股份'),
            ('600968.SS', '海油发展'), ('600971.SS', '恒源煤电'), ('600973.SS', '宝胜股份'),
            ('600975.SS', '新五丰'), ('600976.SS', '健民集团'), ('600977.SS', '中国电影'),
            ('600978.SS', '宜华生活'), ('600979.SS', '广安爱众'), ('600981.SS', '汇鸿集团'),
            ('600982.SS', '宁波热电'), ('600985.SS', '淮北矿业'), ('600986.SS', '科达股份'),
            ('600987.SS', '航民股份'), ('600989.SS', '宝丰能源'), ('600990.SS', '四创电子'),
            ('600995.SS', '文山电力'), ('600996.SS', '贵广网络'), ('600997.SS', '开滦股份'),
            ('600998.SS', '九州通'), ('600999.SS', '招商证券'), ('601006.SS', '大秦铁路'),
            ('601008.SS', '连云港'), ('601012.SZ', '隆基绿能'), ('601016.SS', '节能风电'),
            ('601018.SS', '宁波港'), ('601066.SS', '中信建投'), ('601088.SS', '中国神华'),
            ('601117.SS', '中国化学'), ('601138.SS', '工业富联'), ('601139.SS', '深圳燃气'),
            ('601166.SS', '兴业银行'), ('601168.SS', '西部矿业'), ('601169.SS', '北京银行'),
            ('601186.SS', '中国铁建'), ('601198.SS', '东兴证券'), ('601211.SS', '国泰君安'),
            ('601229.SS', '上海银行'), ('601288.SS', '农业银行'), ('601318.SS', '中国平安'),
            ('601328.SS', '交通银行'), ('601336.SS', '新华保险'), ('601398.SS', '工商银行'),
            ('601601.SS', '中国太保'), ('601618.SS', '中国中冶'), ('601628.SS', '中国人寿'),
            ('601658.SS', '邮储银行'), ('601668.SS', '中国建筑'), ('601669.SS', '中国电建'),
            ('601688.SS', '华泰证券'), ('601698.SS', '中国卫通'), ('601728.SS', '中国电信'),
            ('601766.SS', '中国中车'), ('601800.SS', '中国交建'), ('601816.SS', '京沪高铁'),
            ('601818.SS', '光大银行'), ('601857.SS', '中国石油'), ('601858.SS', '中国科传'),
            ('601878.SS', '浙商证券'), ('601888.SS', '中国中免'), ('601899.SS', '紫金矿业'),
            ('601919.SS', '中远海控'), ('601985.SS', '中国核电'), ('601988.SS', '中国银行'),
            ('601989.SS', '中国重工'), ('601990.SS', '南京证券'), ('601998.SS', '中信银行'),
            ('603087.SS', '甘李药业'), ('603259.SS', '药明康德'), ('603288.SS', '海天味业'),
            ('603303.SS', '得邦照明'), ('603605.SS', '珀莱雅'), ('603799.SS', '华友钴业'),
            ('603986.SS', '兆易创新'), ('688041.SS', '海光信息'), ('688981.SS', '中芯国际'),
            ('300001.SZ', '特锐德'), ('300002.SZ', '神州泰岳'), ('300003.SZ', '乐普医疗'),
            ('300015.SZ', '爱尔眼科'), ('300033.SZ', '同花顺'), ('300059.SZ', '东方财富'),
            ('300122.SZ', '智飞生物'), ('300124.SZ', '汇川技术'), ('300142.SZ', '沃森生物'),
            ('300223.SZ', '北京君正'), ('300274.SZ', '阳光电源'), ('300347.SZ', '泰格医药'),
            ('300363.SZ', '博腾股份'), ('300408.SZ', '三环集团'), ('300450.SZ', '先导智能'),
            ('300496.SZ', '中科创达'), ('300529.SZ', '健帆生物'), ('300595.SZ', '欧普康视'),
            ('300601.SZ', '康泰生物'), ('300750.SZ', '宁德时代'), ('300760.SZ', '迈瑞医疗'),
            ('300759.SZ', '康龙化成'), ('300122.SZ', '智飞生物'),
        ]
        stocks = indices
    
    return stocks

# ============ 核心9: 单标的形态识别 ============
def detect_signals(df: pd.DataFrame, symbol: str):
    """对单只标的进行全形态检测"""
    if len(df) < 25: return [], {}, []
    
    n = len(df)
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values, 
    body = np.abs(c - o)
    ranges = h - l
    body_ratio = np.where(ranges > 0, body / ranges, 0)
    atr = calc_atr(pd.Series(h), pd.Series(l), pd.Series(c))
    bb_upper, bb_lower, bb_mid = calc_bollinger_bands(df.close)
    vol_ma = calc_volume_ma(df.volume)
    
    trend = get_trend(df.close)
    
    trend_score = {'up': 1.0, 'sideways': 0.5, 'down': 0.0}[trend]
    
    signals = []
    
    # 基础形态检测（PDF8种）
    for i in range(2, n):
        cur_close = c[i]
        atr_val = atr[i] if not np.isnan(atr[i]) else 0
        vol_now = df.volume.iloc[i]
        vol_ma_val = vol_ma.iloc[i] if not np.isnan(vol_ma.iloc[i]) and vol_ma.iloc[i] > 0 else 1
        bb_u, bb_l, bb_m = bb_upper.iloc[i], bb_lower.iloc[i], bb_mid.iloc[i]
        
        # 吞没
        if is_bullish_engulfing(o, h, l, c, body_ratio, atr, ranges, i):
            conf = min(body_ratio[i] * 100 + ranges[i] / atr_val * 10 if atr_val > 0 else 50, 100)
            if vol_now >= vol_ma_val * 1.2:
                sc = score_signal('bullish_engulfing', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
                signals.append({'index': i, 'type': 'bullish_engulfing', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_bearish_engulfing(o, h, l, c, body_ratio, atr, ranges, i):
            conf = min(body_ratio[i] * 100 + ranges[i] / atr_val * 10 if atr_val > 0 else 50, 100)
            if vol_now >= vol_ma_val * 1.2:
                sc = score_signal('bearish_engulfing', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
                signals.append({'index': i, 'type': 'bearish_engulfing', 'confidence': conf, 'score': sc, 'trend': trend})
        
        # 晨星/晚星
        if is_morning_star(o, h, l, c, body_ratio, atr, ranges, i):
            conf = 75
            sc = score_signal('morning_star', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'morning_star', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_evening_star(o, h, l, c, body_ratio, atr, ranges, i):
            conf = 75
            sc = score_signal('evening_star', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'evening_star', 'confidence': conf, 'score': sc, 'trend': trend})
        
        # 刺穿线/暗云盖顶
        if is_piercing(o, h, l, c, body_ratio, atr, ranges, i):
            conf = 70
            sc = score_signal('piercing', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'piercing', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_dark_cloud(o, h, l, c, body_ratio, atr, ranges, i):
            conf = 70
            sc = score_signal('dark_cloud', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'dark_cloud', 'confidence': conf, 'score': sc, 'trend': trend})
        
        # 针棒
        if is_hammer(o, h, l, c, body_ratio, atr, ranges, i):
            conf = min(ranges[i] / atr_val * 20 if atr_val > 0 else 30, 95)
            sc = score_signal('hammer', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'hammer', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_shooting_star(o, h, l, c, body_ratio, atr, ranges, i):
            conf = min(ranges[i] / atr_val * 20 if atr_val > 0 else 30, 95)
            sc = score_signal('shooting_star', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'shooting_star', 'confidence': conf, 'score': sc, 'trend': trend})
        
        # 三内柱
        if is_bullish_three_inside(o, h, l, c, body_ratio, atr, ranges, i):
            conf = 80
            sc = score_signal('bullish_three_inside', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'bullish_three_inside', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_bearish_three_inside(o, h, l, c, body_ratio, atr, ranges, i):
            conf = 80
            sc = score_signal('bearish_three_inside', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'bearish_three_inside', 'confidence': conf, 'score': sc, 'trend': trend})
        
        # 新增10种形态
        if is_double_bottom(o, h, l, c, atr, ranges, i):
            conf = 72
            sc = score_signal('double_bottom', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'double_bottom', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_double_top(o, h, l, c, atr, ranges, i):
            conf = 72
            sc = score_signal('double_top', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'double_top', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_head_shoulders(o, h, l, c, atr, ranges, i):
            conf = 78
            sc = score_signal('head_shoulders', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'head_shoulders', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_inverted_head_shoulders(o, h, l, c, atr, ranges, i):
            conf = 78
            sc = score_signal('inverted_hs', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'inverted_hs', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_rising_wedge(o, h, l, c, atr, ranges, i):
            conf = 68
            sc = score_signal('rising_wedge', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'rising_wedge', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_falling_wedge(o, h, l, c, atr, ranges, i):
            conf = 68
            sc = score_signal('falling_wedge', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'falling_wedge', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_bull_flag(o, h, l, c, atr, ranges, i):
            conf = 65
            sc = score_signal('bull_flag', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'bull_flag', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_bear_flag(o, h, l, c, atr, ranges, i):
            conf = 65
            sc = score_signal('bear_flag', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'bear_flag', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_bullish_triangle(o, h, l, c, atr, ranges, i):
            conf = 70
            sc = score_signal('bullish_triangle', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'bullish_triangle', 'confidence': conf, 'score': sc, 'trend': trend})
        if is_bearish_triangle(o, h, l, c, atr, ranges, i):
            conf = 70
            sc = score_signal('bearish_triangle', conf, cur_close, atr_val, vol_now, vol_ma_val, trend_score, bb_u, bb_l, bb_m)
            signals.append({'index': i, 'type': 'bearish_triangle', 'confidence': conf, 'score': sc, 'trend': trend})
    
    # 多周期共振检测
    return signals, {'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_mid': bb_mid}, trend

# ============ 核心10: 全市场扫描 ============
def scan_market(symbols, max_stocks=300, timeout_per_stock=15):
    """全市场扫描"""
    all_signals = []
    success, failed = 0, 0
    
    for idx, (sym, name) in enumerate(symbols[:max_stocks]):
        try:
            df = yf.Ticker(sym).history(period="3mo", interval="1wk")
            if df.empty or len(df) < 20:
                failed += 1
                continue
            
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'close', 'volume']
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            signals, bb_data, trend = detect_signals(df, sym)
            
            # 趋势+成交量过滤
            filtered = [s for s in signals 
                        if s['trend'] in ['up', 'down'] 
                        and s['score'] >= 55]
            
            if filtered:
                best = max(filtered, key=lambda x: x['score'])
                all_signals.append({
                    'symbol': sym,
                    'name': name,
                    'type': best['type'],
                    'confidence': best['confidence'],
                    'score': best['score'],
                    'trend': best['trend'],
                    'all_signals': filtered,
                    'df': df,
                    'bb_data': bb_data
                })
                # 生成标注图
                img_path = f"{IMG_DIR}/{sym.replace('.', '_')}.png"
                draw_annotated_kline(df, filtered, f"{sym} {name}", img_path)
            
            success += 1
            
            if idx % 50 == 0:
                print(f"  进度: {idx}/{min(len(symbols), max_stocks)} 已扫描, {len(all_signals)}个候选")
            
            time.sleep(0.1)
            
        except Exception as e:
            failed += 1
            continue
    
    # 按综合评分排序
    all_signals.sort(key=lambda x: x['score'], reverse=True)
    return all_signals, success, failed

# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 60)
    print("K线形态识别系统 v2.0  全市场扫描启动")
    print("=" * 60)
    
    # 1. 获取全市场标的
    print("\n📡 正在获取A股全市场标的列表...")
    stocks = get_china_a_stocks()
    print(f"   共加载 {len(stocks)} 个标的")
    
    # 2. 扫描前300只（快速验证，可扩展到全量）
    print(f"\n🔍 开始扫描前{min(300, len(stocks))}只标的...")
    start = time.time()
    results, success, failed = scan_market(stocks, max_stocks=300)
    elapsed = time.time() - start
    
    # 3. 输出结果
    print(f"\n\n{'='*60}")
    print(f"🎯 扫描完成！耗时 {elapsed:.0f}秒")
    print(f"   成功扫描: {success} 只 | 失败: {failed} 只")
    print(f"   共检出: {len(results)} 只标的")
    print(f"{'='*60}")
    
    # 4. 形态分布统计
    type_count = {}
    for r in results:
        t = r['type']
        type_count[t] = type_count.get(t, 0) + 1
    
    print("\n📊 形态分布:")
    for t, cnt in sorted(type_count.items(), key=lambda x: -x[1]):
        label_map = {
            'bullish_engulfing': '看涨吞没', 'bearish_engulfing': '看跌吞没',
            'morning_star': '晨星', 'evening_star': '晚星',
            'piercing': '刺穿线', 'dark_cloud': '暗云盖顶',
            'hammer': '锤子线', 'shooting_star': '射击之星',
            'bullish_three_inside': '三内柱(牛)', 'bearish_three_inside': '三内柱(熊)',
            'double_bottom': 'W底', 'double_top': 'M顶',
            'head_shoulders': '头肩顶', 'inverted_hs': '倒头肩底',
            'rising_wedge': '上升楔形', 'falling_wedge': '下降楔形',
            'bull_flag': '牛旗形', 'bear_flag': '熊旗形',
            'bullish_triangle': '上升三角形', 'bearish_triangle': '下降三角形',
        }
        print(f"   {label_map.get(t, t)}: {cnt}只")
    
    # 5. TOP候选池
    print(f"\n🏆 TOP候选股票池（按综合评分排序）:")
    print(f"{'排名':<4} {'代码':<12} {'名称':<8} {'形态':<12} {'置信度':>8} {'评分':>8} {'趋势':>6}")
    print("-" * 62)
    for i, r in enumerate(results[:30], 1):
        label_map = {
            'bullish_engulfing': '看涨吞没', 'bearish_engulfing': '看跌吞没',
            'morning_star': '晨星', 'evening_star': '晚星',
            'piercing': '刺穿线', 'dark_cloud': '暗云盖顶',
            'hammer': '锤子线', 'shooting_star': '射击之星',
            'bullish_three_inside': '三内柱(牛)', 'bearish_three_inside': '三内柱(熊)',
            'double_bottom': 'W底', 'double_top': 'M顶',
            'head_shoulders': '头肩顶', 'inverted_hs': '倒头肩底',
            'rising_wedge': '上升楔形', 'falling_wedge': '下降楔形',
            'bull_flag': '牛旗形', 'bear_flag': '熊旗形',
            'bullish_triangle': '上升三角形', 'bearish_triangle': '下降三角形',
        }
        print(f"{i:<4} {r['symbol']:<12} {r['name'][:6]:<8} {label_map.get(r['type'], r['type']):<12} "
              f"{r['confidence']:>6.0f}%  {r['score']:>6.0f}  {r['trend']:>6}")
    
    # 6. 导出CSV
    if results:
        export_df = pd.DataFrame([{
            'rank': i+1,
            'symbol': r['symbol'],
            'name': r['name'],
            'pattern': label_map.get(r['type'], r['type']),
            'confidence': f"{r['confidence']:.0f}%",
            'score': r['score'],
            'trend': r['trend'],
            'signal_count': len(r['all_signals']),
            'img_path': f"{IMG_DIR}/{r['symbol'].replace('.', '_')}.png"
        } for i, r in enumerate(results)])
        csv_path = f"{REPORT_DIR}/signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 候选池已导出: {csv_path}")
        print(f"📁 标注图已保存: {IMG_DIR}/")
    
    print("\n✅ v2.0全流程完成！")
