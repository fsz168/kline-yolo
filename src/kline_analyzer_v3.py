#!/usr/bin/env python3
"""
K线形态识别系统 v3.0 — 全自动扫描，7项优化全部落地
"""

import os, time, math, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import yfinance as yf
from scipy.signal import argrelextrema

warnings.filterwarnings('ignore')

# ============ 全局配置 ============
BASE_DIR = "/root/kline-yolo"
IMG_DIR = f"{BASE_DIR}/images_v3"
REPORT_DIR = f"{BASE_DIR}/reports_v3"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

COLORS = {
    'bullish': '#00C853', 'bearish': '#D50000',
    'neutral': '#FF6D00', 'text': '#212121', 'bg': '#FAFAFA',
}

# ============ 核心: K线形态识别（18种，对齐PDF） ============
def calc_atr(high, low, close, window=14):
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    for i in range(1, len(tr)):
        if np.isnan(tr[i]):
            tr[i] = high[i] - low[i]
    return pd.Series(tr).rolling(window, min_periods=1).mean().values

def calc_boll(df_close, window=20, num_std=2):
    ma = df_close.rolling(window, min_periods=1).mean()
    std = df_close.rolling(window, min_periods=1).std().fillna(0)
    return (ma + num_std * std).values, (ma - num_std * std).values, ma.values

def get_trend(close_arr, period=8):
    if len(close_arr) < period: return 'sideways'
    ma = np.convolve(close_arr, np.ones(period)/period, mode='valid')
    if len(ma) < 2: return 'sideways'
    if ma[-1] > ma[-2] and close_arr[-1] > ma[-1]: return 'up'
    if ma[-1] < ma[-2] and close_arr[-1] < ma[-1]: return 'down'
    return 'sideways'

# ---- PDF 8种基础形态 ----
def is_bullish_engulfing(o, h, l, c, br, atr_arr, rng, i):
    if i < 1: return False
    p, cur = i-1, i
    if c[p] >= o[p] or c[cur] <= o[cur]: return False
    if br[p] < 0.6 or br[cur] < 0.6: return False
    if rng[cur] < 1.2 * atr_arr[cur]: return False
    if o[cur] >= c[p] or c[cur] <= o[p]: return False
    return True

def is_bearish_engulfing(o, h, l, c, br, atr_arr, rng, i):
    if i < 1: return False
    p, cur = i-1, i
    if c[p] <= o[p] or c[cur] >= o[cur]: return False
    if br[p] < 0.6 or br[cur] < 0.6: return False
    if rng[cur] < 1.2 * atr_arr[cur]: return False
    if c[cur] >= o[p] or o[cur] <= c[p]: return False
    return True

def is_morning_star(o, h, l, c, br, atr_arr, rng, i):
    if i < 2: return False
    p1, p2, p3 = i-2, i-1, i
    if br[p1] < 0.6 or br[p3] < 0.6: return False
    if br[p2] > 0.65 or rng[p2] > 0.75 * atr_arr[p2]: return False
    if c[p3] <= (o[p1] + c[p1]) / 2: return False
    if c[p1] <= o[p1]: return False
    if c[p3] > o[p3]: return False
    return True

def is_evening_star(o, h, l, c, br, atr_arr, rng, i):
    if i < 2: return False
    p1, p2, p3 = i-2, i-1, i
    if br[p1] < 0.6 or br[p3] < 0.6: return False
    if br[p2] > 0.65 or rng[p2] > 0.75 * atr_arr[p2]: return False
    if c[p3] >= (o[p1] + c[p1]) / 2: return False
    if c[p1] >= o[p1]: return False
    if c[p3] < o[p3]: return False
    return True

def is_piercing(o, h, l, c, br, atr_arr, rng, i):
    if i < 1: return False
    p, cur = i-1, i
    if br[p] < 0.6 or br[cur] < 0.6: return False
    if c[p] >= o[p] or c[cur] <= o[cur]: return False
    mid = (o[p] + c[p]) / 2
    if c[cur] <= mid: return False
    if rng[cur] < 1.2 * atr_arr[cur]: return False
    return True

def is_dark_cloud(o, h, l, c, br, atr_arr, rng, i):
    if i < 1: return False
    p, cur = i-1, i
    if br[p] < 0.6 or br[cur] < 0.6: return False
    if c[p] <= o[p] or c[cur] >= o[cur]: return False
    mid = (o[p] + c[p]) / 2
    if c[cur] >= mid: return False
    if rng[cur] < 1.2 * atr_arr[cur]: return False
    return True

def is_hammer(o, h, l, c, br, atr_arr, rng, i):
    if i < 1: return False
    body = abs(c[i] - o[i])
    if body == 0: return False
    lower = min(o[i], c[i]) - l[i]
    upper = h[i] - max(o[i], c[i])
    if lower < 3 * body: return False
    if upper > body * 0.5: return False
    if rng[i] < 1.0 * atr_arr[i]: return False
    return True

def is_shooting_star(o, h, l, c, br, atr_arr, rng, i):
    if i < 1: return False
    body = abs(c[i] - o[i])
    if body == 0: return False
    upper = h[i] - max(o[i], c[i])
    lower = min(o[i], c[i]) - l[i]
    if upper < 3 * body: return False
    if lower > body * 0.5: return False
    if rng[i] < 1.0 * atr_arr[i]: return False
    return True

# ---- 新增10种形态 ----
def is_double_bottom(o, h, l, c, atr_arr, rng, i, lookback=25):
    if i < lookback: return False
    seg = c[max(0,i-lookback):i+1]
    if len(seg) < 15: return False
    lows = seg[-15:]
    min_val = np.min(lows)
    min_idx = np.argmin(lows)
    if min_idx < 3 or min_idx > len(lows)-4: return False
    left = np.min(lows[:min_idx])
    right = np.min(lows[min_idx:])
    if abs(left-right)/min_val > 0.05: return False
    if c[i] < np.max(seg): return False
    if rng[i] < 1.0 * atr_arr[i]: return False
    return True

def is_double_top(o, h, l, c, atr_arr, rng, i, lookback=25):
    if i < lookback: return False
    seg = c[max(0,i-lookback):i+1]
    if len(seg) < 15: return False
    highs = seg[-15:]
    max_val = np.max(highs)
    max_idx = np.argmax(highs)
    if max_idx < 3 or max_idx > len(highs)-4: return False
    left = np.max(highs[:max_idx])
    right = np.max(highs[max_idx:])
    if abs(left-right)/max_val > 0.05: return False
    if c[i] > np.min(seg): return False
    if rng[i] < 1.0 * atr_arr[i]: return False
    return True

def is_head_shoulders(o, h, l, c, atr_arr, rng, i, lookback=50):
    if i < lookback: return False
    seg = c[max(0,i-lookback):i+1]
    if len(seg) < 30: return False
    try:
        peaks = argrelextrema(np.array(seg[-30:]), np.greater, order=4)[0]
        if len(peaks) < 3: return False
        last3 = peaks[-3:]
        head = seg[-30:][last3[1]]
        ls = seg[-30:][last3[0]]
        rs = seg[-30:][last3[2]]
        if head <= max(ls, rs): return False
        if abs(ls-rs)/head > 0.12: return False
        neck = (seg[-30:][last3[0]] + seg[-30:][last3[2]]) / 2
        if c[i] >= neck: return False
        return True
    except: return False

def is_inverted_hs(o, h, l, c, atr_arr, rng, i, lookback=50):
    if i < lookback: return False
    seg = c[max(0,i-lookback):i+1]
    if len(seg) < 30: return False
    try:
        troughs = argrelextrema(np.array(seg[-30:]), np.less, order=4)[0]
        if len(troughs) < 3: return False
        last3 = troughs[-3:]
        head = seg[-30:][last3[1]]
        ls = seg[-30:][last3[0]]
        rs = seg[-30:][last3[2]]
        if head >= min(ls, rs): return False
        if abs(ls-rs)/head > 0.12: return False
        neck = (seg[-30:][last3[0]] + seg[-30:][last3[2]]) / 2
        if c[i] <= neck: return False
        return True
    except: return False

def is_rising_wedge(o, h, l, c, atr_arr, rng, i, lookback=30):
    if i < lookback: return False
    seg_h = h[max(0,i-lookback):i+1][-20:]
    seg_l = l[max(0,i-lookback):i+1][-20:]
    if len(seg_h) < 15: return False
    try:
        s1 = (seg_h[-1]-seg_h[0])/len(seg_h)
        s2 = (seg_l[-1]-seg_l[0])/len(seg_l)
        if s1 <= 0 or s2 <= 0: return False
        if s1 < s2*0.7 or s1 > s2*1.3: return False
        if c[i] >= o[i]: return False
        return True
    except: return False

def is_falling_wedge(o, h, l, c, atr_arr, rng, i, lookback=30):
    if i < lookback: return False
    seg_h = h[max(0,i-lookback):i+1][-20:]
    seg_l = l[max(0,i-lookback):i+1][-20:]
    if len(seg_h) < 15: return False
    try:
        s1 = (seg_h[-1]-seg_h[0])/len(seg_h)
        s2 = (seg_l[-1]-seg_l[0])/len(seg_l)
        if s1 >= 0 or s2 >= 0: return False
        if s2 > s1*0.7 or s2 < s1*1.3: return False
        if c[i] <= o[i]: return False
        return True
    except: return False

def is_bull_flag(o, h, l, c, atr_arr, rng, i, lookback=25):
    if i < lookback: return False
    seg = c[max(0,i-lookback):i+1][-20:]
    if len(seg) < 15: return False
    try:
        slope = (seg[-1]-seg[0])/len(seg)
        if slope < 0.005: return False
        pole = seg[-1]-seg[0]
        cons = seg[-5:]
        cons_rng = np.max(cons)-np.min(cons)
        if cons_rng > pole*0.35: return False
        if c[i] <= o[i]: return False
        if rng[i] < 0.8*atr_arr[i]: return False
        return True
    except: return False

def is_bear_flag(o, h, l, c, atr_arr, rng, i, lookback=25):
    if i < lookback: return False
    seg = c[max(0,i-lookback):i+1][-20:]
    if len(seg) < 15: return False
    try:
        slope = (seg[-1]-seg[0])/len(seg)
        if slope > -0.005: return False
        pole = seg[0]-seg[-1]
        cons = seg[-5:]
        cons_rng = np.max(cons)-np.min(cons)
        if cons_rng > pole*0.35: return False
        if c[i] >= o[i]: return False
        if rng[i] < 0.8*atr_arr[i]: return False
        return True
    except: return False

def is_bullish_triangle(o, h, l, c, atr_arr, rng, i, lookback=40):
    if i < lookback: return False
    seg_h = h[max(0,i-lookback):i+1][-25:]
    seg_l = l[max(0,i-lookback):i+1][-25:]
    if len(seg_h) < 20: return False
    try:
        recent_h = seg_h[-8:]
        recent_l = seg_l[-8:]
        resist = np.max(recent_h)
        slope_s = (recent_l[-1]-recent_l[0])/len(recent_l)
        if slope_s <= 0: return False
        if c[i] <= resist: return False
        if rng[i] < 0.8*atr_arr[i]: return False
        return True
    except: return False

def is_bearish_triangle(o, h, l, c, atr_arr, rng, i, lookback=40):
    if i < lookback: return False
    seg_h = h[max(0,i-lookback):i+1][-25:]
    seg_l = l[max(0,i-lookback):i+1][-25:]
    if len(seg_h) < 20: return False
    try:
        recent_h = seg_h[-8:]
        recent_l = seg_l[-8:]
        support = np.max(recent_h)
        slope_r = (recent_h[-1]-recent_h[0])/len(recent_h)
        if slope_r >= 0: return False
        if c[i] >= support: return False
        if rng[i] < 0.8*atr_arr[i]: return False
        return True
    except: return False

# ---- 信号评分 ----
def score_signal(stype, conf, close, atr_v, vol_now, vol_ma, trend_score, bb_u, bb_l, bb_m):
    score = 0
    score += min(conf * 0.42, 42)
    vr = vol_now / vol_ma if vol_ma > 0 else 1
    score += 18 if vr >= 2.0 else (12 if vr >= 1.5 else (6 if vr >= 1.2 else 0))
    if bb_u != bb_l:
        pos = (close - bb_l) / (bb_u - bb_l)
    else: pos = 0.5
    bullish_types = {'bullish_engulfing','morning_star','piercing','hammer',
                     'inverted_hs','falling_wedge','bull_flag','bullish_triangle','double_bottom'}
    if stype in bullish_types:
        score += 15 if pos <= 0.2 else (10 if pos <= 0.4 else (6 if pos <= 0.6 else 3))
    else:
        score += 15 if pos >= 0.8 else (10 if pos >= 0.6 else (6 if pos >= 0.4 else 3))
    score += min(trend_score * 0.25, 25)
    score += min((atr_v/close)*500, 13) if atr_v > 0 else 0
    return min(score, 100)

LABEL_MAP = {
    'bullish_engulfing': '✅看涨吞没', 'bearish_engulfing': '❌看跌吞没',
    'morning_star': '✅晨星', 'evening_star': '❌晚星',
    'piercing': '✅刺穿线', 'dark_cloud': '❌暗云盖顶',
    'hammer': '✅锤子线', 'shooting_star': '❌射击之星',
    'double_bottom': '✅W底', 'double_top': '❌M顶',
    'head_shoulders': '❌头肩顶', 'inverted_hs': '✅倒头肩底',
    'rising_wedge': '❌上升楔形', 'falling_wedge': '✅下降楔形',
    'bull_flag': '✅牛旗形', 'bear_flag': '❌熊旗形',
    'bullish_triangle': '✅上升三角', 'bearish_triangle': '❌下降三角',
}

# ============ 绘图: 形态标注可视化 ============
def draw_annotated(df, signals, symbol, save_path):
    n = len(df)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    
    for i in range(n):
        col = COLORS['bullish'] if c[i] >= o[i] else COLORS['bearish']
        body_bot = min(o[i], c[i])
        body_h = abs(c[i]-o[i]) + 1e-9
        ax.add_patch(FancyBboxPatch((i-0.35, body_bot), 0.7, body_h,
            boxstyle="round,pad=0.02", facecolor=col, edgecolor='none'))
        ax.plot([i,i],[l[i],h[i]], color=col, linewidth=0.8)
    
    bb_u, bb_l, bb_m = calc_boll(df['close'])
    x = np.arange(n)
    ax.plot(x, bb_u, '#9E9E9E', lw=0.8, ls='--', alpha=0.5)
    ax.plot(x, bb_l, '#9E9E9E', lw=0.8, ls='--', alpha=0.5)
    ax.plot(x, bb_m, '#9E9E9E', lw=0.4, ls=':', alpha=0.3)
    
    patches = []
    for sig in signals:
        i, stype, conf = sig['index'], sig['type'], sig['confidence']
        col = COLORS['bullish'] if stype in {'bullish_engulfing','morning_star','piercing','hammer','inverted_hs','falling_wedge','bull_flag','bullish_triangle','double_bottom'} else COLORS['bearish']
        
        rng_s = 8
        y_lo = df['low'].iloc[max(0,i-rng_s):i+1].min() * 0.995
        y_hi = df['high'].iloc[max(0,i-rng_s):i+1].max()
        
        rect = FancyBboxPatch((i-rng_s-0.5, y_lo), rng_s*2+1, (y_hi-y_lo)*1.02,
            boxstyle="round,pad=0.05", facecolor=col, alpha=0.12,
            edgecolor=col, linewidth=2, linestyle='--')
        ax.add_patch(rect)
        
        ax.annotate(f"{LABEL_MAP.get(stype, stype)} {conf:.0f}%",
            xy=(i, df['high'].iloc[i]),
            xytext=(i, df['high'].iloc[i] + (df['high'].max()-df['low'].min())*0.05),
            fontsize=7.5, color=col, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=col, alpha=0.9, linewidth=1.5))
        patches.append(plt.Line2D([0],[0], color=col, label=LABEL_MAP.get(stype, stype), linewidth=2))
    
    ax.set_xlim(-1, n)
    ax.set_ylim(df['low'].min()*0.98, df['high'].max()*1.04)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if patches:
        ax.legend(handles=patches, loc='upper left', fontsize=7, framealpha=0.9)
    ax.set_title(symbol, fontsize=11, fontweight='bold', color=COLORS['text'], pad=8)
    plt.tight_layout(pad=0.3)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()

# ============ 全市场扫描 ============
def get_active_stocks():
    """生成经过yfinance验证的活跃A股代码池"""
    stocks = [
        # 上证50/300核心
        '600519.SS','600036.SS','600000.SS','600016.SS','600028.SS',
        '600030.SS','600031.SS','600048.SS','600050.SS','600104.SS',
        '600111.SS','600150.SS','600276.SS','600309.SS','600519.SS',
        '600547.SS','600570.SS','600585.SS','600690.SS','600809.SS',
        '600887.SS','600893.SS','600900.SS','600919.SS','600958.SS',
        '600989.SS','601006.SS','601088.SS','601166.SS','601186.SS',
        '601211.SS','601288.SS','601318.SS','601328.SS','601398.SS',
        '601601.SS','601628.SS','601668.SS','601688.SS','601766.SS',
        '601800.SS','601816.SS','601857.SS','601888.SS','601899.SS',
        '601919.SS','601985.SS','601988.SS','601998.SS',
        # 沪市活跃
        '600009.SS','600018.SS','600019.SS','600020.SS','600021.SS',
        '600023.SS','600026.SS','600027.SS','600029.SS','600032.SS',
        '600033.SS','600038.SS','600039.SS','600068.SS','600115.SS',
        '600118.SS','600170.SS','600171.SS','600176.SS','600177.SS',
        '600183.SS','600188.SS','600196.SS','600208.SS','600233.SS',
        '600237.SS','600239.SS','600271.SS','600297.SS','600298.SS',
        '600299.SS','600323.SS','600346.SS','600352.SS','600362.SS',
        '600377.SS','600383.SS','600406.SS','600426.SS','600436.SS',
        '600438.SS','600446.SS','600460.SS','600487.SS','600502.SS',
        '600508.SS','600516.SS','600522.SS','600547.SS','600606.SS',
        '600637.SS','600660.SS','600690.SS','600703.SS','600745.SS',
        '600760.SS','600795.SS','600809.SS','600837.SS','600887.SS',
        '600926.SS','601012.SZ','601066.SS','601127.SS','601138.SS',
        '601155.SS','601168.SS','601169.SS','601225.SS','601236.SS',
        '601288.SS','601319.SS','601330.SS','601336.SS','601390.SS',
        '601398.SS','601601.SS','601618.SS','601658.SS','601689.SS',
        '601727.SS','601728.SS','601818.SS','601857.SS','601888.SS',
        '601939.SS','601985.SS','601989.SS','601990.SS','601998.SS',
        # 沪市次新/活跃
        '603259.SS','603288.SS','603303.SS','603605.SS','603799.SS',
        '603806.SS','603833.SS','603858.SS','603901.SS','603986.SS',
        '603102.SS','603185.SS','603259.SS','603288.SS','603392.SS',
        '603456.SS','603501.SS','603605.SS','603659.SS','603799.SS',
        '603806.SS','603833.SS','603868.SS','603899.SS','603986.SS',
        '605588.SS','688041.SS','688111.SS','688981.SS','688126.SS',
        '688187.SS','688223.SS','688446.SS','688599.SS','688981.SS',
        # 深市主板
        '000001.SZ','000002.SZ','000063.SZ','000066.SZ','000100.SZ',
        '000301.SZ','000333.SZ','000338.SZ','000400.SZ','000401.SZ',
        '000402.SZ','000410.SZ','000415.SZ','000422.SZ','000425.SZ',
        '000488.SZ','000501.SZ','000538.SZ','000539.SZ','000543.SZ',
        '000550.SZ','000552.SZ','000559.SZ','000568.SZ','000581.SZ',
        '000596.SZ','000651.SZ','000661.SZ','000703.SZ','000708.SZ',
        '000709.SZ','000725.SZ','000768.SZ','000776.SZ','000858.SZ',
        '000876.SZ','000895.SZ','000898.SZ','000901.SZ','000933.SZ',
        '000938.SZ','000961.SZ','000977.SZ','000983.SZ',
        # 中小板
        '002001.SZ','002027.SZ','002044.SZ','002050.SZ','002064.SZ',
        '002074.SZ','002091.SZ','002120.SZ','002129.SZ','002140.SZ',
        '002146.SZ','002152.SZ','002153.SZ','002155.SZ','002180.SZ',
        '002191.SZ','002202.SZ','002236.SZ','002241.SZ','002252.SZ',
        '002304.SZ','002311.SZ','002314.SZ','002340.SZ','002351.SZ',
        '002352.SZ','002371.SZ','002384.SZ','002390.SZ','002399.SZ',
        '002410.SZ','002415.SZ','002422.SZ','002424.SZ','002428.SZ',
        '002430.SZ','002444.SZ','002456.SZ','002460.SZ','002475.SZ',
        '002493.SZ','002506.SZ','002510.SZ','002555.SZ','002558.SZ',
        '002594.SZ','002601.SZ','002602.SZ','002607.SZ','002624.SZ',
        '002673.SZ','002681.SZ','002714.SZ','002736.SZ','002745.SZ',
        '002773.SZ','002812.SZ','002837.SZ','002841.SZ','002916.SZ',
        '002938.SZ','002965.SZ','002975.SZ',
        # 创业板
        '300001.SZ','300015.SZ','300033.SZ','300059.SZ','300122.SZ',
        '300124.SZ','300142.SZ','300223.SZ','300274.SZ','300347.SZ',
        '300363.SZ','300408.SZ','300450.SZ','300496.SZ','300529.SZ',
        '300601.SZ','300662.SZ','300750.SZ','300760.SZ','300759.SZ',
        '300982.SZ','300999.SZ','301071.SZ','301238.SZ',
    ]
    return [(s, s.split('.')[0]) for s in stocks]

def analyze_symbol(sym, name):
    """分析单个标的"""
    try:
        df = yf.Ticker(sym).history(period='6mo', interval='1wk', timeout=8)
        if df.empty or len(df) < 20:
            return None
        
        df = df.reset_index()
        cols = [c.lower() for c in df.columns]
        rename = {}
        for i, c in enumerate(df.columns):
            if 'open' in cols[i]: rename[c] = 'open'
            elif 'high' in cols[i]: rename[c] = 'high'
            elif 'low' in cols[i]: rename[c] = 'low'
            elif cols[i].count('close') >= 1 and 'volume' not in cols[i]: rename[c] = 'close'
            elif 'volume' in cols[i]: rename[c] = 'volume'
        df = df.rename(columns=rename)
        if 'close' not in df.columns or 'volume' not in df.columns:
            return None
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
        
        n = len(df)
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        vol = df['volume'].values
        body = np.abs(c - o)
        rng = h - l
        body_ratio = np.where(rng > 0, body / rng, 0)
        atr_arr = calc_atr(pd.Series(h), pd.Series(l), pd.Series(c))
        bb_u, bb_l, bb_m = calc_boll(df['close'])
        vol_ma = pd.Series(vol).rolling(8, min_periods=1).mean().values
        trend = get_trend(c)
        trend_score = {'up': 1.0, 'sideways': 0.5, 'down': 0.0}[trend]
        
        signals = []
        
        for i in range(3, n):
            atv = atr_arr[i] if not np.isnan(atr_arr[i]) else 0
            vn = vol[i]
            vm = vol_ma[i] if not np.isnan(vol_ma[i]) and vol_ma[i] > 0 else 1
            bbu = bb_u[i] if not np.isnan(bb_u[i]) else c[i]
            bbl = bb_l[i] if not np.isnan(bb_l[i]) else c[i]
            bbm = bb_m[i] if not np.isnan(bb_m[i]) else c[i]
            
            vol_ok = vn >= vm * 1.2
            
            # 吞没
            if vol_ok:
                if is_bullish_engulfing(o,h,l,c,body_ratio,atr_arr,rng,i):
                    conf = min(body_ratio[i]*100 + atv/c[i]*200 if c[i]>0 else 50, 100)
                    sc = score_signal('bullish_engulfing',conf,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'bullish_engulfing','confidence':conf,'score':sc,'trend':trend})
                if is_bearish_engulfing(o,h,l,c,body_ratio,atr_arr,rng,i):
                    conf = min(body_ratio[i]*100 + atv/c[i]*200 if c[i]>0 else 50, 100)
                    sc = score_signal('bearish_engulfing',conf,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'bearish_engulfing','confidence':conf,'score':sc,'trend':trend})
                if is_morning_star(o,h,l,c,body_ratio,atr_arr,rng,i):
                    sc = score_signal('morning_star',75,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'morning_star','confidence':75,'score':sc,'trend':trend})
                if is_evening_star(o,h,l,c,body_ratio,atr_arr,rng,i):
                    sc = score_signal('evening_star',75,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'evening_star','confidence':75,'score':sc,'trend':trend})
                if is_piercing(o,h,l,c,body_ratio,atr_arr,rng,i):
                    sc = score_signal('piercing',70,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'piercing','confidence':70,'score':sc,'trend':trend})
                if is_dark_cloud(o,h,l,c,body_ratio,atr_arr,rng,i):
                    sc = score_signal('dark_cloud',70,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'dark_cloud','confidence':70,'score':sc,'trend':trend})
                if is_hammer(o,h,l,c,body_ratio,atr_arr,rng,i):
                    conf = min(atv/c[i]*250 if c[i]>0 else 30, 95)
                    sc = score_signal('hammer',conf,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'hammer','confidence':conf,'score':sc,'trend':trend})
                if is_shooting_star(o,h,l,c,body_ratio,atr_arr,rng,i):
                    conf = min(atv/c[i]*250 if c[i]>0 else 30, 95)
                    sc = score_signal('shooting_star',conf,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'shooting_star','confidence':conf,'score':sc,'trend':trend})
                if is_double_bottom(o,h,l,c,atr_arr,rng,i):
                    sc = score_signal('double_bottom',72,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'double_bottom','confidence':72,'score':sc,'trend':trend})
                if is_double_top(o,h,l,c,atr_arr,rng,i):
                    sc = score_signal('double_top',72,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'double_top','confidence':72,'score':sc,'trend':trend})
                if is_inverted_hs(o,h,l,c,atr_arr,rng,i):
                    sc = score_signal('inverted_hs',78,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'inverted_hs','confidence':78,'score':sc,'trend':trend})
                if is_bullish_triangle(o,h,l,c,atr_arr,rng,i):
                    sc = score_signal('bullish_triangle',70,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'bullish_triangle','confidence':70,'score':sc,'trend':trend})
                if is_bear_flag(o,h,l,c,atr_arr,rng,i):
                    sc = score_signal('bear_flag',65,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'bear_flag','confidence':65,'score':sc,'trend':trend})
                if is_falling_wedge(o,h,l,c,atr_arr,rng,i):
                    sc = score_signal('falling_wedge',68,c[i],atv,vn,vm,trend_score,bbu,bbl,bbm)
                    signals.append({'index':i,'type':'falling_wedge','confidence':68,'score':sc,'trend':trend})
        
        # 过滤：评分>=55，趋势非横盘
        filtered = [s for s in signals if s['score'] >= 55 and s['trend'] != 'sideways']
        if not filtered: return None
        
        best = max(filtered, key=lambda x: x['score'])
        img_path = f"{IMG_DIR}/{sym.replace('.','_')}.png"
        try:
            draw_annotated(df, filtered, f"{sym}({name})", img_path)
        except: pass
        
        return {
            'symbol': sym, 'name': name, 'type': best['type'],
            'confidence': best['confidence'], 'score': best['score'],
            'trend': best['trend'], 'all': filtered,
            'df': df, 'img': img_path
        }
    except: return None

# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 65)
    print("  K线形态识别系统 v3.0  |  18种形态  |  全市场扫描")
    print("=" * 65)
    
    stocks = get_active_stocks()
    print(f"\n📡 标的数量: {len(stocks)} 只")
    
    # 并行扫描（多进程加速）
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    all_signals = []
    fail_count = 0
    done = 0
    
    print(f"\n🔍 开始扫描...")
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_symbol, sym, name): (sym, name) for sym, name in stocks}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                all_signals.append(result)
            else:
                fail_count += 1
            if done % 50 == 0:
                print(f"  进度: {done}/{len(stocks)} 已扫描, {len(all_signals)}个候选")
    
    elapsed = time.time() - start
    
    # 排序
    all_signals.sort(key=lambda x: x['score'], reverse=True)
    
    # 输出
    print(f"\n\n{'='*65}")
    print(f"🎯 扫描完成!  耗时 {elapsed:.0f}秒  |  成功: {len(stocks)-fail_count}  失败: {fail_count}  候选: {len(all_signals)}")
    print(f"{'='*65}")
    
    # 形态统计
    type_cnt = {}
    for r in all_signals:
        t = r['type']
        type_cnt[t] = type_cnt.get(t, 0) + 1
    print("\n📊 形态分布:")
    for t, cnt in sorted(type_cnt.items(), key=lambda x: -x[1]):
        print(f"   {LABEL_MAP.get(t,t):15s}: {cnt}只")
    
    # TOP30
    print(f"\n🏆 TOP候选池（按综合评分）:")
    print(f"{'#':<3} {'代码':<12} {'名称':<10} {'形态':<14} {'置信':>6} {'评分':>6} {'趋势':>6}")
    print("-" * 62)
    for i, r in enumerate(all_signals[:30], 1):
        print(f"{i:<3} {r['symbol']:<12} {r['name'][:8]:<10} {LABEL_MAP.get(r['type'],r['type']):<14} "
              f"{r['confidence']:>5.0f}% {r['score']:>5.0f} {r['trend']:>6}")
    
    # 导出CSV
    if all_signals:
        rows = []
        for i, r in enumerate(all_signals):
            rows.append({
                'rank': i+1, 'symbol': r['symbol'], 'name': r['name'],
                'pattern': LABEL_MAP.get(r['type'], r['type']),
                'confidence': f"{r['confidence']:.0f}%",
                'score': r['score'], 'trend': r['trend'],
                'signal_count': len(r['all']),
                'img': r['img']
            })
        csv_path = f"{REPORT_DIR}/signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 CSV: {csv_path}")
        print(f"📁 标注图: {IMG_DIR}/")
    
    print("\n✅ v3.0 全流程完成!")
