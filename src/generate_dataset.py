#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成新的纯K线+成交量训练样本
无任何文字/标注/干扰元素
"""
import os
import pandas as pd
import yfinance as yf
from draw_kline import draw_kline_chart

os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)

# 热门ETF列表（你关注的标的）
ETF_LIST = [
    "510050.SS", "510300.SS", "510500.SS", "588000.SS", "159915.SZ",
    "512880.SS", "512690.SS", "512660.SS", "515030.SS", "515790.SS",
    "515050.SS", "518880.SS", "159949.SZ", "159995.SZ", "159825.SZ",
    "159790.SZ", "513050.SS", "513500.SS", "513100.SS", "513030.SS",
]

# A股核心蓝筹
STOCK_LIST = [
    "600519.SS", "600036.SS", "601318.SS", "002594.SZ", "000858.SZ",
    "600030.SS", "601899.SS", "600438.SS", "600745.SS", "601012.SS",
    "002415.SZ", "300750.SZ", "002475.SZ", "002714.SZ", "300059.SZ",
]

# 拉取周线K线数据
def fetch_kline(symbol: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval="1wk", auto_adjust=False)
        df = df.reset_index()
        if "Date" not in df.columns:
            return pd.DataFrame()
        # 标准化列名
        df = df.rename(columns={
            "Date": "Date", "Open": "Open", "High": "High", 
            "Low": "Low", "Close": "Close", "Volume": "Volume"
        })
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()

# 生成样本
count = 0
for symbol in ETF_LIST + STOCK_LIST:
    print(f"正在生成 {symbol} 样本...")
    df = fetch_kline(symbol)
    if len(df) < 30:
        continue
    # 每个标的生成5个滚动窗口样本，丰富多样性
    for offset in range(0, len(df)-30, 6):
        df_window = df.iloc[offset:offset+30].reset_index(drop=True)
        save_path = f"dataset/images/train/{symbol}_{offset}.png"
        ok = draw_kline_chart(df_window, save_path)
        if ok:
            count +=1
            # 验证集取10%的样本
            if count % 10 == 0:
                os.rename(save_path, save_path.replace('/train/', '/val/'))

print(f"✅ 样本生成完成，共生成 {count} 张无干扰K线+成交量图，训练集: {int(count*0.9)} 张，验证集: {int(count*0.1)} 张")