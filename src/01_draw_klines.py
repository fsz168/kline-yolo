"""
阶段1：数据源验证 + 批量K线图生成
- 支持 LB (Longbridge) 和 yfinance 双数据源
- 输出标准化640x640纯K线图（无坐标轴、无网格）
- 自动生成样本库供后续标注/预测使用
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 强制使用非GUI后端渲染matplotlib
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
import yfinance as yf
from draw_kline import draw_kline_chart

# ─────────────────────────────────────────
# 1. 数据源配置
# ─────────────────────────────────────────
# A股蓝筹股前100只（按市值+流动性筛选，覆盖各行业板块）
STOCK_LIST_A = [
    "600519.SS", "600036.SS", "601318.SS", "600276.SS", "601166.SS",
    "600887.SS", "600030.SS", "601328.SS", "601088.SS", "601601.SS",
    "600016.SS", "600028.SS", "601288.SS", "601398.SS", "601939.SS",
    "601668.SS", "601186.SS", "601668.SS", "601390.SS", "601857.SS",
    "600050.SS", "601766.SS", "600104.SS", "601229.SS", "601012.SS",
    "601899.SS", "601628.SS", "603259.SS", "688981.SS", "688599.SS",
    "600585.SS", "600690.SS", "600031.SS", "600150.SS", "600588.SS",
    "600570.SS", "601888.SS", "603288.SS", "603259.SS", "688041.SS",
    "601336.SS", "601319.SS", "601816.SS", "601377.SS", "601688.SS",
    "600009.SS", "600115.SS", "600170.SS", "600221.SS", "600276.SS",
    "601727.SS", "601800.SS", "601808.SS", "601818.SS", "601838.SS",
    "601985.SS", "603160.SS", "603259.SS", "603288.SS", "603501.SS",
    "603799.SS", "603986.SS", "688012.SS", "688111.SS", "688126.SS",
    "688187.SS", "688223.SS", "688303.SS", "688396.SS", "688981.SS",
    "002415.SZ", "002475.SZ", "002594.SZ", "002714.SZ", "002475.SZ",
    "002230.SZ", "002352.SZ", "002371.SZ", "002460.SZ", "002466.SZ",
    "002493.SZ", "002594.SZ", "002607.SZ", "002714.SZ", "002736.SZ",
    "002812.SZ", "002920.SZ", "300015.SZ", "300059.SZ", "300122.SZ",
    "300124.SZ", "300142.SZ", "300274.SZ", "300357.SZ", "300433.SZ",
    "300496.SZ", "300750.SZ", "300896.SZ", "300999.SZ", "301012.SZ",
]

# 港股蓝筹（前20只）
STOCK_LIST_HK = [
    "0700.HK", "0941.HK", "9988.HK", "3690.HK", "9618.HK",
    "1810.HK", "1024.HK", "2382.HK", "2319.HK", "6690.HK",
    "6618.HK", "6030.HK", "1833.HK", "6623.HK", "1810.HK",
    "0386.HK", "0857.HK", "2628.HK", "2331.HK", "1109.HK",
]

# 美股科技蓝筹（前20只）
STOCK_LIST_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "BRK-B", "JPM", "JNJ",
    "V", "UNH", "HD", "MA", "PG",
    "XOM", "CVX", "MRK", "ABBV", "PFE",
]

STOCK_LIST = STOCK_LIST_A + STOCK_LIST_HK + STOCK_LIST_US


def draw_kline(df: pd.DataFrame, save_path: str) -> bool:
    """生成无干扰纯K线图，640x640，正方形"""
    try:
        ok = draw_kline_chart(df, save_path=save_path)
        return ok
    except Exception as e:
        return False

def fetch_kline_yf(symbol: str, interval: str = "1wk", period: str = "3mo") -> pd.DataFrame:
    """用 yfinance 拉取K线（主力数据源，完全免费无限制）"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)
        df = df.reset_index()
        if "Date" not in df.columns and "Datetime" not in df.columns:
            return pd.DataFrame()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={date_col: "Date"})
        return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


def fetch_kline_lb(symbol: str, period: str = "1w", count: int = 30):
    """用 Longbridge 拉取K线（备用数据源，已配置过环境变量）"""
    try:
        from longbridge.openapi import TradeContext, Config
        config = Config.from_env()
        ctx = TradeContext(config)
        resp = ctx.kline(symbol, period=period, count=count)
        df = pd.DataFrame([{
            "Date": item.timestamp,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume
        } for item in resp])
        return df
    except Exception as e:
        print(f"    ⚠️ LB拉取失败 [{symbol}]: {e}")
        return pd.DataFrame()


def main():
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kline_samples")
    os.makedirs(samples_dir, exist_ok=True)

    results = {"success": [], "failed": []}
    total = len(STOCK_LIST)

    print(f"\n{'='*60}")
    print(f"📊 开始批量生成K线图，共 {total} 只标的")
    print(f"📁 输出目录: {samples_dir}")
    print(f"{'='*60}\n")

    for idx, symbol in enumerate(STOCK_LIST, 1):
        # 确定标的类型
        suffix = symbol.split(".")[-1] if "." in symbol else ""
        if suffix == "SS":
            mtype = "A股"
        elif suffix == "SZ":
            mtype = "A股"
        elif suffix == "HK":
            mtype = "港股"
        else:
            mtype = "美股"

        interval = "1wk" if mtype != "美股" else "1wk"

        # 尝试LB，失败则用yfinance
        df = fetch_kline_lb(symbol)
        if df.empty:
            df = fetch_kline_yf(symbol, interval=interval)

        if df.empty:
            results["failed"].append(symbol)
            print(f"  [{idx:3d}/{total}] ❌ 数据拉取失败: {symbol} ({mtype})")
            continue

        save_path = os.path.join(samples_dir, f"{symbol.replace('.', '_')}.png")
        ok = draw_kline(df, save_path)

        if ok:
            results["success"].append(symbol)
            print(f"  [{idx:3d}/{total}] ✅ 生成成功: {symbol} ({mtype})")
        else:
            results["failed"].append(symbol)
            print(f"  [{idx:3d}/{total}] ❌ 绘图失败: {symbol} ({mtype})")

    # 输出统计报告
    print(f"\n{'='*60}")
    print(f"📋 生成报告")
    print(f"  ✅ 成功: {len(results['success'])} / {total}")
    print(f"  ❌ 失败: {len(results['failed'])} / {total}")
    if results["failed"]:
        print(f"  失败标的: {results['failed'][:20]}{'...' if len(results['failed']) > 20 else ''}")
    print(f"{'='*60}\n")

    # 保存失败清单
    with open(os.path.join(samples_dir, "failed_list.txt"), "w") as f:
        f.write("\n".join(results["failed"]))

    return results


if __name__ == "__main__":
    main()
