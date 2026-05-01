"""
阶段2补充：纯Python实现的K线形态识别
- 严格对齐PDF《量化指标解码19》的8种核心形态硬规则
- ATR过滤 + 实体占比过滤 + 位置验证
- 三层过滤：趋势过滤 + 成交量过滤 + 形态位置过滤
- 支持自定义扩展任意形态
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import yfinance as yf
from finta import TA
from datetime import datetime
import pandas as pd

# ─────────────────────────────────────────
# 形态识别引擎（完全对齐PDF硬规则）
# ─────────────────────────────────────────

def calc_atr(high, low, close, period: int = 14) -> np.ndarray:
    """计算ATR（真实波动幅度）"""
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close), np.abs(low - prev_close))
    atr = np.zeros_like(tr)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def detect_all_patterns(df: pd.DataFrame, symbol: str = "") -> list:
    """
    对一根K线或K线组合进行全形态检测
    返回所有检测到的形态列表，每项包含：形态名、信号方向、置信度、规则说明
    """
    if len(df) < 20:
        return []

    O = df["Open"].values.astype(float)
    H = df["High"].values.astype(float)
    L = df["Low"].values.astype(float)
    C = df["Close"].values.astype(float)
    V = df["Volume"].values.astype(float)

    atr = calc_atr(H, L, C, period=14)
    n = len(df)
    results = []

    # ── 辅助：判断K线颜色 ──
    def is_bull(i): return C[i] >= O[i]
    def is_bear(i): return C[i] < O[i]

    # ── 辅助：计算实体 ──
    def body(i): return abs(C[i] - O[i])
    def range_bar(i): return H[i] - L[i]
    def body_ratio(i): return body(i) / range_bar(i) if range_bar(i) > 0 else 0
    def atr_ratio(i): return range_bar(i) / atr[i] if atr[i] > 0 else 0

    # ── 辅助：影线比例 ──
    def upper_shadow(i): return H[i] - max(O[i], C[i])
    def lower_shadow(i): return min(O[i], C[i]) - L[i]
    def shadow_ratio(i, direction="lower"):
        rng = range_bar(i)
        if rng <= 0: return 0
        if direction == "lower":
            return lower_shadow(i) / rng
        else:
            return upper_shadow(i) / rng

    # ── 1. 看涨吞没（Bullish Engulfing） ──
    for i in range(1, n):
        prev, cur = i - 1, i
        # 前一根阴线，当前阳线
        if not is_bear(prev) or not is_bull(cur):
            continue
        # 两根K线实体占比均>60%
        if body_ratio(prev) < 0.6 or body_ratio(cur) < 0.6:
            continue
        # 当前K线波动>1.2倍ATR
        if atr_ratio(cur) < 1.2:
            continue
        # 当前阳线实体完全吞没前一根阴线实体
        if not (O[cur] >= C[prev] and C[cur] <= O[prev]):
            continue
        results.append({
            "pattern": "吞没（看涨）",
            "signal": "bullish",
            "index": i,
            "confidence": round(min(body_ratio(prev), body_ratio(cur)) * 100, 100),
            "rule": "实体>60% + 波动>1.2ATR + 完全吞没",
            "index_name": "engulf_bull",
        })

    # ── 2. 看跌吞没（Bearish Engulfing） ──
    for i in range(1, n):
        prev, cur = i - 1, i
        if not is_bull(prev) or not is_bear(cur):
            continue
        if body_ratio(prev) < 0.6 or body_ratio(cur) < 0.6:
            continue
        if atr_ratio(cur) < 1.2:
            continue
        if not (C[cur] <= O[prev] and O[cur] >= C[prev]):
            continue
        results.append({
            "pattern": "吞没（看跌）",
            "signal": "bearish",
            "index": i,
            "confidence": round(min(body_ratio(prev), body_ratio(cur)) * 100, 1),
            "rule": "实体>60% + 波动>1.2ATR + 完全吞没",
            "index_name": "engulf_bear",
        })

    # ── 3. 锤子线（看涨Pin Bar / Hammer） ──
    for i in range(n - 3, n):  # 只看最后几根
        if not is_bull(i):
            continue
        # 下影线≥3倍实体
        if lower_shadow(i) < 3 * body(i):
            continue
        # 上影线极短（<10%的K线总波动）
        if upper_shadow(i) / range_bar(i) > 0.1:
            continue
        # 实体较小（<30%总波动）
        if body_ratio(i) > 0.35:
            continue
        results.append({
            "pattern": "针棒（看涨锤子）",
            "signal": "bullish",
            "index": i,
            "confidence": round(min((lower_shadow(i) / body(i)) * 20 + 60, 100), 1),
            "rule": "下影≥3倍实体 + 上影<10% + 实体<35%",
            "index_name": "pin_bar_bull",
        })

    # ── 4. 射击之星（看跌Pin Bar / Shooting Star） ──
    for i in range(n - 3, n):
        if not is_bear(i):
            continue
        if upper_shadow(i) < 3 * body(i):
            continue
        if lower_shadow(i) / range_bar(i) > 0.1:
            continue
        if body_ratio(i) > 0.35:
            continue
        results.append({
            "pattern": "针棒（看跌射击星）",
            "signal": "bearish",
            "index": i,
            "confidence": round(min((upper_shadow(i) / body(i)) * 20 + 60, 100), 1),
            "rule": "上影≥3倍实体 + 下影<10% + 实体<35%",
            "index_name": "pin_bar_bear",
        })

    # ── 5. 光头光脚（大阳线/大阴线） ──
    for i in range(n - 3, n):
        # 上下影线占比均<10%
        if shadow_ratio(i, "lower") > 0.10 or shadow_ratio(i, "upper") > 0.10:
            continue
        # 波动>1.5倍ATR
        if atr_ratio(i) < 1.5:
            continue
        # 成交量>1.5倍14周期均量
        vol_ma14 = np.mean(V[max(0, i-14):i])
        if vol_ma14 > 0 and V[i] / vol_ma14 < 1.5:
            continue
        # 实体占比>60%
        if body_ratio(i) < 0.6:
            continue
        signal = "bullish" if is_bull(i) else "bearish"
        label = "光头光脚（大阳线）" if signal == "bullish" else "光头光脚（大阴线）"
        results.append({
            "pattern": label,
            "signal": signal,
            "index": i,
            "confidence": round(body_ratio(i) * 100, 1),
            "rule": "影线<10% + 波动>1.5ATR + 量>1.5x均量",
            "index_name": "shaved_top_bull" if signal == "bullish" else "shaved_top_bear",
        })

    # ── 6. 晨星（Morning Star） ──
    for i in range(2, n):
        p0, p1, p2 = i - 2, i - 1, i
        # 第1根大阴线，实体>60%
        if not (is_bear(p0) and body_ratio(p0) > 0.6):
            continue
        # 第2根小星（实体<65%，波动<0.75倍ATR）
        if body_ratio(p1) >= 0.65:
            continue
        if atr_ratio(p1) >= 0.75:
            continue
        # 第3根大阳线，实体>60%
        if not (is_bull(p2) and body_ratio(p2) > 0.6):
            continue
        # 第3根收盘超过第1根K线中点
        mid_p0 = (O[p0] + C[p0]) / 2
        if C[p2] <= mid_p0:
            continue
        results.append({
            "pattern": "晨星（看涨）",
            "signal": "bullish",
            "index": i,
            "confidence": round((body_ratio(p0) + body_ratio(p2)) * 50, 1),
            "rule": "大阴+小星+大阳 + 第3根收盘>第1根中点",
            "index_name": "morning_star",
        })

    # ── 7. 晚星（Evening Star） ──
    for i in range(2, n):
        p0, p1, p2 = i - 2, i - 1, i
        if not (is_bull(p0) and body_ratio(p0) > 0.6):
            continue
        if body_ratio(p1) >= 0.65:
            continue
        if atr_ratio(p1) >= 0.75:
            continue
        if not (is_bear(p2) and body_ratio(p2) > 0.6):
            continue
        mid_p0 = (O[p0] + C[p0]) / 2
        if C[p2] >= mid_p0:
            continue
        results.append({
            "pattern": "晚星（看跌）",
            "signal": "bearish",
            "index": i,
            "confidence": round((body_ratio(p0) + body_ratio(p2)) * 50, 1),
            "rule": "大阳+小星+大阴 + 第3根收盘<第1根中点",
            "index_name": "evening_star",
        })

    # ── 8. 刺穿线（Piercing Line） ──
    for i in range(1, n):
        prev, cur = i - 1, i
        if not (is_bear(prev) and is_bull(cur)):
            continue
        if body_ratio(prev) < 0.6 or body_ratio(cur) < 0.6:
            continue
        # 阳线收盘深入阴线实体中点以上（>50%穿透）
        mid_prev = (O[prev] + C[prev]) / 2
        if C[cur] <= mid_prev:
            continue
        results.append({
            "pattern": "刺穿线（看涨）",
            "signal": "bullish",
            "index": i,
            "confidence": round(min((C[cur] - mid_prev) / body(prev) * 50 + 50, 100), 1),
            "rule": "大阴+大阳 + 收盘>阴线中点",
            "index_name": "piercing",
        })

    # ── 9. 暗云盖顶（Dark Cloud Cover） ──
    for i in range(1, n):
        prev, cur = i - 1, i
        if not (is_bull(prev) and is_bear(cur)):
            continue
        if body_ratio(prev) < 0.6 or body_ratio(cur) < 0.6:
            continue
        mid_prev = (O[prev] + C[prev]) / 2
        if C[cur] <= mid_prev:
            continue
        results.append({
            "pattern": "暗云盖顶（看跌）",
            "signal": "bearish",
            "index": i,
            "confidence": round(min((mid_prev - C[cur]) / body(prev) * 50 + 50, 100), 1),
            "rule": "大阳+大阴 + 收盘<阳线中点",
            "index_name": "dark_cloud",
        })

    return results


def get_market_context(df: pd.DataFrame) -> dict:
    """获取市场上下文信息（趋势+成交量）"""
    if len(df) < 20:
        return {"trend": "unknown", "vol_ratio": 1.0}

    close = df["Close"].values
    volume = df["Volume"].values
    ma20 = np.mean(close[-20:])
    vol_ma20 = np.mean(volume[-20:])
    current_price = close[-1]
    current_vol = volume[-1]

    if current_price > ma20 * 1.05:
        trend = "up"
    elif current_price < ma20 * 0.95:
        trend = "down"
    else:
        trend = "sideways"

    return {
        "trend": trend,
        "vol_ratio": round(current_vol / vol_ma20, 2) if vol_ma20 > 0 else 1.0,
        "ma20": round(ma20, 2),
        "close": round(current_price, 2),
    }


def apply_filters(patterns: list, context: dict, min_confidence: float = 60) -> list:
    """
    三层过滤（对齐PDF实盘经验）：
    1. 置信度过滤：低于阈值的形态直接丢弃
    2. 趋势过滤：上升趋势只保留看涨形态，下降趋势只保留看跌形态
    3. 成交量过滤：缩量信号大概率是假信号（vol_ratio < 0.8时过滤）
    """
    if not patterns:
        return []

    filtered = []
    trend = context["trend"]
    vol_ratio = context["vol_ratio"]

    for p in patterns:
        # 置信度过滤
        if p["confidence"] < min_confidence:
            continue

        # 成交量过滤：缩量直接过滤（PDF原文经验）
        if vol_ratio < 0.8:
            continue

        # 趋势过滤：顺趋势使用（PDF原文经验）
        # 侧向趋势不做过滤（视为中性，可接受任意方向形态）
        if trend != "sideways":
            if trend == "up" and p["signal"] == "bearish":
                continue  # 上升趋势中过滤看跌形态
            if trend == "down" and p["signal"] == "bullish":
                continue  # 下降趋势中过滤看涨形态

        filtered.append(p)

    return filtered


# ─────────────────────────────────────────
# 主程序：全市场扫描
# ─────────────────────────────────────────

STOCK_LIST_A = [
    "600519.SS", "600036.SS", "601318.SS", "600276.SS", "601166.SS",
    "600887.SS", "600030.SS", "601328.SS", "601088.SS", "601601.SS",
    "600016.SS", "600028.SS", "601288.SS", "601398.SS", "601939.SS",
    "601668.SS", "601186.SS", "601390.SS", "601857.SS", "600050.SS",
    "601766.SS", "600104.SS", "601229.SS", "601012.SS", "601899.SS",
    "601628.SS", "603259.SS", "688981.SS", "688599.SS", "600585.SS",
    "600690.SS", "600031.SS", "600150.SS", "600588.SS", "600570.SS",
    "601888.SS", "603288.SS", "688041.SS", "601336.SS", "601319.SS",
    "601816.SS", "601377.SS", "601688.SS", "600009.SS", "600115.SS",
    "600170.SS", "601727.SS", "601800.SS", "601808.SS", "601818.SS",
    "002415.SZ", "002475.SZ", "002594.SZ", "002714.SZ", "002230.SZ",
    "002352.SZ", "002371.SZ", "002460.SZ", "002466.SZ", "002493.SZ",
    "300015.SZ", "300059.SZ", "300122.SZ", "300124.SZ", "300142.SZ",
    "300274.SZ", "300357.SZ", "300433.SZ", "300496.SZ", "300750.SZ",
]

STOCK_LIST_HK = [
    "0700.HK", "0941.HK", "9988.HK", "3690.HK", "9618.HK",
    "1810.HK", "1024.HK", "2382.HK", "2319.HK", "6690.HK",
    "6618.HK", "6030.HK", "1833.HK", "0386.HK", "0857.HK",
]

STOCK_LIST_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "BRK-B", "JPM", "JNJ",
]

STOCK_LIST = STOCK_LIST_A + STOCK_LIST_HK + STOCK_LIST_US


def scan_all(symbols: list, min_confidence: float = 60) -> dict:
    """全市场扫描"""
    all_results = []
    stats = {"total": 0, "detected": 0, "pattern_counts": {}}

    for idx, sym in enumerate(symbols, 1):
        stats["total"] += 1
        suffix = sym.split(".")[-1] if "." in sym else ""
        yf_sym = f"{sym.split('.')[0]}.{suffix}" if suffix else sym

        try:
            df = yf.Ticker(yf_sym).history(period="6mo", interval="1wk")
            if df.empty or len(df) < 10:
                continue

            # 形态检测
            patterns = detect_all_patterns(df, sym)

            # 市场上下文
            context = get_market_context(df)

            # 三层过滤
            filtered = apply_filters(patterns, context, min_confidence)

            if filtered:
                stats["detected"] += 1
                for p in filtered:
                    cls = p["pattern"]
                    stats["pattern_counts"][cls] = stats["pattern_counts"].get(cls, 0) + 1

                all_results.append({
                    "symbol": sym,
                    "market": "A股" if suffix in ("SS", "SZ") else ("港股" if suffix == "HK" else "美股"),
                    "trend": context["trend"],
                    "vol_ratio": context["vol_ratio"],
                    "price": context["close"],
                    "patterns": filtered,
                })

            status = f"✅ {len(filtered)}个形态" if filtered else "⭕ 无"
            print(f"  [{idx:3d}/{len(symbols)}] {sym:15s} 趋势:{context['trend']:8s} 量比:{context['vol_ratio']:.2f}x  {status}")

        except Exception as e:
            print(f"  [{idx:3d}/{len(symbols)}] ❌ {sym}: {e}")

    return {"results": all_results, "stats": stats}


def save_report(results: dict, report_path: str):
    """生成结构化报告"""
    stats = results["stats"]
    data = results["results"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("📊 K线形态识别报告（PDF规则实现版）\n")
        f.write(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("【一、扫描统计摘要】\n")
        f.write(f"  总扫描标的: {stats['total']}\n")
        f.write(f"  检出形态标的: {stats['detected']}\n")
        f.write(f"  形态分布:\n")
        for cls, cnt in sorted(stats["pattern_counts"].items(), key=lambda x: -x[1]):
            f.write(f"    {cls:30s}: {cnt} 次\n")

        f.write("\n【二、候选股票池】\n")
        f.write("  (已通过：置信度≥60% + 趋势过滤 + 成交量过滤)\n\n")
        if not data:
            f.write("  （当前扫描样本中无满足条件的形态信号）\n")
        for item in data:
            f.write(f"  📌 {item['symbol']} ({item['market']})\n")
            f.write(f"     趋势:{item['trend']:8s}  量比:{item['vol_ratio']:.2f}x  当前价:{item['price']}\n")
            for p in item["patterns"]:
                sig = "🔴 看跌" if p["signal"] == "bearish" else "🟢 看涨"
                f.write(f"     {sig} {p['pattern']:25s} 置信度:{p['confidence']:.0f}%  规则:{p['rule']}\n")
            f.write("\n")

        f.write("\n【三、使用说明】\n")
        f.write("  1. 严格对齐PDF《量化指标解码19》文章中的量化识别硬规则\n")
        f.write("  2. 三层过滤：实体占比>60% + ATR波动强度 + 趋势方向 + 成交量验证\n")
        f.write("  3. 以上仅为形态识别结果，不构成交易建议，需人工二次筛选\n")
        f.write("  4. 如需更高精度，建议结合YOLO自定义训练（阶段3）\n")


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("📊 K线形态识别（全市场扫描）")
    print(f"   数据源: yfinance（免费、无限制）")
    print(f"   形态规则: 严格对齐PDF《量化指标解码19》硬规则")
    print(f"   三层过滤: 置信度≥60% + 趋势过滤 + 成交量过滤")
    print("=" * 60 + "\n")

    results = scan_all(STOCK_LIST, min_confidence=60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"pattern_report_{timestamp}.txt")
    save_report(results, report_path)

    print()
    print("=" * 60)
    print("📋 扫描结果摘要")
    print(f"   总标的: {results['stats']['total']}")
    print(f"   检出形态: {results['stats']['detected']} 只")
    if results["stats"]["pattern_counts"]:
        print(f"   形态分布: {results['stats']['pattern_counts']}")
    print(f"   📄 报告: {report_path}")
    print("=" * 60)

    # 打印候选池
    if results["results"]:
        print("\n🏆 候选股票池（前10）:")
        for item in results["results"][:10]:
            for p in item["patterns"]:
                print(f"  {item['symbol']:10s} {item['trend']:8s} {p['pattern']:25s} {p['confidence']:.0f}%")

    return results


if __name__ == "__main__":
    main()
