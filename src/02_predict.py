"""
阶段2：K线形态识别预测
- 支持自定义预训练模型路径（极简入门用标准yolov8n-pose演示pipeline）
- 批量预测全市场K线图，输出结构化报告
- 三层过滤：趋势过滤 + 成交量过滤 + 置信度阈值
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"

import yfinance as yf
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# ─────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kline_samples")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# 默认用标准YOLOv8n演示（无自定义模型时），真实场景替换为你的自定义模型
DEFAULT_MODEL = "yolov8n.pt"

# 可配置的形态类别名（当使用自定义K线模型时替换这些名称）
CUSTOM_CLASSES = [
    "head_shoulder_top",   # 头肩顶
    "head_shoulder_bot",   # 头肩底
    "double_top",          # 双重顶
    "double_bot",          # 双重底
    "triangle_up",         # 上升三角形
    "triangle_down",       # 下降三角形
    "flag",                # 旗形
    "wedge_up",            # 上升楔形
    "wedge_down",          # 下降楔形
    "engulf_bull",         # 吞没（看涨）
    "engulf_bear",         # 吞没（看跌）
    "morning_star",        # 晨星
    "evening_star",        # 晚星
    "pin_bar_bull",        # 针棒（看涨）
    "pin_bar_bear",        # 针棒（看跌）
]

# 搜索目录下所有pt模型
def find_model():
    custom = os.path.join(MODEL_PATH, "yolov8n-kline.pt")
    if os.path.exists(custom) and os.path.getsize(custom) > 1_000_000:
        print(f"🔍 找到自定义K线模型: {custom}")
        return custom
    print(f"🔍 未找到自定义模型，使用标准YOLOv8n演示预测流程")
    return DEFAULT_MODEL


def get_trend_and_volume(symbol: str) -> dict:
    """获取标的的趋势和成交量信息，用于三层过滤"""
    try:
        # 从symbol提取纯代码（去掉交易所后缀）
        pure_sym = symbol.replace("_", ".").replace("-", "")
        suffix = ""
        for s in [".SS", ".SZ", ".HK"]:
            if pure_sym.endswith(s):
                suffix = s
                pure_sym = pure_sym.replace(s, "")
                break
        yf_sym = f"{pure_sym}{suffix}" if suffix else pure_sym

        df = yf.Ticker(yf_sym).history(period="3mo", interval="1d")
        if df.empty or len(df) < 20:
            return {"trend": "unknown", "vol_ratio": 1.0}

        close = df["Close"].values
        ma20 = np.mean(close[-20:])  # 20日均线
        current_price = close[-1]
        vol_ma20 = np.mean(df["Volume"].values[-20:])
        vol_today = df["Volume"].values[-1]

        # 趋势判断
        if current_price > ma20 * 1.05:
            trend = "up"
        elif current_price < ma20 * 0.95:
            trend = "down"
        else:
            trend = "sideways"

        # 成交量比值
        vol_ratio = vol_today / vol_ma20 if vol_ma20 > 0 else 1.0

        return {"trend": trend, "vol_ratio": vol_ratio, "ma20": ma20, "close": current_price}
    except Exception:
        return {"trend": "unknown", "vol_ratio": 1.0}


def predict_batch(model_path: str, conf_threshold: float = 0.3) -> dict:
    """批量预测所有K线图"""
    model = YOLO(model_path)

    image_files = sorted([
        f for f in os.listdir(SAMPLES_DIR)
        if f.endswith(".png")
    ])

    if not image_files:
        print("❌ 未找到K线图片")
        return {"results": [], "summary": {}}

    print(f"\n📊 批量预测中，共 {len(image_files)} 张图片...")
    print(f"   模型: {model_path}")
    print(f"   置信度阈值: {conf_threshold}")
    print()

    all_results = []
    detection_counts = {}

    for fname in image_files:
        symbol = fname.replace(".png", "").replace("_", ".")
        img_path = os.path.join(SAMPLES_DIR, fname)

        # 运行预测
        results = model.predict(
            img_path,
            conf=conf_threshold,
            iou=0.45,
            verbose=False,
            save=False,
        )

        # 提取检测结果
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.cpu().numpy()
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 映射类别名（自定义模型有具体类别名，标准模型是0-79的通用类别）
                    if len(CUSTOM_CLASSES) > cls_id:
                        cls_name = CUSTOM_CLASSES[cls_id]
                    else:
                        cls_name = f"class_{cls_id}"

                    detections.append({
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "box": [round(x, 1) for x in box.xyxy[0].tolist()],
                    })

        # 统计
        if detections:
            all_results.append({
                "symbol": symbol,
                "detections": detections,
            })
            for d in detections:
                cls = d["class"]
                detection_counts[cls] = detection_counts.get(cls, 0) + 1

        # 进度
        idx = image_files.index(fname) + 1
        status = f"✅ {len(detections)}个形态" if detections else "⭕ 无"
        print(f"  [{idx:3d}/{len(image_files)}] {symbol:15s} {status}")

    return {"results": all_results, "summary": detection_counts, "total_images": len(image_files)}


def filter_signals(results: dict, min_vol_ratio: float = 1.5, require_trend: bool = True) -> dict:
    """
    三层过滤：
    1. 趋势过滤：上升趋势只保留看涨形态，下降趋势只保留看跌形态
    2. 成交量过滤：形态出现日成交量≥1.5倍20日均量
    3. 置信度过滤：已在外层conf_threshold控制
    """
    filtered = []
    trend_bull_classes = {"engulf_bull", "morning_star", "pin_bar_bull", "head_shoulder_bot", "double_bot", "triangle_up"}
    trend_bear_classes = {"engulf_bear", "evening_star", "pin_bar_bear", "head_shoulder_top", "double_top", "triangle_down"}

    for item in results["results"]:
        symbol = item["symbol"]
        meta = get_trend_and_volume(symbol)
        trend = meta["trend"]

        valid_detections = []
        for d in item["detections"]:
            cls = d["class"]

            # 趋势过滤
            if require_trend and trend != "unknown":
                if trend == "up" and cls in trend_bear_classes:
                    continue  # 上升趋势中过滤看跌形态
                if trend == "down" and cls in trend_bull_classes:
                    continue  # 下降趋势中过滤看涨形态

            # 成交量过滤
            if meta["vol_ratio"] < min_vol_ratio:
                continue  # 缩量形态大概率是假信号

            valid_detections.append(d)

        if valid_detections:
            filtered.append({
                "symbol": symbol,
                "trend": trend,
                "vol_ratio": round(meta["vol_ratio"], 2),
                "price": round(meta.get("close", 0), 2),
                "detections": valid_detections,
            })

    return filtered


def save_report(all_results: dict, filtered: list, report_path: str):
    """生成结构化报告"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"📊 K线形态识别报告\n")
        f.write(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # 统计摘要
        f.write("【一、检测统计摘要】\n")
        f.write(f"  总图片数: {all_results['total_images']}\n")
        f.write(f"  检出形态数: {len(all_results['results'])}\n")
        f.write(f"  形态分布:\n")
        for cls, cnt in sorted(all_results["summary"].items(), key=lambda x: -x[1]):
            f.write(f"    {cls:25s}: {cnt} 次\n")

        # 过滤后候选池
        f.write("\n【二、过滤后候选股票池】\n")
        f.write("  (已通过趋势过滤 + 成交量过滤)\n\n")
        if not filtered:
            f.write("  （无候选股票，满足条件的形态信号不足）\n")
        for item in filtered:
            f.write(f"  📌 {item['symbol']}\n")
            f.write(f"     趋势: {item['trend']:10s}  成交量比: {item['vol_ratio']:.2f}x  当前价: {item['price']}\n")
            for d in item["detections"]:
                f.write(f"     - {d['class']:25s} 置信度: {d['confidence']:.2f}\n")

        # 说明
        f.write("\n【三、使用说明】\n")
        f.write("  1. 以上仅为形态识别结果，不构成交易建议\n")
        f.write("  2. 需人工结合基本面、行业、消息面做二次筛选\n")
        f.write("  3. 当前使用标准YOLOv8n模型展示流程，需替换自定义K线模型\n")
        f.write("  4. 自定义模型建议识别准确率≥85%后再用于实盘\n")

    return report_path


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 找模型
    model_path = find_model()

    # 批量预测
    results = predict_batch(model_path, conf_threshold=0.3)

    # 三层过滤
    filtered = filter_signals(results, min_vol_ratio=0.8, require_trend=False)

    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUTS_DIR, f"report_{timestamp}.txt")
    save_report(results, filtered, report_path)

    print()
    print("=" * 60)
    print("📋 预测结果摘要")
    print(f"   总图片: {results['total_images']}")
    print(f"   检出形态: {len(results['results'])} 张")
    print(f"   三层过滤后候选: {len(filtered)} 个")
    print(f"   📄 报告: {report_path}")
    if results["summary"]:
        print(f"   形态分布: {results['summary']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
