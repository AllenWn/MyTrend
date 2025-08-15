import pandas as pd
import numpy as np


# --------------------------
# 共通计算函数（提取重复逻辑）
# --------------------------
def calculate_technical_indicators(df):
    """计算所有技术指标（供各特征函数共用）"""
    df = df.copy()
    # 均线
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    # MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_line"] = df["ema12"] - df["ema26"]
    df["signal_line"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["signal_line"]
    # RSI
    delta = df["close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi14"] = 100 - (100 / (1 + (gain / loss).replace([np.inf, -np.inf], 0)))
    # 布林带
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    return df


def get_wave_periods(df):
    """
    基于5日/10日均线划分40天滑动窗口的波段
    处理逻辑：
    1. 过滤均线计算不足的前9天数据（不参与波段判断）
    2. 确保所有场景返回明确波段，无异常值填充
    3. 返回结构化波段信息，供后续特征函数调用
    """
    df = calculate_technical_indicators(df.copy())
    n_days = len(df)
    if n_days != 40:
        raise ValueError("输入必须为40天数据")

    # --------------------------
    # 1. 处理均线计算不足的前9天数据
    # --------------------------
    # MA10需要至少10天数据，前9天（索引0-8）的MA10无效，标记为无效区域
    valid_start = 9  # 从第10天（索引9）开始，MA10有效
    valid_df = df.iloc[valid_start:].copy()  # 仅用有效数据判断波段
    if len(valid_df) < 10:  # 有效数据不足10天，无法判断趋势
        return {
            "is_valid": False,  # 标记数据无效
            "overall_trend": "数据不足",
            "error": "有效数据不足10天，无法计算波段",
            "bands": None,  # 无波段信息
        }

    # --------------------------
    # 2. 计算均线指标（仅基于有效数据）
    # --------------------------
    # 金叉/死叉（1=金叉，-1=死叉）
    valid_df["ma_cross"] = 0
    valid_df.loc[
        (valid_df["ma5"] > valid_df["ma10"])
        & (valid_df["ma5"].shift(1) <= valid_df["ma10"].shift(1)),
        "ma_cross",
    ] = 1
    valid_df.loc[
        (valid_df["ma5"] < valid_df["ma10"])
        & (valid_df["ma5"].shift(1) >= valid_df["ma10"].shift(1)),
        "ma_cross",
    ] = -1

    # 均线角度（斜率）和方向
    valid_df["ma5_angle"] = valid_df["ma5"].pct_change(5) * 100  # 近5天变化率
    valid_df["ma10_angle"] = valid_df["ma10"].pct_change(5) * 100
    valid_df["ma5_dir"] = valid_df["ma5_angle"].apply(
        lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
    )
    valid_df["ma10_dir"] = valid_df["ma10_angle"].apply(
        lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
    )

    # --------------------------
    # 3. 识别波段（仅用有效数据）
    # --------------------------
    bands = []  # 存储所有波段信息（start/end/type）

    # 3.1 持续多头排列（5>10>20，无死叉，方向向上）
    is_persistent_bull = (
        (valid_df["ma5"] > valid_df["ma10"]).all()
        and (valid_df["ma10"] > valid_df["ma20"]).all()
        and (valid_df["ma_cross"] != -1).all()  # 无死叉
        and (valid_df["ma5_dir"] == 1).all()
        and (valid_df["ma10_dir"] == 1).all()
    )
    if is_persistent_bull:
        bands.append(
            {
                "start": valid_start,  # 转换为原始数据索引（+valid_start）
                "end": 39,
                "type": "持续多头",
                "strength": "强" if valid_df["ma5_angle"].mean() > 3 else "弱",
            }
        )

    # 3.2 横盘整理（非多头，无交叉，波动小）
    elif (valid_df["ma_cross"] == 0).all():  # 无任何金叉/死叉
        price_volatility = valid_df["close"].pct_change().abs().mean()
        if price_volatility < 0.01:
            bands.append(
                {
                    "start": valid_start,
                    "end": 39,
                    "type": "横盘整理",
                    "volatility": price_volatility,
                }
            )

    # 3.3 震荡波段（交叉频繁，方向混乱）
    else:
        cross_count = (valid_df["ma_cross"] != 0).sum()
        dir_change = (valid_df["ma5_dir"].diff() != 0).sum() + (
            valid_df["ma10_dir"].diff() != 0
        ).sum()
        if cross_count >= 3 and dir_change >= 3:
            bands.append(
                {
                    "start": valid_start,
                    "end": 39,
                    "type": "震荡波段",
                    "cross_count": cross_count,
                }
            )

    # 3.4 未匹配上述类型，则按金叉/死叉划分上涨/下跌波段
    if not bands:
        # 取最后一次金叉/死叉作为波段起点
        last_cross_idx = (
            valid_df[valid_df["ma_cross"] != 0].index[-1]
            if (valid_df["ma_cross"] != 0).any()
            else None
        )
        if last_cross_idx is not None:
            cross_type = valid_df.loc[last_cross_idx, "ma_cross"]
            band_type = "上涨波段" if cross_type == 1 else "下跌波段"
            bands.append(
                {
                    "start": last_cross_idx + valid_start,  # 转换为原始索引
                    "end": 39,
                    "type": band_type,
                    "signal": "金叉" if cross_type == 1 else "死叉",
                }
            )
        else:
            # 无任何交叉，视为平缓趋势
            bands.append(
                {
                    "start": valid_start,
                    "end": 39,
                    "type": "平缓趋势",
                    "ma5_dir": valid_df["ma5_dir"].iloc[-1],
                }
            )

    # --------------------------
    # 4. 构造返回值（确保后续特征可调用）
    # --------------------------
    current_day = df.iloc[-1]
    prev_day = df.iloc[-2] if n_days >= 2 else None

    return {
        "is_valid": True,  # 标记数据有效
        "overall_trend": bands[-1]["type"],  # 最终趋势
        "bands": bands,  # 所有波段详情（供特征函数调用）
        "current_period": {
            "start": bands[-1]["start"],
            "end": bands[-1]["end"],
            "type": bands[-1]["type"],
        },
        "current_day": current_day,
        "prev_day": prev_day,
        "current_idx": 39,  # 40天数据的最后索引
        "ma_metrics": {  # 关键均线指标，供特征函数使用
            "ma5_angle": valid_df["ma5_angle"].iloc[-1],
            "ma10_angle": valid_df["ma10_angle"].iloc[-1],
            "last_cross": valid_df["ma_cross"].iloc[-1],
        },
    }


# --------------------------
# 特征函数（输入为(df, periods)元组）
# --------------------------


def check_drawdown_controlled(input_data):
    """特征1：调整幅度可控（兼容所有波段类型，增强回调识别精度）"""
    df, periods = input_data
    if not periods["is_valid"]:
        return False

    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]
    band_data = df.iloc[band_start : band_end + 1]

    if overall_trend in ["持续多头", "上涨波段"]:
        # 优化：更精准定位第一波高点（波段内前半段高点）
        split_idx = band_start + (band_end - band_start) // 2
        first_wave_high = band_data.iloc[:split_idx]["high"].max()
        # 回调低点取调整期最低（而非固定近5天）
        adjustment_data = band_data.iloc[split_idx:]
        current_low = (
            adjustment_data["low"].min()
            if not adjustment_data.empty
            else band_data["low"].min()
        )

        if first_wave_high == 0:  # 避免除零错误
            return False
        drawdown = (first_wave_high - current_low) / first_wave_high
        # 增加趋势强度过滤：强趋势允许更大回调（30%），弱趋势更严格（20%）
        strength_factor = 0.3 if current_band.get("strength") == "强" else 0.2
        return drawdown <= strength_factor

    elif overall_trend == "横盘整理":
        # 优化：用波段内最高价和最低价计算真实波动幅度
        band_high = band_data["high"].max()
        band_low = band_data["low"].min()
        volatility = (band_high - band_low) / band_low
        return volatility < 0.05  # 替代依赖ma_metrics的volatility

    else:  # 下跌/震荡波段
        return False


def check_break_adjustment_high(input_data):
    """特征2：突破调整区间上沿（增强调整区间识别逻辑）"""
    df, periods = input_data
    if not periods["is_valid"] or periods["current_day"] is None:
        return False

    current_day = periods["current_day"]
    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 优化：根据趋势动态确定调整区间长度
    lookback_days = 15 if overall_trend == "震荡波段" else 10
    recent_data = df.iloc[max(0, band_end - lookback_days) : band_end + 1]
    recent_high = recent_data["high"].max()

    # 增强突破有效性：收盘价突破+成交量放大
    volume_condition = current_day["volume"] > 1.2 * recent_data["volume"].mean()

    if overall_trend in ["横盘整理", "震荡波段"]:
        # 横盘/震荡突破需连续2天站稳
        if band_end < 1:
            return current_day["close"] > recent_high and volume_condition
        prev_close = df["close"].iloc[band_end - 1]
        return (
            current_day["close"] > recent_high and prev_close > recent_high
        ) and volume_condition

    else:  # 上涨/持续多头
        # 从波段数据中提取调整高点（而非依赖ma_metrics）
        peak_idx = recent_data["high"].idxmax()
        adjustment_high = recent_data.loc[peak_idx]["high"]
        return current_day["close"] > adjustment_high and volume_condition


def check_break_first_wave_high(input_data):
    """特征3：突破第一波高点（完善第一波识别逻辑）"""
    df, periods = input_data
    if not periods["is_valid"]:
        return False

    # 扩展适用场景：除上涨趋势外，横盘突破也可视为突破第一波
    valid_trends = ["持续多头", "上涨波段", "横盘整理"]
    if periods["overall_trend"] not in valid_trends:
        return False

    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 优化：分阶段识别第一波（前40%为第一波，后60%为调整）
    split_ratio = 0.4 if periods["overall_trend"] in ["持续多头", "上涨波段"] else 0.6
    split_idx = band_start + int((band_end - band_start) * split_ratio)
    first_wave_data = df.iloc[band_start:split_idx]

    if first_wave_data.empty:
        return False

    first_wave_high = first_wave_data["high"].max()
    current_close = periods["current_day"]["close"]

    # 增强验证：突破时伴随量能放大且收盘价站稳高点
    volume_condition = (
        periods["current_day"]["volume"] > 1.3 * first_wave_data["volume"].mean()
    )
    return current_close > first_wave_high and volume_condition


def check_volume_pattern(input_data):
    """特征4：调整期缩量且起爆点放量+量价齐升（兼容所有波段类型）"""
    df, periods = input_data
    # 数据无效时直接返回False
    if not periods["is_valid"]:
        return False

    current_day = periods["current_day"]
    prev_day = periods["prev_day"]
    if prev_day is None:
        return False

    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]  # 获取当前波段信息
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 1. 针对有明确调整期的典型上涨波段（保留原逻辑核心）
    if overall_trend in ["持续多头", "上涨波段"]:
        # 自动识别调整期（当前波段内价格回调超过5%的阶段）
        peak_price = df["close"].iloc[band_start : band_end + 1].max()
        adjustment_mask = (
            df["close"].iloc[band_start : band_end + 1] < peak_price * 0.95
        )
        if adjustment_mask.any():
            adjustment_start = band_start + adjustment_mask.idxmax()
            adjustment_end = band_end - (adjustment_mask[::-1].idxmax() - band_start)
            adjustment_data = df.iloc[adjustment_start : adjustment_end + 1]
            first_wave_data = df.iloc[band_start:adjustment_start]

            # 原逻辑：调整期缩量、起爆点放量、量价齐升
            adj_vol_mean = adjustment_data["volume"].mean()
            first_vol_mean = (
                first_wave_data["volume"].mean()
                if not first_wave_data.empty
                else adj_vol_mean
            )
            is_shrinking = adj_vol_mean < 0.7 * first_vol_mean
            is_expanded = current_day["volume"] >= 1.5 * adj_vol_mean
            is_price_up = current_day["close"] > prev_day["close"]
            is_vol_up = current_day["volume"] > prev_day["volume"]
            return is_shrinking and is_expanded and is_price_up and is_vol_up

    # 2. 横盘整理波段（缩量后放量）
    elif overall_trend == "横盘整理":
        # 取波段内前80%作为整理期，后20%作为起爆观察期
        split_idx = band_start + int((band_end - band_start) * 0.8)
        consolidation_data = df.iloc[band_start:split_idx]
        consolidation_vol_mean = consolidation_data["volume"].mean()

        # 保留缩量后放量+量价齐升逻辑
        is_shrinking = current_day["volume"] > 1.5 * consolidation_vol_mean
        is_price_up = current_day["close"] > prev_day["close"]
        is_vol_up = current_day["volume"] > prev_day["volume"]
        return is_shrinking and is_price_up and is_vol_up

    # 3. 其他趋势（震荡/下跌）保留量价齐升核心判断
    else:
        # 至少满足量价齐升基本条件
        return (
            current_day["close"] > prev_day["close"]
            and current_day["volume"] > prev_day["volume"]
        )


def check_ma_support(input_data):
    """特征5：调整期站稳20日均线（兼容所有波段类型）"""
    df, periods = input_data
    # 数据无效时直接返回False
    if not periods["is_valid"]:
        return False

    current_idx = periods["current_idx"]
    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 1. 针对有明确调整期的波段（保留原逻辑核心）
    if overall_trend in ["持续多头", "上涨波段"]:
        # 自动识别调整期（同量能函数的调整期定义）
        peak_price = df["close"].iloc[band_start : band_end + 1].max()
        adjustment_mask = (
            df["close"].iloc[band_start : band_end + 1] < peak_price * 0.95
        )
        if adjustment_mask.any():
            adjustment_start = band_start + adjustment_mask.idxmax()
            adjustment_end = band_end - (adjustment_mask[::-1].idxmax() - band_start)
            adjustment_data = df.iloc[adjustment_start : adjustment_end + 1]
            adj_ma20 = df["ma20"].iloc[adjustment_start : adjustment_end + 1]

            # 原逻辑：调整期全程站稳20日均线
            return (adjustment_data["close"] > adj_ma20).all()

    # 2. 横盘整理波段（全程在20日均线上方）
    elif overall_trend == "横盘整理":
        band_ma20 = df["ma20"].iloc[band_start : band_end + 1]
        band_close = df["close"].iloc[band_start : band_end + 1]
        return (band_close > band_ma20).all()

    # 3. 震荡波段（近5天站稳20日均线）
    elif overall_trend == "震荡波段":
        recent_close = df["close"].iloc[-5:]
        recent_ma20 = df["ma20"].iloc[-5:]
        return (recent_close > recent_ma20).all()

    # 4. 其他情况（至少当前价格在20日均线上方）
    return df["close"].iloc[current_idx] > df["ma20"].iloc[current_idx]


def check_ma_bullish(input_data):
    """特征6：均线多头排列（5>10>20且角度为正）"""
    df, periods = input_data
    # 数据无效时直接返回False
    if not periods["is_valid"]:
        return False

    current_idx = periods["current_idx"]
    # 获取当前波段信息
    current_band = periods["bands"][-1]
    band_type = current_band["type"]

    # 基础多头排列判断（5>10>20）
    ma5 = df["ma5"].iloc[current_idx]
    ma10 = df["ma10"].iloc[current_idx]
    ma20 = df["ma20"].iloc[current_idx]
    is_basic_bull = ma5 > ma10 and ma10 > ma20

    # 不同波段类型的增强判断
    if band_type in ["持续多头", "上涨波段"]:
        # 上涨趋势中要求均线角度为正（趋势延续）
        return (
            is_basic_bull
            and periods["ma_metrics"]["ma5_angle"] > 0
            and periods["ma_metrics"]["ma10_angle"] > 0
        )
    elif band_type == "横盘整理":
        # 横盘突破时要求短期均线角度转正
        return is_basic_bull and periods["ma_metrics"]["ma5_angle"] > 0
    else:  # 震荡/下跌波段
        # 仅需基础多头排列（视为潜在反转信号）
        return is_basic_bull


def check_ma_golden_cross(input_data):
    """特征7：5日与10日均线金叉（结合波段趋势）"""
    df, periods = input_data
    # 数据无效时直接返回False
    if not periods["is_valid"]:
        return False

    current_idx = periods["current_idx"]
    # 利用波段指标中的交叉信号快速判断
    last_cross = periods["ma_metrics"]["last_cross"]

    # 1. 优先使用波段计算的交叉信号（提高效率）
    if last_cross == 1:
        # 金叉有效性验证：不同波段对金叉的要求不同
        if periods["overall_trend"] in ["持续多头", "上涨波段"]:
            # 上涨趋势中，金叉需伴随均线角度增大
            return (
                periods["ma_metrics"]["ma5_angle"] > periods["ma_metrics"]["ma10_angle"]
            )
        elif periods["overall_trend"] == "横盘整理":
            # 横盘趋势中，金叉需突破整理区间
            current_band = periods["bands"][-1]
            recent_high = (
                df["high"].iloc[current_band["start"] : current_band["end"]].max()
            )
            return periods["current_day"]["close"] > recent_high
        else:  # 震荡波段
            # 震荡中，金叉需伴随成交量放大
            return (
                periods["current_day"]["volume"] > 1.2 * df["volume"].iloc[-5:].mean()
            )

    # 2. 未检测到交叉信号时，执行原始逐日验证（兼容所有场景）
    if current_idx < 1:
        return False
    prev_ma5 = df["ma5"].iloc[current_idx - 1]
    prev_ma10 = df["ma10"].iloc[current_idx - 1]
    curr_ma5 = df["ma5"].iloc[current_idx]
    curr_ma10 = df["ma10"].iloc[current_idx]

    # 原始金叉判断逻辑
    is_golden_cross = prev_ma5 <= prev_ma10 and curr_ma5 > curr_ma10

    # 针对震荡波段的额外过滤（减少假信号）
    if periods["overall_trend"] == "震荡波段" and is_golden_cross:
        # 震荡中需连续2天维持金叉才算有效
        if current_idx < 2:
            return False
        return df["ma5"].iloc[current_idx - 1] > df["ma10"].iloc[current_idx - 1]

    return is_golden_cross


def check_macd_second_red(input_data):
    """特征8：MACD二次翻红（0轴附近金叉，兼容所有波段类型）"""
    df, periods = input_data
    # 数据无效时直接返回False
    if not periods["is_valid"]:
        return False

    current_idx = periods["current_idx"]
    if current_idx < 1:
        return False  # 确保有前一天数据

    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 1. 上涨/持续多头波段（保留原始二次翻红逻辑）
    if overall_trend in ["持续多头", "上涨波段"]:
        # 自动识别第一波和调整期（第一波为波段前半段，调整期为后半段）
        split_idx = band_start + (band_end - band_start) // 2
        first_wave_data = df.iloc[band_start:split_idx]
        adjustment_data = df.iloc[split_idx:band_end]

        if first_wave_data.empty or adjustment_data.empty:
            return False

        # 原始逻辑：调整期MACD贴近0轴，当前出现金叉且翻红
        first_macd_peak = first_wave_data["macd_hist"].abs().max()
        is_near_zero = (
            adjustment_data["macd_hist"].abs() < 0.5 * first_macd_peak
        ).all()

        curr_macd = df["macd_line"].iloc[current_idx]
        curr_signal = df["signal_line"].iloc[current_idx]
        prev_macd = df["macd_line"].iloc[current_idx - 1]
        prev_signal = df["signal_line"].iloc[current_idx - 1]
        is_cross = prev_macd <= prev_signal and curr_macd > curr_signal

        return is_near_zero and is_cross and df["macd_hist"].iloc[current_idx] > 0

    # 2. 横盘/震荡波段（简化为0轴附近金叉）
    else:
        # 横盘震荡中只需满足MACD在0轴附近金叉且翻红
        curr_macd = df["macd_line"].iloc[current_idx]
        curr_signal = df["signal_line"].iloc[current_idx]
        prev_macd = df["macd_line"].iloc[current_idx - 1]
        prev_signal = df["signal_line"].iloc[current_idx - 1]

        is_cross = prev_macd <= prev_signal and curr_macd > curr_signal
        is_near_zero = abs(df["macd_hist"].iloc[current_idx]) < 0.02  # 贴近0轴阈值

        return is_cross and is_near_zero and df["macd_hist"].iloc[current_idx] > 0


def check_rsi_breakout(input_data):
    """特征9：RSI突破60且调整期未超卖（兼容所有波段类型）"""
    df, periods = input_data
    if not periods["is_valid"]:
        return False

    current_idx = periods["current_idx"]
    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 1. 上涨/持续多头波段（保留原始调整期判断）
    if overall_trend in ["持续多头", "上涨波段"]:
        # 自动识别调整期（波段内价格回调阶段）
        peak_price = df["close"].iloc[band_start : band_end + 1].max()
        adjustment_mask = (
            df["close"].iloc[band_start : band_end + 1] < peak_price * 0.95
        )

        if adjustment_mask.any():
            adjustment_start = band_start + adjustment_mask.idxmax()
            adjustment_end = band_end - (adjustment_mask[::-1].idxmax() - band_start)
            adj_rsi = df["rsi14"].iloc[adjustment_start : adjustment_end + 1]

            # 原始逻辑：调整期RSI未超卖（≥30）且当前突破60
            return (adj_rsi >= 30).all() and 60 < df["rsi14"].iloc[
                current_idx
            ] < 80  # 新增超买过滤

    # 2. 横盘整理波段（整个波段RSI未超卖，当前突破60）
    elif overall_trend == "横盘整理":
        band_rsi = df["rsi14"].iloc[band_start : band_end + 1]
        return (band_rsi >= 30).all() and df["rsi14"].iloc[current_idx] > 60

    # 3. 震荡波段（近5天RSI未超卖，当前突破60）
    else:
        recent_rsi = (
            df["rsi14"].iloc[-5:]
            if current_idx >= 4
            else df["rsi14"].iloc[: current_idx + 1]
        )
        return (recent_rsi >= 30).all() and df["rsi14"].iloc[current_idx] > 60


def check_bb_upper_break(input_data):
    """特征10：突破布林带上轨（增强版，结合波段趋势）"""
    df, periods = input_data
    if not periods["is_valid"]:
        return False

    current_day = periods["current_day"]
    current_idx = periods["current_idx"]
    overall_trend = periods["overall_trend"]

    # 基础判断：收盘价突破布林带上轨（保留原始逻辑）
    is_break = current_day["close"] > df["bb_upper"].iloc[current_idx]

    # 结合趋势增强判断，减少假突破
    if overall_trend in ["持续多头", "上涨波段"]:
        # 上涨趋势中，要求突破伴随量能放大
        return is_break and current_day["volume"] > 1.2 * df["volume"].iloc[-5:].mean()
    elif overall_trend == "横盘整理":
        # 横盘突破需连续2天站稳上轨
        if current_idx < 1:
            return is_break
        return (
            is_break
            and df["close"].iloc[current_idx - 1] > df["bb_upper"].iloc[current_idx - 1]
        )
    else:  # 震荡波段
        # 震荡中突破需成交量显著放大（1.5倍）
        return is_break and current_day["volume"] > 1.5 * df["volume"].iloc[-5:].mean()


def check_big_bull_candle(input_data):
    """特征11：起爆点大阳线（涨幅≥3%，实体占比≥70%，结合波段增强判断）"""
    df, periods = input_data
    # 数据无效时直接返回False
    if not periods["is_valid"] or periods["current_day"] is None:
        return False

    current_day = periods["current_day"]
    overall_trend = periods["overall_trend"]

    # 核心判断逻辑（保留原始定义）
    change = (current_day["close"] - current_day["open"]) / current_day["open"]
    body = abs(current_day["close"] - current_day["open"])
    range_ = current_day["high"] - current_day["low"]

    # 避免除以零错误（高低点相等时视为无效）
    if range_ == 0:
        return False

    is_big_bull = change >= 0.03 and (body / range_) >= 0.7

    # 结合波段趋势增强判断（提高信号质量）
    if is_big_bull:
        if overall_trend in ["持续多头", "上涨波段"]:
            # 上涨趋势中，大阳线需伴随成交量放大（确认动能）
            return current_day["volume"] > 1.2 * df["volume"].iloc[-5:].mean()
        elif overall_trend in ["横盘整理", "震荡波段"]:
            # 横盘/震荡中，大阳线需突破近期阻力位
            recent_high = df["high"].iloc[-20:].max()
            return current_day["close"] > recent_high
        # 下跌波段中出现大阳线视为潜在反转信号，保留原始判断

    return is_big_bull


def check_bull_engulfing(input_data):
    """特征12：阳包阴形态（结合波段过滤无效信号）"""
    df, periods = input_data
    # 数据无效或缺少前一天数据时返回False
    if (
        not periods["is_valid"]
        or periods["current_day"] is None
        or periods["prev_day"] is None
    ):
        return False

    current_day = periods["current_day"]
    prev_day = periods["prev_day"]
    overall_trend = periods["overall_trend"]

    # 核心判断逻辑（保留原始定义）
    # 前日为阴线，今日为阳线且完全包裹前日实体
    is_prev_bear = prev_day["close"] < prev_day["open"]
    is_current_bull = current_day["close"] > current_day["open"]
    is_engulf = (
        current_day["open"] < prev_day["close"]
        and current_day["close"] > prev_day["open"]
    )

    base_pattern = is_prev_bear and is_current_bull and is_engulf

    # 结合波段趋势过滤假信号
    if base_pattern:
        if overall_trend == "震荡波段":
            # 震荡中需成交量同步放大才有效
            return current_day["volume"] > 1.5 * prev_day["volume"]
        elif overall_trend == "横盘整理":
            # 横盘突破需收盘价创新高
            return (
                current_day["close"]
                > df["high"]
                .iloc[periods["bands"][-1]["start"] : periods["current_idx"]]
                .max()
            )

    return base_pattern


def is_price_volume_rise(input_data):
    """特征13：当天价涨量增（结合波段趋势判断强度）"""
    df, periods = input_data
    # 数据无效或缺少前一天数据时返回False
    if (
        not periods["is_valid"]
        or periods["current_day"] is None
        or periods["prev_day"] is None
    ):
        return False

    current = periods["current_day"]
    prev = periods["prev_day"]
    overall_trend = periods["overall_trend"]

    # 核心判断逻辑（保留原始定义）
    price_rise = current["close"] > prev["close"]
    volume_rise = current["volume"] > prev["volume"]

    # 基础价涨量增判断
    base_condition = price_rise and volume_rise

    # 结合波段趋势设置量能门槛（提高趋势延续性判断）
    if base_condition:
        # 计算近期平均成交量作为参考基准
        recent_vol_mean = (
            df["volume"].iloc[-10:].mean() if len(df) >= 10 else prev["volume"]
        )

        if overall_trend in ["持续多头", "上涨波段"]:
            # 上涨趋势中需放量至近期均值1.2倍以上
            return volume_rise and current["volume"] > 1.2 * recent_vol_mean
        elif overall_trend == "横盘整理":
            # 横盘突破需放量至近期均值1.5倍以上
            return volume_rise and current["volume"] > 1.5 * recent_vol_mean

    return base_condition


def is_big_bull_candle(input_data):
    """特征14：起爆点出现大阳线（结合波段趋势增强判断）"""
    df, periods = input_data
    # 数据无效或缺少当前天数据时返回False
    if not periods["is_valid"] or periods["current_day"] is None:
        return False

    current = periods["current_day"]
    overall_trend = periods["overall_trend"]
    current_band = periods["bands"][-1]

    # 核心判断逻辑（保留原始定义）
    price_change = (current["close"] - current["open"]) / current["open"]
    body = abs(current["close"] - current["open"])
    range_total = current["high"] - current["low"]

    # 避免除以零错误
    if range_total == 0:
        return False
    body_ratio = body / range_total
    base_condition = (price_change >= 0.03) and (body_ratio >= 0.7)

    # 结合波段趋势判断起爆点有效性
    if base_condition:
        # 起爆点需处于关键位置（波段起点或突破点）
        is_crucial_point = current_band["start"] == periods[
            "current_idx"
        ] or periods[  # 波段第一天
            "current_period"
        ][
            "type"
        ] in [
            "第二波上涨",
            "突破调整区间",
        ]  # 突破阶段

        if overall_trend in ["持续多头", "上涨波段"]:
            # 上涨趋势中，大阳线需伴随量能放大且处于调整后
            return (
                is_crucial_point
                and current["volume"] > 1.3 * df["volume"].iloc[-10:].mean()
            )
        elif overall_trend in ["横盘整理", "震荡波段"]:
            # 横盘/震荡中，大阳线需突破波段高点
            band_high = (
                df["high"].iloc[current_band["start"] : current_band["end"]].max()
            )
            return is_crucial_point and current["close"] > band_high

    return base_condition


def is_break_flag_pattern(input_data):
    """特征15：突破旗形整理（自动识别调整期，兼容新波段结构）"""
    df, periods = input_data
    # 数据无效时返回False
    if not periods["is_valid"]:
        return False

    current = periods["current_day"]
    current_idx = periods["current_idx"]
    current_band = periods["bands"][-1]
    band_start = current_band["start"]
    band_end = current_band["end"]

    # 1. 自动识别旗形整理期（基于波段内的调整阶段）
    # 旗形整理通常出现在上涨波段后的短期调整，取波段内前70%作为潜在整理期
    flag_length = int((band_end - band_start) * 0.7)
    if flag_length < 5:  # 旗形整理至少需要5天
        return False

    flag_start = band_start
    flag_end = band_start + flag_length
    adjustment = df.iloc[flag_start:flag_end]  # 替代原有的adjustment_period['data']

    # 2. 旗形形态判断（保留核心逻辑）
    adjustment_highs = adjustment["high"]
    adjustment_lows = adjustment["low"]

    # 旗形特征：高点和低点均小幅下移（斜率为负但绝对值较小）
    avg_high_change = adjustment_highs.diff().mean()
    avg_low_change = adjustment_lows.diff().mean()
    is_flag = (
        -0.02 < avg_high_change < 0  # 高点缓慢下移
        and -0.02 < avg_low_change < 0  # 低点缓慢下移
        and (adjustment_highs.max() - adjustment_lows.min()) / adjustment_lows.min()
        < 0.05  # 振幅小
    )

    # 3. 突破判断（收盘价突破旗形整理区间高点）
    is_break = current["close"] > adjustment_highs.max()

    # 4. 结合趋势增强有效性
    if is_flag and is_break:
        # 有效突破需伴随成交量放大
        flag_vol_mean = adjustment["volume"].mean()
        return current["volume"] > 1.5 * flag_vol_mean

    return False


def check_ma_convergence(input_data):
    """特征16：均线汇聚（结合绝对价格差和百分比差判断）"""
    df, periods = input_data
    if not periods["is_valid"]:
        return False

    current_idx = periods["current_idx"]
    overall_trend = periods["overall_trend"]

    # 获取当前均线值
    ma5 = df["ma5"].iloc[current_idx]
    ma10 = df["ma10"].iloc[current_idx]
    ma20 = df["ma20"].iloc[current_idx]

    # 定义绝对价格差阈值（5分以内视为汇聚，可根据需求调整）
    ABSOLUTE_THRESHOLD = 0.05  # 绝对价格差阈值（如0.05元）

    # 计算均线间的绝对差值
    abs_diff_5_10 = abs(ma5 - ma10)
    abs_diff_5_20 = abs(ma5 - ma20)
    abs_diff_10_20 = abs(ma10 - ma20)

    # 绝对差判断：只要所有均线间的绝对差都小于阈值，直接判定为汇聚
    if (
        abs_diff_5_10 < ABSOLUTE_THRESHOLD
        and abs_diff_5_20 < ABSOLUTE_THRESHOLD
        and abs_diff_10_20 < ABSOLUTE_THRESHOLD
    ):
        return True

    # 若绝对差不满足，再用百分比差辅助判断（避免低价股误判）
    # 计算均线最小值（避免除零）
    min_ma = min(ma5, ma10, ma20)
    if min_ma <= 0:
        return False  # 排除价格为0或负数的极端情况

    # 计算百分比差
    pct_diff_5_10 = abs_diff_5_10 / min_ma
    pct_diff_5_20 = abs_diff_5_20 / min_ma
    pct_diff_10_20 = abs_diff_10_20 / min_ma
    max_pct_diff = max(pct_diff_5_10, pct_diff_5_20, pct_diff_10_20)

    # 结合趋势动态调整百分比阈值
    if overall_trend in ["持续多头", "上涨波段"]:
        # 上涨趋势：百分比差<2%，且保持多头排列
        return max_pct_diff < 0.02 and ma5 > ma10 > ma20
    elif overall_trend == "横盘整理":
        # 横盘：百分比差<1.5%（更严格）
        return max_pct_diff < 0.015
    elif overall_trend == "震荡波段":
        # 震荡：5日与10日连续3天百分比差<1%
        if current_idx < 2:
            return False
        # 检查前2天的百分比差
        prev_ma5 = df["ma5"].iloc[current_idx - 1]
        prev_ma10 = df["ma10"].iloc[current_idx - 1]
        prev_min = min(prev_ma5, prev_ma10)
        prev_pct = abs(prev_ma5 - prev_ma10) / prev_min if prev_min > 0 else 1

        prev_prev_ma5 = df["ma5"].iloc[current_idx - 2]
        prev_prev_ma10 = df["ma10"].iloc[current_idx - 2]
        prev_prev_min = min(prev_prev_ma5, prev_prev_ma10)
        prev_prev_pct = (
            abs(prev_prev_ma5 - prev_prev_ma10) / prev_prev_min
            if prev_prev_min > 0
            else 1
        )

        return pct_diff_5_10 < 0.01 and prev_pct < 0.01 and prev_prev_pct < 0.01
    else:  # 下跌波段
        # 下跌：百分比差<2.5%（宽松标准，捕捉潜在反转）
        return max_pct_diff < 0.025


def screen_second_wave(input_data):
    """组合所有特征，返回筛选结果"""
    df, periods = input_data
    features = {
        # '调整幅度可控': check_drawdown_controlled(input_data),
        # '突破调整区间上沿': check_break_adjustment_high(input_data),
        # '突破第一波高点': check_break_first_wave_high(input_data),
        # '调整期缩量且起爆点放量': check_volume_pattern(input_data),
        # '价涨量增': is_price_volume_rise(input_data),
        # '调整期站稳20日均线': check_ma_support(input_data),
        # '均线多头排列': check_ma_bullish(input_data),
        # '5日与10日均线金叉': check_ma_golden_cross(input_data),
        # 'MACD二次翻红': check_macd_second_red(input_data),
        # 'RSI突破60': check_rsi_breakout(input_data),
        # '突破布林带上轨': check_bb_upper_break(input_data),
        # '大阳线': is_big_bull_candle(input_data),
        # '阳包阴': check_bull_engulfing(input_data),
        "均线汇聚": check_ma_convergence(input_data)
    }
    return features


# 使用示例
if __name__ == "__main__":
    # 模拟40天数据
    date_range = pd.date_range(start="2023-01-01", periods=40)
    np.random.seed(43)
    first_wave = np.cumsum(np.random.randn(20)) + 100 + np.linspace(0, 10, 20)
    adjustment = (
        first_wave[-1] * (1 - np.random.uniform(0.3, 0.5, 19))
        + np.random.randn(19) * 0.5
    )
    breakout_day = adjustment[-1] * 1.05
    close_prices = np.concatenate([first_wave, adjustment, [breakout_day]])
    df = pd.DataFrame(
        {
            "close": close_prices,
            "high": close_prices + np.random.rand(40) * 2,
            "low": close_prices - np.random.rand(40) * 2,
            "open": close_prices - np.random.rand(40) * 1,
            "volume": np.concatenate(
                [
                    np.random.randint(1000, 2000, 20),
                    np.random.randint(500, 800, 19),
                    [1500],
                ]
            ),
        },
        index=date_range,
    )

    # 预处理数据（模拟preprocess_stock_data返回值）
    processed_df = calculate_technical_indicators(df)
    periods = get_wave_periods(df)
    input_data = (processed_df, periods)

    # 检测特征
    print("单个特征检测：", check_break_adjustment_high(input_data))

    # 检测全部特征
    all_results = screen_second_wave(input_data)
    print("\n全部特征结果：")
    for name, result in all_results.items():
        print(f"{name}: {result}")
