import torch
import pandas as pd
import numpy as np
import tushare as ts
import os
import time
import joblib
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from trading_calendar import TradingCalendar
from config import (
    FUTURE_RETURN_THRESHOLD,
    FEATURE_COLS,
    INPUT_WINDOW,
    INITIAL_START_DATE,
    INITIAL_END_DATE,
    DERIVED_FEATURE_DAYS,
    SCALER_DIR,
)


class KeyFeatureDataLoader:
    def __init__(
        self,
        data_path,
        feature_cols,
        key_feature_name,
        label_col,
        sector_code=None,
        start_date=None,
        end_date=None,
        stocks_file=None,
        trading_calendar=None,
        data_dir="stock_data",
        use_baostock=False,
        tushare_token="8b1ef90e2f704b9d90e09a0de94078ff5ae6c5c18cc3382e75b879b7",
    ):
        # 日期格式配置
        self.date_format = "%Y-%m-%d"
        self.date_format_ymd = "%Y%m%d"

        self.data_path = data_path
        self.feature_cols = feature_cols
        self.key_feature_name = key_feature_name
        self.label_col = label_col
        self.key_feature_idx = None
        self.sector_code = sector_code
        self.start_date = INITIAL_START_DATE
        self.end_date = INITIAL_END_DATE
        self.stocks_file = stocks_file
        self.stock_symbols = []
        self.use_baostock = use_baostock
        self.tushare_token = tushare_token
        self.ts_api = None
        self.scalers = {}

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(SCALER_DIR, exist_ok=True)

        # 初始化交易日历
        self.calendar = TradingCalendar("trading_calendar.csv")
        self.is_tushare_logged_in = False
        self.requires_tushare = False
        self.trading_calendar = trading_calendar
        self.processed_columns = None  # 存储预处理后的特征列

    def _format_date(self, date_str, target_format=None):
        """统一日期格式转换"""
        if not isinstance(date_str, str):
            date_str = str(date_str)

        if target_format is None:
            target_format = self.date_format

        # 尝试多种日期格式
        date_formats = [
            "%Y%m%d",  # 20250809
            "%Y-%m-%d",  # 2025-08-09
            "%Y/%m/%d",  # 2025/8/9 或 2025/08/09
        ]

        for fmt in date_formats:
            try:
                dt = pd.to_datetime(date_str, format=fmt)
                return dt.strftime(target_format)
            except:
                continue

        # 如果所有格式都失败，尝试使用pandas解析
        try:
            dt = pd.to_datetime(date_str)
            return dt.strftime(target_format)
        except:
            print(f"警告：无法识别的日期格式: {date_str}，使用原始值")
            return date_str

    def _format_date_for_cache(self, date_str):
        """转换为缓存文件名格式"""
        return self._format_date(date_str, self.date_format_ymd)

    def _get_date_delta(self, start_date, end_date):
        """计算日期差"""
        start_dt = pd.to_datetime(self._format_date(start_date))
        end_dt = pd.to_datetime(self._format_date(end_date))
        return (end_dt - start_dt).days

    # data_loader.py v2.0（完整版本 - 2/5）
    def tushare_login(self):
        """Tushare登录"""
        if not self.is_tushare_logged_in and self.requires_tushare:
            if not self.tushare_token:
                raise ValueError("使用Tushare需要设置tushare_token")

            try:
                ts.set_token(self.tushare_token)
                self.ts_api = ts.pro_api()
                self.is_tushare_logged_in = True
                print("Tushare登录成功")
                return True
            except Exception as e:
                print(f"Tushare登录失败: {e}")
                return False
        return True

    def tushare_logout(self):
        """Tushare登出"""
        if self.is_tushare_logged_in:
            self.is_tushare_logged_in = False
            print("Tushare登出成功")
            return True
        return False

    def get_stock_code(self, symbol):
        """转换股票代码格式"""
        if isinstance(symbol, str):
            if symbol.startswith(("688", "60", "601", "603")):
                return f"{symbol}.SH"
            else:
                return f"{symbol}.SZ"
        else:
            print(f"错误：股票代码不是字符串类型: {symbol}")
            return symbol

    def get_sector_stocks(self, sector_code=None):
        """加载成分股列表"""
        if not self.stocks_file:
            print("未指定成分股文件路径，请设置stocks_file参数")
            return []

        if not os.path.exists(self.stocks_file):
            print(f"成分股文件不存在: {self.stocks_file}")
            return []

        try:
            print(f"从文件加载{sector_code or '指定'}成分股列表...")
            try:
                df = pd.read_csv(self.stocks_file)
            except UnicodeDecodeError:
                df = pd.read_csv(self.stocks_file, encoding="gbk")

            if "code" not in df.columns:
                print("成分股文件中未找到'code'列")
                return []

            df["code"] = df["code"].astype(str).apply(lambda x: x.zfill(6))
            stocks = [
                {"code": row["code"], "name": row.get("name", "")}
                for _, row in df.iterrows()
            ]
            print(f"成功加载{len(stocks)}只成分股")
            return stocks

        except Exception as e:
            print(f"获取成分股失败: {e}")
            return []

    # data_loader.py v2.0（完整版本 - 3/5）
    def get_stock_data(self, symbol, from_date, to_date, max_retries=5, delay=2):
        """获取单只股票数据"""
        if not self.is_tushare_logged_in:
            raise ValueError("Tushare未登录，请先调用tushare_login()")

        stock_code = self.get_stock_code(symbol)
        formatted_from_date = self._format_date(from_date, "%Y%m%d")
        formatted_to_date = self._format_date(to_date, "%Y%m%d")

        date_delta = self._get_date_delta(formatted_from_date, formatted_to_date)
        if date_delta < 0:
            print(f"错误：结束日期早于开始日期")
            return pd.DataFrame()

        trading_days = self.calendar.get_trading_days(
            self._format_date(formatted_from_date), self._format_date(formatted_to_date)
        )
        if not trading_days:
            print(f"警告：无交易日数据")
            return pd.DataFrame()

        print(f"获取 {symbol} 数据，共{len(trading_days)}个交易日")

        for attempt in range(max_retries):
            try:
                df = self.ts_api.daily(
                    ts_code=stock_code,
                    start_date=formatted_from_date,
                    end_date=formatted_to_date,
                    fields="ts_code,trade_date,open,high,low,close,vol,amount,pre_close,change,pct_chg",
                )

                if df is not None and not df.empty:
                    print(f"成功获取{symbol}数据，共{len(df)}行")
                    break

                print(f"尝试 {attempt+1}/{max_retries}: 数据为空")

            except Exception as e:
                error_msg = str(e)
                print(f"尝试 {attempt+1}/{max_retries}: 错误: {error_msg}")
                if "没有接口访问权限" in error_msg:
                    print("权限不足，请检查Tushare积分")
                    return pd.DataFrame()

                wait_time = delay * (2**attempt)
                if attempt < max_retries - 1:
                    print(f"{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"获取{symbol}数据失败")
                    return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        required_cols = ["trade_date", "open", "high", "low", "close", "vol"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"缺少必要列: {missing_cols}")
            return pd.DataFrame()

        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        # 确保日期格式正确转换
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        except:
            # 如果固定格式失败，尝试自动解析
            df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index(ascending=True)
        df = self._add_technical_indicators(df)

        if not df.empty:
            self.validate_stock_data(df, FEATURE_COLS)
        return df

    def _add_technical_indicators(self, data):
        """计算技术指标（确保与FEATURE_COLS匹配）"""
        if data.empty:
            return data

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # 处理数值列
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        data[numeric_cols] = data[numeric_cols].interpolate(method="time")
        data = data.dropna(subset=numeric_cols)

        # 移动平均线
        data["ma5"] = data["close"].rolling(window=5).mean()
        data["ma10"] = data["close"].rolling(window=10).mean()

        # 成交量均线
        data["vol5"] = data["volume"].rolling(window=5).mean()
        data["vol10"] = data["volume"].rolling(window=10).mean()

        # RSI指标
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean().replace(0, 1e-10)  # 避免除零
        rs = avg_gain / avg_loss
        data["rsi"] = 100 - (100 / (1 + rs))

        # 布林带
        data["bb_middle"] = data["close"].rolling(window=20).mean()
        data["bb_std"] = data["close"].rolling(window=20).std()
        data["bb_upper"] = data["bb_middle"] + 2 * data["bb_std"]
        data["bb_lower"] = data["bb_middle"] - 2 * data["bb_std"]

        # 处理技术指标中的异常值
        technical_features = [
            "ma5",
            "ma10",
            "vol5",
            "vol10",
            "rsi",
            "bb_upper",
            "bb_lower",
        ]
        for feature in technical_features:
            if feature in data.columns:
                data[feature] = data[feature].ffill().fillna(data[feature].mean())
                # 裁剪极端值（均值±5倍标准差）
                mean = data[feature].mean()
                std = data[feature].std()
                data[feature] = data[feature].clip(
                    lower=mean - 5 * std, upper=mean + 5 * std
                )

        # 移除剩余NaN行
        original_len = len(data)
        data = data.dropna(subset=FEATURE_COLS)
        dropped = original_len - len(data)
        if dropped > 0:
            print(f"警告：删除了 {dropped} 行包含NaN的技术指标数据")

        return data

    # data_loader.py v2.0（完整版本 - 4/5）
    def preprocess_data(self, data, target_type="close", print_stats=True):
        """标准化处理（确保与模型输入匹配）"""
        print(f"preprocess_data{data}")
        if data.empty:
            raise ValueError("数据为空，无法预处理")

        missing = [col for col in FEATURE_COLS if col not in data.columns]
        if missing:
            raise ValueError(f"缺少必要特征: {missing}")

        self.processed_columns = FEATURE_COLS
        valid_data = data[FEATURE_COLS].copy()

        # 对成交量取对数（减少极端值影响）
        if "volume" in valid_data.columns:
            valid_data["volume"] = np.log1p(valid_data["volume"])

        # 缺失值处理
        for col in valid_data.columns:
            if valid_data[col].isna().all():
                valid_data[col] = 0
            else:
                valid_data[col] = valid_data[col].ffill().fillna(valid_data[col].mean())

        # 极端值处理（1%和99%分位数裁剪）
        for col in valid_data.columns:
            q1 = valid_data[col].quantile(0.01)
            q99 = valid_data[col].quantile(0.99)
            valid_data[col] = valid_data[col].clip(q1, q99)
            # 处理无穷大
            valid_data[col] = valid_data[col].replace([np.inf, -np.inf], q99)

        # 标准化（RobustScaler对异常值更稳健）
        scaler = RobustScaler()
        X = valid_data.values
        if not np.isfinite(X).all():
            X = np.nan_to_num(X)

        normalized_data = scaler.fit_transform(X)

        # 打印统计信息（调试用）
        if print_stats:
            print("\n===== 标准化前特征统计 =====")
            for col in FEATURE_COLS:
                print(
                    f"{col}：均值={valid_data[col].mean():.4f}，标准差={valid_data[col].std():.4f}"
                )

            print("\n===== 标准化后特征统计 =====")
            for i, col in enumerate(FEATURE_COLS):
                med = np.median(normalized_data[:, i])
                iqr = np.percentile(normalized_data[:, i], 75) - np.percentile(
                    normalized_data[:, i], 25
                )
                print(f"{col}：中位数={med:.4f}，四分位距={iqr:.4f}")

        normalized_df = pd.DataFrame(
            normalized_data, columns=FEATURE_COLS, index=data.index
        )
        normalized_df = normalized_df.fillna(0)  # 最终NaN填充

        return normalized_df, scaler

    def create_sequences(self, data, input_window, output_window, target_type="close"):
        """生成训练序列：输入=前20天特征，标签=未来5天是否涨15%"""
        sequences, labels = [], []
        data_np = data.values
        L = len(data)

        if not hasattr(self, "processed_columns"):
            raise AttributeError("请先调用preprocess_data方法")

        # 目标列索引（如close）
        target_col_idx = self.processed_columns.index(target_type)

        for i in range(L - input_window - output_window):
            # 输入序列：前20天的所有特征
            input_seq = data_np[i : i + input_window]

            # 计算未来5天的最大涨幅
            current_price = input_seq[-1, target_col_idx]  # 输入窗口最后一天的价格
            future_prices = data_np[
                i + input_window : i + input_window + output_window, target_col_idx
            ]  # 未来5天价格
            if len(future_prices) < output_window:
                continue  # 跳过不完整的未来窗口

            max_future_price = np.max(future_prices)
            price_ratio = max_future_price / current_price  # 计算涨幅比例

            # 标签：是否≥15%涨幅
            label = 1.0 if price_ratio >= FUTURE_RETURN_THRESHOLD else 0.0

            # 检查输入序列是否有异常值
            if np.isnan(input_seq).any() or np.isinf(input_seq).any():
                print(f"警告：序列{i}包含异常值，已跳过")
                continue

            sequences.append(input_seq)
            labels.append(label)

        if not sequences:
            raise ValueError("未生成有效训练序列，请检查数据长度和质量")

        return list(zip(sequences, labels))

    def create_feature_screening_data(
        self, data, input_window, output_window, target_type="close"
    ):
        """
        生成特征筛选数据：返回40天数据用于特征筛选

        参数:
            data: 原始股票数据（DataFrame，包含OHLCV列）
            input_window: 输入窗口长度（20天）
            output_window: 输出窗口长度（5天）
            target_type: 目标列类型（如'close'）

        返回:
            feature_data_list: 包含(股票数据, 标签, 序列索引)的列表
            每个股票数据是完整的40天DataFrame，包含所有必要的列
        """
        from config import DERIVED_FEATURE_DAYS

        feature_data_list = []
        L = len(data)

        # 计算特征筛选所需的总天数
        total_days_needed = DERIVED_FEATURE_DAYS + input_window  # 20 + 20 = 40天

        # 确保数据长度足够
        if L < total_days_needed + output_window:
            print(
                f"警告：数据长度{L}不足{total_days_needed + output_window}天，无法生成特征筛选数据"
            )
            return []

        for i in range(L - total_days_needed - output_window):
            # 获取完整的40天数据（用于特征筛选）
            # 注意：这里使用原始DataFrame的切片，保持列名和数据类型
            feature_data = data.iloc[i : i + total_days_needed].copy()

            # 计算未来5天的最大涨幅（用于标签生成）
            current_price = feature_data.iloc[-1][target_type]  # 输入窗口最后一天的价格
            future_prices = data.iloc[
                i + total_days_needed : i + total_days_needed + output_window
            ][
                target_type
            ]  # 未来5天价格

            if len(future_prices) < output_window:
                continue  # 跳过不完整的未来窗口

            max_future_price = future_prices.max()
            price_ratio = max_future_price / current_price  # 计算涨幅比例

            # 标签：是否≥15%涨幅
            label = 1.0 if price_ratio >= FUTURE_RETURN_THRESHOLD else 0.0

            # 检查数据是否有异常值
            if (
                feature_data.isnull().any().any()
                or feature_data.isin([np.inf, -np.inf]).any().any()
            ):
                print(f"警告：序列{i}包含异常值，已跳过")
                continue

            # 确保数据包含必要的列（open, close, high, low, volume）
            required_cols = ["open", "close", "high", "low", "volume"]
            if not all(col in feature_data.columns for col in required_cols):
                print(f"警告：序列{i}缺少必要列，已跳过")
                continue

            # 添加到特征筛选数据列表
            feature_data_list.append(
                {
                    "data": feature_data,
                    "label": label,
                    "sequence_index": i,
                    "current_price": current_price,
                    "future_max_price": max_future_price,
                    "price_ratio": price_ratio,
                    "total_days": len(feature_data),
                }
            )

        if not feature_data_list:
            print("警告：未生成有效特征筛选数据")
            return []

        print(f"成功生成 {len(feature_data_list)} 个特征筛选数据样本")
        print(
            f"每个样本包含 {total_days_needed} 天数据（{DERIVED_FEATURE_DAYS}天技术指标 + {input_window}天输入窗口）"
        )
        positive_count = sum(1 for item in feature_data_list if item["label"] == 1.0)
        print(
            f"正样本数量: {positive_count}, 负样本数量: {len(feature_data_list) - positive_count}"
        )

        return feature_data_list

    def create_sequences_with_raw_data(
        self, data, input_window, output_window, target_type="close"
    ):
        """
        生成训练序列：输入序列为40天（DERIVED_FEATURE_DAYS + input_window）+ 标签（是否为正样本）
        标签逻辑：40天数据结束后，未来5天内最高涨幅≥15%为1，否则为0
        """
        sequences, labels = [], []
        data_np = data.values
        L = len(data)

        # 40天总长度 = 衍生特征天数 + 输入窗口
        total_input_len = DERIVED_FEATURE_DAYS + input_window

        # 校验目标列
        if not hasattr(self, "processed_columns") or self.processed_columns is None:
            self.processed_columns = list(data.columns)
        if target_type not in self.processed_columns:
            raise ValueError(
                f"目标类型{target_type}不在特征列中: {self.processed_columns}"
            )
        target_col_idx = self.processed_columns.index(target_type)

        # 循环范围：确保40天数据 + 5天未来数据不越界
        for i in range(L - total_input_len - output_window + 1):
            # 1. 40天输入序列（核心修改：取足DERIVED_FEATURE_DAYS + input_window）
            input_seq = data_np[i : i + total_input_len]  # 长度=40天

            # 2. 计算未来5天最高涨幅（仅用于标签，不保存未来数据）
            current_price = input_seq[-1, target_col_idx]  # 40天最后一天的价格
            # 未来窗口：40天结束后紧接着的output_window天（如5天）
            future_start = i + total_input_len
            future_end = future_start + output_window
            future_prices = data_np[future_start:future_end, target_col_idx]

            # 跳过不完整的未来窗口
            if len(future_prices) < output_window:
                continue

            # 3. 计算标签（≥15%为正样本）
            max_future_price = np.max(future_prices)
            price_ratio = max_future_price / current_price
            label = 1.0 if price_ratio >= FUTURE_RETURN_THRESHOLD else 0.0

            # 过滤异常值
            if np.isnan(input_seq).any() or np.isinf(input_seq).any():
                print(f"警告：序列{i}包含异常值，已跳过")
                continue

            sequences.append(input_seq)
            labels.append(label)

        if not sequences:
            raise ValueError(
                f"未生成有效训练序列，数据长度需≥{total_input_len + output_window}天（40+5=45天）"
            )

        # 将numpy数组转换为DataFrame格式，保持列名信息
        sequences_with_columns = []
        for seq in sequences:
            # 将numpy数组转换为DataFrame，保持原有的列名
            seq_df = pd.DataFrame(seq, columns=self.processed_columns)
            sequences_with_columns.append(seq_df)

        # 返回DataFrame格式的序列和标签组合
        return list(zip(sequences_with_columns, labels))

    def prepare_raw_data_for_training(
        self, input_window, output_window, train_test_split=0.8, target_type="close"
    ):
        """
        准备板块数据用于训练和测试（使用原始数据，未标准化）

        参数:
            input_window: 输入序列长度
            output_window: 输出序列长度
            train_test_split: 训练集占比
            target_type: 目标类型（如收盘价）

        返回:
            train_sequences: 训练序列列表，每个元素为(原始特征序列, 标签)
            test_sequences: 测试序列列表，每个元素为(原始特征序列, 标签)
        """
        if not self.start_date or not self.end_date:
            raise ValueError("请先设置start_date和end_date属性")

        # 获取板块成分股数据
        df, valid_symbols = self.get_sector_stock_data(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("未获取到任何成分股数据")

        # 初始化训练和测试序列
        train_sequences = []
        test_sequences = []

        # 对每只股票准备数据
        for symbol in tqdm(valid_symbols, desc="处理成分股原始数据"):
            try:
                # 提取单只股票数据
                stock_data = df[df["symbol"] == symbol].copy()
                if stock_data.empty:
                    print(f"警告：{symbol}的数据为空，跳过")
                    continue

                # 确保所有必要特征都存在
                missing_features = [
                    col for col in FEATURE_COLS if col not in stock_data.columns
                ]
                if missing_features:
                    print(f"警告：{symbol}缺少特征{missing_features}，跳过")
                    continue

                # 对成交量取对数（减少极端值影响，但保持原始数据特征）
                if "volume" in stock_data.columns:
                    stock_data["volume"] = np.log1p(stock_data["volume"])

                # 缺失值处理
                for col in FEATURE_COLS:
                    if stock_data[col].isna().all():
                        stock_data[col] = 0
                    else:
                        stock_data[col] = (
                            stock_data[col].ffill().fillna(stock_data[col].mean())
                        )

                # 极端值处理（1%和99%分位数裁剪）
                for col in FEATURE_COLS:
                    q1 = stock_data[col].quantile(0.01)
                    q99 = stock_data[col].quantile(0.99)
                    stock_data[col] = stock_data[col].clip(q1, q99)
                    # 处理无穷大
                    stock_data[col] = stock_data[col].replace([np.inf, -np.inf], q99)

                # 设置processed_columns属性（create_sequences需要）
                self.processed_columns = FEATURE_COLS

                # 生成训练样本（使用原始数据）
                sequences = self.create_sequences_with_raw_data(
                    stock_data[FEATURE_COLS], input_window, output_window, target_type
                )
                if not sequences:
                    print(f"警告：股票{symbol}生成的序列为空，跳过")
                    continue

                # 划分训练集和测试集
                split_idx = int(len(sequences) * train_test_split)
                train_sequences.extend(sequences[:split_idx])
                test_sequences.extend(sequences[split_idx:])

            except Exception as e:
                print(f"处理股票{symbol}时出错: {e}")
                import traceback

                traceback.print_exc()

        if not train_sequences and not test_sequences:
            raise ValueError("未能生成任何训练或测试样本，请检查数据质量")

        print(
            f"已准备{len(train_sequences)}个训练样本和{len(test_sequences)}个测试样本（原始数据）"
        )
        return train_sequences, test_sequences

    # data_loader.py v2.0（完整版本 - 5/5）
    def load_or_append_data(self, symbol, from_date, to_date, force_download=False):
        """加载本地数据或追加新数据"""
        from_date_cache = self._format_date_for_cache(from_date)
        to_date_cache = self._format_date_for_cache(to_date)
        filename = f"a股_{symbol}_{from_date_cache}_{to_date_cache}.joblib"
        file_path = os.path.join(self.data_dir, filename)

        # 确保交易日历是最新的
        self.calendar.update()
        if pd.to_datetime(self.calendar.last_updated) < pd.to_datetime(from_date):
            print("交易日历过期，正在更新...")
            self.calendar.update()

        # 检查本地缓存
        if os.path.exists(file_path) and not force_download:
            try:
                print(f"从缓存加载 {symbol} 数据: {file_path}")
                data = joblib.load(file_path)
                return data
            except Exception as e:
                print(f"缓存文件损坏，重新下载: {e}")

        # 下载数据
        print(f"下载 {symbol} 数据 ({from_date} 至 {to_date})...")
        data = self.get_stock_data(symbol, from_date, to_date)

        if data.empty:
            print(f"{symbol} 无有效数据，跳过缓存")
            return data

        # 保存到本地缓存
        try:
            joblib.dump(data, file_path)
            print(f"数据已缓存至: {file_path}")
        except Exception as e:
            print(f"缓存失败: {e}")

        return data

    def validate_stock_data(self, data, required_cols):
        """验证股票数据完整性"""
        if data.empty:
            print("警告：数据为空")
            return False

        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            print(f"警告：数据缺少必要列: {missing}")
            return False

        # 检查日期连续性（允许最多3天缺口）
        date_diff = data.index.to_series().diff().dt.days.dropna()
        max_gap = date_diff.max()
        if max_gap > 3:
            print(f"警告：数据存在超过3天的缺口，最大缺口{max_gap}天")

        print(
            f"数据验证通过，共{len(data)}行，时间范围: {data.index.min().strftime('%Y-%m-%d')} 至 {data.index.max().strftime('%Y-%m-%d')}"
        )
        return True

    def load_sector_data(self, from_date, to_date, force_download=False):
        """加载板块内所有股票数据"""
        if not self.sector_code and not self.stocks_file:
            raise ValueError("请指定sector_code或stocks_file以加载板块数据")

        # 获取板块成分股
        stocks = self.get_sector_stocks(self.sector_code)
        if not stocks:
            raise ValueError("未获取到有效成分股列表")

        self.stock_symbols = [stock["code"] for stock in stocks]
        all_data = {}

        for stock in tqdm(stocks, desc="加载板块数据"):
            symbol = stock["code"]
            name = stock.get("name", "未知名称")
            print(f"\n===== 处理 {name} ({symbol}) =====")

            data = self.load_or_append_data(symbol, from_date, to_date, force_download)
            if not data.empty:
                all_data[symbol] = {"name": name, "data": data}
            else:
                print(f"{name} ({symbol}) 无有效数据，已跳过")

        print(f"\n板块数据加载完成，有效股票数: {len(all_data)}/{len(stocks)}")
        return all_data

    def create_dataset(
        self, stock_data_dict, input_window, output_window, target_type="close"
    ):
        """为板块内所有股票创建训练数据集"""
        if not stock_data_dict:
            raise ValueError("无有效股票数据，无法创建数据集")

        all_sequences = []
        all_labels = []
        scalers = {}

        for symbol, stock_info in tqdm(stock_data_dict.items(), desc="创建数据集"):
            name = stock_info["name"]
            data = stock_info["data"]

            try:
                # 预处理数据（标准化）
                processed_data, scaler = self.preprocess_data(
                    data, target_type=target_type
                )
                scalers[symbol] = scaler

                # 生成序列
                sequences = self.create_sequences(
                    processed_data,
                    input_window=input_window,
                    output_window=output_window,
                    target_type=target_type,
                )

                if sequences:
                    seqs, lbls = zip(*sequences)
                    all_sequences.extend(seqs)
                    all_labels.extend(lbls)
                    print(f"{name} 生成 {len(seqs)} 个序列")
                else:
                    print(f"{name} 未生成有效序列")

            except Exception as e:
                print(f"{name} 处理失败: {e}，已跳过")
                continue

        if not all_sequences:
            raise ValueError("未生成任何有效训练序列")

        # 转换为Tensor
        X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
        y = torch.tensor(np.array(all_labels), dtype=torch.float32).view(-1, 1)

        print(f"\n数据集生成完成：{len(X)} 个样本")
        print(f"正样本比例: {torch.sum(y).item() / len(y):.2%}")

        return X, y, scalers

    def create_feature_screening_dataset(
        self, stock_data_dict, input_window, output_window, target_type="close"
    ):
        """
        为板块内所有股票创建特征筛选数据集

        参数:
            stock_data_dict: 股票数据字典
            input_window: 输入窗口长度（20天）
            output_window: 输出窗口长度（5天）
            target_type: 目标列类型（如'close'）

        返回:
            feature_screening_data: 包含所有股票特征筛选数据的字典
            每个股票包含多个40天数据片段，用于调用features.py中的函数
        """
        if not stock_data_dict:
            raise ValueError("无有效股票数据，无法创建特征筛选数据集")

        feature_screening_data = {}

        for symbol, stock_info in tqdm(
            stock_data_dict.items(), desc="创建特征筛选数据集"
        ):
            name = stock_info["name"]
            data = stock_info["data"]

            try:
                # 注意：特征筛选使用原始数据，不进行标准化
                # 因为features.py中的函数需要原始的OHLCV数据
                if not self.validate_stock_data(
                    data, ["open", "close", "high", "low", "volume"]
                ):
                    print(f"{name} 数据验证失败，缺少必要列，已跳过")
                    continue

                # 生成特征筛选数据（使用原始数据）
                feature_data_list = self.create_feature_screening_data(
                    data,
                    input_window=input_window,
                    output_window=output_window,
                    target_type=target_type,
                )

                if feature_data_list:
                    feature_screening_data[symbol] = {
                        "name": name,
                        "feature_data_list": feature_data_list,
                        "total_samples": len(feature_data_list),
                        "positive_samples": sum(
                            1 for item in feature_data_list if item["label"] == 1.0
                        ),
                        "negative_samples": sum(
                            1 for item in feature_data_list if item["label"] == 0.0
                        ),
                    }
                    print(f"{name} 生成 {len(feature_data_list)} 个特征筛选样本")
                else:
                    print(f"{name} 未生成有效特征筛选样本")

            except Exception as e:
                print(f"{name} 特征筛选数据处理失败: {e}，已跳过")
                continue

        if not feature_screening_data:
            raise ValueError("未生成任何有效特征筛选数据")

        # 统计总体情况
        total_samples = sum(
            info["total_samples"] for info in feature_screening_data.values()
        )
        total_positive = sum(
            info["positive_samples"] for info in feature_screening_data.values()
        )
        total_negative = sum(
            info["negative_samples"] for info in feature_screening_data.values()
        )

        print(f"\n特征筛选数据集生成完成：")
        print(f"总股票数: {len(feature_screening_data)}")
        print(f"总样本数: {total_samples}")
        print(f"正样本数: {total_positive}")
        print(f"负样本数: {total_negative}")
        print(f"正样本比例: {total_positive/total_samples:.2%}")

        return feature_screening_data

    def get_sector_stock_data(
        self, from_date, to_date, max_retries=3, force_download=False
    ):
        """获取板块所有成分股的数据，保持Tushare登录状态"""
        if not self.stocks_file and not self.sector_code:
            raise ValueError("请先设置stocks_file或sector_code属性")

        stocks = self.get_sector_stocks(self.sector_code)
        if not stocks:
            raise ValueError(f"成分股列表为空，请检查{self.stocks_file}文件内容")

        self.stock_symbols = [stock["code"] for stock in stocks]

        all_data = []
        valid_symbols = []

        # 确保交易日历是最新的
        self.calendar.update()

        # 计算日期范围内的交易日数量
        trading_days = self.calendar.get_trading_days(
            self._format_date(from_date), self._format_date(to_date)
        )
        if not trading_days:
            print(f"警告：{from_date} 至 {to_date} 之间没有交易日")
            return pd.DataFrame(), []

        print(f"{from_date} 至 {to_date} 之间有 {len(trading_days)} 个交易日")

        # 登录Tushare（只登录一次）
        self.requires_tushare = True
        if not self.tushare_login():
            raise ValueError("Tushare登录失败，无法获取数据")

        # 控制请求频率，避免Tushare限流
        for i, stock in enumerate(tqdm(stocks, desc="获取成分股数据")):
            symbol = stock["code"]
            try:
                stock_data = self.load_or_append_data(
                    symbol, from_date, to_date, force_download
                )
                if not stock_data.empty:
                    stock_data["symbol"] = symbol
                    all_data.append(stock_data)
                    valid_symbols.append(symbol)

                # Tushare有请求频率限制，控制请求间隔
                if (i + 1) % 20 == 0:  # 每请求20只股票后暂停
                    print(
                        f"已请求{i+1}/{len(stocks)}只股票，暂停3秒避免请求频率超限..."
                    )
                    time.sleep(3)  # 暂停3秒

            except Exception as e:
                print(f"获取{symbol}数据失败: {e}")
                # Tushare请求出错后适当暂停
                time.sleep(2)

        # 所有数据获取完成后登出
        self.tushare_logout()

        if all_data:
            print(f"成功获取{len(valid_symbols)}只成分股的数据")
            return pd.concat(all_data), valid_symbols
        print("警告：未能获取任何成分股数据")
        return pd.DataFrame(), []

    def prepare_sector_data(self, start_date, end_date, force_download=False):
        """
        准备特定板块的股票数据（简化版，只加载数据不生成序列）

        参数:
            start_date: 开始日期
            end_date: 结束日期
            force_download: 是否强制重新下载数据

        返回:
            处理后的股票数据字典
        """
        if not self.sector_code:
            raise ValueError("请先设置sector_code")

        self.start_date = start_date
        self.end_date = end_date

        # 登录Tushare
        if not self.tushare_login():
            raise ValueError("Tushare登录失败，请检查token")

        try:
            # 加载板块内所有股票数据
            print(
                f"开始加载{self.sector_code}板块数据，时间范围: {start_date} 至 {end_date}"
            )
            sector_data = self.load_sector_data(
                from_date=start_date, to_date=end_date, force_download=force_download
            )

            if not sector_data:
                raise ValueError(f"未能加载任何{self.sector_code}板块的股票数据")

            print(f"成功加载{len(sector_data)}只股票数据")
            return sector_data

        finally:
            # 登出Tushare
            self.tushare_logout()

    def prepare_sector_data_for_training(
        self, input_window, output_window, train_test_split=0.8, target_type="close"
    ):
        """
        准备板块数据用于训练和测试（原始版本，生成训练序列）

        参数:
            input_window: 输入序列长度
            output_window: 输出序列长度
            train_test_split: 训练集占比
            target_type: 目标类型（如收盘价）
        """
        if not self.start_date or not self.end_date:
            raise ValueError("请先设置start_date和end_date属性")

        # 获取板块成分股数据
        df, valid_symbols = self.get_sector_stock_data(self.start_date, self.end_date)

        if df.empty:
            raise ValueError("未获取到任何成分股数据")

        # 创建保存scaler的目录
        os.makedirs(SCALER_DIR, exist_ok=True)

        # 初始化训练和测试序列
        train_sequences = []
        test_sequences = []

        # 对每只股票准备数据
        for symbol in tqdm(valid_symbols, desc="处理成分股数据"):
            try:
                # 提取单只股票数据
                stock_data = df[df["symbol"] == symbol].copy()
                if stock_data.empty:
                    print(f"警告：{symbol}的数据为空，跳过")
                    continue

                # 为单只股票生成scaler
                normalized_data, scaler = self.preprocess_data(stock_data, target_type)

                # 保存该股票的scaler（按股票代码命名）
                scaler_path = os.path.join(SCALER_DIR, f"scaler_{symbol}.pth")
                torch.save(scaler, scaler_path)
                print(f"已保存{symbol}的scaler至: {scaler_path}")

                # 生成训练样本
                sequences = self.create_sequences(
                    normalized_data, input_window, output_window, target_type
                )
                if not sequences:
                    print(f"警告：股票{symbol}生成的序列为空，跳过")
                    continue

                # 划分训练集和测试集
                split_idx = int(len(sequences) * train_test_split)
                train_sequences.extend(sequences[:split_idx])
                test_sequences.extend(sequences[split_idx:])

            except Exception as e:
                print(f"处理股票{symbol}时出错: {e}")
                import traceback

                traceback.print_exc()

        if not train_sequences and not test_sequences:
            raise ValueError("未能生成任何训练或测试样本，请检查数据质量")

        print(
            f"已准备{len(train_sequences)}个训练样本和{len(test_sequences)}个测试样本"
        )
        return train_sequences, test_sequences

    def load_scaler(self, symbol):
        """加载指定股票的标准化器"""
        if symbol in self.scalers:
            return self.scalers[symbol]

        scaler_path = os.path.join(SCALER_DIR, f"{symbol}_scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                self.scalers[symbol] = scaler
                return scaler
            except Exception as e:
                print(f"加载标准化器失败: {e}")

        return None

    def save_scaler(self, symbol, scaler):
        """保存标准化器到本地"""
        scaler_path = os.path.join(SCALER_DIR, f"{symbol}_scaler.joblib")
        try:
            joblib.dump(scaler, scaler_path)
            self.scalers[symbol] = scaler
            print(f"标准化器已保存至: {scaler_path}")
        except Exception as e:
            print(f"保存标准化器失败: {e}")
