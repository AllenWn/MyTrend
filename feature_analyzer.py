import pandas as pd
import numpy as np
from itertools import combinations
import json
import os
from features import (
    screen_second_wave,
    get_wave_periods,
    calculate_technical_indicators,
)
from config import (
    FEATURE_COMBINATION_PATH,
    MIN_SUPPORT,
    MAX_COMBINATION_LEN,
    NEG_THRESHOLD,
)  # 从config导入阈值配置


class FeatureCombinationAnalyzer:
    def __init__(self, min_support=None, max_combination_len=None, neg_threshold=None):
        # 使用config中的默认值，如果传入参数则覆盖
        self.min_support = min_support if min_support is not None else MIN_SUPPORT
        self.max_combination_len = (
            max_combination_len
            if max_combination_len is not None
            else MAX_COMBINATION_LEN
        )
        self.neg_threshold = (
            neg_threshold if neg_threshold is not None else NEG_THRESHOLD
        )
        self.positive_patterns = []  # 最终筛选的特征组合

        # 特征语义映射字典
        self.feature_semantics = {
            "drawdown_controlled": "调整幅度可控",
            "break_adjustment_high": "突破调整区间上沿",
            "break_flag_pattern": "突破旗形整理",
            "macd_second_red": "MACD二次翻红",
            "volume_surge": "放量突破",
            "ma_trend_aligned": "均线趋势一致",
            "rsi_oversold_rebound": "RSI超卖反弹",
            "bb_squeeze": "布林带收窄",
            "price_momentum": "价格动量增强",
            "support_resistance": "支撑阻力位突破",
        }

    def _extract_features(self, stock_data):
        """提取单只股票的特征（调用features.py逻辑）"""
        try:
            # 检查数据长度，如果不足40天，返回空字典
            # 特征提取需要40天数据：DERIVED_FEATURE_DAYS(20天) + INPUT_WINDOW(20天)
            if len(stock_data) < 40:
                print(f"数据长度不足40天（实际{len(stock_data)}天），跳过特征提取")
                return {}

            # 确保数据是DataFrame格式
            if not isinstance(stock_data, pd.DataFrame):
                print("输入数据不是DataFrame格式，跳过特征提取")
                return {}

            # 检查必要的列是否存在
            required_cols = ["open", "close", "high", "low", "volume"]
            missing_cols = [
                col for col in required_cols if col not in stock_data.columns
            ]
            if missing_cols:
                print(f"缺少必要列: {missing_cols}，跳过特征提取")
                return {}

            df = calculate_technical_indicators(stock_data.copy())
            periods = get_wave_periods(df)  # 依赖40天数据
            input_data = (df, periods)
            features = screen_second_wave(input_data)
            return {k: v for k, v in features.items() if v}  # 只保留True的特征
        except Exception as e:
            print(f"特征提取失败: {e}")
            return {}

    def _get_feature_semantic(self, feature_name):
        """获取特征的语义描述"""
        return self.feature_semantics.get(feature_name, feature_name)

    def _display_feature_combination_semantics(self, features):
        """显示特征组合的语义内容"""
        semantic_descriptions = []
        for feature in features:
            semantic = self._get_feature_semantic(feature)
            semantic_descriptions.append(f"{feature}({semantic})")
        return " + ".join(semantic_descriptions)

    def analyze_positive_samples(self, positive_data_list):
        """从正样本中归纳高频特征组合"""
        # 1. 提取所有正样本的特征
        all_positive_features = []
        for data in positive_data_list:
            features = self._extract_features(data)
            if features:
                all_positive_features.append(list(features.keys()))

        # 2. 统计不同长度的特征组合频率
        pattern_counts = {}
        total_samples = len(all_positive_features)

        for feat_list in all_positive_features:
            # 生成1到max_combination_len长度的所有组合
            for l in range(1, self.max_combination_len + 1):
                for combo in combinations(feat_list, l):
                    combo_key = frozenset(combo)  # 用frozenset确保组合无序
                    pattern_counts[combo_key] = pattern_counts.get(combo_key, 0) + 1

        # 3. 筛选超过最小支持度的组合
        qualified_patterns = []
        for combo, count in pattern_counts.items():
            support = count / total_samples
            if support >= self.min_support:
                qualified_patterns.append(
                    {"features": list(combo), "support": support, "count": count}
                )

        # 按支持度排序
        self.positive_patterns = sorted(
            qualified_patterns, key=lambda x: x["support"], reverse=True
        )

        # 显示找到的特征组合及其语义内容
        print(f"\n找到 {len(self.positive_patterns)} 个符合条件的特征组合:")
        for i, pattern in enumerate(self.positive_patterns[:10]):  # 显示前10个
            semantic_content = self._display_feature_combination_semantics(
                pattern["features"]
            )
            print(f"  组合{i+1}: {semantic_content}")
            print(f"      特征: {pattern['features']}")
            print(
                f"      支持度: {pattern['support']:.3f} ({pattern['count']}/{total_samples})"
            )

        return self.positive_patterns

    def validate_with_negative_samples(self, negative_data_list):
        """用负样本过滤特征组合"""
        if not self.positive_patterns:
            raise ValueError("请先调用analyze_positive_samples归纳正样本特征")

        # 1. 提取所有负样本的特征
        all_negative_features = []
        for data in negative_data_list:
            features = self._extract_features(data)
            if features:
                all_negative_features.append(list(features.keys()))

        total_neg_samples = len(all_negative_features)
        filtered_patterns = []

        # 2. 检查每个正样本组合在负样本中的出现频率
        for pattern in self.positive_patterns:
            combo_set = set(pattern["features"])
            neg_count = 0

            for neg_features in all_negative_features:
                if combo_set.issubset(set(neg_features)):
                    neg_count += 1

            neg_frequency = (
                neg_count / total_neg_samples if total_neg_samples > 0 else 0
            )

            if neg_frequency <= self.neg_threshold:
                filtered_patterns.append(pattern)
                # 显示过滤后的特征组合语义内容
                semantic_content = self._display_feature_combination_semantics(
                    pattern["features"]
                )
                print(f"  ✓ 保留组合: {semantic_content}")
                print(f"    负样本频率: {neg_frequency:.3f} (≤{self.neg_threshold})")
            else:
                semantic_content = self._display_feature_combination_semantics(
                    pattern["features"]
                )
                print(f"  ✗ 过滤组合: {semantic_content}")
                print(f"    负样本频率: {neg_frequency:.3f} (>{self.neg_threshold})")

        print(
            f"\n负样本过滤完成: {len(self.positive_patterns)} -> {len(filtered_patterns)} 个组合"
        )
        return filtered_patterns

    def save_patterns(self, save_path=FEATURE_COMBINATION_PATH):
        """保存特征组合到文件"""
        if not self.positive_patterns:
            raise ValueError("无有效特征组合可保存")

        # 确保路径存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 转换为可序列化格式（frozenset→list）
        save_data = [
            {
                "features": p["features"],
                "support": p["support"],
                "count": p["count"],
                "neg_ratio": p.get("neg_ratio", 0),
            }
            for p in self.positive_patterns
        ]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        print(f"特征组合已保存至: {save_path}")

    def load_patterns(self, load_path=FEATURE_COMBINATION_PATH):
        """从文件加载特征组合"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"特征组合文件不存在: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            self.positive_patterns = json.load(f)
        print(f"已加载{len(self.positive_patterns)}个特征组合")

    @staticmethod
    def evaluate_patterns_on_test(patterns, test_pos_data, test_neg_data):
        """评估特征组合在测试集上的表现"""
        if not patterns:
            return 0, 1.0  # 无组合时覆盖率0，误判率100%

        # 计算测试集正样本覆盖率（符合至少一个组合的比例）
        pos_covered = 0
        for data in test_pos_data:
            features = FeatureCombinationAnalyzer()._extract_features(data)
            feat_set = set(features.keys())
            for p in patterns:
                if set(p["features"]).issubset(feat_set):
                    pos_covered += 1
                    break
        coverage = pos_covered / len(test_pos_data) if test_pos_data else 0

        # 计算测试集负样本误判率（被错误覆盖的比例）
        neg_false = 0
        for data in test_neg_data:
            features = FeatureCombinationAnalyzer()._extract_features(data)
            feat_set = set(features.keys())
            for p in patterns:
                if set(p["features"]).issubset(feat_set):
                    neg_false += 1
                    break
        false_rate = neg_false / len(test_neg_data) if test_neg_data else 0

        return coverage, false_rate
