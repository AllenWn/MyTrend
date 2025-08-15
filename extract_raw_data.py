#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取标准化之前的贴好标签的数据
用于获取原始特征数据，便于分析和调试
"""

import torch
import pandas as pd
import numpy as np
import os
from data_loader import KeyFeatureDataLoader
from config import (
    FEATURE_COLS,
    INPUT_WINDOW,
    OUTPUT_WINDOW,
    INITIAL_START_DATE,
    INITIAL_END_DATE,
    FUTURE_RETURN_THRESHOLD,
)


def extract_raw_labeled_data():
    """
    提取标准化之前的贴好标签的数据
    """
    print("开始提取原始标签数据...")

    # 初始化数据加载器
    data_loader = KeyFeatureDataLoader(
        data_path="",
        feature_cols=FEATURE_COLS,
        key_feature_name="future_return",
        start_date=INITIAL_START_DATE,
        end_date=INITIAL_END_DATE,
        label_col="target",
        use_baostock=True,
        stocks_file="data/hs300_stocks.csv",
        tushare_token="8b1ef90e2f704b9d90e09a0de94078ff5ae6c5c18cc3382e75b879b7",
    )
    data_loader.requires_tushare = True
    data_loader.tushare_login()

    # 使用新的方法提取原始数据
    print("正在准备原始数据...")
    train_sequences, test_sequences = data_loader.prepare_raw_data_for_training(
        input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, train_test_split=0.8
    )

    print(f"成功提取数据：")
    print(f"  训练样本数量: {len(train_sequences)}")
    print(f"  测试样本数量: {len(test_sequences)}")

    # 分析数据分布
    train_labels = [label for _, label in train_sequences]
    test_labels = [label for _, label in test_sequences]

    print(f"\n标签分布分析：")
    print(
        f"  训练集 - 正样本(1): {sum(train_labels)}, 负样本(0): {len(train_labels) - sum(train_labels)}"
    )
    print(
        f"  测试集 - 正样本(1): {sum(test_labels)}, 负样本(0): {len(test_labels) - sum(test_labels)}"
    )

    # 保存原始数据到文件
    save_raw_data(train_sequences, test_sequences)

    return train_sequences, test_sequences


def save_raw_data(train_sequences, test_sequences):
    """
    保存原始数据到文件
    """
    # 创建保存目录
    output_dir = "raw_data"
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练数据
    train_data = []
    for i, (sequence, label) in enumerate(train_sequences):
        # 将序列转换为DataFrame格式
        sequence_df = pd.DataFrame(sequence, columns=FEATURE_COLS)
        sequence_df["sample_id"] = f"train_{i:06d}"
        sequence_df["label"] = label
        sequence_df["sequence_index"] = range(len(sequence))
        train_data.append(sequence_df)

    train_df = pd.concat(train_data, ignore_index=True)
    train_file = os.path.join(output_dir, "train_raw_data.csv")
    train_df.to_csv(train_file, index=False)
    print(f"训练数据已保存至: {train_file}")

    # 保存测试数据
    test_data = []
    for i, (sequence, label) in enumerate(test_sequences):
        sequence_df = pd.DataFrame(sequence, columns=FEATURE_COLS)
        sequence_df["sample_id"] = f"test_{i:06d}"
        sequence_df["label"] = label
        sequence_df["sequence_index"] = range(len(sequence))
        test_data.append(sequence_df)

    test_df = pd.concat(test_data, ignore_index=True)
    test_file = os.path.join(output_dir, "test_raw_data.csv")
    test_df.to_csv(test_file, index=False)
    print(f"测试数据已保存至: {test_file}")

    # 保存数据统计信息
    stats = {
        "train_samples": len(train_sequences),
        "test_samples": len(test_sequences),
        "total_samples": len(train_sequences) + len(test_sequences),
        "train_positive_ratio": sum([label for _, label in train_sequences])
        / len(train_sequences),
        "test_positive_ratio": sum([label for _, label in test_sequences])
        / len(test_sequences),
        "input_window": INPUT_WINDOW,
        "output_window": OUTPUT_WINDOW,
        "feature_columns": FEATURE_COLS,
        "future_return_threshold": FUTURE_RETURN_THRESHOLD,
    }

    stats_file = os.path.join(output_dir, "data_statistics.json")
    import json

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"数据统计信息已保存至: {stats_file}")


def analyze_raw_data(train_sequences, test_sequences):
    """
    分析原始数据的特征分布
    """
    print("\n=== 原始数据特征分析 ===")

    # 合并所有序列数据
    all_sequences = train_sequences + test_sequences
    all_features = []

    for sequence, _ in all_sequences:
        all_features.append(sequence)

    # 转换为numpy数组
    all_features = np.array(all_features)

    # 计算每个特征的统计信息
    for i, feature_name in enumerate(FEATURE_COLS):
        feature_data = all_features[:, :, i].flatten()

        print(f"\n{feature_name}:")
        print(f"  均值: {np.mean(feature_data):.4f}")
        print(f"  标准差: {np.std(feature_data):.4f}")
        print(f"  最小值: {np.min(feature_data):.4f}")
        print(f"  最大值: {np.max(feature_data):.4f}")
        print(f"  中位数: {np.median(feature_data):.4f}")


def main():
    """
    主函数
    """
    try:
        # 提取原始数据
        train_sequences, test_sequences = extract_raw_labeled_data()

        # 分析数据
        analyze_raw_data(train_sequences, test_sequences)

        print("\n=== 数据提取完成 ===")
        print("原始数据已保存到 'raw_data' 目录中")
        print("文件包括：")
        print("  - train_raw_data.csv: 训练集原始数据")
        print("  - test_raw_data.csv: 测试集原始数据")
        print("  - data_statistics.json: 数据统计信息")

    except Exception as e:
        print(f"数据提取失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
