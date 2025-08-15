import torch
import torch.nn as nn
import torch.optim as optim

from feature_analyzer import FeatureCombinationAnalyzer
from torch.utils.data import DataLoader, Dataset  # 替换TensorDataset为自定义Dataset
import numpy as np
import json  # 新增：用于处理JSON格式

# import chardet
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from data_loader import KeyFeatureDataLoader
from model import KeyFeatureModel
from trainer import check_data_distribution

# 从config导入所需配置（确保与实际config.py中的变量名一致）
from config import (
    FEATURE_COLS,
    INPUT_WINDOW,
    OUTPUT_WINDOW,
    INITIAL_START_DATE,
    INITIAL_END_DATE,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_CLASSES,
    KEY_FEATURE_IDX,
    FEATURE_WEIGHT_MODE,
    MODEL_PATH,
    METRICS_PATH,
    EPOCHS,
    LEARNING_RATE,
    FUTURE_RETURN_THRESHOLD,
    PATIENCE,
    MIN_IMPROVEMENT,
    MIN_SUPPORT,
    MAX_COMBINATION_LEN,
    NEG_THRESHOLD,
)


# 复用第一版中的SequenceDataset类，处理序列数据
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = (
            sequences  #  sequences格式为[(序列1, 标签1), (序列2, 标签2), ...]
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]  # 每个元素是(输入序列, 标签)
        return torch.FloatTensor(x), torch.tensor(y, dtype=torch.long)  # 转换为tensor


def train_model(
    model, train_loader, test_loader, criterion, optimizer, device, epochs, patience
):
    # 保持原train_model逻辑不变（早停、指标计算等）
    check_data_distribution(train_loader)  # 需要实现这个函数
    best_f1 = -1.0
    early_stop_counter = 0
    best_model_weights = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0  # 每轮训练的损失

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * sequences.size(0)  # 累加批次损失

        train_loss /= len(train_loader.dataset)  # 计算平均训练损失

        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device).long()
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * sequences.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_loss /= len(test_loader.dataset)

        # 计算指标（改进版）
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        # 安全计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        if cm.size == 4:  # 确保混淆矩阵有4个元素
            tn, fp, fn, tp = cm.ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            print(f"警告：混淆矩阵异常 - 形状: {cm.shape}, 值: {cm}")
            false_positive_rate = 0  # 默认为0

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f}")
        print(f"精确率: {precision:.4f} | 召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f} | 假阳性率: {false_positive_rate:.4f}\n")

        # 早停机制（改进版）
        if f1 > best_f1 + (MIN_IMPROVEMENT or 0.0):  # 添加默认值
            best_f1 = f1
            best_model_weights = model.state_dict().copy()
            early_stop_counter = 0
            print(f"Epoch {epoch+1}: 保存最佳模型 (F1: {f1:.4f})")
        else:
            early_stop_counter += 1
            print(f"Epoch {epoch+1}: 未改进 ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"早停触发：连续{patience}轮无显著改善")
                break

    # 确保使用最佳模型权重
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"使用最佳模型权重 (F1: {best_f1:.4f})")
    else:
        print("警告：未找到最佳模型权重，使用最后一轮训练的模型")

    # 在返回前添加训练总结
    print("\n==== 训练总结 ====")
    print(f"最佳F1分数: {best_f1:.4f}")
    print(f"最终召回率: {recall:.4f}")
    print(f"最终假阳性率: {false_positive_rate:.4f}")
    return model, {
        "f1": best_f1,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
    }


# 替代chardet的简化编码检测（不依赖第三方库）
def detect_encoding(file_path):
    try:
        # 尝试常见编码
        for encoding in ["utf-8", "gbk", "latin-1"]:
            with open(file_path, "r", encoding=encoding) as f:
                try:
                    f.read(10000)  # 读取前10KB测试
                    return encoding
                except UnicodeDecodeError:
                    continue
        return "utf-8"  # 默认使用UTF-8
    except Exception:
        return "utf-8"


def check_existing_stock_data():
    """检查是否已存在股票数据，避免重复下载"""
    stock_data_dir = "stock_data"
    if os.path.exists(stock_data_dir):
        # 检查是否有.joblib文件
        joblib_files = [f for f in os.listdir(stock_data_dir) if f.endswith(".joblib")]
        if joblib_files:
            print(f"发现现有股票数据文件: {len(joblib_files)} 个")
            return True
    return False


def get_user_choice():
    """获取用户选择"""
    print("\n" + "=" * 60)
    print("股票特征分析与预测系统")
    print("=" * 60)
    print("请选择要执行的操作:")
    print("1. 完整流程：数据加载 + 模型训练 + 特征分析")
    print("2. 仅特征筛选：跳过模型训练，直接进行特征分析")
    print("3. 仅模型训练：跳过特征分析，直接训练模型")
    print("4. 退出程序")

    while True:
        try:
            choice = input("\n请输入选择 (1-4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                return int(choice)
            else:
                print("无效选择，请输入1-4之间的数字")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            return 4
        except Exception as e:
            print(f"输入错误: {e}")


def run_feature_analysis_only(data_loader):
    """仅运行特征分析部分"""
    print("\n开始特征分析...")

    # 准备原始数据用于特征分析
    train_sequences = []
    test_sequences = []
    try:
        train_sequences, test_sequences = data_loader.prepare_raw_data_for_training(
            input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, train_test_split=0.8
        )

        # 检查是否有足够的数据进行特征分析
        if not train_sequences or not test_sequences:
            print("警告：没有足够的数据进行特征分析")
            return

        # 提取正样本和负样本数据
        train_pos_data = []
        train_neg_data = []
        for seq, label in train_sequences:
            if label == 1:
                train_pos_data.append(seq)
            else:
                train_neg_data.append(seq)

        test_pos_data = []
        test_neg_data = []
        for seq, label in test_sequences:
            if label == 1:
                test_pos_data.append(seq)
            else:
                test_neg_data.append(seq)

        # 检查是否有足够的正样本和负样本
        if len(train_pos_data) == 0 or len(train_neg_data) == 0:
            print("警告：正样本或负样本数量不足，无法进行特征分析")
            return

        print(f"正样本数量: {len(train_pos_data)}, 负样本数量: {len(train_neg_data)}")

        # 使用config中的配置初始化分析器
        analyzer = FeatureCombinationAnalyzer()

        # 多轮迭代优化
        best_score = -1
        best_patterns = []
        for epoch in range(3):  # 最多迭代3轮
            try:
                # 1. 训练集归纳+过滤
                print(f"\n迭代{epoch+1}：开始分析正样本特征...")
                analyzer.analyze_positive_samples(train_pos_data)

                if not analyzer.positive_patterns:
                    print(f"迭代{epoch+1}：未找到符合条件的特征组合，跳过")
                    continue

                print(f"迭代{epoch+1}：找到{len(analyzer.positive_patterns)}个特征组合")

                current_patterns = analyzer.validate_with_negative_samples(
                    train_neg_data
                )

                if not current_patterns:
                    print(f"迭代{epoch+1}：经过负样本过滤后无有效特征组合，跳过")
                    continue

                print(
                    f"迭代{epoch+1}：经过负样本过滤后剩余{len(current_patterns)}个特征组合"
                )

                # 2. 测试集验证
                coverage, false_rate = (
                    FeatureCombinationAnalyzer.evaluate_patterns_on_test(
                        current_patterns, test_pos_data, test_neg_data
                    )
                )
                print(
                    f"迭代{epoch+1}：测试集覆盖率={coverage:.2f}，误判率={false_rate:.2f}"
                )

                # 3. 保存最优组合
                current_score = coverage - false_rate  # 综合评分
                if current_score > best_score:
                    best_score = current_score
                    best_patterns = current_patterns
                    analyzer.positive_patterns = best_patterns
                    analyzer.save_patterns()  # 保存本轮最优组合
                    print(f"迭代{epoch+1}：发现更好的特征组合，已保存")

            except Exception as e:
                print(f"迭代{epoch+1}失败: {e}")
                continue

        if best_patterns:
            print(f"\n最优特征组合数量: {len(best_patterns)}")
            for i, pattern in enumerate(best_patterns[:5]):  # 只显示前5个
                semantic_content = analyzer._display_feature_combination_semantics(
                    pattern["features"]
                )
                print(f"  组合{i+1}: {semantic_content}")
                print(f"    支持度: {pattern['support']:.3f}")
        else:
            print("\n未找到有效的特征组合")

    except Exception as e:
        print(f"特征分析失败: {e}")
        import traceback

        traceback.print_exc()


def run_model_training_only(data_loader):
    """仅运行模型训练部分"""
    print("\n开始模型训练...")

    # 准备训练数据
    train_sequences, test_sequences = data_loader.prepare_sector_data_for_training(
        input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, train_test_split=0.8
    )

    # 检查数据有效性
    if not train_sequences or not test_sequences:
        raise ValueError("未能生成有效训练或测试样本")

    # 使用第一版的SequenceDataset处理数据
    train_dataset = SequenceDataset(train_sequences)
    test_dataset = SequenceDataset(test_sequences)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KeyFeatureModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        key_feature_idx=KEY_FEATURE_IDX,
        weight_mode=FEATURE_WEIGHT_MODE,
    ).to(device)

    # 加载历史模型
    load_success = False
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            if model.input_size != INPUT_SIZE:
                raise ValueError(
                    f"模型输入特征数不匹配（文件：{model.input_size}，当前：{INPUT_SIZE}）"
                )
            print(f"成功加载历史模型（输入特征数：{INPUT_SIZE}），将继续训练")
            load_success = True
        except Exception as e:
            print(f"加载历史模型失败：{e}，将从头训练新模型")

    # 定义优化器和损失函数
    pos_samples = sum([1 for _, label in train_sequences if label == 1])
    neg_samples = len(train_sequences) - pos_samples

    if pos_samples == 0 or neg_samples == 0:
        class_weights = torch.tensor([1.0, 1.0], device=device)
    else:
        total_samples = len(train_sequences)
        class_weights = torch.tensor(
            [total_samples / neg_samples, total_samples / pos_samples], device=device
        )

    print(f"修正后的类别权重: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练模型
    print(f"开始训练（输入特征数：{INPUT_SIZE}，最大轮次：{EPOCHS}）")
    trained_model, new_metrics = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=EPOCHS,
        patience=PATIENCE,
    )

    # 比较新模型与历史模型的性能，决定是否保存
    should_save_model = True
    if load_success and os.path.exists(METRICS_PATH):
        try:
            # 加载历史指标
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                history_metrics = json.load(f)

            print(f"\n模型性能对比:")
            print(f"历史最佳F1分数: {history_metrics['f1']:.4f}")
            print(f"新训练F1分数: {new_metrics['f1']:.4f}")

            # 检查新模型是否显著优于历史模型
            if new_metrics["f1"] > history_metrics["f1"] + MIN_IMPROVEMENT:
                print(
                    f"✓ 新模型性能显著提升 (F1提升: {new_metrics['f1'] - history_metrics['f1']:.4f})，将保存新模型"
                )
                should_save_model = True
            else:
                print(
                    f"✗ 新模型性能未显著提升 (F1提升: {new_metrics['f1'] - history_metrics['f1']:.4f} < {MIN_IMPROVEMENT})，保留历史模型"
                )
                should_save_model = False

        except Exception as e:
            print(f"加载历史指标失败: {e}，将保存新模型")
            should_save_model = True
    else:
        print("首次训练或未找到历史指标，将保存新模型")
        should_save_model = True

    # 根据性能比较结果决定是否保存模型
    if should_save_model:
        torch.save(trained_model.state_dict(), MODEL_PATH)
        print(f"✓ 新模型已保存到: {MODEL_PATH}")

        # 保存训练指标
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(new_metrics, f, ensure_ascii=False, indent=2)
        print(f"✓ 训练指标已保存到: {METRICS_PATH}")
    else:
        print("✗ 新模型未保存，保留历史最优模型")
        # 恢复历史模型状态
        if load_success:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("已恢复历史最优模型状态")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查是否已有股票数据
    has_existing_data = check_existing_stock_data()
    if has_existing_data:
        print("检测到现有股票数据，将跳过数据下载步骤")

    # 获取用户选择
    user_choice = get_user_choice()

    if user_choice == 4:
        print("程序退出")
        return

    # 初始化数据加载器
    data_loader = KeyFeatureDataLoader(
        data_path="",
        feature_cols=FEATURE_COLS,
        key_feature_name="future_return",
        start_date=INITIAL_START_DATE,
        end_date=INITIAL_END_DATE,
        label_col="target",
        use_baostock=True,
        stocks_file="data_list/hs300_stocks.csv",
        tushare_token="8b1ef90e2f704b9d90e09a0de94078ff5ae6c5c18cc3382e75b879b7",
    )
    data_loader.requires_tushare = True

    # 只有在没有现有数据时才进行数据下载
    if not has_existing_data:
        print("开始下载股票数据...")
        data_loader.tushare_login()
    else:
        print("使用现有股票数据，跳过下载步骤")

    # 根据用户选择执行相应操作
    if user_choice == 1:  # 完整流程
        print("\n执行完整流程：数据加载 + 模型训练 + 特征分析")

        # 准备训练数据
        train_sequences, test_sequences = data_loader.prepare_sector_data_for_training(
            input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW, train_test_split=0.8
        )

        # 检查数据有效性
        if not train_sequences or not test_sequences:
            raise ValueError("未能生成有效训练或测试样本")

        # 使用第一版的SequenceDataset处理数据
        train_dataset = SequenceDataset(train_sequences)
        test_dataset = SequenceDataset(test_sequences)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型
        model = KeyFeatureModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_classes=NUM_CLASSES,
            key_feature_idx=KEY_FEATURE_IDX,
            weight_mode=FEATURE_WEIGHT_MODE,
        ).to(device)

        # 加载历史模型
        load_success = False
        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                if model.input_size != INPUT_SIZE:
                    raise ValueError(
                        f"模型输入特征数不匹配（文件：{model.input_size}，当前：{INPUT_SIZE}）"
                    )
                print(f"成功加载历史模型（输入特征数：{INPUT_SIZE}），将继续训练")
                load_success = True
            except Exception as e:
                print(f"加载历史模型失败：{e}，将从头训练新模型")

        # 定义优化器和损失函数
        pos_samples = sum([1 for _, label in train_sequences if label == 1])
        neg_samples = len(train_sequences) - pos_samples

        if pos_samples == 0 or neg_samples == 0:
            class_weights = torch.tensor([1.0, 1.0], device=device)
        else:
            total_samples = len(train_sequences)
            class_weights = torch.tensor(
                [total_samples / neg_samples, total_samples / pos_samples],
                device=device,
            )

        print(f"修正后的类别权重: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 训练模型
        print(f"开始训练（输入特征数：{INPUT_SIZE}，最大轮次：{EPOCHS}）")
        trained_model, new_metrics = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            epochs=EPOCHS,
            patience=PATIENCE,
        )

        # 比较新模型与历史模型的性能，决定是否保存
        should_save_model = True
        if load_success and os.path.exists(METRICS_PATH):
            try:
                # 加载历史指标
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    history_metrics = json.load(f)

                print(f"\n模型性能对比:")
                print(f"历史最佳F1分数: {history_metrics['f1']:.4f}")
                print(f"新训练F1分数: {new_metrics['f1']:.4f}")

                # 检查新模型是否显著优于历史模型
                if new_metrics["f1"] > history_metrics["f1"] + MIN_IMPROVEMENT:
                    print(
                        f"✓ 新模型性能显著提升 (F1提升: {new_metrics['f1'] - history_metrics['f1']:.4f})，将保存新模型"
                    )
                    should_save_model = True
                else:
                    print(
                        f"✗ 新模型性能未显著提升 (F1提升: {new_metrics['f1'] - history_metrics['f1']:.4f} < {MIN_IMPROVEMENT})，保留历史模型"
                    )
                    should_save_model = False

            except Exception as e:
                print(f"加载历史指标失败: {e}，将保存新模型")
                should_save_model = True
        else:
            print("首次训练或未找到历史指标，将保存新模型")
            should_save_model = True

        # 根据性能比较结果决定是否保存模型
        if should_save_model:
            torch.save(trained_model.state_dict(), MODEL_PATH)
            print(f"✓ 新模型已保存到: {MODEL_PATH}")

            # 保存训练指标
            with open(METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(new_metrics, f, ensure_ascii=False, indent=2)
            print(f"✓ 训练指标已保存到: {METRICS_PATH}")
        else:
            print("✗ 新模型未保存，保留历史最优模型")
            # 恢复历史模型状态
            if load_success:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print("已恢复历史最优模型状态")

        # 进行特征分析
        run_feature_analysis_only(data_loader)

    elif user_choice == 2:  # 仅特征筛选
        print("\n执行特征筛选模式：跳过模型训练，直接进行特征分析")
        run_feature_analysis_only(data_loader)

    elif user_choice == 3:  # 仅模型训练
        print("\n执行模型训练模式：跳过特征分析，直接训练模型")
        run_model_training_only(data_loader)

    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
