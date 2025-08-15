import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)  # 新增
from config import MIN_IMPROVEMENT


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    MIN_IMPROVEMENT,
):  # 新增MIN_IMPROVEMENT参数
    check_data_distribution(train_loader)  # 调用数据分布检查函数
    best_f1 = -1.0
    early_stop_counter = 0
    best_model_weights = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_loader.dataset)  # 计算平均训练损失

        # 验证逻辑（保持原有代码不变）
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

        # 计算指标（保持原有代码不变）
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            print(f"警告：混淆矩阵异常 - 形状: {cm.shape}, 值: {cm}")
            false_positive_rate = 0

        # tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        # false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"训练损失: {train_loss:.4f} | 测试损失: {test_loss:.4f}")

        # 早停机制（改进保存逻辑）
        if f1 > best_f1 + MIN_IMPROVEMENT:
            best_f1 = f1
            best_model_weights = model.state_dict().copy()  # 保存最佳权重
            early_stop_counter = 0
            print(f"Epoch {epoch+1}: 保存最佳模型 (F1: {f1:.4f})")
        else:
            early_stop_counter += 1
            print(f"Epoch {epoch+1}: 未改进 ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"早停触发：连续{patience}轮无显著改善")
                break

    # 确保使用最佳模型权重（如果有）
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"使用最佳模型权重 (F1: {best_f1:.4f})")

    return model, {
        "f1": best_f1,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
    }


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=["下跌", "上涨"]))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(cm)

    # 计算AUC
    try:
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(
                all_labels,
                [p[1] for p in F.softmax(torch.tensor(all_preds), dim=1).numpy()],
            )
            print(f"\nAUC: {auc:.4f}")
    except:
        print("\n无法计算AUC")

    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n准确率: {accuracy:.4f}")

    return accuracy


def check_data_distribution(data_loader):
    """检查数据集中正负样本的分布"""
    total = 0
    positive = 0

    for _, labels in data_loader:
        total += len(labels)
        positive += labels.sum().item()

    negative = total - positive
    print(
        f"数据分布: 正样本 = {positive} ({positive/total*100:.2f}%), 负样本 = {negative} ({negative/total*100:.2f}%)"
    )

    if positive == 0 or negative == 0:
        raise ValueError("数据集中只有单一类别，无法训练分类模型")
