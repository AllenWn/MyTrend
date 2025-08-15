import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INITIAL_KEY_WEIGHT  # 导入初始权重配置

# model.py v2.0
# 核心更新：支持动态特征权重学习，自动发现与未来涨幅相关的特征


class KeyFeatureModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        key_feature_idx,
        weight_mode="dynamic",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.key_feature_idx = key_feature_idx  # 初始关注特征索引（静态模式用）
        self.weight_mode = weight_mode

        # 动态权重：让模型自主学习每个特征的重要性
        if weight_mode == "dynamic":
            self.feature_weights = nn.Parameter(
                torch.ones(input_size)
            )  # 可学习的特征权重
        else:
            self.key_weight = INITIAL_KEY_WEIGHT  # 静态权重（从config导入）

        # LSTM层（处理序列特征）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        # 注意力机制（聚焦重要时间步）
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1), nn.Tanh()  # 双向LSTM输出维度为2*hidden_size
        )

        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 特征权重应用
        if self.weight_mode == "dynamic":
            # 动态模式：对每个特征应用可学习的权重
            weighted_x = x * self.feature_weights.unsqueeze(0).unsqueeze(
                0
            )  # 广播到(batch, seq_len, input_size)
            lstm_out, _ = self.lstm(weighted_x)
        else:
            # 静态模式：仅增强初始关注的特征
            lstm_out, _ = self.lstm(x)
            key_feat = x[:, :, self.key_feature_idx].unsqueeze(2)  # 提取初始关注特征
            key_enhanced = nn.Linear(1, self.hidden_size)(key_feat)  # 增强特征维度
            lstm_out += key_enhanced * self.key_weight  # 放大关键特征影响

        # 限制数值范围，避免溢出
        lstm_out = torch.clamp(lstm_out, -1e4, 1e4)

        # 注意力机制：聚焦对未来涨幅重要的时间步
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # 时间步权重归一化
        weighted_sum = torch.bmm(attn_weights.transpose(1, 2), lstm_out).squeeze(
            1
        )  # (batch, 2*hidden_size)

        # 输出分类结果
        return self.fc(weighted_sum)
