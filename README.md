# 股票特征分析与预测系统 (Mytrend)

## 项目概述

这是一个基于深度学习的股票特征分析与预测系统，主要功能是分析A股股票的技术特征，预测未来5天内是否会出现15%以上的涨幅。系统采用LSTM+注意力机制的深度学习模型，结合16个专业的技术分析特征，实现智能化的股票筛选。

## 项目结构

```
Mytrend/
├── config.py                 # 配置文件
├── main.py                   # 主程序入口
├── data_loader.py            # 数据加载与预处理模块
├── model.py                  # 深度学习模型定义
├── features.py               # 技术特征计算模块
├── feature_analyzer.py       # 特征组合分析器
├── trainer.py                # 模型训练器
├── predict.py                # 股票预测模块
├── trading_calendar.py       # 交易日历管理
├── extract_raw_data.py       # 原始数据提取工具
├── data_list/                # 数据目录
│   ├── hs300_stocks.csv      # 沪深300成分股列表
│   └── hs300_stocks_predict.csv  # 预测用成分股列表
├── stock_data/               # 股票数据目录
│   ├── scalers/              # 数据标准化器
│   └── *.joblib              # 股票数据缓存文件
├── models/                   # 训练好的模型文件
├── trading_calendar.csv      # 交易日历文件
├── metrics/                  # 模型评估指标和特征分析结果
└── README.md                 # 项目说明文档
```

## 文件详细说明

### 1. config.py - 配置文件
**作用**: 集中管理所有配置参数，包括数据配置、模型配置、训练配置等。

**主要配置项**:
- **数据配置**: 特征列定义、时间窗口设置、数据日期范围
- **模型配置**: 网络结构参数、特征权重模式、类别数量
- **训练配置**: 训练轮次、学习率、批次大小、早停参数
- **路径配置**: 缓存目录、模型保存路径、指标保存路径

### 2. main.py - 主程序入口
**作用**: 系统的主控制流程，协调各个模块的执行。

**主要方法**:
- `main()`: 主程序入口，控制整个系统流程
- `get_user_choice()`: 获取用户选择的操作模式
- `run_feature_analysis_only()`: 仅执行特征分析
- `run_model_training_only()`: 仅执行模型训练
- `train_model()`: 模型训练的核心逻辑
- `check_existing_stock_data()`: 检查现有数据缓存

**用户选择模式**:
1. 完整流程：数据加载 + 模型训练 + 特征分析
2. 仅特征筛选：跳过模型训练，直接进行特征分析
3. 仅模型训练：跳过特征分析，直接训练模型
4. 退出程序

### 3. data_loader.py - 数据加载与预处理模块
**作用**: 负责股票数据的获取、预处理、序列生成和缓存管理。

**主要方法**:
- `__init__()`: 初始化数据加载器，设置参数和目录
- `tushare_login()/tushare_logout()`: Tushare API登录登出
- `get_stock_data()`: 获取单只股票的历史数据
- `_add_technical_indicators()`: 计算技术指标（MA、RSI、布林带等）
- `preprocess_data()`: 数据标准化和异常值处理
- `create_sequences()`: 生成训练序列（20天输入+5天标签）
- `prepare_sector_data_for_training()`: 准备板块训练数据
- `load_or_append_data()`: 加载或追加股票数据
- `validate_stock_data()`: 验证数据完整性
- `get_sector_stock_data()`: 获取板块所有成分股数据

**数据流程**:
1. 从Tushare获取原始OHLCV数据
2. 计算12个技术指标特征
3. 数据标准化和异常值处理
4. 生成20天×12特征的训练序列
5. 本地缓存管理

### 4. model.py - 深度学习模型定义
**作用**: 定义专门用于股票预测的深度学习模型架构。

**主要方法**:
- `__init__()`: 初始化模型参数和网络层
- `forward()`: 前向传播，处理输入数据

**模型架构**:
- **特征权重层**: 支持动态权重学习，自动发现重要特征
- **LSTM层**: 双向2层LSTM，处理时序特征依赖
- **注意力机制**: 聚焦对未来涨幅重要的时间步
- **分类层**: 全连接网络，输出涨跌概率

**特点**:
- 输入：20天×12特征的时序数据
- 输出：二分类概率（上涨/下跌）
- 支持动态特征权重学习
- 集成注意力机制

### 5. features.py - 技术特征计算模块
**作用**: 实现16个专业的技术分析特征函数，用于股票形态识别。

**主要方法**:
- `calculate_technical_indicators()`: 计算基础技术指标
- `get_wave_periods()`: 基于均线划分波段
- `screen_second_wave()`: 组合所有特征，返回筛选结果

**特征函数**:
1. `check_drawdown_controlled()`: 调整幅度可控
2. `check_break_adjustment_high()`: 突破调整区间上沿
3. `check_break_first_wave_high()`: 突破第一波高点
4. `check_volume_pattern()`: 调整期缩量且起爆点放量
5. `check_ma_support()`: 调整期站稳20日均线
6. `check_ma_bullish()`: 均线多头排列
7. `check_ma_golden_cross()`: 5日与10日均线金叉
8. `check_macd_second_red()`: MACD二次翻红
9. `check_rsi_breakout()`: RSI突破60且调整期未超卖
10. `check_bb_upper_break()`: 突破布林带上轨
11. `check_big_bull_candle()`: 起爆点大阳线
12. `check_bull_engulfing()`: 阳包阴形态
13. `is_price_volume_rise()`: 当天价涨量增
14. `is_big_bull_candle()`: 起爆点出现大阳线
15. `is_break_flag_pattern()`: 突破旗形整理
16. `check_ma_convergence()`: 均线汇聚

### 6. feature_analyzer.py - 特征组合分析器
**作用**: 从正样本中挖掘高频特征组合，用负样本验证有效性。

**主要方法**:
- `__init__()`: 初始化分析器参数
- `analyze_positive_samples()`: 从正样本中归纳高频特征组合
- `validate_with_negative_samples()`: 用负样本过滤特征组合
- `save_patterns()/load_patterns()`: 保存和加载特征组合
- `evaluate_patterns_on_test()`: 评估特征组合在测试集上的表现

**分析流程**:
1. 提取所有正样本的特征
2. 统计不同长度特征组合的频率
3. 筛选超过最小支持度的组合
4. 用负样本验证组合有效性
5. 多轮迭代优化特征组合质量

### 7. trainer.py - 模型训练器
**作用**: 负责深度学习模型的训练、验证和性能评估。

**主要方法**:
- `train_model()`: 执行模型训练，包括早停机制
- `evaluate_model()`: 评估模型性能
- `check_data_distribution()`: 检查数据集中正负样本分布

**训练特性**:
- 支持早停机制，防止过拟合
- 梯度裁剪，稳定训练过程
- 类别权重平衡，处理样本不平衡
- 性能指标监控（精确率、召回率、F1分数）

### 8. predict.py - 股票预测模块
**作用**: 使用训练好的模型对股票进行实时预测和筛选。

**主要方法**:
- `__init__()`: 初始化预测器，加载模型和数据
- `_load_model()`: 加载预训练模型
- `preprocess_stock_data()`: 预处理股票数据
- `predict_stock()`: 预测单只股票
- `find_qualified_stocks()`: 寻找符合条件的股票
- `_calculate_feature_matches()`: 计算特征匹配数量
- `save_prediction_results()`: 保存预测结果

**预测流程**:
1. 加载预训练模型和特征组合
2. 获取目标股票的实时数据
3. 数据预处理和技术指标计算
4. 模型预测上涨概率
5. 特征匹配度计算
6. 结果排序和保存

### 9. trading_calendar.py - 交易日历管理
**作用**: 管理A股交易日历，提供日期计算和验证功能。

**主要方法**:
- `__init__()`: 初始化交易日历
- `update()`: 更新交易日历数据
- `is_trading_day()`: 判断指定日期是否为交易日
- `get_next_trading_day()`: 获取下一个交易日
- `get_previous_trading_day()`: 获取上一个交易日
- `get_trading_days()`: 获取指定日期范围内的交易日

**数据源**: 使用Baostock API获取交易日历数据

### 10. extract_raw_data.py - 原始数据提取工具
**作用**: 提取标准化之前的原始标签数据，用于数据分析和调试。

**主要方法**:
- `extract_raw_labeled_data()`: 提取原始标签数据
- `save_raw_data()`: 保存原始数据到文件
- `analyze_raw_data()`: 分析原始数据的特征分布
- `main()`: 主函数，执行数据提取流程

## 完整运行流程

### 1. 系统启动阶段
```
main.py:main() 
├── 设备检测 (GPU/CPU)
├── 数据缓存检查 check_existing_stock_data()
└── 用户模式选择 get_user_choice()
```

### 2. 数据加载阶段
```
data_loader.py:KeyFeatureDataLoader.__init__()
├── 初始化配置参数
├── 创建缓存目录
└── 初始化交易日历 TradingCalendar

data_loader.py:tushare_login()
├── 设置Tushare token
└── 创建API连接

data_loader.py:prepare_sector_data_for_training()
├── 获取沪深300成分股列表 get_sector_stocks()
├── 下载股票数据 get_sector_stock_data()
├── 计算技术指标 _add_technical_indicators()
├── 数据预处理 preprocess_data()
└── 生成训练序列 create_sequences()
```

### 3. 模型训练阶段
```
model.py:KeyFeatureModel.__init__()
├── 初始化网络层（LSTM、注意力、分类层）
└── 设置特征权重模式

trainer.py:train_model()
├── 数据分布检查 check_data_distribution()
├── 训练循环（前向传播、损失计算、反向传播）
├── 验证评估（计算精确率、召回率、F1分数）
├── 早停机制（连续无改善则停止）
└── 梯度裁剪（防止梯度爆炸）

main.py:模型性能比较
├── 加载历史指标
├── 比较F1分数提升
└── 决定是否保存新模型
```

### 4. 特征分析阶段
```
feature_analyzer.py:FeatureCombinationAnalyzer.__init__()
├── 设置分析参数（最小支持度、最大组合长度等）

feature_analyzer.py:analyze_positive_samples()
├── 提取正样本特征 _extract_features()
├── 统计特征组合频率
└── 筛选高频组合

feature_analyzer.py:validate_with_negative_samples()
├── 提取负样本特征
├── 验证组合在负样本中的出现频率
└── 过滤无效组合

features.py:特征提取
├── calculate_technical_indicators() - 计算技术指标
├── get_wave_periods() - 识别波段
└── 16个特征函数 - 形态识别
```

### 5. 预测应用阶段
```
predict.py:StockPredictor.__init__()
├── 加载预训练模型 _load_model()
├── 初始化数据加载器
└── 加载特征组合

predict.py:find_qualified_stocks()
├── 获取目标日期
├── 加载股票数据 _load_and_format_data()
├── 数据预处理 preprocess_stock_data()
├── 模型预测 predict_stock()
├── 特征匹配计算 _calculate_feature_matches()
└── 结果排序和保存
```

### 6. 数据流转过程
```
原始OHLCV数据 → 技术指标计算 → 数据标准化 → 序列生成 → 模型训练 → 特征分析 → 模型预测 → 结果输出
```

### 7. 关键配置参数
- **输入窗口**: 20天历史数据
- **预测窗口**: 未来5天
- **涨幅阈值**: 15%
- **特征数量**: 12个技术指标
- **模型结构**: 64维隐藏层，2层双向LSTM
- **训练参数**: 10轮训练，学习率0.001，批次大小32

## 使用方法

### 环境要求
- Python 3.9+
- PyTorch 1.8+
- 其他依赖见requirements.txt

### 运行步骤
1. 配置Tushare API token
2. 运行主程序：`python main.py`
3. 选择执行模式
4. 等待训练和分析完成
5. 查看预测结果

### 输出结果
- 训练好的模型文件
- 特征组合分析结果
- 股票预测筛选结果
- 训练性能指标

## 技术特点

1. **专业性**: 融合传统技术分析与现代深度学习
2. **实用性**: 直接输出可操作的股票筛选结果
3. **可扩展性**: 模块化设计，易于添加新特征和模型
4. **稳定性**: 完善的异常处理和数据验证机制
5. **智能化**: 自动学习特征权重，适应市场变化

## 注意事项

1. 需要有效的Tushare API token
2. 首次运行需要下载大量股票数据
3. 模型训练需要一定的计算资源
4. 预测结果仅供参考，不构成投资建议
