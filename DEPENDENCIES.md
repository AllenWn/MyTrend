# 项目依赖库详细说明

## 核心依赖库

### 1. 深度学习框架
- **torch>=1.8.0**: PyTorch深度学习框架，用于构建和训练神经网络
- **torchvision>=0.9.0**: PyTorch计算机视觉工具包，提供额外的模型和工具

### 2. 数据处理和分析
- **pandas>=1.3.0**: 强大的数据分析库，用于处理股票时间序列数据
- **numpy>=1.21.0**: 数值计算库，提供高效的数组操作和数学函数

### 3. 机器学习工具
- **scikit-learn>=1.0.0**: 机器学习库，提供数据预处理、模型评估等工具
  - `RobustScaler`: 用于数据标准化，对异常值更稳健
  - `MinMaxScaler`: 用于数据归一化
  - 各种评估指标：precision_score, recall_score, f1_score等

### 4. 股票数据API
- **tushare>=1.2.0**: 专业的金融数据接口，用于获取A股实时数据
- **baostock>=0.7.0**: 免费的证券数据平台，用于获取交易日历

### 5. 进度条和工具
- **tqdm>=4.62.0**: 进度条库，用于显示长时间运行任务的进度

### 6. 数据缓存和序列化
- **joblib>=1.1.0**: 用于保存和加载大型数据对象，如股票数据和模型

### 7. 日期时间处理
- **python-dateutil>=2.8.0**: 日期时间处理工具，增强datetime功能

## 标准库依赖

### 内置模块
- **os**: 操作系统接口，用于文件路径操作
- **time**: 时间相关函数，用于API请求延迟
- **json**: JSON数据处理，用于保存配置和结果
- **itertools**: 迭代器工具，用于特征组合生成
- **datetime**: 日期时间处理
- **torch.utils.data**: PyTorch数据加载工具

## 安装说明

### 基础安装
```bash
pip install -r requirements.txt
```

### GPU支持安装（可选）
如果需要GPU加速，安装CUDA版本的PyTorch：
```bash
# CUDA 11.8版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 开发环境安装（可选）
```bash
pip install jupyter matplotlib seaborn
```

## 版本兼容性

### Python版本
- **推荐**: Python 3.9+
- **最低**: Python 3.8
- **最高**: Python 3.11

### 操作系统
- **Windows**: 10/11
- **macOS**: 10.15+ (支持Apple Silicon)
- **Linux**: Ubuntu 18.04+, CentOS 7+

## 依赖库用途说明

### 在项目中的具体使用

1. **torch**: 
   - 构建KeyFeatureModel神经网络
   - 训练和推理
   - 张量操作和GPU加速

2. **pandas**: 
   - 处理股票OHLCV数据
   - 时间序列操作
   - 数据清洗和转换

3. **numpy**: 
   - 数值计算
   - 数组操作
   - 随机数生成

4. **scikit-learn**: 
   - 数据标准化
   - 模型性能评估
   - 特征工程

5. **tushare**: 
   - 获取实时股票数据
   - 历史行情数据
   - 基本面数据

6. **baostock**: 
   - 交易日历管理
   - 市场状态查询

7. **joblib**: 
   - 缓存股票数据
   - 保存训练好的模型
   - 数据持久化

## 常见问题解决

### 1. PyTorch安装失败
```bash
# 尝试使用清华源
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. Tushare权限问题
- 需要注册Tushare账号获取token
- 免费账号有API调用次数限制

### 3. 内存不足
- 减少batch_size
- 使用数据生成器而不是一次性加载所有数据

### 4. GPU相关问题
- 确保CUDA版本与PyTorch版本匹配
- 检查GPU驱动是否正确安装
