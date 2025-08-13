from data_loader import KeyFeatureDataLoader
from model import KeyFeatureModel
from trading_calendar import TradingCalendar
import torch
import pandas as pd
import numpy as np
import tushare as ts
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.serialization import add_safe_globals
from sklearn.preprocessing import MinMaxScaler
from config import MODEL_PATH, FEATURE_COLS, INPUT_WINDOW, DERIVED_FEATURE_DAYS, KEY_FEATURE_IDX  # 先导入配置


class StockPredictor:
    def __init__(self, 
                 model_path=MODEL_PATH,  # 从config导入动态模型路径
                 stocks_file='data/hs300_stocks_predict.csv',
                 data_dir='stock_data', 
                 cache_dir='stock_data/cache',
                 tushare_token='8b1ef90e2f704b9d90e09a0de94078ff5ae6c5c18cc3382e75b879b7'):
        self.model_path = model_path
        self.feature_cols = FEATURE_COLS  # 特征列也从config导入，确保与训练一致
        self.stocks_file = stocks_file
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.tushare_token = tushare_token
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 特征列（与训练时严格一致）
        self.model = self._load_model()  # 加载模型
        self.calendar = TradingCalendar(os.path.join(self.cache_dir, 'trading_calendar.csv'))
        self.calendar.update()
        
        # 初始化数据加载器
        self.ts_api = self._init_tushare()
        self.data_loader = KeyFeatureDataLoader(
            data_path='',
            feature_cols=self.feature_cols,
            key_feature_name='future_return',
            label_col='target',
            use_baostock=True,
            stocks_file=self.stocks_file,
            tushare_token=self.tushare_token
        )
        self.data_loader.requires_tushare = True
        self.data_loader.tushare_login()
        if self.ts_api:
            self.data_loader.ts_api = self.ts_api
            
    def _init_tushare(self):
        """初始化Tushare API（用于行情数据）"""
        if not self.tushare_token:
            print("警告：未设置Tushare token，无法获取数据")
            return None
        
        try:
            ts.set_token(self.tushare_token)
            api = ts.pro_api()
            # 验证基础接口权限
            try:
                api.daily(ts_code='600000.SH', start_date='20230101', end_date='20230102')
                print("Tushare行情接口验证通过")
            except Exception as e:
                print(f"Tushare权限警告: {e}")
            return api
        except Exception as e:
            print(f"Tushare初始化失败: {e}")
            return None
    
    def _load_model(self):
        """加载模型（处理PyTorch 2.6+安全限制）"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 模型参数（与训练时一致）
        input_size = len(self.feature_cols)
        model = KeyFeatureModel(
            input_size=input_size,
            hidden_size=64,
            num_classes=2,
            #  key_feature_idx=input_size - 1,
            key_feature_idx=KEY_FEATURE_IDX,
            weight_mode="dynamic"  # 指定动态权重模式
            # 不传递 key_weight 参数
        ).to(self.device)
        
        # 解决模型加载的安全限制（允许numpy底层对象）
        import numpy as np
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray._reconstruct, np.dtype])
        # 显式关闭weights_only（信任模型来源时）
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
        model.eval()
        print(f"模型加载成功（{input_size}个输入特征）: {self.model_path}")
        return model

    
    def _load_and_format_data(self, symbol, start_date, end_date):
        """加载并格式化股票数据，确保'date'列存在且不与索引冲突"""
        try:
            # 使用原始的加载和追加逻辑
            stock_data = self.data_loader.load_or_append_data(
                symbol, start_date, end_date, force_download=False
            )
            
            if stock_data.empty:
                return stock_data
            
            # 深拷贝数据，避免修改原始数据
            data = stock_data.copy()
            
            # 确保日期列名为'date'
            if 'trade_date' in data.columns:
                data = data.rename(columns={'trade_date': 'date'})
            
            # 处理索引与列名冲突问题
            if 'date' in data.index.names:
                data = data.reset_index()  # 确保'date'仅作为列存在
            
            # 若日期在索引中且列中没有'date'，转换为列
            if 'date' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                data['date'] = data.index
                data = data.reset_index(drop=True)
            
            # 确保日期格式正确
            data['date'] = pd.to_datetime(data['date'])
            
            # 按日期排序并重置索引
            data = data.sort_values('date').reset_index(drop=True)
            
            return data
        except Exception as e:
            print(f"加载{symbol}数据出错: {e}")
            return pd.DataFrame()   
    
    def _load_and_format_data_OLD(self, symbol, start_date, end_date):
        """加载数据并统一列名（确保'date'列存在）"""
        stock_data = self.data_loader.load_or_append_data(
            symbol, start_date, end_date, force_download=False
        )
        
        if stock_data.empty:
            return stock_data
        
        # 确保日期列名为'date'
        if 'trade_date' in stock_data.columns:
            stock_data = stock_data.rename(columns={'trade_date': 'date'})
        # 若日期在索引中，转换为列
        if 'date' not in stock_data.columns and isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data['date'] = stock_data.index
        
        # 确保日期格式正确
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        return stock_data
    
    def preprocess_stock_data_OLD(self, stock_data, symbol, target_date, input_window=INPUT_WINDOW):
        """预处理数据（含技术指标计算，确保10个特征完整）"""
        if stock_data.empty:
            raise ValueError("股票数据为空")
        
        # 确保'date'列存在
        if 'date' not in stock_data.columns:
            raise ValueError("数据中缺少'date'列（加载时未正确转换）")
        
        stock_data = stock_data.sort_values(by='date')
        
        # 1. 确保基础特征存在
        basic_features = ['open', 'close', 'high', 'low', 'volume']
        missing_basic = [col for col in basic_features if col not in stock_data.columns]
        if missing_basic:
            raise ValueError(f"缺少基础特征: {missing_basic}")
        
        # 2. 计算缺失的技术指标（如果数据中没有）
        # 计算5日均线
        if 'ma5' not in stock_data.columns:
            stock_data['ma5'] = stock_data['close'].rolling(window=5).mean()
        
        # 计算5日平均量（新增）
        if 'vol5' not in stock_data.columns:
            stock_data['vol5'] = stock_data['volume'].rolling(window=5).mean()
        
        # 计算RSI
        if 'rsi' not in stock_data.columns:
            delta = stock_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            stock_data['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算布林带
        if 'bb_upper' not in stock_data.columns or 'bb_lower' not in stock_data.columns:
            stock_data['ma20'] = stock_data['close'].rolling(window=20).mean()
            stock_data['std20'] = stock_data['close'].rolling(window=20).std()
            stock_data['bb_upper'] = stock_data['ma20'] + 2 * stock_data['std20']
            stock_data['bb_lower'] = stock_data['ma20'] - 2 * stock_data['std20']
        
        # 3. 筛选连续交易日
        start_date = stock_data['date'].min().strftime('%Y-%m-%d')
        end_date = stock_data['date'].max().strftime('%Y-%m-%d')
        valid_trading_days = self.calendar.get_trading_days(start_date, end_date)
        valid_trading_days = pd.to_datetime(valid_trading_days)
        
        stock_trading_days = stock_data[stock_data['date'].isin(valid_trading_days)]['date']
        if len(stock_trading_days) < input_window:
            raise ValueError(f"有效交易日不足{input_window}天（实际{len(stock_trading_days)}天）")
        
        # 4. 截取最后input_window个交易日
        latest_days = stock_trading_days.tail(input_window)
        stock_data = stock_data[stock_data['date'].isin(latest_days)]
        
        # 5. 移除计算指标后产生的NaN值
        stock_data = stock_data.dropna(subset=self.feature_cols)
        
        # 6. 校验所有10个特征是否存在
        missing_features = [col for col in self.feature_cols if col not in stock_data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")
        
        # 7. 加载并验证缩放器（与10个特征匹配）
        # 加载该股票的专属scaler
        scaler_dir = os.path.join(self.cache_dir, 'scalers')
        scaler_path = os.path.join(scaler_dir, f'scaler_{symbol}.pth')
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"未找到{symbol}的scaler文件: {scaler_path}")
        
        # 关键修改：允许加载MinMaxScaler类
        from sklearn.preprocessing import MinMaxScaler
        add_safe_globals([MinMaxScaler])
        scaler = torch.load(scaler_path)
        # 检查scaler特征数量是否匹配
        if scaler.n_features_in_ != len(self.feature_cols):
            raise ValueError(f"{symbol}的scaler特征数量不匹配（预期{len(self.feature_cols)}，实际{scaler.n_features_in_}）")
        
        # 用该股票的scaler标准化数据
        normalized_data = scaler.transform(stock_data[self.feature_cols].values)
        return torch.FloatTensor(normalized_data).unsqueeze(0).to(self.device)

    def preprocess_stock_data(self, stock_data, symbol, target_date, input_window=INPUT_WINDOW):
        """预处理数据（支持预训练scaler和临时scaler生成）"""
        if stock_data.empty:
            raise ValueError("股票数据为空")
        
        # 确保'date'列存在
        if 'date' not in stock_data.columns:
            raise ValueError("数据中缺少'date'列")
        
        stock_data = stock_data.sort_values(by='date')
        
        # 1. 检查基础特征
        basic_features = ['open', 'close', 'high', 'low', 'volume']
        missing_basic = [col for col in basic_features if col not in stock_data.columns]
        if missing_basic:
            raise ValueError(f"缺少基础特征: {missing_basic}")
        
        # 2. 计算技术指标（确保特征完整）
        if 'ma5' not in stock_data.columns:
            stock_data['ma5'] = stock_data['close'].rolling(window=5).mean()
        if 'vol5' not in stock_data.columns:
            stock_data['vol5'] = stock_data['volume'].rolling(window=5).mean()
        if 'rsi' not in stock_data.columns:
            delta = stock_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            stock_data['rsi'] = 100 - (100 / (1 + rs))
        if 'bb_upper' not in stock_data.columns or 'bb_lower' not in stock_data.columns:
            stock_data['ma20'] = stock_data['close'].rolling(window=20).mean()
            stock_data['std20'] = stock_data['close'].rolling(window=20).std()
            stock_data['bb_upper'] = stock_data['ma20'] + 2 * stock_data['std20']
            stock_data['bb_lower'] = stock_data['ma20'] - 2 * stock_data['std20']
        
        # 3. 筛选有效交易日
        start_date = stock_data['date'].min().strftime('%Y-%m-%d')
        end_date = stock_data['date'].max().strftime('%Y-%m-%d')
        valid_trading_days = self.calendar.get_trading_days(start_date, end_date)
        valid_trading_days = pd.to_datetime(valid_trading_days)
        
        stock_trading_days = stock_data[stock_data['date'].isin(valid_trading_days)]['date']
        if len(stock_trading_days) < input_window:
            raise ValueError(f"有效交易日不足{input_window}天（实际{len(stock_trading_days)}天）")
        
        # 4. 截取所需窗口数据
        latest_days = stock_trading_days.tail(input_window)
        stock_data = stock_data[stock_data['date'].isin(latest_days)]
        
        # 5. 移除NaN值
        stock_data = stock_data.dropna(subset=self.feature_cols)
        
        # 6. 检查特征完整性
        missing_features = [col for col in self.feature_cols if col not in stock_data.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")
        
        # 7. 加载或生成scaler（核心修改）
        scaler_dir = os.path.join(self.cache_dir, 'scalers')
        scaler_path = os.path.join(scaler_dir, f'scaler_{symbol}.pth')
        
        # 优先加载预训练scaler
        if os.path.exists(scaler_path):
            try:
                # 允许必要的全局对象（解决安全限制）
                import numpy as np
                from torch.serialization import add_safe_globals
                from sklearn.preprocessing import MinMaxScaler
                add_safe_globals([MinMaxScaler, np._core.multiarray._reconstruct, np.dtype])
                # 关闭weights_only以加载scaler
                scaler = torch.load(scaler_path, weights_only=False)
                
                # 检查特征数量匹配
                if scaler.n_features_in_ != len(self.feature_cols):
                    raise ValueError(f"scaler特征数量不匹配（预期{len(self.feature_cols)}，实际{scaler.n_features_in_}）")
            except Exception as e:
                print(f"加载预训练scaler失败: {e}，将生成临时scaler")
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler.fit(stock_data[self.feature_cols].values)
        else:
            # 对新股票生成临时scaler
            print(f"未找到{symbol}的预训练scaler，使用临时scaler")
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(stock_data[self.feature_cols].values)
        
        # 标准化数据
        normalized_data = scaler.transform(stock_data[self.feature_cols].values)
        return torch.FloatTensor(normalized_data).unsqueeze(0).to(self.device)

        
    def _check_stock_listed(self, symbol):
        """检查股票是否上市"""
        if not self.ts_api:
            return True, "未验证上市状态"
        
        try:
            ts_code = self._convert_to_ts_code(symbol)
            stock_basic = self.ts_api.stock_basic(ts_code=ts_code, fields='list_status')
            if stock_basic.empty or stock_basic['list_status'].iloc[0] != 'L':
                return False, "非上市状态"
            return True, "正常上市"
        except Exception as e:
            return True, f"上市状态检查失败: {e}"
    
    def _convert_to_ts_code(self, symbol):
        """转换股票代码格式"""
        if symbol.startswith(('688', '60', '601', '603')):
            return f"{symbol}.SH"
        else:
            return f"{symbol}.SZ"
    
    def _check_data_integrity(self, stock_data, target_date, input_window=INPUT_WINDOW):
        """检查数据完整性"""
        if stock_data.empty:
            return False, "数据为空"
        
        if not self.calendar.is_trading_day(target_date):
            return False, f"目标日期{target_date}非交易日"
        
        # 检查目标日期是否在数据中（判断停牌）
        stock_dates = stock_data['date'].dt.strftime('%Y-%m-%d').tolist()
        if target_date not in stock_dates:
            return False, f"目标日期{target_date}停牌（无数据）"
        
        if len(stock_data) < input_window:
            return False, f"历史数据不足{input_window}天"
        
        return True, "数据完整"
    
    def predict_stock(self, stock_data, symbol, target_date, input_window=INPUT_WINDOW):
        """预测单只股票（修复参数传递）"""
        # 检查上市状态
        is_listed, listed_msg = self._check_stock_listed(symbol)
        if not is_listed:
            print(f"股票{symbol}状态异常: {listed_msg}，跳过")
            return False, 0.0, listed_msg
        
        # 检查数据完整性
        is_complete, data_msg = self._check_data_integrity(stock_data, target_date, input_window)
        if not is_complete:
            print(f"股票{symbol}数据异常: {data_msg}，跳过")
            return False, 0.0, data_msg
        
        # 执行预测（修复参数传递格式）
        try:
            # 显式指定参数名，避免位置参数混淆
            input_tensor = self.preprocess_stock_data(
                stock_data=stock_data,
                symbol=symbol,
                target_date=target_date,
                input_window=input_window
            )
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                up_prob = probs[0, 1].item()
                return up_prob > 0.5, up_prob, "正常"
        except Exception as e:
            print(f"预测{symbol}出错: {e}")
            return False, 0.0, f"预测错误: {e}"
    
    
    def find_qualified_stocks(self, target_date=None, threshold=0.6, input_window=INPUT_WINDOW):
        """寻找符合条件的股票"""
        # 确定目标日期
        if target_date is None:
            target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        target_date = self._format_date(target_date)
        
        if not self.calendar.is_trading_day(target_date):
            target_date = self.calendar.get_next_trading_day(target_date)
            print(f"目标日期调整为: {target_date}")
        
        # 计算数据范围

        # 替换原n_days计算逻辑（约在第380行左右）
        n_days = INPUT_WINDOW + DERIVED_FEATURE_DAYS  # 动态计算总需求天数
        start_date = self.calendar.get_previous_trading_day(target_date, n=n_days)

        print(f"数据范围: {start_date} 至 {target_date}")
        
        # 获取成分股
        stocks = self.data_loader.get_sector_stocks()
        if not stocks:
            raise ValueError("未获取到成分股列表")
        
        # 加载特征组合分析器
        from feature_analyzer import FeatureCombinationAnalyzer
        analyzer = FeatureCombinationAnalyzer()
        analyzer.load_patterns()
        self.positive_patterns = analyzer.positive_patterns
        
        qualified = []
        qualified_feature=[]
        for stock in tqdm(stocks, desc="预测成分股"):
            symbol, name = stock['code'], stock['name']
            try:
                # 加载并格式化数据（确保'date'列存在）
                # end_date = self.calendar.get_previous_trading_day(target_date)
                stock_data = self._load_and_format_data(symbol, start_date, target_date)
                
                if stock_data.empty:
                    print(f"股票{symbol}无数据，跳过")
                    continue
                
                # 确保'date'是普通列而不是索引
                if 'date' in stock_data.index.names:
                    stock_data = stock_data.reset_index()
                
                # 确保'date'列是datetime类型
                if 'date' in stock_data.columns:
                    stock_data['date'] = pd.to_datetime(stock_data['date'])
                
                # 预测
                is_qualified, prob, status = self.predict_stock(
                    stock_data, symbol, target_date, input_window  # 用配置中的INPUT_WINDOW
                )
 
                if status == "正常" and is_qualified and prob >= threshold:
                    # 计算特征匹配数量
                    
                    qualified.append({
                        'code': symbol, 
                        'name': name, 
                        'probability': round(prob, 4), 
                        'target_date': target_date
                    })
                    feature_matches = self._calculate_feature_matches(stock_data, symbol)
                    qualified_feature.append({
                        'code': symbol, 
                        'name': name, 
                        'probability': round(prob, 4), 
                        'target_date': target_date,
                        'feature_matches': feature_matches
                    })
            except Exception as e:
                print(f"处理{symbol}出错: {e}")
        qualified.sort(key=lambda x: x['probability'], reverse=False)
        # 按特征匹配数量升序排列（符合特征最少的排在前面）
        qualified_feature.sort(key=lambda x: x['feature_matches'], reverse=False)
        print(f"预测完成，找到{len(qualified)}只符合条件的股票,找到{len(qualified_feature)}只符合条件的股票")
        return qualified,qualified_feature
    
    def _calculate_feature_matches(self, stock_data, symbol):
        """计算股票数据符合特征的数量"""
        try:
            # 确保数据有40天用于特征计算
            if len(stock_data) < 40:
                # 如果数据不足40天，尝试扩展数据范围
                extended_start = self.calendar.get_previous_trading_day(
                    stock_data['date'].min().strftime('%Y-%m-%d'), n=40
                )
                extended_data = self._load_and_format_data(symbol, extended_start, 
                                                         stock_data['date'].max().strftime('%Y-%m-%d'))
                if len(extended_data) >= 40:
                    stock_data = extended_data.tail(40)
                else:
                    return 0
            
            # 确保数据是DataFrame格式
            if isinstance(stock_data.index, pd.DatetimeIndex):
                stock_data = stock_data.reset_index()
            
            # 检查必要的列是否存在
            required_cols = ['open', 'close', 'high', 'low', 'volume']
            missing_cols = [col for col in required_cols if col not in stock_data.columns]
            if missing_cols:
                return 0
            
            # 计算技术指标
            from features import calculate_technical_indicators, get_wave_periods, screen_second_wave
            
            df = calculate_technical_indicators(stock_data.copy())
            
            # 如果数据不足40天，返回0
            if len(df) < 40:
                return 0
            
            # 获取波段信息
            periods = get_wave_periods(df)
            if not periods.get('is_valid', False):
                return 0
            
            # 计算特征
            input_data = (df, periods)
            features = screen_second_wave(input_data)
            
            # 计算单个特征匹配数
            single_feature_count = sum(1 for v in features.values() if v)
            
            # 计算组合特征匹配数（每个组合特征算3个）
            combination_count = 0
            if hasattr(self, 'positive_patterns') and self.positive_patterns:
                for pattern in self.positive_patterns:
                    pattern_features = pattern.get('features', [])
                    if all(features.get(feat, False) for feat in pattern_features):
                        combination_count += 1
            
            # 总特征匹配数 = 单个特征数 + 组合特征数 * 3
            total_matches = single_feature_count + combination_count * 3
            
            return total_matches
            
        except Exception as e:
            print(f"计算{symbol}特征匹配数出错: {e}")
            return 0

    def save_prediction_results(self, results, target_date):
        """保存预测结果"""
        if not results:
            print("无结果可保存")
            return
        
        output_dir = os.path.join(self.data_dir, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"prediction_{target_date.replace('-', '')}.csv")
        
        df = pd.DataFrame(results)
        df['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"结果保存至: {output_file}")
    
    def save_prediction_results_feature(self, results, target_date):
        """保存预测结果，按概率分段和特征数量排序"""
        if not results:
            print("无结果可保存")
            return
        
        output_dir = os.path.join(self.data_dir, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"prediction_{target_date.replace('-', '')}.csv")
        
        # 按概率分段（从低到高，让高概率段在后面）
        probability_groups = {
            '60%及以上': [],
            '65%及以上': [],
            '70%及以上': [],
            '75%及以上': [],
            '80%及以上': [],
            '85%及以上': [],
            '90%及以上': [],
            '95%及以上': []
        }
        
        # 将股票分配到对应概率组
        for stock in results:
            prob = stock['probability']
            if prob >= 0.95:
                probability_groups['95%及以上'].append(stock)
            elif prob >= 0.90:
                probability_groups['90%及以上'].append(stock)
            elif prob >= 0.85:
                probability_groups['85%及以上'].append(stock)
            elif prob >= 0.80:
                probability_groups['80%及以上'].append(stock)
            elif prob >= 0.75:
                probability_groups['75%及以上'].append(stock)
            elif prob >= 0.70:
                probability_groups['70%及以上'].append(stock)
            elif prob >= 0.65:
                probability_groups['65%及以上'].append(stock)
            elif prob >= 0.60:
                probability_groups['60%及以上'].append(stock)
        
        # 在每个概率组内按特征匹配数量升序排列（特征最少的在前面）
        for group_name, group_stocks in probability_groups.items():
            if group_stocks:
                group_stocks.sort(key=lambda x: x['feature_matches'], reverse=False)
        
        # 创建最终排序结果
        final_results = []
        for group_name, group_stocks in probability_groups.items():
            if group_stocks:
                for stock in group_stocks:
                    stock_copy = stock.copy()
                    stock_copy['probability_group'] = group_name
                    final_results.append(stock_copy)
        
        # 保存到CSV
        df = pd.DataFrame(final_results)
        df['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 打印分组结果
        print(f"\n按概率分组的预测结果:")
        for group_name, group_stocks in probability_groups.items():
            if group_stocks:
                print(f"\n{group_name} ({len(group_stocks)}只):")
                for stock in group_stocks[:10]:  # 只显示前10只
                    print(f"  {stock['code']} {stock['name']} - 概率:{stock['probability']:.4f} 特征匹配:{stock['feature_matches']}")
                if len(group_stocks) > 10:
                    print(f"  ... 还有{len(group_stocks)-10}只股票")
        
        print(f"\n结果保存至: {output_file}")
        return final_results
    
    def _format_date(self, date_str):
        """统一日期格式"""
        try:
            if len(date_str) == 8 and date_str.isdigit():
                return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError(f"无效日期格式: {date_str}（需YYYY-MM-DD或YYYYMMDD）")


def main():
    try:
        default_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        while True:
            user_input = input(f"请输入目标日期(格式YYYY-MM-DD，默认{default_date}): ")
            target_date = user_input or default_date
            try:
                target_date = StockPredictor()._format_date(target_date)
                break
            except ValueError as e:
                print(f"输入错误: {e}，请重新输入")
        
        predictor = StockPredictor()
        qualified_stocks,qualified_feature = predictor.find_qualified_stocks(target_date, threshold=0.6)
        
        if qualified_stocks:
            print("\n符合条件的股票:")
            for stock in qualified_stocks:
                print(f"{stock['code']} {stock['name']} - 上涨概率: {stock['probability']:.4f}")
            predictor.save_prediction_results(qualified_stocks, target_date)
        else:
            print("未找到符合条件的股票")
            
        if qualified_feature:
            print("\n符合条件的股票（按特征匹配数量升序排列）:")
            for i, stock in  enumerate(qualified_feature[:20], 1):   # 显示前20只
                print(f"{i:2d}. {stock['code']} {stock['name']} - 上涨概率: {stock['probability']:.4f} 特征匹配: {stock['feature_matches']}")
            
            # 保存结果（包含概率分组和特征排序）
            # final_results = predictor.save_prediction_results(qualified_stocks, target_date)
            print(f"\n总共找到 {len(qualified_feature)} 只符合条件的股票")
        else:
            print("未找到符合条件的股票")
        
    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()