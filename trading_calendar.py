import pandas as pd
import os
from datetime import datetime, timedelta
import baostock as bs

class TradingCalendar:
    """交易日历处理类，负责获取、存储和管理A股交易日历"""
    def __init__(self, cache_path='trading_calendar.csv'):
        self.cache_path = cache_path
        self.trading_days = []  # 使用列表代替集合，保持顺序
        self.date_format = '%Y-%m-%d'  # 统一日期格式
        self.last_updated = None
        
        # 尝试从缓存加载交易日历
        self._load_from_cache()
    
    def _convert_to_standard_format(self, date_str):
        """将日期字符串转换为标准格式 '%Y-%m-%d'"""
        if isinstance(date_str, str):
            # 尝试多种日期格式
            date_formats = [
                '%Y%m%d',      # 20250809
                '%Y-%m-%d',    # 2025-08-09
                '%Y/%m/%d',    # 2025/8/9 或 2025/08/09
            ]
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime(self.date_format)
                except ValueError:
                    continue
            
            # 如果所有格式都失败，尝试使用pandas解析
            try:
                dt = pd.to_datetime(date_str)
                return dt.strftime(self.date_format)
            except:
                # 如果已经是标准格式或其他格式，直接返回
                return date_str
        elif isinstance(date_str, (int, float)):
            # 如果是数字类型，转换为字符串并确保格式正确
            date_str = str(int(date_str))
            if len(date_str) == 8:
                dt = datetime.strptime(date_str, '%Y%m%d')
                return dt.strftime(self.date_format)
            else:
                raise ValueError(f"日期格式错误: {date_str}")
        else:
            raise TypeError(f"不支持的日期类型: {type(date_str)}")
    
    # 标注：以下函数原用于格式转换，现统一使用 _convert_to_standard_format，可能在其他文件有调用，保留
    def _convert_to_tushare_format(self, date_str):
        pass
    
    def _convert_to_tushare_format111(self, date_str):
        pass
    
    def _convert_from_tushare_format(self, date_str):
        pass
        
    def _load_from_cache(self):
        """从缓存文件加载交易日历"""
        if os.path.exists(self.cache_path):
            try:
                df = pd.read_csv(self.cache_path)
                # 确保所有日期都转换为标准格式
                self.trading_days = []
                for date_str in df['date'].tolist():
                    try:
                        standard_date = self._convert_to_standard_format(str(date_str))
                        if standard_date:
                            self.trading_days.append(standard_date)
                    except Exception as e:
                        print(f"跳过无效日期 {date_str}: {e}")
                
                self.trading_days = sorted(self.trading_days)  # 确保有序
                self.last_updated = df['last_updated'].iloc[0] if 'last_updated' in df.columns else None
                print(f"成功从缓存加载交易日历，共{len(self.trading_days)}个交易日，最后更新时间: {self.last_updated}")
                return True
            except Exception as e:
                print(f"从缓存加载交易日历失败: {e}")
        return False
    
    
    def _save_to_cache(self):
        """将交易日历保存到缓存文件"""
        if self.trading_days:
            df = pd.DataFrame({'date': self.trading_days})
            df['last_updated'] = datetime.now().strftime(self.date_format)
            df.to_csv(self.cache_path, index=False)
            print(f"交易日历已保存至缓存: {self.cache_path}")
    
    def _is_update_needed(self):
        """检查是否需要更新（一天只更新一次）"""
        today = datetime.now().strftime(self.date_format)
        return self.last_updated != today

    def update(self, start_date=None, end_date=None, force=False):
        """
        更新交易日历
        
        Args:
            start_date: 开始日期，默认为10年前
            end_date: 结束日期，默认为今天
            force: 是否强制更新
        """
        # 检查是否需要更新（一天只更新一次）
        if not force and not self._is_update_needed():
            print(f"交易日历今天已经更新过(截至{self.last_updated})，无需再次更新")
            return True

        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime(self.date_format)
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3650)).strftime(self.date_format)
        
        # 检查是否需要更新
        if not force and self.trading_days:
            if not self.trading_days:
                print("交易日历为空，需要更新")
            else:
                latest_date = max(self.trading_days)
                end_date_str = end_date if isinstance(end_date, str) else str(end_date)
                if latest_date >= end_date_str:
                    print(f"交易日历是最新的(截至{latest_date})，无需更新")
                    return True
                else:
                    # 需要追加数据
                    start_date = latest_date
                    print(f"需要追加从{start_date}到{end_date}的交易日数据")
        
        print(f"开始更新交易日历，日期范围: {start_date} 至 {end_date}")
        
        try:
            # 登录Baostock
            lg = bs.login()
            if lg.error_code != '0':
                print(f"Baostock登录失败: {lg.error_msg}")
                return False
                
            # 获取交易日历
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            if rs.error_code != '0':
                print(f"获取交易日历失败: {rs.error_msg}")
                bs.logout()
                return False
                
            # 解析交易日历，转换为标准格式
            new_trading_days = []
            while rs.next():
                row = rs.get_row_data()
                if row[1] == '1':  # is_trading_day
                    new_trading_days.append(self._convert_to_standard_format(row[0]))
            
            # 登出Baostock
            bs.logout()
            
            # 验证获取的交易日历
            if new_trading_days:
                print(f"成功获取{len(new_trading_days)}个交易日数据")
                
                # 合并新旧数据，保持唯一性和顺序
                if self.trading_days:
                    # 合并并去重
                    all_days = list(set(self.trading_days + new_trading_days))
                    self.trading_days = sorted(all_days)
                    print(f"合并后共有{len(self.trading_days)}个交易日")
                else:
                    self.trading_days = sorted(new_trading_days)
                
                self._save_to_cache()
                return True
            else:
                print("获取的交易日历为空")
                return False
                
        except Exception as e:
            print(f"更新交易日历异常: {e}")
            return False
    
    def is_trading_day(self, date):
        """判断指定日期是否为交易日"""
        # 确保交易日历已加载
        if not self.trading_days:
            print("交易日历未加载，尝试更新...")
            if not self.update():
                return False
                
        # 转换为标准格式
        date_standard = self._convert_to_standard_format(date)
        return date_standard in self.trading_days
    
    def get_next_trading_day(self, date, n=1):
        """获取指定日期之后的第n个交易日"""
        if not self.trading_days:
            if not self.update():
                return None
                
        # 转换为标准格式
        date_standard = self._convert_to_standard_format(date)
        
        # 找到日期在列表中的位置
        try:
            idx = self.trading_days.index(date_standard)
            next_idx = idx + n
            if next_idx < len(self.trading_days):
                return self.trading_days[next_idx]
            else:
                print(f"请求的交易日超出了当前日历范围，返回最后一个交易日")
                return self.trading_days[-1]
        except ValueError:
            # 如果日期不在列表中，找到大于该日期的最小交易日
            for day in self.trading_days:
                if day > date_standard:
                    return day
            print(f"未找到大于{date_standard}的交易日")
            return None
    
    def get_previous_trading_day(self, date, n=1):
        """获取指定日期之前的第n个交易日"""
        if not self.trading_days:
            if not self.update():
                return None
                
        # 转换为标准格式
        date_standard = self._convert_to_standard_format(date)
        
        # 找到日期在列表中的位置
        try:
            idx = self.trading_days.index(date_standard)
            prev_idx = idx - n
            if prev_idx >= 0:
                return self.trading_days[prev_idx]
            else:
                print(f"请求的交易日超出了当前日历范围，返回第一个交易日")
                return self.trading_days[0]
        except ValueError:
            # 如果日期不在列表中，找到小于该日期的最大交易日
            for i in range(len(self.trading_days) - 1, -1, -1):
                if self.trading_days[i] < date_standard:
                    return self.trading_days[i]
            print(f"未找到小于{date_standard}的交易日")
            return None
    
    def get_trading_days(self, start_date, end_date):
        """获取指定日期范围内的所有交易日"""
        if not self.trading_days:
            if not self.update():
                return []
                
        # 转换为标准格式
        try:
            start_date_standard = self._convert_to_standard_format(start_date)
            end_date_standard = self._convert_to_standard_format(end_date)
        except Exception as e:
            print(f"日期格式转换失败: {e}")
            return []
        
        # 筛选范围内的交易日
        try:
            return [d for d in self.trading_days if start_date_standard <= d <= end_date_standard]
        except Exception as e:
            print(f"日期比较失败: {e}")
            return []
    
    def format_for_standard(self, date_str):
        """将日期字符串转换为标准格式"""
        return self._convert_to_standard_format(date_str)