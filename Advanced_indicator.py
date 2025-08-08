import pandas as pd
import numpy as np
from Basic_indicator import BI_basic_indicator


class AI_advanced_indicator:


    def __init__(self, data):

        self.data = data

    def UT_Bot(self, utbot_key_value, atr_period, use_heikin_ashi=False):
        """
        df: pandas DataFrame with columns ['open', 'high', 'low', 'close']
        utbot_key_value: 调整灵敏度，越大越不容易给信号
        atr_period: ATR 的周期
        use_heikin_ashi: 是否使用 Heikin Ashi 蜡烛图
        """

        if use_heikin_ashi:
            ha = self.data.copy()
            ha['Close'] = (self.data['Open'] + self.data['High'] + self.data['Low'] + self.data['Close']) / 4
            ha['Open'] = (self.data['Open'].shift(1) + self.data['Close'].shift(1)) / 2
            ha['High'] = self.data[['High', 'Open', 'Close']].max(axis=1)
            ha['Low'] = self.data[['Low', 'Open', 'Close']].min(axis=1)
            src = ha['Close']
        else:
            src = self.data['Close']

        atr = BI_basic_indicator(self.data).ATR(atr_period)








        print(atr)
        return atr

