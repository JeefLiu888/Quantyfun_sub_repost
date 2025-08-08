import pandas as pd
import numpy as np



class BI_basic_indicator:

    def __init__(self, data):
        self.data = data


    def ATR(self, atr_period):
        # calculate ATR
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period, min_periods=1).mean()

        return atr












