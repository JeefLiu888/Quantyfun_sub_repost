import pandas as pd
import numpy as np
from main import load_data
from datetime import date


def RMA(series, period):
    '''
    RMA（Wilder's MA）
    '''
    return series.ewm(alpha=1/period, adjust=False).mean()

def WMA(series, period):
    '''
    WMA（Weighted Moving Average） 线性加权，最新数据权重最大
    '''
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

def ATR(df, period = 14, smoothing="RMA"):
    '''
    df 必须包含列: High, Low, Close. smoothing: "RMA", "SMA", "EMA", "WMA"
    '''
    high = df['High']
    low = df['Low']
    close = df['Close']

    prev_close = close.shift(1) #上一个收盘价

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 选择平滑方法
    smoothing = smoothing.upper()
    if smoothing == "RMA":
        atr = RMA(tr, period)
    elif smoothing == "SMA":
        atr = tr.rolling(window=period, min_periods=1).mean()
    elif smoothing == "EMA":
        atr = tr.ewm(span=period, adjust=False).mean()
    elif smoothing == "WMA":
        atr = WMA(tr, period)
    else:
        raise ValueError("Unknown smoothing method")

    atr.to_csv('atr_check.csv')

    return atr

df = load_data('BTC', start='2023-01-01', end= date.today(), interval='1d')

ATR(df, period=10, smoothing='EMA')




if __name__ == "__main__":
    print('ff')