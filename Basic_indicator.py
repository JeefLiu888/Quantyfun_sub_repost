import pandas as pd
import numpy as np



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



def ATR(df, atr_period = 14, smoothing ='EMA'):
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
        atr = RMA(tr, atr_period)
    elif smoothing == "SMA":
        atr = tr.rolling(window=atr_period, min_periods=1).mean()
    elif smoothing == "EMA":
        atr = tr.ewm(span= atr_period, adjust=False).mean()
    elif smoothing == "WMA":
        atr = WMA(tr, atr_period)
    else:
        raise ValueError("Unknown smoothing method")

    #atr.to_csv('atr_check.csv')
    #print(atr)

    return atr




def heikin_ashi(df):

    ha_df = pd.DataFrame(index=df.index)

    ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_df['Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    ha_df['High'] = ha_df[['Open', 'Close']].join(df['High']).max(axis=1)
    ha_df['Low'] = ha_df[['Open', 'Close']].join(df['Low']).min(axis=1)
    return ha_df



def UT_BOT_alerts(df, a=1, atr_period =14 , use_heikin=False):

    '''

    :param df: original data
    :param a: 灵敏倍数，越大越不敏感
    :param atr_period: atr用的回滚周期长度
    :param use_heikin: 是否使用这个参数

    '''

    if use_heikin:
        src_df = heikin_ashi(df)
        src = src_df['Close']
    else:
        src = df['Close']

    xATR = ATR(df, atr_period=atr_period, smoothing='EMA')
    nLoss = a * xATR

    xATRTrailingStop = np.zeros(len(df))
    pos_trend = np.zeros(len(df))   # 原趋势方向
    holding   = np.zeros(len(df))   # 真实持仓状态（0=空仓, 1=多仓）

    buy_signal  = np.zeros(len(df), dtype=bool)
    sell_signal = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if i == 0:
            xATRTrailingStop[i] = src.iloc[i] - nLoss.iloc[i]
            pos_trend[i] = 0
            holding[i] = 0
            continue

        prev_stop = xATRTrailingStop[i-1]

        # 计算趋势上的 stop 轨迹
        if src.iloc[i] > prev_stop and src.iloc[i-1] > prev_stop:
            xATRTrailingStop[i] = max(prev_stop, src.iloc[i] - nLoss.iloc[i])
        elif src.iloc[i] < prev_stop and src.iloc[i-1] < prev_stop:
            xATRTrailingStop[i] = min(prev_stop, src.iloc[i] + nLoss.iloc[i])
        elif src.iloc[i] > prev_stop:
            xATRTrailingStop[i] = src.iloc[i] - nLoss.iloc[i]
        else:
            xATRTrailingStop[i] = src.iloc[i] + nLoss.iloc[i]

        # 时间顺序更新趋势方向
        if src.iloc[i-1] < prev_stop and src.iloc[i] > prev_stop:
            pos_trend[i] = 1
        elif src.iloc[i-1] > prev_stop and src.iloc[i] < prev_stop:
            pos_trend[i] = -1
        else:
            pos_trend[i] = pos_trend[i-1]

        # ------ 新增：真正的交易持仓逻辑 ------
        # 只在空仓时买
        if pos_trend[i] == 1 and holding[i-1] == 0:
            buy_signal[i] = True
            holding[i] = 1
        # 只在持多仓时卖
        elif pos_trend[i] == -1 and holding[i-1] == 1:
            sell_signal[i] = True
            holding[i] = 0
        else:
            holding[i] = holding[i-1]
        # -------------------------------------

    df['ut_bot_buy_signal'] = buy_signal
    df['ut_bot_sell_signal'] = sell_signal
    df['ut_bot_pos_trend'] = pos_trend
    df['ut_bot_holding'] = holding


    '''
    df_copy = df.copy()
    df_copy['ATR'] = xATR
    df_copy['ATR_Stop'] = pd.Series(xATRTrailingStop, index=df.index)
    df_copy['pos_trend'] = pos_trend
    df_copy['holding'] = holding
    df_copy['buy_signal'] = buy_signal
    df_copy['sell_signal'] = sell_signal
    df_copy.to_csv('btc_ut_bot_test.csv')   
    '''

    return df



def ADX(df, di_len=14, adx_period=14):
    """
    :param df: 必须包含 ['high', 'low', 'close'] 列的 pandas DataFrame
    :param di_len: DI Length
    :param adx_len: ADX Smoothing Length
    """

    # True Range
    df['prev_close'] = df['Close'].shift(1)
    df['tr'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['prev_close']),
                                     abs(df['Low'] - df['prev_close'])))

    # Up Move & Down Move
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']

    # PlusDM & MinusDM
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)


    # RMA = Wilder's smoothing
    def rma(series, period):
        rma = series.rolling(window=period).mean()
        rma.iloc[period:] = np.nan
        rma.iloc[period-1] = series.iloc[:period].mean()  # first value
        for i in range(period, len(series)):
            rma.iloc[i] = (rma.iloc[i-1] * (period - 1) + series.iloc[i]) / period
        return rma

    # Smoothed DI
    tr_rma = rma(df['tr'], di_len)
    plus_rma = rma(df['plus_dm'], di_len)
    minus_rma = rma(df['minus_dm'], di_len)

    df['plus_di'] = 100 * plus_rma / tr_rma
    df['minus_di'] = 100 * minus_rma / tr_rma

    # ADX
    dx = (abs(df['plus_di'] - df['minus_di']) /
          (df['plus_di'] + df['minus_di']).replace(0, np.nan)) * 100
    df['adx'] = rma(dx, adx_period)

    # 清理临时列
    df.drop(['prev_close', 'tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm'], axis=1, inplace=True)

    return df


def AroonOscillator(df, length=14):
    """
    :param df: 必须包含 ['High', 'Low'] 列的 pandas DataFrame
    :param length: 窗口长度 (默认 14)
    """

    def highestbars(arr, period):
        # 找到过去 period 根K线中最高点距离当前的bar数
        return arr.rolling(period).apply(lambda x: period - 1 - np.argmax(x), raw=True)

    def lowestbars(arr, period):
        # 找到过去 period 根K线中最低点距离当前的bar数
        return arr.rolling(period).apply(lambda x: period - 1 - np.argmin(x), raw=True)

    # Aroon upper and lower
    upper = 100 * (highestbars(df['High'], length + 1) + length) / length
    lower = 100 * (lowestbars(df['Low'], length + 1) + length) / length


    df['aroon_osc'] = upper - lower

    return df















