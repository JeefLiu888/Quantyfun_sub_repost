import pandas as pd
import numpy as np






def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period).mean()





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





def MACD(df, fast_length=12, slow_length=26, signal_length=9,
         ma_source="EMA", ma_signal="EMA"):
    """

    :param df: 必须包含 ['Close'] 列的 pandas DataFrame
    :param fast_length:快均线周期 (默认 12)
    :param slow_length:慢均线周期 (默认 26)
    :param signal_length: 信号线周期 (默认 9)
    :param ma_source: "EMA" 或 "SMA"，用于 fast/slow 均线
    :param ma_signal:"EMA" 或 "SMA"，用于信号线
    :return: 修改后的df 含有macd的信息
    """

    # 选择均线类型
    ma_func_source = ema if ma_source.upper() == "EMA" else sma
    ma_func_signal = ema if ma_signal.upper() == "EMA" else sma

    # 计算 MACD
    fast_ma = ma_func_source(df['Close'], fast_length)
    slow_ma = ma_func_source(df['Close'], slow_length)

    df['macd'] = fast_ma - slow_ma
    df['macd_signal'] = ma_func_signal(df['macd'], signal_length)
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df



def LinearRegressionSlope(df, clen=50, slen=5, glen=13, src_col="Close"):
    '''
    计算 Linear Regression Slope (UCS-LRS) 指标，并并入 df
    :param df:必须包含 [src_col] 的 pandas DataFrame
    :param clen:回归曲线长度 (默认 50)
    :param slen: Slope 平滑周期 (默认 5, EMA)
    :param glen:Signal 平滑周期 (默认 13, SMA)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return:
    '''

    src = df[src_col]

    # === Linear Regression Curve (lrc) ===
    def linreg(series):
        y = series.values
        x = np.arange(len(y))
        x_mean, y_mean = x.mean(), y.mean()
        # slope & intercept
        m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        b = y_mean - m * x_mean
        return m * x[-1] + b  # 预测最后一点的拟合值

    df['lrc'] = src.rolling(clen).apply(linreg, raw=False)

    #Linear Regression Slope
    df['lrs'] = df['lrc'] - df['lrc'].shift(1)

    # Smooth LRS(EMA)
    df['slrs'] = df['lrs'].ewm(span=slen, adjust=False).mean()

    #Signal line(SMA)
    df['alrs'] = df['slrs'].rolling(window=glen).mean()

    return df







