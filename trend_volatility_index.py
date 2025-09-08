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


def HurstExponent(df, period=100, src_col="Close"):
    '''
    计算Hurst指数 - 测量时间序列的长期记忆性
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 100)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    H > 0.5：趋势性市场，价格具有持续性
    H = 0.5：随机游走，震荡市场
    H < 0.5：均值回归，强震荡市场
    '''
    src = df[src_col]

    def calculate_hurst(series):
        if len(series) < 10:
            return np.nan

        lags = range(2, min(20, len(series) // 2))
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

        # 线性回归计算Hurst指数
        log_lags = np.log(lags)
        log_tau = np.log(tau)

        if len(log_tau) < 3:
            return np.nan

        slope, _ = np.polyfit(log_lags, log_tau, 1)
        return slope

    df['hurst'] = src.rolling(period).apply(calculate_hurst, raw=False)
    return df


def ATR(df, period=14, high_col="High", low_col="Low", close_col="Close"):
    '''
    计算Average True Range (ATR)
    :param df: 必须包含 [high_col, low_col, close_col] 的 pandas DataFrame
    :param period: ATR周期 (默认 14)
    :return: df
    ATR上升：波动性增加，可能趋势开始或加速
    ATR下降：波动性减少，可能进入震荡或趋势结束
    ATR突破历史高位：强趋势信号
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # True Range计算
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))

    df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
    df['atr'] = df['tr'].rolling(period).mean()

    return df


def BollingerBandWidth(df, period=20, std_dev=2, src_col="Close"):
    '''
    计算布林带宽度
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 移动平均周期 (默认 20)
    :param std_dev: 标准差倍数 (默认 2)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    bb_width收缩：震荡整理，准备突破
    bb_width扩张：趋势开始或加速
    bb_percent：价格在布林带中的位置
    '''
    src = df[src_col]

    sma = src.rolling(period).mean()
    std = src.rolling(period).std()

    df['bb_upper'] = sma + (std * std_dev)
    df['bb_lower'] = sma - (std * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
    df['bb_percent'] = (src - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    return df


def DonchianChannelFeatures(df, period=20, high_col="High", low_col="Low", close_col="Close"):
    """
    Donchian Channel 特征集
    包含：通道宽度、宽度变化、相对位置、突破事件、突破强度、趋势强度
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    df['dc_upper'] = high.rolling(period).max()
    df['dc_lower'] = low.rolling(period).min()
    df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
    df['dc_width'] = (df['dc_upper'] - df['dc_lower']) / close
    df['dc_width_change'] = df['dc_width'].diff()
    df['dc_percent'] = (close - df['dc_lower']) / (df['dc_upper'] - df['dc_lower'])

    # 突破事件
    df['dc_breakout_up'] = (close > df['dc_upper']).astype(int)
    df['dc_breakout_down'] = (close < df['dc_lower']).astype(int)

    # 突破强度（避免单纯布尔）
    df['dc_breakout_strength'] = np.where(close > df['dc_upper'],
                                          (close - df['dc_upper']) / df['dc_width'], 0)
    df['dc_breakdown_strength'] = np.where(close < df['dc_lower'],
                                           (df['dc_lower'] - close) / df['dc_width'], 0)

    # 趋势强度（相对位置 × 宽度变化）
    df['dc_trend_strength'] = (df['dc_percent'] - 0.5) * df['dc_width_change']

    return df


def KeltnerChannelFeatures(df, period=20, atr_period=14, multiplier=2,
                           high_col="High", low_col="Low", close_col="Close"):
    """
    Keltner Channel 特征集
    包含：通道宽度、宽度变化、相对位置、突破事件、突破强度、趋势强度
    """
    close = df[close_col]

    # 计算ATR（假设你已有 ATR 函数）
    df = ATR(df, atr_period, high_col, low_col, close_col)

    ema = close.ewm(span=period).mean()
    df['kc_upper'] = ema + (df['atr'] * multiplier)
    df['kc_lower'] = ema - (df['atr'] * multiplier)

    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / close
    df['kc_width_change'] = df['kc_width'].diff()
    df['kc_percent'] = (close - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])

    # 突破事件
    df['kc_breakout_up'] = (close > df['kc_upper']).astype(int)
    df['kc_breakout_down'] = (close < df['kc_lower']).astype(int)

    # 突破强度
    df['kc_breakout_strength'] = np.where(close > df['kc_upper'],
                                          (close - df['kc_upper']) / df['kc_width'], 0)
    df['kc_breakdown_strength'] = np.where(close < df['kc_lower'],
                                           (df['kc_lower'] - close) / df['kc_width'], 0)

    # 趋势强度（相对位置 × 宽度变化）
    df['kc_trend_strength'] = (df['kc_percent'] - 0.5) * df['kc_width_change']

    return df



def HistoricalVolatility(df, period=20, annualize=252, src_col="Close"):
    '''
    计算历史波动率
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 20)
    :param annualize: 年化因子 (默认 252)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    波动率上升：市场不确定性增加，可能转向趋势
    波动率下降：市场平静，震荡概率大
    波动率突破：重要的制度转换信号
    '''
    src = df[src_col]

    # 对数收益率
    log_returns = np.log(src / src.shift(1))

    # 历史波动率
    df['hist_vol'] = log_returns.rolling(period).std() * np.sqrt(annualize)
    df['parkinson_vol'] = np.sqrt(np.log(df.get('High', src) / df.get('Low', src)) ** 2)

    return df


def YangZhangVolatility(df, period=20, high_col="High", low_col="Low",
                        open_col="Open", close_col="Close"):
    '''
    计算Yang-Zhang波动率估计量
    :param df: 必须包含 [high_col, low_col, open_col, close_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 20)
    :return: df
    同上historicalvolatility
    '''
    high = df[high_col]
    low = df[low_col]
    open_price = df[open_col]
    close = df[close_col]

    # Yang-Zhang组件
    overnight = np.log(open_price / close.shift(1))
    rs = np.log(high / close) * np.log(high / open_price) + np.log(low / close) * np.log(low / open_price)

    df['yz_vol'] = np.sqrt(overnight.rolling(period).var() + rs.rolling(period).mean())

    return df


def GARCH_Volatility(df, period=50, p=1, q=1, src_col="Close"):
    '''
    简化的GARCH(1,1)波动率预测
    注意：这是简化版本，真正的GARCH需要专门的库如arch
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param p: GARCH的p参数 (默认 1)
    :param q: GARCH的q参数 (默认 1)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    # 计算收益率
    returns = src.pct_change().dropna()

    def simple_garch(ret_series, alpha=0.1, beta=0.85, omega=0.05):
        if len(ret_series) < 10:
            return np.nan

        # 初始化条件方差
        sigma2 = ret_series.var()
        garch_var = [sigma2]

        for i in range(1, len(ret_series)):
            # GARCH(1,1): sigma2_t = omega + alpha*r2_{t-1} + beta*sigma2_{t-1}
            sigma2 = omega + alpha * (ret_series.iloc[i - 1] ** 2) + beta * sigma2
            garch_var.append(sigma2)

        return np.sqrt(garch_var[-1])  # 返回最后一期的波动率

    df['garch_vol'] = returns.rolling(period).apply(simple_garch, raw=False)
    return df



#################### TA 继续处理 ###########################33

# ---------- helpers ----------
def rolling_zscore(series, window, min_periods=None):
    if min_periods is None:
        min_periods = window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / (std + 1e-9)

def rolling_pct_rank(series, window, min_periods=None):
    if min_periods is None:
        min_periods = window
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def bars_since_event(event_series):
    """
    返回从上次事件发生到当前 bar 的距离（事件当日 = 0）。
    如果历史上从未发生过事件，返回 i+1（即从序列开始算起的距离）。
    说明：用纯 numpy 循环实现，直观且不易出边界 bug。
    """
    s = event_series.astype(bool).values
    out = np.empty(len(s), dtype=int)
    last = -1
    for i, v in enumerate(s):
        if v:
            last = i
            out[i] = 0
        else:
            out[i] = i - last if last != -1 else i + 1
    return pd.Series(out, index=event_series.index)

# ---------- build_features ----------
def build_features(df,
                   macd_fast=12, macd_slow=26, macd_signal=9,
                   adx_len=14, adx_smooth=14,
                   atr_len=14,
                   bb_len=20, bb_std=2,
                   dc_len=20, kc_len=20,
                   z_windows=(50, 100)):

    df = df.copy()

    # 1) 算已有ta（确保这些函数在脚本中已定义并使用相同列名）
    df = ADX(df, di_len=adx_len, adx_period=adx_smooth)
    df = MACD(df, fast_length=macd_fast, slow_length=macd_slow, signal_length=macd_signal)
    df = ATR(df, period=atr_len)
    df = BollingerBandWidth(df, period=bb_len, std_dev=bb_std)
    df = DonchianChannelFeatures(df, period=dc_len)
    df = KeltnerChannelFeatures(df, period=kc_len, atr_period=atr_len)
    df = HurstExponent(df, period=100)
    df = LinearRegressionSlope(df)
    df = HistoricalVolatility(df)

    # 2) ATR 归一（把一些尺度不一的指标换到同一波动率尺度）
    for col in ["macd", "macd_signal", "macd_hist", "lrs", "slrs", "alrs"]:
        if (col in df.columns) and ('atr' in df.columns):
            df[f"{col}_atr_norm"] = df[col] / (df['atr'] + 1e-9)

    # 3) MACD 衍生特征（动量与事件型）
    if {'macd', 'macd_signal', 'macd_hist'}.issubset(df.columns):
        df['macd_gap'] = df['macd'] - df['macd_signal']                    # 差距（力度）
        df['macd_gap_atr'] = df['macd_gap'] / (df['atr'] + 1e-9)           # 相对波动率下的力度
        df['d_macd'] = df['macd'].diff()                                   # 一阶差分（斜率）
        df['d_macd_hist'] = df['macd_hist'].diff()                         # hist 的一阶差分（加速/减速）

        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) &
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

        df['bars_since_macd_cross_up'] = bars_since_event(df['macd_cross_up'] == 1)
        df['bars_since_macd_cross_down'] = bars_since_event(df['macd_cross_down'] == 1)
        df['macd_above_zero'] = (df['macd'] > 0).astype(int)

    # 4) ADX 特征
    if {'plus_di', 'minus_di', 'adx'}.issubset(df.columns):
        df['trend_direction'] = np.sign(df['plus_di'] - df['minus_di'])
        df['trend_confidence'] = (df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9)
        df['adx_turning'] = df['adx'].diff()

    # 5) 通道 / 带宽 的触碰与 bars_since
    for prefix in ['bb', 'dc', 'kc']:
        up_col = f'{prefix}_upper'
        low_col = f'{prefix}_lower'
        if (up_col in df.columns) and (low_col in df.columns):
            df[f'{prefix}_touch_upper'] = (df['Close'] >= df[up_col]).astype(int)
            df[f'{prefix}_touch_lower'] = (df['Close'] <= df[low_col]).astype(int)
            df[f'{prefix}_bars_since_touch_upper'] = bars_since_event(df[f'{prefix}_touch_upper'] == 1)
            df[f'{prefix}_bars_since_touch_lower'] = bars_since_event(df[f'{prefix}_touch_lower'] == 1)

    # squeeze（bb 宽度处于过去 50 根的 10% 分位）
    if 'bb_width' in df.columns:
        df['squeeze'] = (df['bb_width'] <= df['bb_width'].rolling(50, min_periods=1).quantile(0.1)).astype(int)

    # 6) 波动率特征
    if ('hist_vol' in df.columns) and ('yz_vol' in df.columns):
        df['vol_ratio'] = rolling_zscore(df['hist_vol'], 50) - rolling_zscore(df['yz_vol'], 50)
        df['d_vol'] = df['hist_vol'].diff()

    # 7) 斜率翻转
    if 'slrs' in df.columns:
        df['slope_flip'] = ((df['slrs'] > 0) & (df['slrs'].shift(1) <= 0)).astype(int)
        df['bars_since_slope_flip'] = bars_since_event(df['slope_flip'] == 1)

    # 8) Hurst 状态
    if 'hurst' in df.columns:
        df['hurst_state'] = np.where(df['hurst'] > 0.5, 1, np.where(df['hurst'] < 0.5, -1, 0))

    # 9) 多窗口 Z-score & 百分位
    cols = ['adx', 'macd_gap', 'bb_width', 'dc_width', 'kc_width', 'hist_vol']
    for col in cols:
        if col in df.columns:
            for w in z_windows:
                df[f'{col}_z{w}'] = rolling_zscore(df[col], w)
                df[f'{col}_pct{w}'] = rolling_pct_rank(df[col], w)

    # 10) 防信息泄露：把除 OHLCV 的所有派生列整体 shift(1)
    protected = [c for c in df.columns if c.lower() in ('open', 'high', 'low', 'close', 'volume')]
    feature_cols = [c for c in df.columns if c not in protected]
    df[feature_cols] = df[feature_cols].shift(1)

    return df


