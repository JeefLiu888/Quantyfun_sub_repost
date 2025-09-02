import pandas as pd
import numpy as np


def OBV(df, close_col="Close", volume_col="Volume"):
    '''
    计算On Balance Volume (OBV)
    :param df: 必须包含 [close_col, volume_col] 的 pandas DataFrame
    :param close_col: 收盘价列 (默认 'Close')
    :param volume_col: 成交量列 (默认 'Volume')
    :return: df
    OBV与价格同向：趋势确认
    OBV与价格背离：趋势可能反转
    OBV横盘：震荡整理
    '''
    close = df[close_col]
    volume = df[volume_col]

    # OBV计算
    price_change = close.diff()
    obv_change = np.where(price_change > 0, volume,
                          np.where(price_change < 0, -volume, 0))

    df['obv'] = obv_change.cumsum()
    return df


def PVT(df, close_col="Close", volume_col="Volume"):
    '''
    计算Price Volume Trend (PVT)
    :param df: 必须包含 [close_col, volume_col] 的 pandas DataFrame
    :param close_col: 收盘价列 (默认 'Close')
    :param volume_col: 成交量列 (默认 'Volume')
    :return: df
    '''
    close = df[close_col]
    volume = df[volume_col]

    # PVT计算
    price_change_pct = close.pct_change()
    pvt_change = price_change_pct * volume

    df['pvt'] = pvt_change.cumsum()
    return df


def ChaikinMoneyFlow(df, period=20, high_col="High", low_col="Low",
                     close_col="Close", volume_col="Volume"):
    '''
    计算Chaikin Money Flow (CMF)
    :param df: 必须包含 [high_col, low_col, close_col, volume_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 20)
    :return: df
    CMF > 0.1：资金流入，上升趋势
    CMF < -0.1：资金流出，下降趋势
    -0.1 < CMF < 0.1：震荡整理
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]

    # Money Flow Multiplier
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.fillna(0)  # 处理high=low的情况

    # Money Flow Volume
    mf_volume = mf_multiplier * volume

    # Chaikin Money Flow
    df['cmf'] = mf_volume.rolling(period).sum() / volume.rolling(period).sum()

    return df


def EaseOfMovement(df, period=14, scale=1000000, high_col="High", low_col="Low", volume_col="Volume"):
    '''
    计算Ease of Movement (EOM)
    :param df: 必须包含 [high_col, low_col, volume_col] 的 pandas DataFrame
    :param period: 移动平均周期 (默认 14)
    :param scale: 缩放因子 (默认 1000000)
    :return: df
    '''
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]

    # Distance Moved
    distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2

    # Box Height
    box_height = volume / scale / (high - low)

    # 1-Period EMV
    emv_1 = distance_moved / box_height
    emv_1 = emv_1.replace([np.inf, -np.inf], np.nan).fillna(0)

    # N-Period EOM
    df['eom'] = emv_1.rolling(period).mean()

    return df


def ForceIndex(df, period=13, close_col="Close", volume_col="Volume"):
    '''
    计算Force Index
    :param df: 必须包含 [close_col, volume_col] 的 pandas DataFrame
    :param period: 移动平均周期 (默认 13)
    :param close_col: 收盘价列 (默认 'Close')
    :param volume_col: 成交量列 (默认 'Volume')
    :return: df
    Force Index持续为正：强势上升趋势
    Force Index持续为负：强势下降趋势
    Force Index围绕零线震荡：震荡市场
    '''
    close = df[close_col]
    volume = df[volume_col]

    # Force Index = (Close - Previous Close) * Volume
    df['force_index_raw'] = (close - close.shift(1)) * volume
    df['force_index'] = df['force_index_raw'].ewm(span=period).mean()

    return df


def AccumulationDistribution(df, high_col="High", low_col="Low",
                             close_col="Close", volume_col="Volume"):
    '''
    计算累积/派发线 (A/D Line)
    :param df: 必须包含相应列的 pandas DataFrame
    :return: df
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]

    # Money Flow Multiplier
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = mf_multiplier.fillna(0)

    # Money Flow Volume
    mf_volume = mf_multiplier * volume

    # A/D Line (累积)
    df['ad_line'] = mf_volume.cumsum()

    return df


def MoneyFlowIndex(df, period=14, high_col="High", low_col="Low",
                   close_col="Close", volume_col="Volume"):
    '''
    计算Money Flow Index (MFI)
    :param df: 必须包含相应列的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :return: df
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]

    # 典型价格
    typical_price = (high + low + close) / 3

    # Raw Money Flow
    raw_money_flow = typical_price * volume

    # 正负Money Flow
    price_change = typical_price.diff()
    positive_mf = raw_money_flow.where(price_change > 0, 0)
    negative_mf = raw_money_flow.where(price_change < 0, 0)

    # Money Flow Ratio
    positive_mf_sum = positive_mf.rolling(period).sum()
    negative_mf_sum = negative_mf.rolling(period).sum()

    mf_ratio = positive_mf_sum / negative_mf_sum

    # MFI
    df['mfi'] = 100 - (100 / (1 + mf_ratio))
    df['mfi'] = df['mfi'].fillna(50)

    return df


def KlingerOscillator(df, fast=34, slow=55, signal=13,
                      high_col="High", low_col="Low", close_col="Close", volume_col="Volume"):
    '''
    计算Klinger Oscillator
    :param df: 必须包含相应列的 pandas DataFrame
    :param fast: 快速EMA周期 (默认 34)
    :param slow: 慢速EMA周期 (默认 55)
    :param signal: 信号线周期 (默认 13)
    :return: df
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]

    # Typical Price
    tp = (high + low + close) / 3

    # Trend direction
    trend = np.where(tp > tp.shift(1), 1, -1)

    # Volume Force
    vf = volume * trend * abs(2 * ((close - low) - (high - close)) / (high - low)) * 100
    vf = vf.fillna(0)

    # Klinger Oscillator
    fast_ema = vf.ewm(span=fast).mean()
    slow_ema = vf.ewm(span=slow).mean()

    df['klinger'] = fast_ema - slow_ema
    df['klinger_signal'] = df['klinger'].ewm(span=signal).mean()

    return df
