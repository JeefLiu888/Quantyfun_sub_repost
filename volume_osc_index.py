import pandas as pd
import numpy as np

"""
vol_osc
volume_relative_strength
vwap_deviation
"""

def VolumeOscillator(df, short_period=5, long_period=20, volume_col="Volume"):
    '''
    计算Volume Oscillator
    :param df: 必须包含 [volume_col] 的 pandas DataFrame
    :param short_period: 短周期 (默认 5)
    :param long_period: 长周期 (默认 20)
    :param volume_col: 成交量列 (默认 'Volume')
    :return: df
    vol_osc > 0：成交量放大，可能趋势开始
    vol_osc < 0：成交量萎缩，可能震荡整理
    '''
    volume = df[volume_col]

    short_ma = volume.rolling(short_period).mean()
    long_ma = volume.rolling(long_period).mean()

    df['vol_osc'] = ((short_ma - long_ma) / long_ma) * 100

    return df


def VolumeRelativeStrength(df, period=14, volume_col="Volume"):
    '''
    计算成交量相对强度
    :param df: 必须包含 [volume_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :param volume_col: 成交量列 (默认 'Volume')
    :return: df
    '''
    volume = df[volume_col]

    # 成交量移动平均
    volume_ma = volume.rolling(period).mean()

    # 相对强度
    df['volume_relative_strength'] = volume / volume_ma
    df['volume_rs_smoothed'] = df['volume_relative_strength'].rolling(5).mean()

    return df


def VWAPDeviation(df, period=20, high_col="High", low_col="Low",
                  close_col="Close", volume_col="Volume"):
    '''
    计算成交量加权平均价格偏离度
    :param df: 必须包含相应列的 pandas DataFrame
    :param period: 计算周期 (默认 20)
    :return: df
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]

    # 典型价格
    typical_price = (high + low + close) / 3

    # VWAP计算
    vwap_numerator = (typical_price * volume).rolling(period).sum()
    vwap_denominator = volume.rolling(period).sum()
    vwap = vwap_numerator / vwap_denominator

    # 偏离度
    df['vwap_deviation'] = (close - vwap) / vwap * 100

    return df


def VolumeRateOfChange(df, period=12, volume_col="Volume"):
    '''
    计算Volume Rate of Change
    :param df: 必须包含 [volume_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 12)
    :param volume_col: 成交量列 (默认 'Volume')
    :return: df
    '''
    volume = df[volume_col]

    # Volume ROC
    df['volume_roc'] = ((volume - volume.shift(period)) / volume.shift(period)) * 100
    df['volume_roc'] = df['volume_roc'].fillna(0)

    return df



