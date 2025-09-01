import pandas as pd
import numpy as np


def DominantCyclePeriod(df, period=50, src_col="Close"):
    '''
    计算主导周期长度
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def dominant_cycle(series):
        if len(series) < 20:
            return np.nan

        # 使用FFT找到主导频率
        detrended = series - series.mean()
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(series))

        # 找到最大幅度对应的频率
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude) // 2]) + 1
        dominant_freq = freqs[dominant_freq_idx]

        if dominant_freq == 0:
            return np.nan

        # 转换为周期
        dominant_period = 1 / abs(dominant_freq)
        return dominant_period

    df['dominant_cycle'] = src.rolling(period).apply(dominant_cycle, raw=False)
    return df


def KaufmanEfficiencyRatio(df, period=14, src_col="Close"):
    '''
    计算Kaufman's Efficiency Ratio
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    # Direction
    direction = abs(src - src.shift(period))

    # Volatility
    volatility = abs(src - src.shift(1)).rolling(period).sum()

    # Efficiency Ratio
    df['efficiency_ratio'] = direction / volatility
    df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0)

    return df


def MomentumQuality(df, period=14, close_col="Close", volume_col="Volume"):
    '''
    计算动量质量指标 - 结合价格和成交量的动量强度
    :param df: 必须包含 [close_col, volume_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :return: df
    '''
    close = df[close_col]
    volume = df[volume_col]

    # 价格动量
    price_momentum = (close - close.shift(period)) / close.shift(period)

    # 成交量相对强度
    volume_ratio = volume / volume.rolling(period).mean()

    # 动量质量 = 价格动量 * 成交量权重
    df['momentum_quality'] = price_momentum * np.log(volume_ratio)
    df['momentum_quality'] = df['momentum_quality'].fillna(0)

    return df


def TrendConsistency(df, short_period=5, med_period=14, long_period=30, src_col="Close"):
    '''
    计算趋势一致性指标 - 多时间框架趋势方向一致度
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param short_period: 短期周期 (默认 5)
    :param med_period: 中期周期 (默认 14)
    :param long_period: 长期周期 (默认 30)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df['src_col']

    # 不同周期的趋势方向
    short_trend = np.sign(src - src.shift(short_period))
    med_trend = np.sign(src - src.shift(med_period))
    long_trend = np.sign(src - src.shift(long_period))

    # 一致性得分 (-3 到 3)
    df['trend_consistency'] = short_trend + med_trend + long_trend

    # 归一化到 0-1
    df['trend_consistency_norm'] = (df['trend_consistency'] + 3) / 6

    return df
