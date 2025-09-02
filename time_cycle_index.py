import pandas as pd
import numpy as np


def CyclicalityStrength(df, period=50, src_col="Close"):
    '''
    周期性强度指标
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def cyclical_strength(series):
        if len(series) < 20:
            return np.nan

        # 使用FFT检测周期性
        detrended = series - series.mean()
        fft_result = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(series))

        # 功率谱
        power_spectrum = np.abs(fft_result) ** 2

        # 找到最强的周期性成分
        # 排除DC成分 (频率=0)
        valid_indices = np.where(frequencies > 0)[0][:len(frequencies) // 2]

        if len(valid_indices) == 0:
            return np.nan

        max_power_idx = valid_indices[np.argmax(power_spectrum[valid_indices])]
        max_power = power_spectrum[max_power_idx]
        total_power = np.sum(power_spectrum[valid_indices])

        # 周期性强度 = 最大功率 / 总功率
        cyclical_strength = max_power / total_power if total_power > 0 else 0
        return cyclical_strength

    df['cyclical_strength'] = src.rolling(period).apply(cyclical_strength, raw=False)
    return df


def IntradayMomentumIndex(df, period=14, src_col="Close"):
    '''
    IntradayMomentumIndex - 适用于有开盘价的数据
    :param df: 必须包含 [src_col] 和 'Open' 的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :param src_col: 收盘价列 (默认 'Close')
    :return: df
    '''
    close = df[src_col]

    # 如果没有开盘价，使用前一日收盘价作为替代
    if 'Open' in df.columns:
        open_price = df['Open']
    else:
        open_price = close.shift(1)
        print("警告: 没有开盘价列，使用前一日收盘价作为替代")

    # 日内动量
    intraday_momentum = close - open_price

    # RSI类型的计算
    gains = intraday_momentum.where(intraday_momentum > 0, 0)
    losses = -intraday_momentum.where(intraday_momentum < 0, 0)

    avg_gains = gains.rolling(period).mean()
    avg_losses = losses.rolling(period).mean()

    rs = avg_gains / avg_losses
    df['intraday_momentum_index'] = 100 - (100 / (1 + rs))
    df['intraday_momentum_index'] = df['intraday_momentum_index'].fillna(50)

    return df


def HilbertTransform(df, period=50, src_col="Close"):
    '''
    Hilbert变换相位
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def hilbert_phase(series):
        if len(series) < 20:
            return np.nan

        # 简化的Hilbert变换 - 使用90度相移
        # 实际应用中需要更复杂的实现
        analytic_signal = series.values + 1j * np.gradient(series.values)
        instantaneous_phase = np.angle(analytic_signal)

        return instantaneous_phase[-1]

    df['hilbert_phase'] = src.rolling(period).apply(hilbert_phase, raw=False)
    return df


def SpectralAnalysis(df, period=50, src_col="Close"):
    '''
    谱分析功率谱密度
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def power_spectral_density(series):
        if len(series) < 16:
            return np.nan

        # 去除趋势
        detrended = series - series.mean()

        # FFT
        fft_result = np.fft.fft(detrended)
        power_spectrum = np.abs(fft_result) ** 2

        # 返回功率谱的能量
        return np.sum(power_spectrum[:len(power_spectrum) // 2])

    df['spectral_power'] = src.rolling(period).apply(power_spectral_density, raw=False)
    return df


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