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


def PriceJumpAnalysis(df, threshold_std=2, period=20, src_col="Close"):
    '''
    价格跳跃频率和幅度分析
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param threshold_std: 跳跃阈值(标准差倍数) (默认 2)
    :param period: 计算周期 (默认 20)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    # 计算收益率
    returns = src.pct_change()

    # 滚动标准差
    rolling_std = returns.rolling(period).std()

    # 识别跳跃
    jump_threshold = rolling_std * threshold_std
    jumps = np.abs(returns) > jump_threshold

    # 跳跃频率 (滚动窗口内跳跃次数)
    df['jump_frequency'] = jumps.rolling(period).sum() / period

    # 跳跃幅度 (滚动窗口内平均跳跃大小)
    jump_magnitude = np.abs(returns) * jumps
    df['jump_magnitude'] = jump_magnitude.rolling(period).mean()

    # 跳跃方向偏好 (向上跳跃 vs 向下跳跃)
    up_jumps = (returns > jump_threshold).rolling(period).sum()
    down_jumps = (returns < -jump_threshold).rolling(period).sum()
    total_jumps = up_jumps + down_jumps
    df['jump_direction_bias'] = np.where(total_jumps > 0,
                                         (up_jumps - down_jumps) / total_jumps, 0)

    return df


def QuantumPriceBehavior(df, period=20, src_col="Close"):
    '''
    价格行为的"量子"特征 - 基于价格的不确定性和观测效应
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 20)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def uncertainty_measure(series):
        if len(series) < 10:
            return np.nan

        # "测不准原理" - 价格和动量的不确定性乘积
        price_std = series.std()
        momentum_std = series.diff().std()

        uncertainty_product = price_std * momentum_std
        return uncertainty_product

    def coherence_measure(series):
        if len(series) < 10:
            return np.nan

        # 价格"相干性" - 基于自相关的衰减
        autocorr_sum = 0
        for lag in range(1, min(10, len(series) // 2)):
            try:
                autocorr = series.autocorr(lag=lag)
                if not np.isnan(autocorr):
                    autocorr_sum += abs(autocorr)
            except:
                continue

        return autocorr_sum

    df['quantum_uncertainty'] = src.rolling(period).apply(uncertainty_measure, raw=False)
    df['price_coherence'] = src.rolling(period).apply(coherence_measure, raw=False)

    return df


def MarketEfficiencyScore(df, period=30, src_col="Close"):
    '''
    市场效率综合得分
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 30)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def efficiency_score(series):
        if len(series) < 15:
            return np.nan

        returns = series.pct_change().dropna()
        if len(returns) < 10:
            return np.nan

        # 1. 随机游走检验 (方差比)
        var_1 = returns.var()
        var_2 = returns.rolling(2).sum().var() / 2
        variance_ratio = var_2 / var_1 if var_1 > 0 else 1

        # 2. 自相关检验
        autocorr_1 = returns.autocorr(lag=1)
        if np.isnan(autocorr_1):
            autocorr_1 = 0

        # 3. 收益分布正态性检验 (简化)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        normality_score = 1 / (1 + abs(skewness) + abs(kurtosis - 3))

        # 综合效率得分
        efficiency = (
                abs(1 - variance_ratio) * 0.4 +  # 方差比偏离1的程度
                abs(autocorr_1) * 0.3 +  # 自相关强度
                (1 - normality_score) * 0.3  # 非正态性程度
        )

        # 转换为效率得分 (0-1，1表示高效)
        efficiency_score = max(0, 1 - efficiency)

        return efficiency_score

    df['market_efficiency'] = src.rolling(period).apply(efficiency_score, raw=False)
    return df


def TrendQuality(df, period=14, src_col="Close"):
    '''
    趋势质量指标 - 综合多个趋势指标
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    # 线性回归R方 - 趋势拟合度
    def trend_r_squared(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        try:
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            return r_value ** 2
        except:
            return np.nan

    # 价格变化的一致性
    def price_consistency(series):
        if len(series) < 3:
            return np.nan
        changes = series.diff().dropna()
        if len(changes) == 0:
            return np.nan
        # 同向变化的比例
        positive_changes = np.sum(changes > 0)
        negative_changes = np.sum(changes < 0)
        consistency = abs(positive_changes - negative_changes) / len(changes)
        return consistency

    # 计算趋势质量组件
    df['trend_r2'] = src.rolling(period).apply(trend_r_squared, raw=False)
    df['price_consistency'] = src.rolling(period).apply(price_consistency, raw=False)

    # 综合趋势质量得分
    df['trend_quality'] = (df['trend_r2'] * 0.6 + df['price_consistency'] * 0.4)
    df['trend_quality'] = df['trend_quality'].fillna(0)

    return df
