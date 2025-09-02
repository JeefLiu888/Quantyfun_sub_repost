import pandas as pd
import numpy as np


def FractalDimension(df, period=50, src_col="Close"):
    '''
    计算分形维数 - 测量价格序列复杂度
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    分形维数接近1：价格路径简单，趋势性强
    分形维数接近2：价格路径复杂，震荡性强
    维数1.5左右：随机游走
    '''
    src = df[src_col]

    def higuchi_fd(series, kmax=10):
        if len(series) < kmax:
            return np.nan

        N = len(series)
        lk = np.zeros(kmax)

        for k in range(1, kmax + 1):
            lm = np.zeros(int(np.floor((N - 1) / k)))
            for m in range(int(np.floor((N - 1) / k))):
                ll = 0
                for i in range(1, int(np.floor((N - m - 1) / k)) + 1):
                    ll += abs(series[m + i * k] - series[m + (i - 1) * k])
                lm[m] = ll * (N - 1) / (int(np.floor((N - m - 1) / k)) * k * k)
            lk[k - 1] = np.mean(lm)

        # 线性回归计算分形维数
        loglk = np.log(lk)
        logk = np.log(range(1, kmax + 1))

        slope, _ = np.polyfit(logk, loglk, 1)
        return -slope

    df['fractal_dim'] = src.rolling(period).apply(lambda x: higuchi_fd(x.values), raw=False)
    return df


def PriceDensityDistribution(df, period=50, bins=10, src_col="Close"):
    '''
    计算价格密度分布
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param bins: 分布区间数 (默认 10)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    高熵值：价格分布均匀，震荡市场
    低熵值：价格集中分布，可能突破
    '''
    src = df[src_col]

    def price_density(series):
        if len(series) < 10:
            return np.nan

        hist, _ = np.histogram(series, bins=bins, density=True)
        # 返回熵作为价格分布的复杂度度量
        hist = hist[hist > 0]  # 避免log(0)
        return -np.sum(hist * np.log(hist))

    df['price_density'] = src.rolling(period).apply(price_density, raw=False)
    return df


def SupportResistanceStrength(df, period=20, threshold_pct=0.02,
                              high_col="High", low_col="Low", close_col="Close"):
    '''
    计算支撑阻力突破强度
    :param df: 必须包含 [high_col, low_col, close_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 20)
    :param threshold_pct: 突破阈值百分比 (默认 2%)
    :return: df
    break_strength > 0：向上突破阻力，趋势信号
    break_strength < 0：向下跌破支撑，趋势信号
    在支撑阻力间震荡：震荡市场
    '''
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # 阻力位 (最高价)
    resistance = high.rolling(period).max()
    # 支撑位 (最低价)
    support = low.rolling(period).min()

    # 突破强度
    resistance_break = np.where(close > resistance.shift(1) * (1 + threshold_pct),
                                (close - resistance.shift(1)) / resistance.shift(1), 0)
    support_break = np.where(close < support.shift(1) * (1 - threshold_pct),
                             (support.shift(1) - close) / support.shift(1), 0)

    df['resistance_level'] = resistance
    df['support_level'] = support
    df['break_strength'] = resistance_break - support_break

    return df


def PriceActionScore(df, period=14, high_col="High", low_col="Low",
                     open_col="Open", close_col="Close"):
    '''
    量化价格行为模式得分
    :param df: 必须包含 [high_col, low_col, open_col, close_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 14)
    :return: df
    pa_momentum持续上升：上升趋势
    pa_momentum持续下降：下降趋势
    pa_momentum在零轴附近震荡：震荡市场
    '''
    high = df[high_col]
    low = df[low_col]
    open_price = df[open_col]
    close = df[close_col]

    # K线实体大小
    body_size = abs(close - open_price) / (high - low)
    body_size = body_size.fillna(0)

    # 上影线长度
    upper_shadow = (high - np.maximum(open_price, close)) / (high - low)
    upper_shadow = upper_shadow.fillna(0)

    # 下影线长度
    lower_shadow = (np.minimum(open_price, close) - low) / (high - low)
    lower_shadow = lower_shadow.fillna(0)

    # K线方向 (1: 阳线, -1: 阴线)
    candle_direction = np.sign(close - open_price)

    # 价格行为得分 (综合实体大小和方向)
    df['pa_body_size'] = body_size
    df['pa_upper_shadow'] = upper_shadow
    df['pa_lower_shadow'] = lower_shadow
    df['pa_score'] = body_size * candle_direction
    df['pa_momentum'] = df['pa_score'].rolling(period).mean()

    return df


def ZigzagAmplitude(df, threshold_pct=5, src_col="Close"):
    '''
    计算Zigzag指标的摆动幅度
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param threshold_pct: 反转阈值百分比 (默认 5%)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]
    threshold = threshold_pct / 100

    # 简化的Zigzag实现
    zigzag = []
    peaks_valleys = []
    current_trend = 0  # 0: 未定义, 1: 上升, -1: 下降
    last_extreme = src.iloc[0]
    last_extreme_idx = 0

    for i, price in enumerate(src):
        if np.isnan(price):
            continue

        if current_trend == 0:  # 初始化
            if price > last_extreme * (1 + threshold):
                current_trend = 1
                peaks_valleys.append((last_extreme_idx, last_extreme, 'valley'))
                last_extreme = price
                last_extreme_idx = i
            elif price < last_extreme * (1 - threshold):
                current_trend = -1
                peaks_valleys.append((last_extreme_idx, last_extreme, 'peak'))
                last_extreme = price
                last_extreme_idx = i

        elif current_trend == 1:  # 上升趋势
            if price > last_extreme:
                last_extreme = price
                last_extreme_idx = i
            elif price < last_extreme * (1 - threshold):
                peaks_valleys.append((last_extreme_idx, last_extreme, 'peak'))
                current_trend = -1
                last_extreme = price
                last_extreme_idx = i

        elif current_trend == -1:  # 下降趋势
            if price < last_extreme:
                last_extreme = price
                last_extreme_idx = i
            elif price > last_extreme * (1 + threshold):
                peaks_valleys.append((last_extreme_idx, last_extreme, 'valley'))
                current_trend = 1
                last_extreme = price
                last_extreme_idx = i

    # 计算摆动幅度
    amplitudes = []
    for i in range(1, len(peaks_valleys)):
        prev_price = peaks_valleys[i - 1][1]
        curr_price = peaks_valleys[i][1]
        amplitude = abs(curr_price - prev_price) / prev_price
        amplitudes.append(amplitude)

    # 将振幅映射回原序列
    df['zigzag_amplitude'] = 0.0
    if len(amplitudes) > 0:
        avg_amplitude = np.mean(amplitudes)
        df['zigzag_amplitude'] = avg_amplitude

    return df


def ElliottWaveCount(df, period=100, src_col="Close"):
    '''
    简化的Elliott波浪计数 - 基于价格摆动
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 100)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def elliott_pattern_score(series):
        if len(series) < 50:
            return np.nan

        # 寻找波峰波谷
        from scipy.signal import find_peaks

        # 寻找局部最大值和最小值
        peaks, _ = find_peaks(series.values, distance=5)
        valleys, _ = find_peaks(-series.values, distance=5)

        # 合并并排序关键点
        turning_points = np.concatenate([peaks, valleys])
        turning_points.sort()

        if len(turning_points) < 5:
            return 0

        # 分析波浪长度比率
        wave_lengths = []
        for i in range(1, min(6, len(turning_points))):  # 最多看前5波
            start_idx = turning_points[i - 1]
            end_idx = turning_points[i]
            wave_length = abs(series.iloc[end_idx] - series.iloc[start_idx])
            wave_lengths.append(wave_length)

        if len(wave_lengths) < 3:
            return 0

        # 检查是否符合Elliott波浪比率
        # 第3波通常是最长的，第2波和第4波相对较短
        elliott_score = 0

        # 简化的评分系统
        if len(wave_lengths) >= 3:
            # 第3波 > 第1波
            if wave_lengths[2] > wave_lengths[0]:
                elliott_score += 0.3

            # 第2波和第4波相对较小
            if len(wave_lengths) >= 4:
                avg_impulse = (wave_lengths[0] + wave_lengths[2]) / 2
                if wave_lengths[1] < avg_impulse * 0.8:
                    elliott_score += 0.2
                if wave_lengths[3] < avg_impulse * 0.8:
                    elliott_score += 0.2

        return elliott_score

    df['elliott_wave_score'] = src.rolling(period).apply(elliott_pattern_score, raw=False)
    return df


