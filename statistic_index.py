import pandas as pd
import numpy as np


def AutoCorrelation(df, lags=20, src_col="Close"):
    '''
    计算自相关系数
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param lags: 滞后期数 (默认 20)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    for lag in range(1, lags + 1):
        df[f'autocorr_{lag}'] = src.rolling(50).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
        )

    return df


def RunsTest(df, period=50, src_col="Close"):
    '''
    计算Runs Test统计量 - 检验随机性
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def runs_test(series):
        if len(series) < 10:
            return np.nan

        # 计算收益率
        returns = series.pct_change().dropna()
        if len(returns) < 5:
            return np.nan

        # 转换为二进制序列
        median_return = returns.median()
        binary_seq = (returns > median_return).astype(int)

        # 计算runs
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq.iloc[i] != binary_seq.iloc[i - 1]:
                runs += 1

        # 期望runs数
        n1 = np.sum(binary_seq)
        n2 = len(binary_seq) - n1

        if n1 == 0 or n2 == 0:
            return np.nan

        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                        ((n1 + n2) ** 2 * (n1 + n2 - 1))

        if variance_runs <= 0:
            return np.nan

        # Z统计量
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        return z_stat

    df['runs_test'] = src.rolling(period).apply(runs_test, raw=False)
    return df


def VarianceRatio(df, period=50, k=2, src_col="Close"):
    '''
    计算方差比检验 (Lo-MacKinlay)
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param k: 方差比的k值 (默认 2)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def variance_ratio_test(series, k=k):
        if len(series) < k * 5:
            return np.nan

        # 对数收益率
        log_returns = np.log(series / series.shift(1)).dropna()

        if len(log_returns) < k * 3:
            return np.nan

        # k期收益率方差
        k_period_returns = log_returns.rolling(k).sum().dropna()
        var_k = k_period_returns.var()

        # 1期收益率方差
        var_1 = log_returns.var()

        if var_1 == 0:
            return np.nan

        # 方差比
        vr = var_k / (k * var_1)
        return vr

    df['variance_ratio'] = src.rolling(period).apply(variance_ratio_test, raw=False)
    return df


def LjungBoxTest(df, period=50, lags=10, src_col="Close"):
    '''
    计算Ljung-Box统计量 - 序列相关性检验
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param lags: 滞后阶数 (默认 10)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def ljung_box_stat(series, h=lags):
        if len(series) < h + 10:
            return np.nan

        # 计算收益率
        returns = series.pct_change().dropna()
        if len(returns) < h + 5:
            return np.nan

        n = len(returns)
        # 自相关系数
        autocorrs = [returns.autocorr(lag=i) for i in range(1, h + 1)]
        autocorrs = [ac for ac in autocorrs if not np.isnan(ac)]

        if len(autocorrs) == 0:
            return np.nan

        # Ljung-Box统计量
        lb_stat = n * (n + 2) * sum([(ac ** 2) / (n - k) for k, ac in enumerate(autocorrs, 1)])
        return lb_stat

    df['ljung_box'] = src.rolling(period).apply(ljung_box_stat, raw=False)
    return df


def KPSSTest(df, period=50, src_col="Close"):
    '''
    计算KPSS检验统计量 - 平稳性检验
    :param df: 必须包含 [src_col] 的 pandas DataFrame
    :param period: 计算周期 (默认 50)
    :param src_col: 用于计算的价格列 (默认 'Close')
    :return: df
    '''
    src = df[src_col]

    def kpss_stat(series):
        if len(series) < 20:
            return np.nan

        # 去趋势
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        detrended = series - (slope * x + intercept)

        # 累积和
        cumsum = np.cumsum(detrended)

        # 长期方差估计
        n = len(series)
        s2 = np.sum(detrended ** 2) / n

        # KPSS统计量
        kpss = np.sum(cumsum ** 2) / (n ** 2 * s2)
        return kpss

    df['kpss_stat'] = src.rolling(period).apply(kpss_stat, raw=False)
    return df






