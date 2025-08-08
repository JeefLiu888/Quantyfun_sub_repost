import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from datetime import date
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from backtesting import Strategy
from backtesting.lib import FractionalBacktest
from main import calc_factors
from TopCryptoTicker import Get_Tickers

pd.set_option('display.max_columns', None)
load_dotenv(fr"C:\Users\ljfgk\Desktop\telebot.env")

ticker_info = Get_Tickers().get_top_n_tickers_exclude_stablecoins(10)







def load_data(ticker_name, start, end, time_internal):
    df = yf.download(ticker_name, start=start, end=end, interval=time_internal, progress=False)

    if df.empty:
        print("Error! Can't reach data!")
        return pd.DataFrame()

    # 解决 MultiIndex 问题
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # 统一列名为首字母大写
    df.columns = [col.capitalize() for col in df.columns]

    if not all(col in df.columns for col in ["Close", "Open", "High", "Low", "Volume"]):
        print("下载的数据缺少关键列:", df.columns)
        return pd.DataFrame()

    df["OHLC"] = (df["Close"] + df["High"] + df["Low"] + df["Open"]) / 4
    return df



def calc_factors(df):
    """
    计算所需因子（需根据实际数据实现）
    :param df: 原始数据
    :return: 包含因子的DataFrame
    """
    # 示例因子计算（实际实现需替换为真实计算逻辑）
    df["ret_3d"] = df["OHLC"].pct_change(3)
    df["ret_7d"] = df["OHLC"].pct_change(7)
    df["ret_30d"] = df["OHLC"].pct_change(30)
    df["volatility_30d"] = df["OHLC"].pct_change().rolling(30).std() * np.sqrt(365)
    df["volume_change_7d"] = df["Volume"].pct_change(7)
    # NVT代理 = 市值 / (交易量 * 价格) 这个先获取不了
    return df


def add_multi_factor_score(df, params):
    """
    计算多因子综合得分
    :param df: 包含原始数据的DataFrame
    :param params: 包含因子权重和信号阈值的参数字典
    :return: 添加了'score'列的DataFrame
    """

    df = calc_factors()



