import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import requests
import datetime
import mplfinance as mpf
from datetime import date, timedelta
from dotenv import load_dotenv
from lifelines.fitters.npmle import interval
from sklearn.preprocessing import StandardScaler
from backtesting import Backtest, Strategy
import ta
import sqlite3

import trend_strength_index
from TopCryptoTicker import Get_Tickers
from strategies import MyStrategy
import Basic_indicator

pd.set_option('display.max_columns', None)
load_dotenv(fr"C:\Users\ljfgk\Desktop\telebot.env")


# this returns top n ticker for crypto market (excludes stablecoins and wrapped asset)
top_n = 10
top_n_tickers = Get_Tickers().get_top_n_tickers_exclude_stablecoins(top_n)




#############################  Load the basic dataset #################################################################

#here with yfinance, free, but with many restrictions
def load_data(ticker_name, start, end, interval):
    df = yf.download(ticker_name+'-USD', start=start, end=end, interval=interval, progress=False, auto_adjust=False)

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

    df.to_csv(fr'{ticker_name}_data.csv')
    return df



test_ticker = 'BTC'

test_df = load_data(test_ticker, start= '2023-01-01', end = date.today(), interval= '1d')
test_df.to_csv(f'test_df_{test_ticker}')


Basic_indicator.UT_BOT_alerts(test_df,a=2,atr_period=1,use_heikin= False)
Basic_indicator.ADX(test_df,14,14)

trend_strength_index.AroonOscillator(test_df)
trend_strength_index.MACD(test_df)

test_df.to_csv('test_df_result.csv')
#print(test_df)



'''
adx_plot = mpf.make_addplot(test_df['adx'], panel=1, color='red', ylabel='ADX')
mpf.plot(
    test_df,
    type='candle',
    addplot=adx_plot,
    volume=False,
    style='yahoo',
    title='K线与ADX',
    mav=(),           # 可选: 添加均线
    figratio=(12,8)   # 图宽高比
)
'''



'''
# WTF 神了个大奇 yfinance获取数据居然缺了一天， 25年8月6号直接消失了 我日
Basic_indicator.ATR(load_data('BTC', start='2023-01-01', end= date.today(), interval='1d'),14, smoothing='EMA')
Basic_indicator.ut_bot_alerts(load_data('BTC', start='2023-01-01', end= date.today(), interval='1d'),1.5, 14, True )
'''




'''

########################### backtest test ##############################
test_df = pd.read_csv('test_df_result.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df.set_index('Date', inplace=True)
test_df['buy_signal'] = test_df['ut_bot_buy_signal'].astype(bool)
test_df['sell_signal'] = test_df['ut_bot_sell_signal'].astype(bool)



################# test strategy ######################
bt = Backtest(
    test_df,
    MyStrategy,
    cash=1000000,
    commission=0.001,
    exclusive_orders=True
)
stats = bt.run()
print(stats)
bt.plot()

'''

'''
################# strategy optimization ###########################
a_values = np.arange(1, 3.1, 0.1)        # 一维浮点数组
atr_period_values = list(range(10, 20)) # 一维整数列表

print(type(a_values), a_values.shape)
print(type(atr_period_values), len(atr_period_values))



stats = bt.optimize(
    a=a_values.tolist(),
    atr_period=atr_period_values,
    maximize='Sharpe Ratio'
)

print(stats)
bt.plot()

'''


if __name__ == "__main__":
    print('ff15')


