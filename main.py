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
from TopCryptoTicker import Get_Tickers
import ta




pd.set_option('display.max_columns', None)
load_dotenv(fr"C:\Users\ljfgk\Desktop\telebot.env")


# this returns top n ticker for crypto market (excludes stablecoins and wrapped asset)
top_n = 10
top_n_tickers = Get_Tickers().get_top_n_tickers_exclude_stablecoins(top_n)




#############################  Load the basic dataset #################################################################

#here with yfinance, free, but with many restrictions

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
########################################################################################################################





'''
def calc_factors(df):
    df = df.copy()
    df["ret_7d"] = df["Close"].pct_change(7)
    df["ret_30d"] = df["Close"].pct_change(30)
    df["volatility_30d"] = df["Close"].pct_change().rolling(30).std()
    df["volume_change_7d"] = df["Volume"].pct_change(7)
    df["nvt_proxy"] = df["Close"] / (df["Volume"] + 1e-9)
    return df


def normalize_factors(df, factor_list):
    scaler = StandardScaler()
    df[factor_list] = scaler.fit_transform(df[factor_list])
    return df


def add_multi_factor_score(df):
    df = calc_factors(df)
    factor_list = ["ret_7d", "ret_30d", "volatility_30d", "volume_change_7d", "nvt_proxy"]
    df = normalize_factors(df, factor_list)

    weights = {"ret_7d": 0.3,
               "ret_30d": 0.3,
               "volatility_30d": -0.2,
               "volume_change_7d": 0.1,
               "nvt_proxy": -0.1}

    df["score"] = sum(df[f] * w for f, w in weights.items())
    return df


class MultiFactorStrategy(Strategy):
    def init(self):
        self.scores = self.I(lambda: self.data.df["score"])

    def next(self):
        score = self.scores[-1]
        if score > 0:
            if not self.position.is_long:
                self.position.close()
                self.buy()
        else:
            if not self.position.is_short:
                self.position.close()
                self.sell()

'''






'''
if __name__ == "__main__":
    start = "2023-01-01"
    end = str(date.today())
    ticker = "BTC-USD"

    df = load_data(ticker, start, end, "1d")
    df = add_multi_factor_score(df)

    bt = FractionalBacktest(df, MultiFactorStrategy, cash=10000, commission=0.001)
    stats = bt.run()

    # 定义资金曲线变量
    df_bt = bt._results._equity_curve

    bt.plot()

    trades = stats._trades

    if not trades.empty:
        buy_trades = trades[trades['Size'] > 0]
        sell_trades = trades[trades['Size'] < 0]

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label="BTC Close Price", color="gray")

        plt.scatter(buy_trades['EntryTime'], buy_trades['EntryPrice'], marker='^', color='green', s=100, label="Buy")
        plt.scatter(sell_trades['ExitTime'], sell_trades['ExitPrice'], marker='v', color='red', s=100, label="Sell")

        plt.legend()
        plt.title("BTC Close Price with Buy/Sell Signals")
        plt.show()
    else:
        print("No trades generated, skipping trade plot.")

    print(stats)
    
'''

'''
def OI_checker_binance(ticker):
    url = "https://fapi.binance.com/fapi/v1/openInterest"

    def get_open_interest(ticker):
        params = {"symbol": ticker}
        response = requests.get(url, params=params)
        data = response.json()
        if "openInterest" in data:
            return float(data["openInterest"])
        else:
            print("Error:", data)
            return None
'''