import requests
import yfinance as yf


class PremiumChecker:
    def __init__(self, ticker):
        self.ticker = ticker
        self.binance_symbol = f"{ticker}USDT"
        self.coinbase_symbol = f"{ticker}-USD"
        self.upbit_symbol = f"KRW-{ticker}"
        self.cme_symbol = f"{ticker}=F"

    def get_binance_price(self):
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.binance_symbol}"
        return float(requests.get(url).json()["price"])

    def get_coinbase_price(self):
        url = f"https://api.exchange.coinbase.com/products/{self.coinbase_symbol}/ticker"
        return float(requests.get(url).json()["price"])

    def get_usd_to_krw(self):
        url = "https://open.er-api.com/v6/latest/USD"
        data = requests.get(url).json()
        if "rates" in data and "KRW" in data["rates"]:
            return float(data["rates"]["KRW"])
        else:
            raise ValueError(f"无法获取 USD/KRW 汇率，返回数据: {data}")

    def get_upbit_price(self):
        url = f"https://api.upbit.com/v1/ticker?markets={self.upbit_symbol}"
        return float(requests.get(url).json()[0]["trade_price"])

    def get_cme_futures_price(self):
        try:
            ticker = yf.Ticker(self.cme_symbol)
            price = ticker.history(period="1d")["Close"].iloc[-1]
            return float(price)
        except Exception as e:
            print(f"无法获取 CME 期货价格: {e}")
            return None

    def get_coinbase_premium(self):
        cb = self.get_coinbase_price()
        bn = self.get_binance_price()
        diff = cb - bn
        return {
            "coinbase": cb,
            "binance": bn,
            "premium": diff,
            "premium_pct": diff / bn * 100
        }

    def get_korea_premium(self):
        up = self.get_upbit_price()
        bn = self.get_binance_price()
        fx = self.get_usd_to_krw()
        bn_krw = bn * fx
        diff = up - bn_krw
        return {
            "upbit": up,
            "krw_binance": bn_krw,
            "fx": fx,
            "premium": diff,
            "premium_pct": diff / bn_krw * 100
        }

    def get_cme_premium(self):
        cme = self.get_cme_futures_price()
        bn = self.get_binance_price()
        if cme is None:
            return None
        diff = cme - bn
        return {
            "cme": cme,
            "binance": bn,
            "premium": diff,
            "premium_pct": diff / bn * 100
        }


if __name__ == "__main__":
    checker = PremiumChecker('ADA')
    print("Coinbase premium：", checker.get_coinbase_premium())
    print("Upbit premium：", checker.get_korea_premium())
    cme_p = checker.get_cme_premium()
    print("CME future premium：", cme_p if cme_p else "无法获取 CME 数据")
