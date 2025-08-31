from http.client import responses
import requests
from typing import List, Dict, Union
from fontTools.misc.cython import returns


class Get_Tickers:

    def __init__(self):
        self.url = "https://api.coingecko.com/api/v3/coins/markets"

        #here contains also wrapped asset
        self.stablecoins = {'usdt','usdc','busd','dai','tusd','wbeth','weeth','wsteth','usdp','usdy','steth','weth','wbtc'}

    def get_top_n_tickers_exclude_stablecoins(self, n: int) -> List[Dict[str, Union[str,float]]]:
        """获取前n个非稳定币的加密货币交易对及其市值信息
        参数:
        n: 要获取的非稳定币数量
        返回:
        包含交易对和市值信息的字典列表
        格式: [{"ticker": "BTC-USD", "market_cap": 1234567890}, ...]
        """

        # get top_n tickers

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": n * 2,  # request for more pages, insure enough data
            "page": 1,
            "sparkline": False
        }

        response = requests.get(self.url, params=params)

        if response.status_code == 200:
            data = response.json()

            non_stablecoins = [
                coin for coin in data if coin['symbol'].lower() not in self.stablecoins
            ][:n]

            results  = [
                {
                    "ticker":f"{coin['symbol'].upper()}",
                    "market_cap":float(coin['market_cap']) if coin['market_cap'] is not None else 0.0
                } for coin in non_stablecoins
            ]
            print(results)

            ticker_symbol = []
            for item in results:
                base = item["ticker"].upper()
                symbol = base+"USDT"
                ticker_symbol.append(symbol)

            print(ticker_symbol)
            return [results,ticker_symbol]

        else:
            print(f"Error:{response.status_code}")
            return []
