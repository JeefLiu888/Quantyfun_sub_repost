import os
import telegram
import requests
import pandas as pd
from dotenv import load_dotenv
import logging



load_dotenv(fr"C:\Users\ljfgk\Desktop\telebot.env")


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deribit_whale_alert.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


#telegram setting check for OI checker
print("telegram token:", os.getenv("TELEGRAM_BOT_TOKEN"))
print("telegram chat id:", os.getenv("TELEGRAM_CHAT_ID"))



#here need to add the ticker that you wanna inspec

class DerbitWhaleAlert:

    def __init__(self):

        self.deribit_url = "https://deribit.com/api/v2/public"
        self.binance_url = "https://api.binance.com/api/v3"
        self.last_trade_id_btc = None
        self.last_trade_id_eth = None
        self.processed_trades = set()








