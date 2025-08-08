import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone
from TopCryptoTicker import Get_Tickers






BASE_URL = "https://fapi.binance.com/futures/data/openInterestHist"
DB_PATH = "btc_oi_db.csv"


def fetch_open_interest(symbol="BTCUSDT", period="1h", limit=500, start_time=None, end_time=None):
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }
    if start_time:
        params["startTime"] = int(start_time.timestamp() * 1000)
    if end_time:
        params["endTime"] = int(end_time.timestamp() * 1000)

    print(f"请求URL: {BASE_URL}", params)
    r = requests.get(BASE_URL, params=params)
    print(f"返回状态: {r.status_code}")
    if r.status_code != 200:
        print(f"请求失败: {r.text}")
        return []
    try:
        data = r.json()
    except Exception as e:
        print(f"无法解析JSON: {e}, {r.text}")
        return []
    print(f"返回数量: {len(data)}")
    return data


def update_oi_database(symbol="BTCUSDT", period="1h", lookback_days=10):
    now = datetime.now(timezone.utc)

    # 第一步：尝试读取已有数据库文件
    if os.path.exists(DB_PATH):
        df_existing = pd.read_csv(DB_PATH, parse_dates=["timestamp"])
        last_ts = df_existing["timestamp"].max()
        start_time = last_ts + timedelta(hours=1)  # 避免重复
        print(f"从数据库中读取最新时间: {last_ts}，从 {start_time} 开始增量更新")
    else:
        df_existing = pd.DataFrame()
        start_time = now - timedelta(days=lookback_days)
        print(f"数据库不存在，将从过去 {lookback_days} 天开始抓取")

    end_time = now
    step = timedelta(days=10)  # 每次最多拉10天

    all_data = []
    current_start = start_time

    while current_start < end_time:
        current_end = min(current_start + step, end_time)
        data = fetch_open_interest(symbol, period, 500, current_start, current_end)
        if not data:
            print(f"当前区间{current_start}~{current_end}拉取为空，跳过")
            current_start = current_end
            time.sleep(0.3)
            continue

        all_data.extend(data)
        current_start = current_end
        time.sleep(0.3)

    # 第二步：构建新数据 DataFrame 并去重
    df_new = pd.DataFrame(all_data)
    if df_new.empty:
        print("无新数据，退出")
        return

    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
    df_new = df_new.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    else:
        df_combined = df_new

    df_combined.to_csv(DB_PATH, index=False)
    print(f"数据库已更新，当前总共 {len(df_combined)} 条记录，已保存为 {DB_PATH}")


if __name__ == "__main__":
    update_oi_database("BTCUSDT", "1h")
