import pandas as pd

pd.set_option('display.max_columns', None)


# 假设 test_df 已经包含所有字段，并已读入
df = pd.read_csv("test_df_result.csv").reset_index()




'''
# 提取信号位置
buy_idxs = df.index[df['ut_bot_buy_signal'] == True].tolist()
sell_idxs = df.index[df['ut_bot_sell_signal'] == True].tolist()

# 有效交易对列表
trades = []
sell_ptr = 0

for buy_idx in buy_idxs:
    # 找到最近买入后的 sell 信号
    sells_after_buy = [s for s in sell_idxs if s > buy_idx]
    if sells_after_buy:
        sell_idx = sells_after_buy[0]
        trade = {
            'buy_date': df.loc[buy_idx, 'Date'],
            'sell_date': df.loc[sell_idx, 'Date'],
            'buy_adx': df.loc[buy_idx, 'adx'],
            'sell_adx': df.loc[sell_idx, 'adx'],
            'buy_price': df.loc[buy_idx, 'Close'],
            'sell_price': df.loc[sell_idx, 'Close'],
            'profit': df.loc[sell_idx, 'Close'] - df.loc[buy_idx, 'Close'],
            'buy_idx': buy_idx,
            'sell_idx': sell_idx
        }
        trades.append(trade)

# 转为 DataFrame
trades_df = pd.DataFrame(trades)
# 筛出有效（定义为盈利的）交易
trades_df['valid'] = trades_df['profit'] > 0
valid_trades = trades_df[trades_df['valid']]

print(valid_trades[['buy_date', 'sell_date', 'buy_adx', 'sell_adx', 'profit']])


df.to_excel('full_result.xlsx', index=False)

# 保存有效交易表
valid_trades[['buy_date', 'buy_adx', 'sell_date', 'sell_adx', 'profit']].to_excel('valid_trades_adx.xlsx', index=False)







##############################每一组buy sell对应的adx #####################333333
# 假设 df 是你的交易 DataFrame，索引为日期或有 Date 列
df = df.reset_index()  # 若 index 为时间戳，则保留

buy_indices = df.index[df['ut_bot_buy_signal'] == True].tolist()
sell_indices = df.index[df['ut_bot_sell_signal'] == True].tolist()

results = []
for b_idx in buy_indices:
    # 找到后面第一个 sell_idx
    after_sells = [s for s in sell_indices if s > b_idx]
    if after_sells:
        s_idx = after_sells[0]
        results.append({
            'buy_date': df.loc[b_idx, 'Date'] if 'Date' in df.columns else df.loc[b_idx].name,
            'sell_date': df.loc[s_idx, 'Date'] if 'Date' in df.columns else df.loc[s_idx].name,
            'buy_adx': df.loc[b_idx, 'adx'],
            'sell_adx': df.loc[s_idx, 'adx'],
            'buy_price': df.loc[b_idx, 'Close'],
            'sell_price': df.loc[s_idx, 'Close'],
            'profit': df.loc[s_idx, 'Close'] - df.loc[b_idx, 'Close']
        })

trade_adx_df = pd.DataFrame(results)
print(trade_adx_df)
'''




#################################
def pair_trades_strict(df):
    trades = []
    position = 0  # 0 空仓，1 持仓
    buy_idx = None

    # 遍历所有行，按时间顺序模拟交易仓位
    for i, row in df.iterrows():
        if position == 0 and row['ut_bot_buy_signal']:
            # 空仓时遇买信号，开仓买入
            buy_idx = i
            position = 1
        elif position == 1 and row['ut_bot_sell_signal']:
            # 持仓时遇卖信号，平仓卖出
            sell_idx = i
            profit = df.at[sell_idx, 'Close'] - df.at[buy_idx, 'Close']
            profit_perc = (df.at[sell_idx, 'Close'] - df.at[buy_idx, 'Close'])/df.at[buy_idx, 'Close']*100
            trades.append({
                'buy_date': df.at[buy_idx, 'Date'] if 'Date' in df.columns else df.index[buy_idx],
                'sell_date': df.at[sell_idx, 'Date'] if 'Date' in df.columns else df.index[sell_idx],
                'buy_adx': df.at[buy_idx, 'adx'],
                'sell_adx': df.at[sell_idx, 'adx'],
                'buy_price': df.at[buy_idx, 'Close'],
                'sell_price': df.at[sell_idx, 'Close'],
                'profit': profit,
                'profit %': profit_perc
            })
            position = 0
            buy_idx = None

    return pd.DataFrame(trades)

# 使用示例
df = df.reset_index()  # 确保有按顺序的行号索引，且Date列存在

trades_df = pair_trades_strict(df)

print(f"交易对数量: {len(trades_df)}")
print(trades_df)
print(trades_df['profit'].sum())

trades_df.to_csv('paired_trades_strict.csv', index=False)



