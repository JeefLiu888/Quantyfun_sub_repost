import pandas as pd
import numpy as np
import volume_osc_index


df = pd.read_csv('BTC_data.csv')
short_range = (1,10)
long_range = (5,30)
step = 1



# 生成 Volume Oscillator 因子 (如果模块可用)
new_cols = []
for short in range(short_range[0], short_range[1], step):
    for long in range(long_range[0], long_range[1], step):
        if long <= short:
            continue
        print(f"生成因子: VolumeOsc_s{short}_l{long}")
        # 调用函数，计算新的振荡器列，但不直接赋值给df
        temp_df = volume_osc_index.VolumeOscillator(df.copy(), short_period=short, long_period=long)
        col_name = f'vol_osc_short{short}_long{long}'
        # 收集这一列数据
        new_cols.append(temp_df[col_name])

# 一次性合并所有新列，避免多次单列插入
df = pd.concat([df] + new_cols, axis=1)

base_cols = ['Unnamed: 0', "Date", "Adj close", "Close", "High", "Low", "Open", "Volume", "OHLC"]
factor_cols = [col for col in df.columns if col not in base_cols]

def zscore(df, cols):
    return df[cols].apply(lambda x: (x - x.mean()) / x.std())
df_zscore = zscore(df, factor_cols)


for col in df_zscore.columns:
    df[col + '_zscore'] = df_zscore[col]

df.to_csv('test_loop_data_zscore.csv', index=False)

