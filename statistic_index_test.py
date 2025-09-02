import pandas as pd
import numpy as np
import statistic_index
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('BTC_data.csv')

df = data.copy()

statistic_index.RunsTest(df,period=25)
#statistic_index.KPSSTest(df,period=30)
statistic_index.LjungBoxTest(df,period=35)
#statistic_index.AutoCorrelation(df, period=40)
statistic_index.VarianceRatio(df)

df.to_csv('test_df.csv')




factor_cols = [col for col in df.columns if col not in ["Date", "Adj close", "Close", "High", "Low", "Open", "Volume", "OHLC"]]

df_factors = df[factor_cols]
df_factors = df_factors.dropna(how="all", axis=1)
df_factors = df_factors.loc[:, df_factors.std() > 0]   # 去掉常数列


def correlation_filter(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop

filtered_df, dropped = correlation_filter(df_factors, threshold=0.5)

print("剔除的因子:", dropped)



corr_matrix = df_factors.corr()
corr_matrix.to_csv("factor_corr_matrix.csv")
print(corr_matrix)



### plt cor heatmap ###
plt.figure(figsize=(12,8))
plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation Coefficient")
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Factor Correlation Heatmap")
plt.tight_layout()
plt.show()




###################VIF test################


def calculate_vif(df):
    # 去除非数值型列
    df = df.select_dtypes(include=[np.number])
    # 丢掉含 NaN 或 Inf 的行
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values, i)
        for i in range(df.shape[1])
    ]
    return vif_data

# 使用
vif_result = calculate_vif(filtered_df)
print(vif_result)






if __name__ == '__main__':
    print('test df finish')

