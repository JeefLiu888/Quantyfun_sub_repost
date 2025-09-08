import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


df = pd.read_csv('BTC_data.csv')
factor_cols = [col for col in df.columns if col not in ["Date", "Adj close", "Close", "High", "Low", "Open", "Volume", "OHLC"]]

def professional_factor_pipeline(df, factor_cols, price_col="Adj close",
                                 output_report="factor_pipeline_report.txt"):
    """
    专业因子筛选流水线

    筛选顺序：数据质量 → 有效性 → 冗余性
    """

    print("开始专业因子筛选流水线...")
    report_lines = []

    # ========== 阶段1: 数据质量筛选 ==========
    print("阶段1: 数据质量筛选...")

    initial_count = len(factor_cols)
    report_lines.append("=" * 60)
    report_lines.append("专业因子筛选流水线报告")
    report_lines.append("=" * 60)
    report_lines.append(f"初始因子数量: {initial_count}")

    # 1.1 去除缺失值过多的因子
    missing_threshold = 0.2
    valid_factors = []
    for col in factor_cols:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio < missing_threshold:
            valid_factors.append(col)
        else:
            print(f"  删除因子 {col}: 缺失率{missing_ratio:.3f}")

    report_lines.append(f"\n1. 缺失值筛选 (保留缺失率<{missing_threshold}):")
    report_lines.append(f"   保留因子: {len(valid_factors)}")
    report_lines.append(f"   删除因子: {initial_count - len(valid_factors)}")

    # 1.2 去除常数因子和标准差过小因子
    df_factors = df[valid_factors]
    std_threshold = 1e-6
    non_constant_factors = []
    for col in valid_factors:
        if df_factors[col].std() > std_threshold:
            non_constant_factors.append(col)

    report_lines.append(f"\n2. 常数因子筛选:")
    report_lines.append(f"   保留因子: {len(non_constant_factors)}")
    report_lines.append(f"   删除常数因子: {len(valid_factors) - len(non_constant_factors)}")

    # 1.3 极值处理和标准化
    df_clean = df[non_constant_factors + [price_col]].copy()
    for col in non_constant_factors:
        # 3σ极值处理
        mean_val = df_clean[col].mean()
        std_val = df_clean[col].std()
        df_clean[col] = df_clean[col].clip(
            lower=mean_val - 3 * std_val,
            upper=mean_val + 3 * std_val
        )

    # 标准化
    scaler = StandardScaler()
    df_clean[non_constant_factors] = scaler.fit_transform(df_clean[non_constant_factors])

    # ========== 阶段2: 有效性筛选 ==========
    print("阶段2: 有效性筛选...")

    # 2.1 计算未来收益
    horizons = [1, 5, 10, 20]
    for h in horizons:
        df_clean[f"fwd_return_{h}d"] = df_clean[price_col].shift(-h) / df_clean[price_col] - 1

    # 2.2 因子有效性评估
    effective_factors = []
    factor_evaluation = {}

    ic_threshold = 0.02  # IC绝对值阈值
    ic_pval_threshold = 0.1  # IC显著性阈值 (放宽)
    ols_pval_threshold = 0.1  # OLS显著性阈值
    min_significant_periods = 2  # 至少2个周期显著

    for factor in non_constant_factors:
        significant_periods = 0
        max_ic = 0
        ols_significant = 0
        factor_details = {}

        for h in horizons:
            ret_col = f"fwd_return_{h}d"
            tmp = df_clean[[factor, ret_col]].dropna()

            if len(tmp) < 50:  # 样本量太小
                continue

            # IC检验
            ic_val, ic_pval = spearmanr(tmp[factor], tmp[ret_col])
            max_ic = max(max_ic, abs(ic_val))

            # OLS检验
            X = sm.add_constant(tmp[factor])
            model = sm.OLS(tmp[ret_col], X).fit()
            ols_pval = model.pvalues[factor]

            # 分组收益检验
            tmp['quantile'] = pd.qcut(tmp[factor], 5, labels=False, duplicates='drop')
            q_returns = tmp.groupby('quantile')[ret_col].mean()

            # 检查单调性
            returns_list = q_returns.tolist()
            is_monotonic = all(returns_list[i] <= returns_list[i + 1] for i in range(len(returns_list) - 1)) or \
                           all(returns_list[i] >= returns_list[i + 1] for i in range(len(returns_list) - 1))

            factor_details[f'{h}d'] = {
                'ic': ic_val, 'ic_pval': ic_pval,
                'ols_pval': ols_pval, 'monotonic': is_monotonic,
                'q1_q5_spread': abs(returns_list[-1] - returns_list[0]) if len(returns_list) >= 5 else 0
            }

            # 统计显著期数
            if ic_pval < ic_pval_threshold and abs(ic_val) > ic_threshold:
                significant_periods += 1
            if ols_pval < ols_pval_threshold:
                ols_significant += 1

        factor_evaluation[factor] = factor_details

        # 判断是否为有效因子
        is_effective = (
                significant_periods >= min_significant_periods and  # IC显著性要求
                max_ic > ic_threshold and  # 最大IC要求
                ols_significant >= min_significant_periods  # OLS显著性要求
        )

        if is_effective:
            effective_factors.append(factor)
            print(f"  ✓ 保留有效因子 {factor}: 最大IC={max_ic:.4f}, 显著期数={significant_periods}")

    report_lines.append(f"\n阶段2: 有效性筛选结果:")
    report_lines.append(f"   输入因子: {len(non_constant_factors)}")
    report_lines.append(f"   有效因子: {len(effective_factors)}")
    report_lines.append(
        f"   筛选标准: IC>{ic_threshold}, p<{ic_pval_threshold}, 至少{min_significant_periods}个周期显著")

    # ========== 阶段3: 冗余性筛选 ==========
    print("阶段3: 冗余性筛选...")

    if len(effective_factors) <= 1:
        final_factors = effective_factors
        report_lines.append(f"\n阶段3: 因子数量过少，跳过冗余性筛选")
    else:
        # 3.1 相关性筛选
        corr_threshold = 0.7
        df_effective = df_clean[effective_factors]
        corr_matrix = df_effective.corr().abs()

        # 逐步删除高相关因子 (保留IC更高的)
        factors_to_keep = effective_factors.copy()
        removed_due_to_corr = []

        for i in range(len(effective_factors)):
            for j in range(i + 1, len(effective_factors)):
                factor1, factor2 = effective_factors[i], effective_factors[j]

                if factor1 in factors_to_keep and factor2 in factors_to_keep:
                    if corr_matrix.loc[factor1, factor2] > corr_threshold:
                        # 保留IC更高的因子
                        ic1 = max([abs(factor_evaluation[factor1][f'{h}d']['ic']) for h in horizons
                                   if f'{h}d' in factor_evaluation[factor1]])
                        ic2 = max([abs(factor_evaluation[factor2][f'{h}d']['ic']) for h in horizons
                                   if f'{h}d' in factor_evaluation[factor2]])

                        if ic1 >= ic2:
                            factors_to_keep.remove(factor2)
                            removed_due_to_corr.append(
                                f"{factor2} (与{factor1}相关{corr_matrix.loc[factor1, factor2]:.3f})")
                        else:
                            factors_to_keep.remove(factor1)
                            removed_due_to_corr.append(
                                f"{factor1} (与{factor2}相关{corr_matrix.loc[factor1, factor2]:.3f})")

        # 3.2 VIF检验
        vif_threshold = 5.0
        if len(factors_to_keep) > 2:
            df_vif = df_clean[factors_to_keep].dropna()

            vif_data = []
            for i, factor in enumerate(factors_to_keep):
                try:
                    vif_val = variance_inflation_factor(df_vif.values, i)
                    vif_data.append((factor, vif_val))
                except:
                    vif_data.append((factor, np.inf))

            # 移除高VIF因子
            final_factors = [factor for factor, vif in vif_data if vif < vif_threshold]
            removed_due_to_vif = [f"{factor} (VIF={vif:.2f})" for factor, vif in vif_data if vif >= vif_threshold]
        else:
            final_factors = factors_to_keep
            removed_due_to_vif = []

        report_lines.append(f"\n阶段3: 冗余性筛选结果:")
        report_lines.append(f"   输入因子: {len(effective_factors)}")
        report_lines.append(f"   相关性筛选后: {len(factors_to_keep)}")
        report_lines.append(f"   VIF筛选后: {len(final_factors)}")

        if removed_due_to_corr:
            report_lines.append(f"   高相关删除: {removed_due_to_corr}")
        if removed_due_to_vif:
            report_lines.append(f"   高VIF删除: {removed_due_to_vif}")

    # ========== 最终结果 ==========
    report_lines.append(f"\n" + "=" * 60)
    report_lines.append(f"最终筛选结果:")
    report_lines.append(f"初始因子: {initial_count}")
    report_lines.append(f"最终因子: {len(final_factors)}")
    report_lines.append(f"筛选率: {len(final_factors) / initial_count * 100:.1f}%")
    report_lines.append(f"\n最终保留因子:")

    for factor in final_factors:
        # 计算该因子的最佳IC
        best_ic = 0
        best_period = ""
        for h in horizons:
            if f'{h}d' in factor_evaluation.get(factor, {}):
                ic_val = abs(factor_evaluation[factor][f'{h}d']['ic'])
                if ic_val > best_ic:
                    best_ic = ic_val
                    best_period = f"{h}d"

        report_lines.append(f"  ✓ {factor}: 最佳IC={best_ic:.4f} ({best_period})")

    # 保存报告
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n✓ 筛选完成!")
    print(f"✓ 最终保留 {len(final_factors)} 个优质因子")
    print(f"✓ 详细报告保存至: {output_report}")

    return final_factors, factor_evaluation


professional_factor_pipeline(df, factor_cols)
# 使用示例
if __name__ == "__main__":
    # 替换为你的数据
    # final_factors, evaluation = professional_factor_pipeline(df, factor_cols)
    pass