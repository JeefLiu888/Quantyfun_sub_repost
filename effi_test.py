import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
import pprint



'''
计算未来收益
针对每个预测期（1天、5天、10天），计算未来收益率作为目标变量。未来收益的计算为当前价格与未来价格之比减一。

信息系数（IC）
计算因子值与未来收益的斯皮尔曼秩相关系数（Spearman Rank Correlation），衡量因子排序与未来收益排序的一致性。IC值越大，代表因子越有预测力，p值表示显著性。

分组收益（Quantile Returns）
将样本按因子值分为5组，计算每组的平均未来收益。通常高分组（如Q5）收益显著优于低分组（Q1）时，说明因子有效。

线性回归检验（OLS）
因子值对未来收益的线性影响检验：回归系数(beta)、t统计量(tstat)和p值(pval)衡量因子影响力度及显著性。

'''



# 读取 CSV
df = pd.read_csv('test_df.csv')
print("数据列:", df.columns)

def evaluate_factors(df, factor_cols, price_col="Adj close", horizons=[1,5,10,15,20,30], quantiles=5):
    """
    评估因子有效性（IC + 分组收益 + OLS回归检验）

    参数：
    df: pd.DataFrame，包含价格和因子
    factor_cols: list，因子列名
    price_col: str，价格列名（默认 Adj close）
    horizons: list，预测期，比如 [1,5,10]
    quantiles: int，分组数
    """
    df = df.copy()

    # Step 1: 构造未来收益
    for h in horizons:
        df[f"fwd_return_{h}d"] = df[price_col].shift(-h) / df[price_col] - 1

    results = {}

    # Step 2: 循环评估每个因子
    for factor in factor_cols:
        factor_res = {}
        for h in horizons:
            ret_col = f"fwd_return_{h}d"
            tmp = df[[factor, ret_col]].dropna()

            if tmp.empty:
                continue

            # --- 2.1 IC (Spearman Rank Correlation)
            ic, pval = spearmanr(tmp[factor], tmp[ret_col])
            factor_res[f"IC_{h}d"] = (ic, pval)

            # --- 2.2 分组测试 (Quantile Returns)
            if tmp[factor].nunique() > 1:  # 非常数列
                tmp["quantile"] = pd.qcut(tmp[factor], quantiles, labels=False, duplicates='drop') + 1
                q_ret = tmp.groupby("quantile")[ret_col].mean()
                factor_res[f"Q{quantiles}_{h}d"] = q_ret.to_dict()
            else:
                factor_res[f"Q{quantiles}_{h}d"] = None

            # --- 2.3 回归检验 (OLS)
            X = sm.add_constant(tmp[factor])
            y = tmp[ret_col]
            model = sm.OLS(y, X).fit()
            factor_res[f"OLS_{h}d"] = {
                "beta": model.params[factor],
                "tstat": model.tvalues[factor],
                "pval": model.pvalues[factor]
            }

        results[factor] = factor_res

    return results

# 使用示例：选择你已有的因子列, 可以自己添加或删除
factor_cols = [col for col in df.columns if col not in ["Date", "Adj close", "Close", "High", "Low", "Open", "Volume", "OHLC"]]  # 可自行增减

results = evaluate_factors(df, factor_cols)
pprint.pprint(results)




# 保存结果为 pickle 或 CSV
import pickle

with open("factor_evaluation.pkl", "wb") as f:
    pickle.dump(results, f)

# 也可以把 IC 结果整理成 DataFrame 保存
ic_records = []
for factor, res in results.items():
    for key, val in res.items():
        if key.startswith("IC"):
            ic_records.append({"factor": factor, "horizon": key, "IC": val[0], "pval": val[1]})
ic_df = pd.DataFrame(ic_records)
ic_df.to_csv("factor_IC.csv", index=False)
print("IC 保存完成")



############################# 新加评估筛选代码  机器完成 ########################

import pandas as pd
import numpy as np
from datetime import datetime
import json


def filter_excellent_factors(results, output_file="selected_factors_report.txt",
                             ic_threshold=0.03, ic_pval_threshold=0.05,
                             ols_pval_threshold=0.05, ols_tstat_threshold=1.96,
                             monotonic_ratio=0.6, min_score=6.0):
    """
    综合评估并筛选优秀因子

    参数：
    results: dict, evaluate_factors()的输出结果
    output_file: str, 筛选报告输出文件名
    ic_threshold: float, IC绝对值阈值
    ic_pval_threshold: float, IC显著性阈值
    ols_pval_threshold: float, OLS回归p值阈值
    ols_tstat_threshold: float, OLS t统计量绝对值阈值
    monotonic_ratio: float, 分组收益单调性比例要求
    min_score: float, 最低综合得分要求

    返回：
    selected_factors: list, 筛选出的优秀因子列表
    """

    factor_scores = {}
    detailed_reports = {}

    # 定义评分权重
    weights = {
        'ic_score': 0.3,  # IC得分权重
        'ic_significance': 0.2,  # IC显著性权重
        'ols_score': 0.25,  # OLS回归得分权重
        'monotonic_score': 0.15,  # 单调性得分权重
        'stability_score': 0.1  # 稳定性得分权重
    }

    for factor_name, factor_data in results.items():
        scores = {}
        reasons = []
        warnings = []

        # 提取所有预测期
        horizons = []
        for key in factor_data.keys():
            if key.startswith('IC_'):
                horizon = key.split('_')[1]
                horizons.append(horizon)

        if not horizons:
            continue

        # ========== 1. IC评分 ==========
        ic_values = []
        ic_pvals = []
        significant_ics = 0

        for h in horizons:
            ic_key = f'IC_{h}'
            if ic_key in factor_data:
                ic_val, ic_pval = factor_data[ic_key]
                ic_values.append(abs(ic_val))
                ic_pvals.append(ic_pval)
                if ic_pval < ic_pval_threshold and abs(ic_val) > ic_threshold:
                    significant_ics += 1

        if ic_values:
            avg_ic = np.mean(ic_values)
            max_ic = np.max(ic_values)

            # IC得分 (0-10分)
            ic_score = min(10, max_ic * 100)  # |IC|*100作为基础分
            if avg_ic > 0.05:
                ic_score += 2  # 平均IC优秀加分
            scores['ic_score'] = ic_score

            # IC显著性得分 (0-10分)
            sig_ratio = significant_ics / len(horizons)
            ic_sig_score = sig_ratio * 10
            scores['ic_significance'] = ic_sig_score

            if max_ic > 0.05:
                reasons.append(f"最大IC绝对值{max_ic:.4f}，显示良好预测能力")
            if sig_ratio > 0.5:
                reasons.append(f"{sig_ratio * 100:.1f}%的预测期IC显著")
            if avg_ic < 0.02:
                warnings.append(f"平均IC仅{avg_ic:.4f}，预测能力偏弱")
        else:
            scores['ic_score'] = 0
            scores['ic_significance'] = 0
            warnings.append("无有效IC数据")

        # ========== 2. OLS回归评分 ==========
        ols_scores = []
        significant_ols = 0
        consistent_direction = True
        beta_signs = []

        for h in horizons:
            ols_key = f'OLS_{h}'
            if ols_key in factor_data:
                ols_data = factor_data[ols_key]
                beta = ols_data['beta']
                tstat = abs(ols_data['tstat'])
                pval = ols_data['pval']

                beta_signs.append(np.sign(beta))

                # 单个OLS得分
                score = 0
                if pval < ols_pval_threshold:
                    score += 5  # 显著性基础分
                    significant_ols += 1
                if tstat > ols_tstat_threshold:
                    score += 3  # t统计量加分
                if tstat > 3:
                    score += 2  # 高t统计量额外加分

                ols_scores.append(score)

        if ols_scores:
            # 检查beta符号一致性
            if len(set(beta_signs)) > 1:
                consistent_direction = False

            avg_ols_score = np.mean(ols_scores)
            scores['ols_score'] = avg_ols_score

            ols_sig_ratio = significant_ols / len(horizons)
            if ols_sig_ratio > 0.5:
                reasons.append(f"{ols_sig_ratio * 100:.1f}%的预测期OLS回归显著")
            if consistent_direction:
                reasons.append("因子作用方向一致，逻辑稳定")
            else:
                warnings.append("不同预测期beta符号不一致")
        else:
            scores['ols_score'] = 0
            warnings.append("无有效OLS回归数据")

        # ========== 3. 分组收益单调性评分 ==========
        monotonic_scores = []

        for h in horizons:
            q_key = f'Q5_{h}'
            if q_key in factor_data and factor_data[q_key] is not None:
                q_returns = factor_data[q_key]
                if len(q_returns) >= 5:
                    returns_list = [q_returns[i] for i in sorted(q_returns.keys())]

                    # 检查单调性
                    monotonic_pairs = 0
                    total_pairs = len(returns_list) - 1

                    for i in range(total_pairs):
                        # 正向或负向单调性都可以
                        if (returns_list[i + 1] > returns_list[i]) or (returns_list[i + 1] < returns_list[i]):
                            if i == 0:  # 确定方向
                                direction = 1 if returns_list[i + 1] > returns_list[i] else -1
                            if (returns_list[i + 1] - returns_list[i]) * direction > 0:
                                monotonic_pairs += 1

                    monotonic_ratio_actual = monotonic_pairs / total_pairs if total_pairs > 0 else 0
                    monotonic_scores.append(monotonic_ratio_actual * 10)

                    # 检查首尾收益差异
                    spread = abs(returns_list[-1] - returns_list[0])
                    if spread > 0.01:  # 1%收益差异
                        reasons.append(f"{h}期分组收益差异{spread * 100:.2f}%，区分度良好")

        if monotonic_scores:
            scores['monotonic_score'] = np.mean(monotonic_scores)
            avg_monotonic = np.mean([s / 10 for s in monotonic_scores])
            if avg_monotonic > monotonic_ratio:
                reasons.append(f"分组收益单调性{avg_monotonic * 100:.1f}%，因子排序有效")
            else:
                warnings.append(f"分组收益单调性仅{avg_monotonic * 100:.1f}%，低于要求")
        else:
            scores['monotonic_score'] = 0
            warnings.append("无有效分组收益数据")

        # ========== 4. 跨周期稳定性评分 ==========
        if len(horizons) > 1 and ic_values:
            # IC稳定性：方差越小越稳定
            ic_cv = np.std(ic_values) / np.mean(ic_values) if np.mean(ic_values) > 0 else 1
            stability_score = max(0, 10 - ic_cv * 10)  # 变异系数越小得分越高
            scores['stability_score'] = stability_score

            if ic_cv < 0.5:
                reasons.append(f"IC跨周期变异系数{ic_cv:.3f}，稳定性良好")
            else:
                warnings.append(f"IC跨周期变异系数{ic_cv:.3f}，稳定性较差")
        else:
            scores['stability_score'] = 5  # 默认中等分

        # ========== 5. 计算综合得分 ==========
        total_score = sum(scores[k] * weights[k] for k in weights.keys() if k in scores)

        # ========== 6. 生成评级 ==========
        if total_score >= 8:
            grade = "优秀"
        elif total_score >= 6:
            grade = "良好"
        elif total_score >= 4:
            grade = "一般"
        else:
            grade = "较差"

        factor_scores[factor_name] = {
            'total_score': total_score,
            'grade': grade,
            'detailed_scores': scores,
            'reasons': reasons,
            'warnings': warnings
        }

    # ========== 筛选优秀因子 ==========
    selected_factors = [
        factor for factor, info in factor_scores.items()
        if info['total_score'] >= min_score
    ]

    # ========== 生成详细报告 ==========
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("因子筛选与评估报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"筛选标准: 综合得分 >= {min_score}\n")
        f.write(f"候选因子总数: {len(factor_scores)}\n")
        f.write(f"筛选出优秀因子数: {len(selected_factors)}\n\n")

        # 筛选结果摘要
        f.write("筛选结果摘要:\n")
        f.write("-" * 40 + "\n")
        for factor in selected_factors:
            info = factor_scores[factor]
            f.write(f"✓ {factor}: {info['total_score']:.2f}分 ({info['grade']})\n")
        f.write("\n")

        # 详细分析
        f.write("详细分析报告:\n")
        f.write("=" * 40 + "\n\n")

        for factor_name in sorted(factor_scores.keys(),
                                  key=lambda x: factor_scores[x]['total_score'],
                                  reverse=True):
            info = factor_scores[factor_name]

            f.write(f"因子名称: {factor_name}\n")
            f.write(f"综合得分: {info['total_score']:.2f} 分\n")
            f.write(f"评级等级: {info['grade']}\n")
            f.write(f"筛选结果: {'✓ 通过' if factor_name in selected_factors else '✗ 未通过'}\n\n")

            # 详细得分
            f.write("详细得分:\n")
            for score_type, score_val in info['detailed_scores'].items():
                weight = weights.get(score_type, 0)
                f.write(f"  {score_type}: {score_val:.2f}/10 (权重{weight:.1%})\n")
            f.write("\n")

            # 筛选原因
            if info['reasons']:
                f.write("✓ 优势:\n")
                for reason in info['reasons']:
                    f.write(f"  • {reason}\n")
                f.write("\n")

            # 警告信息
            if info['warnings']:
                f.write("⚠ 注意事项:\n")
                for warning in info['warnings']:
                    f.write(f"  • {warning}\n")
                f.write("\n")

            f.write("-" * 60 + "\n\n")

        # 评分标准说明
        f.write("评分标准说明:\n")
        f.write("=" * 40 + "\n")
        f.write(f"IC得分 (权重{weights['ic_score']:.1%}): 基于IC绝对值大小\n")
        f.write(f"IC显著性 (权重{weights['ic_significance']:.1%}): 基于显著IC比例\n")
        f.write(f"OLS得分 (权重{weights['ols_score']:.1%}): 基于回归显著性和t统计量\n")
        f.write(f"单调性得分 (权重{weights['monotonic_score']:.1%}): 基于分组收益单调性\n")
        f.write(f"稳定性得分 (权重{weights['stability_score']:.1%}): 基于跨周期IC稳定性\n\n")

        f.write("筛选参数:\n")
        f.write(f"  IC阈值: {ic_threshold}\n")
        f.write(f"  IC显著性p值: {ic_pval_threshold}\n")
        f.write(f"  OLS显著性p值: {ols_pval_threshold}\n")
        f.write(f"  t统计量阈值: {ols_tstat_threshold}\n")
        f.write(f"  单调性比例要求: {monotonic_ratio}\n")
        f.write(f"  最低综合得分: {min_score}\n")

    print(f"✓ 筛选完成！共筛选出 {len(selected_factors)} 个优秀因子")
    print(f"✓ 详细报告已保存至: {output_file}")
    print("✓ 筛选出的优秀因子:", selected_factors)

    return selected_factors


# 使用示例
if __name__ == "__main__":
    # 假设你已经有了results数据
    # selected = filter_excellent_factors(results)

    # 可调整参数示例
    selected = filter_excellent_factors(
        results,
        output_file="my_factor_selection.txt",
        ic_threshold=0.025,  # 降低IC要求
        min_score=5.5,  # 降低最低分数要求
        monotonic_ratio=0.5  # 降低单调性要求
    )