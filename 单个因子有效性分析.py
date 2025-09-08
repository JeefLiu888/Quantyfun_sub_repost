import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle
import pprint
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块 (假设这些模块存在)
try:
    import volume_osc_index
    import statistic_index
except ImportError:
    print("警告: 无法导入 volume_osc_index 或 statistic_index 模块")
    print("请确保这些模块在当前目录或Python路径中")


class FactorAnalysisSystem:
    def __init__(self, data_path='BTC_data.csv', output_dir='./'):
        """
        初始化因子分析系统

        参数:
        data_path: str, 数据文件路径
        output_dir: str, 输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.factor_results = None
        self.selected_factors = None

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成，形状: {self.df.shape}")
        return self.df.copy()

    def generate_factors_with_params(self, short_range=(5, 11), long_range=(10, 21), step=5):
        """
        使用不同参数组合生成因子

        参数:
        short_range: tuple, 短周期范围 (start, end)
        long_range: tuple, 长周期范围 (start, end)
        step: int, 步长
        """
        print("正在生成因子...")
        df = self.df.copy()

        # 生成 Volume Oscillator 因子 (如果模块可用)
        try:
            for short in range(short_range[0], short_range[1], step):
                for long in range(long_range[0], long_range[1], step):
                    if long <= short:
                        continue
                    print(f"生成因子: VolumeOsc_s{short}_l{long}")
                    volume_osc_index.VolumeOscillator(df, short_period=short, long_period=long)
                    # 保存每个参数组合的数据
                    df.to_csv(f"{self.output_dir}/df_s{short}_l{long}.csv")
        except:
            print("Volume Oscillator 因子生成失败，跳过...")

        # 生成统计类因子 (如果模块可用)
        try:
            print("生成统计类因子...")
            statistic_index.RunsTest(df, period=25)
            statistic_index.KPSSTest(df, period=30)
            statistic_index.LjungBoxTest(df, period=35)
            statistic_index.VarianceRatio(df)
        except:
            print("统计类因子生成失败，跳过...")

        self.df = df
        print("因子生成完成")
        return df

    def evaluate_factors(self, df=None, price_col="Adj close", horizons=[1, 5, 10, 15, 20, 30], quantiles=5):
        """
        评估因子有效性（IC + 分组收益 + OLS回归检验）

        参数：
        df: pd.DataFrame，包含价格和因子
        price_col: str，价格列名（默认 Adj close）
        horizons: list，预测期，比如 [1,5,10]
        quantiles: int，分组数
        """
        if df is None:
            df = self.df.copy()

        print("正在评估因子有效性...")

        # 获取因子列
        factor_cols = [col for col in df.columns if col not in
                       ['Unnamed: 0', "Date", "Adj close", "Close", "High", "Low", "Open", "Volume", "OHLC"]]

        # 清理因子数据
        df_factors = df[factor_cols]
        df_factors = df_factors.dropna(how="all", axis=1)
        df_factors = df_factors.loc[:, df_factors.std() > 0]

        # 重新组合数据
        df_clean = pd.concat([df[["Date", price_col]], df_factors], axis=1)
        factor_cols = df_factors.columns.tolist()

        print(f"有效因子数量: {len(factor_cols)}")

        # Step 1: 构造未来收益
        for h in horizons:
            df_clean[f"fwd_return_{h}d"] = df_clean[price_col].shift(-h) / df_clean[price_col] - 1

        results = {}

        # Step 2: 循环评估每个因子
        for i, factor in enumerate(factor_cols):
            print(f"评估因子 {i + 1}/{len(factor_cols)}: {factor}")

            factor_res = {}
            for h in horizons:
                ret_col = f"fwd_return_{h}d"
                tmp = df_clean[[factor, ret_col]].dropna()

                if tmp.empty or len(tmp) < 10:
                    continue

                # --- 2.1 IC (Spearman Rank Correlation)
                try:
                    ic, pval = spearmanr(tmp[factor], tmp[ret_col])
                    factor_res[f"IC_{h}d"] = (ic, pval)
                except:
                    continue

                # --- 2.2 分组测试 (Quantile Returns)
                if tmp[factor].nunique() > 1:  # 非常数列
                    try:
                        tmp["quantile"] = pd.qcut(tmp[factor], quantiles, labels=False, duplicates='drop') + 1
                        q_ret = tmp.groupby("quantile")[ret_col].mean()
                        factor_res[f"Q{quantiles}_{h}d"] = q_ret.to_dict()
                    except:
                        factor_res[f"Q{quantiles}_{h}d"] = None
                else:
                    factor_res[f"Q{quantiles}_{h}d"] = None

                # --- 2.3 回归检验 (OLS)
                try:
                    X = sm.add_constant(tmp[factor])
                    y = tmp[ret_col]
                    model = sm.OLS(y, X).fit()
                    factor_res[f"OLS_{h}d"] = {
                        "beta": model.params[factor],
                        "tstat": model.tvalues[factor],
                        "pval": model.pvalues[factor]
                    }
                except:
                    continue

            results[factor] = factor_res

        self.factor_results = results

        # 保存结果
        with open(f"{self.output_dir}/factor_evaluation.pkl", "wb") as f:
            pickle.dump(results, f)

        # 保存IC结果为CSV
        ic_records = []
        for factor, res in results.items():
            for key, val in res.items():
                if key.startswith("IC"):
                    ic_records.append({"factor": factor, "horizon": key, "IC": val[0], "pval": val[1]})

        if ic_records:
            ic_df = pd.DataFrame(ic_records)
            ic_df.to_csv(f"{self.output_dir}/factor_IC.csv", index=False)
            print("IC结果保存完成")

        print("因子评估完成")
        return results

    def filter_excellent_factors(self, results=None, output_file=None,
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
        if results is None:
            results = self.factor_results

        if output_file is None:
            output_file = f"{self.output_dir}/selected_factors_report.txt"

        print("正在筛选优秀因子...")

        factor_scores = {}

        # 定义评分权重
        weights = {
            'ic_score': 0.3,  # IC得分权重
            'ic_significance': 0.2,  # IC显著性权重
            'ols_score': 0.25,  # OLS回归得分权重
            'monotonic_score': 0.15,  # 单调性得分权重
            'stability_score': 0.1  # 稳定性得分权重
        }

        for factor_name, factor_data in results.items():
            if not factor_data:
                continue

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

        # 生成报告
        self._generate_factor_report(factor_scores, selected_factors, output_file, weights,
                                     ic_threshold, ic_pval_threshold, ols_pval_threshold,
                                     ols_tstat_threshold, monotonic_ratio, min_score)

        self.selected_factors = selected_factors
        print(f"✓ 筛选完成！共筛选出 {len(selected_factors)} 个优秀因子")
        print(f"✓ 详细报告已保存至: {output_file}")

        return selected_factors

    def _generate_factor_report(self, factor_scores, selected_factors, output_file, weights,
                                ic_threshold, ic_pval_threshold, ols_pval_threshold,
                                ols_tstat_threshold, monotonic_ratio, min_score):
        """生成详细报告"""
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

    def correlation_analysis(self, df=None, correlation_threshold=0.8, plot_heatmap=True):
        """
        相关性分析和处理

        参数:
        df: DataFrame, 输入数据
        correlation_threshold: float, 相关性阈值
        plot_heatmap: bool, 是否绘制热力图
        """
        if df is None:
            df = self.df.copy()

        print("正在进行相关性分析...")

        # 获取因子列
        factor_cols = [col for col in df.columns if col not in
                       ['Unnamed: 0', "Date", "Adj close", "Close", "High", "Low", "Open", "Volume", "OHLC"]]

        df_factors = df[factor_cols]

        # 去除全空列与常数列
        df_factors = df_factors.dropna(how="all", axis=1)
        df_factors = df_factors.loc[:, df_factors.std() > 0]

        # 标准化因子数据
        scaler = StandardScaler()
        df_factors_scaled = pd.DataFrame(
            scaler.fit_transform(df_factors),
            columns=df_factors.columns,
            index=df_factors.index
        )

        # 相关性过滤
        def correlation_filter(df, threshold=correlation_threshold):
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            return df.drop(columns=to_drop), to_drop

        filtered_df, dropped_factors = correlation_filter(df_factors_scaled, threshold=correlation_threshold)

        print(f"原始因子数: {len(df_factors.columns)}")
        print(f"剔除高相关性因子数: {len(dropped_factors)}")
        print(f"保留因子数: {len(filtered_df.columns)}")
        print("剔除的因子:", dropped_factors)

        # 计算相关性矩阵
        corr_matrix = df_factors.corr()
        corr_matrix.to_csv(f"{self.output_dir}/factor_corr_matrix.csv")

        # 绘制相关性热力图
        if plot_heatmap and len(df_factors.columns) <= 50:  # 避免因子过多导致图形不清晰
            plt.figure(figsize=(12, 8))
            plt.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
            plt.colorbar(label="Correlation Coefficient")
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            plt.title("Factor Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/factor_correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.show()
        elif len(df_factors.columns) > 50:
            print("因子数量过多，跳过热力图绘制")

        # VIF分析
        vif_result = self._calculate_vif(filtered_df)
        print("\nVIF分析结果:")
        print(vif_result)
        vif_result.to_csv(f"{self.output_dir}/vif_analysis.csv", index=False)

        return filtered_df, dropped_factors, vif_result

    def _calculate_vif(self, df):
        """计算方差膨胀因子(VIF)"""
        # 去除非数值型列
        df = df.select_dtypes(include=[np.number])
        # 丢掉含 NaN 或 Inf 的行
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df.empty or df.shape[1] < 2:
            return pd.DataFrame({"Feature": [], "VIF": []})

        vif_data = pd.DataFrame()
        vif_data["Feature"] = df.columns

        try:
            vif_data["VIF"] = [
                variance_inflation_factor(df.values, i)
                for i in range(df.shape[1])
            ]
        except Exception as e:
            print(f"VIF计算错误: {e}")
            vif_data["VIF"] = [np.nan] * len(df.columns)

        return vif_data

    def run_complete_analysis(self, short_range=(5, 16), long_range=(10, 31), step=5,
                              horizons=[1, 5, 10, 15, 20, 30], min_score=5.5,
                              correlation_threshold=0.7):
        """
        运行完整的因子分析流程

        参数:
        short_range: tuple, 短周期范围
        long_range: tuple, 长周期范围
        step: int, 参数步长
        horizons: list, 预测期列表
        min_score: float, 因子筛选最低分数
        correlation_threshold: float, 相关性阈值
        """
        print("=" * 60)
        print("开始运行完整的因子分析流程")
        print("=" * 60)

        # Step 1: 加载数据
        self.load_data()

        # Step 2: 生成因子
        self.generate_factors_with_params(short_range, long_range, step)

        # Step 3: 评估因子
        self.evaluate_factors(horizons=horizons)

        # Step 4: 筛选优秀因子
        selected = self.filter_excellent_factors(min_score=min_score)

        # Step 5: 相关性分析
        filtered_df, dropped, vif = self.correlation_analysis(correlation_threshold=correlation_threshold)

        # Step 6: 生成最终报告
        self._generate_factor_report(selected, dropped, vif)

        print("=" * 60)
        print("完整分析流程执行完毕！")
        print("=" * 60)

        return selected, filtered_df

        def _generate_final_report(self, selected_factors, dropped_factors, vif_result):
            """生成最终分析报告"""
            report_file = f"{self.output_dir}/final_analysis_report.txt"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("完整因子分析报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据形状: {self.df.shape}\n\n")

                f.write("1. 因子筛选结果:\n")
                f.write("-" * 40 + "\n")
                f.write(f"筛选出的优秀因子 ({len(selected_factors)} 个):\n")
                for factor in selected_factors:
                    f.write(f"  • {factor}\n")
                f.write("\n")

                f.write("2. 相关性分析结果:\n")
                f.write("-" * 40 + "\n")
                f.write(f"因高相关性剔除的因子 ({len(dropped_factors)} 个):\n")
                for factor in dropped_factors:
                    f.write(f"  • {factor}\n")
                f.write("\n")

                f.write("3. VIF分析结果:\n")
                f.write("-" * 40 + "\n")
                if not vif_result.empty:
                    high_vif = vif_result[vif_result['VIF'] > 5]
                    f.write(f"高VIF因子 (VIF > 5, 共{len(high_vif)}个):\n")
                    for _, row in high_vif.iterrows():
                        f.write(f"  • {row['Feature']}: {row['VIF']:.2f}\n")
                f.write("\n")

                f.write("4. 建议:\n")
                f.write("-" * 40 + "\n")
                f.write("• 优先使用筛选出的优秀因子进行模型构建\n")
                f.write("• 注意高相关性因子的多重共线性问题\n")
                f.write("• 考虑对高VIF因子进行进一步处理\n")
                f.write("• 建议进行样本外测试验证因子稳定性\n")

            print(f"✓ 最终分析报告已保存至: {report_file}")

    # 使用示例和主程序
if __name__ == "__main__":
        # 创建因子分析系统实例
        analyzer = FactorAnalysisSystem(
            data_path='BTC_data.csv',
            output_dir='./'
        )

        # 方案1: 运行完整分析流程（推荐）
        print("选择运行模式:")
        print("1. 完整分析流程（推荐）")
        print("2. 分步骤运行")

        choice = input("请输入选择 (1 或 2): ").strip()

        if choice == "1":
            # 完整流程
            selected_factors, filtered_df = analyzer.run_complete_analysis(
                short_range=(5, 16),  # 短周期 5-15
                long_range=(10, 31),  # 长周期 10-30
                step=5,  # 步长 5
                horizons=[1, 5, 10, 15, 20, 30],  # 预测期
                min_score=5.5,  # 最低得分
                correlation_threshold=0.7  # 相关性阈值
            )

            print("\n" + "=" * 50)
            print("分析完成！主要输出文件:")
            print("=" * 50)
            print("• factor_evaluation.pkl - 因子评估结果")
            print("• factor_IC.csv - IC分析结果")
            print("• selected_factors_report.txt - 因子筛选报告")
            print("• factor_corr_matrix.csv - 相关性矩阵")
            print("• factor_correlation_heatmap.png - 相关性热力图")
            print("• vif_analysis.csv - VIF分析结果")
            print("• final_analysis_report.txt - 最终分析报告")

        else:
            # 分步骤运行
            print("\n开始分步骤运行...")

            # Step 1: 加载数据
            df = analyzer.load_data()

            # Step 2: 生成因子（可自定义参数）
            print("\n是否要生成新因子？(y/n):")
            if input().strip().lower() == 'y':
                df = analyzer.generate_factors_with_params(
                    short_range=(5, 11),
                    long_range=(10, 21),
                    step=5
                )

            # Step 3: 评估因子
            results = analyzer.evaluate_factors(horizons=[1, 5, 10, 15, 20])
            print("因子评估结果样例:")
            for i, (factor, result) in enumerate(results.items()):
                if i < 3:  # 只显示前3个
                    print(f"{factor}: {list(result.keys())}")

            # Step 4: 筛选因子
            selected = analyzer.filter_excellent_factors(
                ic_threshold=0.025,
                min_score=5.0,
                monotonic_ratio=0.5
            )

            # Step 5: 相关性分析
            filtered_df, dropped, vif = analyzer.correlation_analysis(
                correlation_threshold=0.8,
                plot_heatmap=True
            )

            print(f"\n最终筛选出 {len(selected)} 个优秀因子:")
            for factor in selected:
                print(f"  • {factor}")

        print("\n分析完成！请查看生成的报告文件。")