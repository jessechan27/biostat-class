# -*- coding: utf-8 -*-
"""
DM.xls 統計分析與 Seaborn 箱形圖繪製腳本
針對 BMI、血糖、收縮壓進行敘述統計、組間差異檢定與可視化
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 確保輸出支援 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 設定中文字體與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DFKai-SB', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取資料
file_path = 'DM.xls'
df = pd.read_excel(file_path)

# 欄位重新定義/提取關鍵變數
target_cols = {
    'BMI (kg/m²)': 'BMI',
    '血糖 (mg/dL)': '血糖',
    '收縮壓 (mmHg)': '收縮壓'
}

# 建立方便分析的工作資料表
df_clean = df.copy()
df_clean['組別標籤'] = df_clean['組別'].map({1: '第1組 (DM組)', 2: '第2組 (對照組)'}).fillna(df_clean['組別'].astype(str))
df_clean['性別標籤'] = df_clean['性別'].map({1: '男', 2: '女'}).fillna(df_clean['性別'].astype(str))

variables = ['BMI (kg/m²)', '血糖 (mg/dL)', '收縮壓 (mmHg)']
var_names = ['BMI (kg/m²)', '血糖 (mg/dL)', '收縮壓 (mmHg)']

print("=" * 70)
print("【1. 全體樣本敘述統計 (N = {})】".format(len(df_clean)))
print("=" * 70)

overall_stats = []
for col in variables:
    series = df_clean[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    overall_stats.append({
        '指標': col,
        '樣本數': len(series),
        '平均值 (Mean)': round(series.mean(), 2),
        '標準差 (Std)': round(series.std(), 2),
        '中位數 (Median)': round(series.median(), 2),
        '第25百分位 (Q1)': round(q1, 2),
        '第75百分位 (Q3)': round(q3, 2),
        '四分位距 (IQR)': round(iqr, 2),
        '最小值 (Min)': round(series.min(), 2),
        '最大值 (Max)': round(series.max(), 2),
    })

df_overall = pd.DataFrame(overall_stats)
print(df_overall.to_string(index=False))

print("\n" + "=" * 70)
print("【2. 依組別分組敘述統計】")
print("=" * 70)

group_summary = df_clean.groupby('組別標籤')[variables].agg(['count', 'mean', 'std', 'median', lambda x: x.quantile(0.75) - x.quantile(0.25)])
group_summary = group_summary.round(2)
group_summary.columns = pd.MultiIndex.from_tuples(
    [(col, {'count': '樣本數', 'mean': '平均值', 'std': '標準差', 'median': '中位數', '<lambda_0>': 'IQR'}[stat]) 
     for col, stat in group_summary.columns]
)
print(group_summary.to_string())

print("\n" + "=" * 70)
print("【3. 兩組差異假設檢定 (第1組 vs 第2組)】")
print("=" * 70)

g1 = df_clean[df_clean['組別'] == 1]
g2 = df_clean[df_clean['組別'] == 2]

test_results = []
for col in variables:
    # 常態性檢定
    shapiro_g1 = stats.shapiro(g1[col])
    shapiro_g2 = stats.shapiro(g2[col])
    
    # 變異數同質性檢定
    levene = stats.levene(g1[col], g2[col])
    equal_var = levene.pvalue > 0.05
    
    # 獨立雙樣本 t-test
    ttest = stats.ttest_ind(g1[col], g2[col], equal_var=equal_var)
    
    # Mann-Whitney U 檢定 (無母數)
    mwu = stats.mannwhitneyu(g1[col], g2[col])
    
    test_results.append({
        '變數': col,
        '第1組 Mean(SD)': f"{g1[col].mean():.2f} ± {g1[col].std():.2f}",
        '第2組 Mean(SD)': f"{g2[col].mean():.2f} ± {g2[col].std():.2f}",
        't值': round(ttest.statistic, 3),
        't檢定 p值': f"{ttest.pvalue:.4e}" if ttest.pvalue < 0.001 else f"{ttest.pvalue:.4f}",
        'Mann-Whitney p值': f"{mwu.pvalue:.4e}" if mwu.pvalue < 0.001 else f"{mwu.pvalue:.4f}",
        '結論 (α=0.05)': '顯著差異 (p < 0.05)' if ttest.pvalue < 0.05 else '無顯著差異'
    })

df_tests = pd.DataFrame(test_results)
print(df_tests.to_string(index=False))

print("\n" + "=" * 70)
print("【4. 變數間相關性矩陣 (Pearson r)】")
print("=" * 70)
corr_matrix = df_clean[variables].corr().round(3)
print(corr_matrix.to_string())

# 繪製圖表
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')
palette = ["#4C72B0", "#55A868"]

# 圖一：依組別比較 (Group Comparison Boxplot)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('DM.xls 臨床指標組別比較箱形圖 (Boxplot by Group)', fontsize=16, fontweight='bold', y=0.98)

titles = ['BMI 分佈比較', '血糖 分佈比較', '收縮壓 分佈比較']
y_labels = ['BMI (kg/m²)', '血糖 (mg/dL)', '收縮壓 (mmHg)']

for i, col in enumerate(variables):
    ax = axes[i]
    # Seaborn boxplot
    sns.boxplot(
        data=df_clean,
        x='組別標籤',
        y=col,
        hue='組別標籤',
        palette=palette,
        legend=False,
        width=0.45,
        fliersize=5,
        ax=ax,
        boxprops=dict(alpha=0.85),
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red", "markersize":"6"}
    )
    
    # 疊加 stripplot 顯示真實數據點
    sns.stripplot(
        data=df_clean,
        x='組別標籤',
        y=col,
        color='black',
        alpha=0.6,
        size=6,
        jitter=0.15,
        ax=ax
    )
    
    # 圖表美化
    ax.set_title(titles[i], fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('臨床分組', fontsize=11)
    ax.set_ylabel(y_labels[i], fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 標示紅點為平均值圖例說明
    if i == 0:
        ax.plot([], [], marker='o', color='red', linestyle='None', markersize=6, label='平均值 (Mean)')
        ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
output_img_group = 'boxplot_by_group.png'
plt.savefig(output_img_group, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n[OK] 分組箱形圖已儲存至: {output_img_group}")

# 圖二：全體分佈箱形圖 (Overall Boxplot)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('DM.xls 全體指標分佈箱形圖 (Overall Distribution Boxplots)', fontsize=15, fontweight='bold')

single_palette = ["#3498db", "#e74c3c", "#2ecc71"]

for i, col in enumerate(variables):
    ax = axes[i]
    sns.boxplot(
        data=df_clean,
        y=col,
        color=single_palette[i],
        width=0.35,
        ax=ax,
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"yellow", "markeredgecolor":"black", "markersize":"7"}
    )
    sns.stripplot(
        data=df_clean,
        y=col,
        color='#333333',
        alpha=0.7,
        size=6,
        jitter=0.12,
        ax=ax
    )
    ax.set_title(f'{titles[i].replace(" 比較", "")}', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_labels[i], fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
output_img_overall = 'boxplot_overall.png'
plt.savefig(output_img_overall, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] 全體箱形圖已儲存至: {output_img_overall}")

# 圖三：依組別與性別之雙因子箱形圖 (Group & Sex Breakdown)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('DM.xls 依組別與性別分析箱形圖 (Boxplot by Group and Gender)', fontsize=16, fontweight='bold')

gender_palette = {"男": "#4575b4", "女": "#d73027"}

for i, col in enumerate(variables):
    ax = axes[i]
    sns.boxplot(
        data=df_clean,
        x='組別標籤',
        y=col,
        hue='性別標籤',
        palette=gender_palette,
        width=0.55,
        ax=ax,
        boxprops=dict(alpha=0.85)
    )
    sns.stripplot(
        data=df_clean,
        x='組別標籤',
        y=col,
        hue='性別標籤',
        dodge=True,
        palette={'男': '#1f4e79', '女': '#800020'},
        alpha=0.7,
        size=5,
        jitter=0.1,
        ax=ax,
        legend=False
    )
    ax.set_title(titles[i], fontsize=13, fontweight='bold')
    ax.set_xlabel('臨床分組', fontsize=11)
    ax.set_ylabel(y_labels[i], fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if i != 2:
        ax.get_legend().remove()
    else:
        ax.legend(title='性別', loc='upper right')

plt.tight_layout()
output_img_gender = 'boxplot_by_group_gender.png'
plt.savefig(output_img_gender, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] 組別與性別交互箱形圖已儲存至: {output_img_gender}")
