# -*- coding: utf-8 -*-
"""DM.xls 統計分析：BMI、血糖、收縮壓
組別 1 vs 2 比較 + seaborn boxplot
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---- 編碼對照（若與實際相反，改這裡即可） ----
GROUP_LABEL = {1: "第1組 (病例組)", 2: "第2組 (對照組)"}
SEX_LABEL = {1: "男", 2: "女"}

VARS = {
    "BMI (kg/m²)": "BMI",
    "血糖 (mg/dL)": "血糖",
    "收縮壓 (mmHg)": "收縮壓",
}

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei",
              rc={"axes.unicode_minus": False})

df = pd.read_excel("DM.xls", sheet_name=0)
df.columns = [c.strip() for c in df.columns]
df["組別名"] = df["組別"].map(GROUP_LABEL)
df["性別名"] = df["性別"].map(SEX_LABEL)

out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


P("=" * 78)
P("DM.xls 統計分析報告")
P("=" * 78)
P("")
P("總樣本數 n = %d" % len(df))
P("")
P("[ 樣本分布 ]")
P(pd.crosstab(df["組別名"], df["性別名"], margins=True,
              margins_name="合計").to_string())
P("")
P("遺漏值檢查：" + ("無遺漏值" if df.isna().sum().sum() == 0
                 else df.isna().sum().to_string()))

# ---------- 1. 描述性統計 ----------
P("")
P("=" * 78)
P("1. 描述性統計 (依組別)")
P("=" * 78)
desc_rows = []
for col, short in list(VARS.items()) + [("年齡", "年齡")]:
    for g, sub in df.groupby("組別名"):
        x = sub[col].astype(float)
        desc_rows.append({
            "變項": short, "組別": g, "n": len(x),
            "平均數": x.mean(), "標準差": x.std(ddof=1),
            "中位數": x.median(),
            "Q1": x.quantile(.25), "Q3": x.quantile(.75),
            "最小值": x.min(), "最大值": x.max(),
            "CV%": x.std(ddof=1) / x.mean() * 100,
        })
    x = df[col].astype(float)
    desc_rows.append({
        "變項": short, "組別": "全體", "n": len(x),
        "平均數": x.mean(), "標準差": x.std(ddof=1), "中位數": x.median(),
        "Q1": x.quantile(.25), "Q3": x.quantile(.75),
        "最小值": x.min(), "最大值": x.max(),
        "CV%": x.std(ddof=1) / x.mean() * 100,
    })
desc = pd.DataFrame(desc_rows)
P(desc.round(2).to_string(index=False))

# ---------- 2. 常態性與變異數同質性 ----------
P("")
P("=" * 78)
P("2. 前提假設檢定 (Shapiro-Wilk 常態性 / Levene 變異數同質性)")
P("=" * 78)
assume_rows = []
for col, short in VARS.items():
    g1 = df.loc[df["組別"] == 1, col].astype(float)
    g2 = df.loc[df["組別"] == 2, col].astype(float)
    w1, p1 = stats.shapiro(g1)
    w2, p2 = stats.shapiro(g2)
    lev_w, lev_p = stats.levene(g1, g2, center="median")
    assume_rows.append({
        "變項": short,
        "組1 W": w1, "組1 p": p1,
        "組2 W": w2, "組2 p": p2,
        "Levene W": lev_w, "Levene p": lev_p,
        "常態(a=.05)": "是" if min(p1, p2) > .05 else "否",
        "等變異(a=.05)": "是" if lev_p > .05 else "否",
    })
assume = pd.DataFrame(assume_rows)
P(assume.round(4).to_string(index=False))

# ---------- 3. 兩組比較 ----------
P("")
P("=" * 78)
P("3. 兩組差異檢定 (Welch t 檢定 + Mann-Whitney U)")
P("=" * 78)


def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                 / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    return d, d - 1.96 * se, d + 1.96 * se


def d_tag(d):
    a = abs(d)
    if a >= 1.2:
        return "極大"
    if a >= .8:
        return "大"
    if a >= .5:
        return "中"
    if a >= .2:
        return "小"
    return "微小"


def stars(p):
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    return "n.s."


test_rows = []
for col, short in VARS.items():
    g1 = df.loc[df["組別"] == 1, col].astype(float)
    g2 = df.loc[df["組別"] == 2, col].astype(float)
    t, pt = stats.ttest_ind(g1, g2, equal_var=False)
    u, pu = stats.mannwhitneyu(g1, g2, alternative="two-sided", method="exact")
    d, lo, hi = cohen_d(g1, g2)
    n1, n2 = len(g1), len(g2)
    v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
    dfw = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1)
                                      + (v2 / n2) ** 2 / (n2 - 1))
    diff = g1.mean() - g2.mean()
    se = np.sqrt(v1 / n1 + v2 / n2)
    tcrit = stats.t.ppf(.975, dfw)
    test_rows.append({
        "變項": short,
        "組1 平均±SD": "%.2f±%.2f" % (g1.mean(), g1.std(ddof=1)),
        "組2 平均±SD": "%.2f±%.2f" % (g2.mean(), g2.std(ddof=1)),
        "平均差": diff,
        "差異95%CI": "[%.2f, %.2f]" % (diff - tcrit * se, diff + tcrit * se),
        "t": t, "df": dfw, "t檢定 p": pt,
        "U": u, "MWU p": pu,
        "Cohen d": d, "d 95%CI": "[%.2f, %.2f]" % (lo, hi),
        "效果量": d_tag(d), "顯著性": stars(pt),
    })
tests = pd.DataFrame(test_rows)
P(tests.round(4).to_string(index=False))
P("")
P("註：* p<.05, ** p<.01, *** p<.001")

# ---------- 4. 性別分層 ----------
P("")
P("=" * 78)
P("4. 性別分層的兩組比較 (Welch t 檢定)")
P("=" * 78)
strat_rows = []
for col, short in VARS.items():
    for s, slab in SEX_LABEL.items():
        g1 = df.loc[(df["組別"] == 1) & (df["性別"] == s), col].astype(float)
        g2 = df.loc[(df["組別"] == 2) & (df["性別"] == s), col].astype(float)
        t, pt = stats.ttest_ind(g1, g2, equal_var=False)
        d, lo, hi = cohen_d(g1, g2)
        strat_rows.append({
            "變項": short, "性別": slab,
            "組1 n": len(g1), "組1 平均": g1.mean(),
            "組2 n": len(g2), "組2 平均": g2.mean(),
            "平均差": g1.mean() - g2.mean(), "t": t, "p": pt,
            "Cohen d": d, "顯著性": stars(pt),
        })
strat = pd.DataFrame(strat_rows)
P(strat.round(4).to_string(index=False))

P("")
P("[ 各組內的性別差異 ]")
sex_rows = []
for col, short in VARS.items():
    for g, glab in GROUP_LABEL.items():
        m = df.loc[(df["組別"] == g) & (df["性別"] == 1), col].astype(float)
        f = df.loc[(df["組別"] == g) & (df["性別"] == 2), col].astype(float)
        t, pt = stats.ttest_ind(m, f, equal_var=False)
        sex_rows.append({"變項": short, "組別": glab,
                         "男平均": m.mean(), "女平均": f.mean(),
                         "差異": m.mean() - f.mean(),
                         "t": t, "p": pt, "顯著性": stars(pt)})
sexdiff = pd.DataFrame(sex_rows)
P(sexdiff.round(4).to_string(index=False))

# ---------- 5. 相關分析 ----------
P("")
P("=" * 78)
P("5. 相關分析 (三項指標 + 年齡)")
P("=" * 78)
cols = list(VARS.keys()) + ["年齡"]
short_names = list(VARS.values()) + ["年齡"]
pear = df[cols].corr(method="pearson")
pear.index = pear.columns = short_names
P("")
P("Pearson r (全體 n=22)：")
P(pear.round(3).to_string())

corr_rows = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        a, b = df[cols[i]].astype(float), df[cols[j]].astype(float)
        r, pr = stats.pearsonr(a, b)
        rho, prho = stats.spearmanr(a, b)
        corr_rows.append({"配對": short_names[i] + " ~ " + short_names[j],
                          "Pearson r": r, "p": pr,
                          "Spearman rho": rho, "p(rho)": prho,
                          "顯著性": stars(pr)})
corr = pd.DataFrame(corr_rows)
P("")
P(corr.round(4).to_string(index=False))

P("")
P("[ 組內相關 (排除組間差異的干擾) ]")
within_rows = []
for g, glab in GROUP_LABEL.items():
    sub = df[df["組別"] == g]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, pr = stats.pearsonr(sub[cols[i]].astype(float),
                                   sub[cols[j]].astype(float))
            within_rows.append({"組別": glab,
                                "配對": short_names[i] + " ~ " + short_names[j],
                                "n": len(sub), "Pearson r": r, "p": pr,
                                "顯著性": stars(pr)})
within = pd.DataFrame(within_rows)
P(within.round(4).to_string(index=False))

# ---------- 6. 臨床切點 ----------
P("")
P("=" * 78)
P("6. 臨床判讀切點分布")
P("=" * 78)
cut = pd.DataFrame({
    "組別名": df["組別名"],
    "BMI>=27 (肥胖)": df["BMI (kg/m²)"] >= 27,
    "BMI>=24 (過重)": df["BMI (kg/m²)"] >= 24,
    "血糖>=126 (糖尿病)": df["血糖 (mg/dL)"] >= 126,
    "血糖>=100 (偏高)": df["血糖 (mg/dL)"] >= 100,
    "收縮壓>=140 (高血壓)": df["收縮壓 (mmHg)"] >= 140,
    "收縮壓>=130 (偏高)": df["收縮壓 (mmHg)"] >= 130,
})
tab = cut.groupby("組別名").agg(["sum", "count"])
cut_rows = []
for c in cut.columns[1:]:
    row = {"指標": c}
    for g in sorted(df["組別名"].unique()):
        s = int(tab.loc[g, (c, "sum")])
        n = int(tab.loc[g, (c, "count")])
        row[g] = "%d/%d (%.0f%%)" % (s, n, s / n * 100)
    cut_rows.append(row)
cuts = pd.DataFrame(cut_rows)
P(cuts.to_string(index=False))

# ================= 繪圖 =================
PAL = {GROUP_LABEL[1]: "#D1495B", GROUP_LABEL[2]: "#3C8DAD"}
order = [GROUP_LABEL[1], GROUP_LABEL[2]]


def annot(ax, p, label):
    """在兩箱之上畫顯著性橫線"""
    lo, hi = ax.get_ylim()
    rng = hi - lo
    h = hi - rng * .12
    ax.plot([0, 0, 1, 1], [h, h + rng * .02, h + rng * .02, h],
            lw=1.2, c="#444")
    txt = ("%s\np = %.2e" % (label, p)) if p < .001 else ("%s\np = %.3f" % (label, p))
    ax.text(.5, h + rng * .035, txt, ha="center", va="bottom",
            fontsize=9.5, color="#222")
    ax.set_ylim(lo, hi + rng * .10)


# --- 圖1：依組別 ---
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
for ax, (col, short) in zip(axes, VARS.items()):
    sns.boxplot(data=df, x="組別名", y=col, order=order, hue="組別名",
                hue_order=order, palette=PAL, legend=False, width=.55,
                linewidth=1.4, fliersize=0, ax=ax,
                medianprops=dict(color="white", lw=2),
                boxprops=dict(alpha=.85))
    sns.stripplot(data=df, x="組別名", y=col, order=order, ax=ax,
                  color="#2b2b2b", size=5.5, alpha=.75, jitter=.12)
    means = df.groupby("組別名")[col].mean().reindex(order)
    ax.scatter(range(len(order)), means, marker="D", s=55, c="white",
               edgecolors="#111", zorder=5, linewidths=1.2, label="平均數")
    r = tests.loc[tests["變項"] == short].iloc[0]
    annot(ax, r["t檢定 p"], r["顯著性"])
    ax.set_title("%s  (Cohen d = %.2f, %s)" % (short, r["Cohen d"], r["效果量"]),
                 fontsize=13, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(col, fontsize=11)
axes[0].legend(loc="lower left", fontsize=9, framealpha=.9)
fig.suptitle("圖1　BMI、血糖、收縮壓之組間比較 (箱形圖 + 個別觀測值)",
             fontsize=15, y=.99)
fig.tight_layout(rect=[0, .02, 1, .95])
fig.savefig("fig1_boxplot_by_group.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# --- 圖2：組別 × 性別 ---
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
for k, (ax, (col, short)) in enumerate(zip(axes, VARS.items())):
    sns.boxplot(data=df, x="組別名", y=col, hue="性別名", order=order,
                hue_order=["男", "女"],
                palette={"男": "#4C78A8", "女": "#E45756"},
                width=.62, linewidth=1.3, fliersize=0, ax=ax,
                boxprops=dict(alpha=.8), medianprops=dict(color="white", lw=1.8))
    sns.stripplot(data=df, x="組別名", y=col, hue="性別名", order=order,
                  hue_order=["男", "女"], dodge=True, ax=ax,
                  palette={"男": "#1b3a5c", "女": "#7d1f20"}, size=5,
                  alpha=.85, jitter=.08, legend=False)
    ax.set_title(short, fontsize=13, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(col, fontsize=11)
    if k == 0:
        ax.legend(title="性別", fontsize=9, title_fontsize=9.5,
                  loc="center left")
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
fig.suptitle("圖2　組別 × 性別之分層箱形圖", fontsize=15, y=.99)
fig.tight_layout(rect=[0, .02, 1, .95])
fig.savefig("fig2_boxplot_group_sex.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# --- 圖3：標準化後三指標並列 + BMI~血糖散布 ---
long = df.melt(id_vars=["組別名", "性別名"], value_vars=list(VARS.keys()),
               var_name="指標", value_name="原始值")
long["指標"] = long["指標"].map(VARS)
long["Z 分數"] = long.groupby("指標")["原始值"].transform(
    lambda s: (s - s.mean()) / s.std(ddof=1))

fig = plt.figure(figsize=(14.5, 5.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
ax = fig.add_subplot(gs[0])
sns.boxplot(data=long, x="指標", y="Z 分數", hue="組別名", hue_order=order,
            palette=PAL, width=.6, linewidth=1.3, fliersize=0, ax=ax,
            boxprops=dict(alpha=.85), medianprops=dict(color="white", lw=1.8))
sns.stripplot(data=long, x="指標", y="Z 分數", hue="組別名", hue_order=order,
              dodge=True, ax=ax,
              palette={order[0]: "#2b2b2b", order[1]: "#2b2b2b"},
              size=4.5, alpha=.7, jitter=.08, legend=False)
ax.axhline(0, ls="--", lw=1, c="#888")
ax.set_title("三項指標標準化 (Z 分數) 後並列比較", fontsize=13, pad=8)
ax.set_xlabel("")
ax.legend(title="", fontsize=9.5, loc="upper right")

ax2 = fig.add_subplot(gs[1])
sns.scatterplot(data=df, x="BMI (kg/m²)", y="血糖 (mg/dL)", hue="組別名",
                hue_order=order, style="性別名", palette=PAL, s=110,
                edgecolor="#222", ax=ax2)
sns.regplot(data=df, x="BMI (kg/m²)", y="血糖 (mg/dL)", scatter=False,
            color="#555", line_kws=dict(lw=1.3, ls="--"), ax=ax2, ci=95)
r_all, p_all = stats.pearsonr(df["BMI (kg/m²)"], df["血糖 (mg/dL)"])
ax2.set_title("BMI 與血糖之關聯  (全體 r = %.3f, p = %.2e)" % (r_all, p_all),
              fontsize=13, pad=8)
ax2.legend(fontsize=8.5, loc="upper left", framealpha=.9)
fig.suptitle("圖3　標準化比較與 BMI–血糖關聯", fontsize=15, y=.99)
fig.tight_layout(rect=[0, .02, 1, .95])
fig.savefig("fig3_zscore_and_scatter.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ---------- 匯出 ----------
with pd.ExcelWriter("DM_statistics.xlsx", engine="openpyxl") as w:
    desc.round(3).to_excel(w, sheet_name="描述性統計", index=False)
    assume.round(4).to_excel(w, sheet_name="前提假設", index=False)
    tests.round(4).to_excel(w, sheet_name="兩組比較", index=False)
    strat.round(4).to_excel(w, sheet_name="性別分層", index=False)
    sexdiff.round(4).to_excel(w, sheet_name="組內性別差異", index=False)
    corr.round(4).to_excel(w, sheet_name="相關分析", index=False)
    within.round(4).to_excel(w, sheet_name="組內相關", index=False)
    cuts.to_excel(w, sheet_name="臨床切點", index=False)

with open("DM_analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))

print("")
print("已輸出：fig1_boxplot_by_group.png / fig2_boxplot_group_sex.png / "
      "fig3_zscore_and_scatter.png / DM_statistics.xlsx / DM_analysis_report.txt")
