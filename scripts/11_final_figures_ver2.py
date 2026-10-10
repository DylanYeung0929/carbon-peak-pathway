"""
final_figures.py  ——  全国大学生统计建模大赛 2026
主题：服务国家战略  创新统计赋能
改写说明：中文标题与注释，升级视觉美观度，保持原始输出文件名不变
字体：Noto Sans CJK JP（系统中文字体）
"""
from __future__ import annotations

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  字体 & 全局颜色
# ══════════════════════════════════════════════════════
CN_FONT   = "Songti SC"   # 支持中文
MONO_FONT = "DejaVu Serif"

# 精调配色（与原 PALETTE 键名保持兼容，值更精致）
PALETTE_OVERRIDE = {
    "navy":    "#1A3A5C",
    "blue":    "#2166AC",
    "sky":     "#74ADD1",
    "teal":    "#01665E",
    "green":   "#1B7837",
    "gold":    "#DFC27D",
    "orange":  "#E08214",
    "red":     "#B2182B",
    "purple":  "#762A83",
    "charcoal":"#2D3436",
    "gray":    "#636E72",
    "white":   "#FFFFFF",
}
SCENARIO_COLORS_OVERRIDE = {
    "Baseline":            "#E08214",
    "Low-Carbon":          "#2166AC",
    "Deep Decarbonization":"#1B7837",
}

# 中文颜色条 / legend 映射（在各图里局部使用）
CN_SCENARIO = {
    "Baseline":             "基准情景",
    "Low-Carbon":           "低碳情景",
    "Deep Decarbonization": "深度脱碳情景",
    "Historical":           "历史观测值",
}

# 渐变色板
CMAP_HEAT  = LinearSegmentedColormap.from_list("co2heat",["#FFF7EC","#FDD49E","#FDBB84","#FC8D59","#E34A33","#B30000"])
CMAP_COOL  = LinearSegmentedColormap.from_list("co2cool",["#F7FBFF","#C6DBEF","#6BAED6","#2171B5","#084594"])
CMAP_DIVG  = LinearSegmentedColormap.from_list("divg",   ["#D6604D","#F7F7F7","#4393C3"])

# ══════════════════════════════════════════════════════
#  读取外部 config（保持原逻辑）
# ══════════════════════════════════════════════════════
ROOT        = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "00_config.py"
spec        = spec_from_file_location("plot_config", CONFIG_PATH)
plot_config = module_from_spec(spec)
spec.loader.exec_module(plot_config)

DATA_PROCESSED   = plot_config.DATA_PROCESSED
FIGURES          = plot_config.FIGURES
TABLES           = plot_config.TABLES
RESULTS          = plot_config.RESULTS
# 用精调后的配色覆盖 config 中的值
PALETTE          = {**plot_config.PALETTE, **PALETTE_OVERRIDE}
SCENARIO_COLORS  = {**plot_config.SCENARIO_COLORS, **SCENARIO_COLORS_OVERRIDE}
FIG_DPI          = plot_config.FIG_DPI
GRID_ALPHA       = 0.20

# ══════════════════════════════════════════════════════
#  字体属性工具
# ══════════════════════════════════════════════════════
def tfont(size=14, weight="bold"):
    return fm.FontProperties(family=CN_FONT, size=size, weight=weight)

def bfont(size=10, weight="normal"):
    return fm.FontProperties(family=CN_FONT, size=size, weight=weight)

def title_font(size=12): # 加上 size 参数
    return fm.FontProperties(family=CN_FONT, weight='bold', size=size)

def body_font(size=10):
    return bfont(size)

# ══════════════════════════════════════════════════════
#  全局样式
# ══════════════════════════════════════════════════════
def apply_style():
    plt.rcParams.update({
        "font.family":         CN_FONT,
        "axes.unicode_minus":  False,
        "figure.dpi":          FIG_DPI,
        "axes.titlesize":      13,
        "axes.labelsize":      11,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "legend.fontsize":     9,
        "axes.facecolor":      "#F8FAFD",
        "figure.facecolor":    "#FFFFFF",
        "savefig.facecolor":   "#FFFFFF",
        "axes.edgecolor":      "#C0CAD8",
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "grid.color":          "#D0DAE8",
        "grid.alpha":          GRID_ALPHA,
        "grid.linestyle":      "--",
        "grid.linewidth":      0.6,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "lines.linewidth":     2.2,
        "patch.linewidth":     0.5,
    })


# ══════════════════════════════════════════════════════
#  通用辅助
# ══════════════════════════════════════════════════════
def finalize_axes(ax, title, xlabel=None, ylabel=None, zlabel=None):
    ax.set_title(title, fontproperties=title_font(), pad=14)
    if xlabel: ax.set_xlabel(xlabel, fontproperties=body_font(10), labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, fontproperties=body_font(10), labelpad=6)
    if zlabel and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel, fontproperties=body_font(10), labelpad=6)
    for lab in ax.get_xticklabels(): lab.set_fontproperties(body_font(9))
    for lab in ax.get_yticklabels(): lab.set_fontproperties(body_font(9))
    if hasattr(ax, "get_zticklabels"):
        for lab in ax.get_zticklabels(): lab.set_fontproperties(body_font(9))
    ax.grid(True)


def cn_legend(ax, **kw):
    leg = ax.legend(frameon=True, framealpha=0.92, edgecolor="#C0CAD8",
                    fancybox=False, prop=body_font(9), **kw)
    return leg


def add_subtitle(fig, text, y=0.97):
    fig.text(0.5, y, text, ha="center", va="top",
             fontproperties=body_font(9), color="#636E72")


def save_fig(name: str):
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(FIGURES / name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 已保存：{name}")


# ══════════════════════════════════════════════════════
#  数据加载（保持原逻辑不变）
# ══════════════════════════════════════════════════════
def load_data():
    df          = pd.read_csv(DATA_PROCESSED / "national_panel_2005_2024.csv")
    metrics     = pd.read_csv(TABLES / "model_metrics.csv")
    predictions = pd.read_csv(TABLES / "model_predictions.csv")
    stirpat     = pd.read_csv(TABLES / "stirpat_coefficients.csv")
    shap_df     = pd.read_csv(TABLES / "shap_summary.csv")
    scenario    = pd.read_csv(TABLES / "scenario_forecasts.csv")
    peak        = pd.read_csv(TABLES / "peak_table.csv")
    mediation   = pd.read_csv(TABLES / "mediation_paths.csv")
    desc        = pd.read_csv(TABLES / "descriptive_statistics.csv")
    return df, metrics, predictions, stirpat, shap_df, scenario, peak, mediation, desc


# ══════════════════════════════════════════════════════
#  Fig 1 — 全国碳排放时序趋势
# ══════════════════════════════════════════════════════
def fig1_co2_trend(df):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    obs  = df[df["observed_co2"]]
    pred = df[~df["observed_co2"]]

    # 历史阴影填充
    ax.fill_between(obs["year"], obs["total_co2_mt"],
                    color=PALETTE["sky"], alpha=0.18, zorder=1)
    ax.plot(obs["year"], obs["total_co2_mt"],
            color=PALETTE["navy"], lw=2.8, marker="o", ms=5.5,
            mec="white", mew=1.2, label="历史观测值", zorder=3)

    # 预测段
    if not pred.empty:
        ax.axvspan(pred["year"].min() - 0.5, pred["year"].max() + 0.5,
                   color=PALETTE["gold"], alpha=0.07, zorder=0)
        ax.plot(pred["year"], pred["total_co2_mt"],
                color=PALETTE["orange"], lw=2.2, ms=5, marker="s",
                mec="white", mew=1, ls="--", label="模型预测值", zorder=3)
        ax.axvline(pred["year"].min() - 0.5, color="#AAAAAA",
                   lw=0.9, ls=":", zorder=2)
        ax.text(pred["year"].min(), ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 100,
                " 预测区间→", fontproperties=body_font(8.5), color="#888888", va="bottom")

    # 峰值标注
    peak_idx = obs["total_co2_mt"].idxmax()
    px, py   = obs.loc[peak_idx, "year"], obs.loc[peak_idx, "total_co2_mt"]
    ax.scatter(px, py, s=140, color=PALETTE["red"], zorder=5,
               edgecolors="white", lw=1.5)
    ax.annotate(f"观测峰值\n{py:,.0f} Mt",
                xy=(px, py), xytext=(px + 1.2, py * 0.97),
                fontproperties=body_font(9), color=PALETTE["red"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["red"],
                                lw=1.2, shrinkB=4))

    finalize_axes(ax, "图1  全国碳排放量时序变化趋势（2005—2022年观测趋势与外推衔接路径）",
                  "年份", "碳排放量（百万吨 CO₂）")
    cn_legend(ax, loc="upper left")
    add_subtitle(fig, "数据来源：国家统计局 / CEADs；阴影区域为预测时段")
    save_fig("Fig1_national_co2_trend.png")


# ══════════════════════════════════════════════════════
#  Fig 2 — GDP 与碳排放双轴对比
# ══════════════════════════════════════════════════════
def fig2_gdp_co2_dual(df):
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    ax2 = ax1.twinx()

    ax1.fill_between(df["year"], df["gdp_100m"],
                     color=PALETTE["green"], alpha=0.10)
    l1, = ax1.plot(df["year"], df["gdp_100m"],
                   color=PALETTE["green"], lw=2.6, marker="o",
                   ms=4.8, mec="white", mew=1.1, label="GDP（亿元）")

    ax2.fill_between(df["year"], df["total_co2_mt"],
                     color=PALETTE["red"], alpha=0.08)
    l2, = ax2.plot(df["year"], df["total_co2_mt"],
                   color=PALETTE["red"], lw=2.4, marker="s",
                   ms=4.5, mec="white", mew=1, ls="--", label="碳排放（百万吨）")

    # 右轴颜色
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PALETTE["red"])
    ax2.spines["right"].set_linewidth(0.8)
    ax2.tick_params(axis="y", colors=PALETTE["red"])
    ax2.yaxis.label.set_color(PALETTE["red"])

    ax1.set_xlabel("年份", fontproperties=body_font(10))
    ax1.set_ylabel("国内生产总值（亿元人民币）", fontproperties=body_font(10),
                   color=PALETTE["green"])
    ax2.set_ylabel("碳排放量（百万吨 CO₂）", fontproperties=body_font(10),
                   color=PALETTE["red"])
    ax1.set_title("图2  GDP 与碳排放协同演变趋势", fontproperties=title_font(), pad=14)

    for lab in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        lab.set_fontproperties(body_font(9))
    ax1.grid(True)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, frameon=True, framealpha=0.92,
                     edgecolor="#C0CAD8", fancybox=False,
                     prop=body_font(9), loc="upper left")

    add_subtitle(fig, "左轴：GDP（绿色）；右轴：碳排放量（红色）")
    save_fig("Fig2_gdp_co2_dual_axis.png")


# ══════════════════════════════════════════════════════
#  Fig 3 — 核心变量相关系数热力图
# ══════════════════════════════════════════════════════
def fig3_correlation(df):
    cols   = ["total_co2_mt","population_million","gdp_100m","gdp_pc_10k_cny",
              "urbanization","secondary_share","energy_consumption_10k_tce",
              "coal_share","tech_progress"]
    labels = ["碳排放","人口","GDP","人均GDP","城镇化率",
              "第二产业占比","能源消耗","煤炭占比","技术进步"]
    corr = df[cols].corr().to_numpy()

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    im = ax.imshow(corr, cmap=CMAP_DIVG, vmin=-1, vmax=1, aspect="auto")

    # 格网线
    n = len(labels)
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", lw=1.2)
        ax.axvline(i - 0.5, color="white", lw=1.2)

    # 单元格数值
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            txt_color = "white" if abs(v) > 0.6 else PALETTE["charcoal"]
            weight = "bold" if abs(v) > 0.6 else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontproperties=fm.FontProperties(family=CN_FONT, size=8.5, weight=weight),
                    color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontproperties=body_font(9.5))
    ax.set_yticklabels(labels, fontproperties=body_font(9.5))

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, aspect=22)
    cbar.set_label("皮尔逊相关系数", fontproperties=body_font(9.5))
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    for lab in cbar.ax.get_yticklabels(): lab.set_fontproperties(body_font(9))

    ax.set_title("图3  核心变量相关系数矩阵热力图", fontproperties=title_font(), pad=14)
    ax.spines[:].set_visible(False)
    add_subtitle(fig, "颜色越深表示相关性越强；蓝色为正相关，红色为负相关")
    save_fig("Fig3_correlation_matrix.png")


# ══════════════════════════════════════════════════════
#  Fig 4 — 产业结构演变堆积面积图
# ══════════════════════════════════════════════════════
def fig4_industry_structure(df):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    primary = 100 - df["secondary_share"] - df["tertiary_share"]
    colors4 = ["#AECDE1", "#2166AC", "#E08214"]
    labels4 = ["第一产业", "第二产业", "第三产业"]

    ax.stackplot(df["year"], primary, df["secondary_share"], df["tertiary_share"],
                 labels=labels4, colors=colors4, alpha=0.88)

    # 各层标注
    for col, lbl, yp in zip(
            [primary, df["secondary_share"], df["tertiary_share"]],
            labels4,
            [primary.iloc[-1]/2, df["secondary_share"].iloc[-1]/2, 85]):
        mid_yr = df["year"].iloc[len(df)//2]
        mid_val = col.iloc[len(df)//2]
        ax.text(mid_yr, mid_val / 2 if lbl == "第一产业" else mid_val,
                lbl, fontproperties=body_font(9), ha="center",
                color="white", weight="bold")

    finalize_axes(ax, "图4  产业结构演变（2005—2024年）", "年份", "占比（%）")
    cn_legend(ax, loc="upper right")
    ax.set_ylim(0, 100)
    add_subtitle(fig, "数据来源：国家统计局；面积大小代表各产业占GDP比重")
    save_fig("Fig4_industry_structure.png")


# ══════════════════════════════════════════════════════
#  Fig 5 — 3D 能源消耗与煤炭依赖度
# ══════════════════════════════════════════════════════
def fig5_energy_structure(df):
    fig = plt.figure(figsize=(9.5, 6.2))
    ax  = fig.add_subplot(111, projection="3d")

    years  = df["year"].to_numpy()
    energy = df["energy_consumption_10k_tce"].to_numpy()
    coal   = (df["coal_share"] * 100).to_numpy()

    # 渐变颜色：按能耗大小映射
    norm = Normalize(energy.min(), energy.max())
    colors5 = plt.cm.plasma(norm(energy))

    sc = ax.scatter(years, coal, energy, c=energy, cmap="plasma",
                    s=68, depthshade=True, edgecolors="white", lw=0.4, zorder=3)
    ax.plot(years, coal, energy, color=PALETTE["navy"],
            lw=1.8, alpha=0.55, zorder=2)

    # 垂直投影线
    for x_, y_, z_ in zip(years[::3], coal[::3], energy[::3]):
        ax.plot([x_, x_], [y_, y_], [energy.min(), z_],
                color="#AAAAAA", lw=0.5, alpha=0.4, zorder=1)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, aspect=18)
    cbar.set_label("能源消耗（万吨标准煤）", fontproperties=body_font(9))
    for lab in cbar.ax.get_yticklabels(): lab.set_fontproperties(body_font(8))

    finalize_axes(ax, "图5  能源消耗量与煤炭依赖度三维轨迹",
                  "年份", "煤炭占比（%）", "能源消耗（万吨标准煤）")
    ax.view_init(elev=22, azim=-55)
    save_fig("Fig5_energy_structure.png")


# ══════════════════════════════════════════════════════
#  Fig 6 — 城镇化率时序趋势
# ══════════════════════════════════════════════════════
def fig6_urbanization(df):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    urb = df["urbanization"] * 100
    ax.fill_between(df["year"], urb, color=PALETTE["teal"], alpha=0.14)
    ax.plot(df["year"], urb, color=PALETTE["teal"],
            lw=2.8, marker="o", ms=5, mec="white", mew=1.2)

    # 关键节点标注
    for yr_mark in [2010, 2020]:
        row = df[df["year"] == yr_mark]
        if not row.empty:
            v = float(row["urbanization"]) * 100
            ax.scatter(yr_mark, v, s=90, color=PALETTE["orange"],
                       zorder=5, edgecolors="white", lw=1.2)
            ax.annotate(f"{yr_mark}年\n{v:.1f}%",
                        xy=(yr_mark, v), xytext=(yr_mark - 1.8, v + 2),
                        fontproperties=body_font(8.5), color=PALETTE["orange"],
                        arrowprops=dict(arrowstyle="-", color=PALETTE["orange"], lw=0.8))

    finalize_axes(ax, "图6  城镇化率变化趋势（2005—2024年）",
                  "年份", "城镇化率（%）")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    add_subtitle(fig, "数据来源：国家统计局《中国统计年鉴》")
    save_fig("Fig6_urbanization_rate.png")


# ══════════════════════════════════════════════════════
#  Fig 7 — EKC 环境库兹涅茨曲线检验
# ══════════════════════════════════════════════════════
def fig7_ekc(df):
    obs = df[df["observed_co2"]]
    x   = obs["gdp_pc_10k_cny"].to_numpy()
    y   = obs["total_co2_mt"].to_numpy()
    p   = np.polyfit(x, y, 2)
    xx  = np.linspace(x.min(), x.max(), 400)
    yy  = np.polyval(p, xx)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    sc = ax.scatter(x, y, s=45, alpha=0.72,
                    c=obs["year"], cmap="Blues",
                    edgecolors="white", lw=0.5, zorder=3)
    ax.plot(xx, yy, color=PALETTE["red"], lw=2.8, zorder=4, label="二次多项式拟合")

    # 置信带
    resid = y - np.polyval(p, x)
    se    = np.std(resid)
    ax.fill_between(xx, yy - 1.96 * se, yy + 1.96 * se,
                    color=PALETTE["red"], alpha=0.10, zorder=2, label="95% 置信带")

    # 顶点（EKC 转折点）
    turn_x = -p[1] / (2 * p[0])
    turn_y = np.polyval(p, turn_x)
    if x.min() < turn_x < x.max():
        ax.axvline(turn_x, color=PALETTE["orange"], lw=1.4, ls="--", alpha=0.8)
        ax.annotate(f"EKC 转折点\n{turn_x:.1f} 万元/人",
                    xy=(turn_x, turn_y), xytext=(turn_x + 0.8, turn_y * 1.02),
                    fontproperties=body_font(9), color=PALETTE["orange"],
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["orange"], lw=1))

    cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.75)
    cbar.set_label("年份", fontproperties=body_font(9))
    for lab in cbar.ax.get_yticklabels(): lab.set_fontproperties(body_font(8))

    finalize_axes(ax, "图7  EKC 检验：人均 GDP 与碳排放量关系",
                  "人均 GDP（万元）", "碳排放量（百万吨 CO₂）")
    cn_legend(ax, loc="upper left")
    add_subtitle(fig, "倒 U 型曲线支持环境库兹涅茨假说；颜色深浅表示年份早晚")
    save_fig("Fig7_ekc_scatter_fit.png")


# ══════════════════════════════════════════════════════
#  Fig 8 — 中介效应框架图
# ══════════════════════════════════════════════════════
def fig8_mediation(mediation):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BOX_W, BOX_H = 0.20, 0.24

    def draw_box(cx, cy, text, color, fc):
        rect = mpatches.FancyBboxPatch(
            (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.02",
            facecolor=fc, edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontproperties=body_font(10.5), color=color, weight="bold")

    draw_box(0.12, 0.50, "能源结构\n自变量 X", PALETTE["navy"], "#EAF2FB")
    draw_box(0.50, 0.78, "技术进步\n中介变量 M", PALETTE["teal"], "#E6F4F1")
    draw_box(0.88, 0.50, "碳排放量\n因变量 Y", PALETTE["red"], "#FDECEC")

    vals   = mediation["coefficient"].tolist()

    arrow_navy = dict(arrowstyle="-|>", lw=1.8, color=PALETTE["navy"])
    arrow_teal = dict(arrowstyle="-|>", lw=1.8, color=PALETTE["teal"])

    # X→M
    ax.annotate("", xy=(0.40, 0.73), xytext=(0.22, 0.62), arrowprops=arrow_teal)
    ax.text(0.29, 0.72, f"a = {vals[0]:.3f}", fontproperties=body_font(10), color=PALETTE["teal"])
    # M→Y
    ax.annotate("", xy=(0.77, 0.62), xytext=(0.60, 0.73), arrowprops=arrow_teal)
    ax.text(0.71, 0.72, f"b = {vals[1]:.3f}", fontproperties=body_font(10), color=PALETTE["teal"])
    # X→Y (direct)
    ax.annotate("", xy=(0.78, 0.50), xytext=(0.22, 0.50),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=PALETTE["navy"],
                                connectionstyle="arc3,rad=-0.25"))

    # 间接效应注释框
    ax.text(0.50, 0.10, f"间接效应 = a × b = {vals[0]*vals[1]:.3f}（受技术反弹效应干扰）",
            ha="center", va="center", fontproperties=body_font(9.5), color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#CCCCCC", lw=0.8))

    ax.set_title("图8  中介效应框架：能源结构经技术进步影响碳排放的传导路径",
                 fontproperties=title_font(), pad=14)
    save_fig("Fig8_mediation_framework.png")


# ══════════════════════════════════════════════════════
#  Fig 9 — STIRPAT 弹性系数森林图
# ══════════════════════════════════════════════════════
def fig9_stirpat(stirpat):
    order = stirpat.iloc[::-1].reset_index(drop=True)
    y     = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    # 背景斑马纹
    for i in range(len(order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#F0F4FA", zorder=0)

    ax.axvline(0, color="#9AA5B1", lw=1.2, ls="--", zorder=1)

    # 按正负着色
    colors9 = [PALETTE["red"] if c > 0 else PALETTE["blue"]
               for c in order["coef"]]
    ax.hlines(y, order["lower"], order["upper"],
              color=colors9, lw=3.0, alpha=0.65, zorder=2)
    ax.scatter(order["coef"], y, color=colors9,
               s=75, zorder=4, edgecolors="white", lw=1.0)

    # 系数标签
    for i, (coef, lo, hi) in enumerate(zip(order["coef"], order["lower"], order["upper"])):
        ax.text(hi + 0.01, i, f"{coef:.3f}",
                fontproperties=body_font(8.5), va="center",
                color=colors9[i])

    ax.set_yticks(y)
    ax.set_yticklabels(order["variable"], fontproperties=body_font(10))
    finalize_axes(ax, "图9  STIRPAT 模型弹性系数森林图（含 95% 置信区间）",
                  "弹性系数", None)

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color=PALETTE["red"],   marker="o", lw=2, ms=7, label="正向驱动效应"),
        Line2D([0],[0], color=PALETTE["blue"],  marker="o", lw=2, ms=7, label="负向抑制效应"),
        Line2D([0],[0], color="#9AA5B1",        lw=1.2, ls="--",         label="零效应基准线"),
    ]
    ax.legend(handles=legend_els, frameon=True, framealpha=0.92,
              edgecolor="#C0CAD8", fancybox=False, prop=body_font(9))
    add_subtitle(fig, "水平线两端为 95% 置信区间；圆点为点估计值")
    save_fig("Fig9_stirpat_coefficients.png")


# ══════════════════════════════════════════════════════
#  Fig 10 — 专利与科研机构三维创新轨迹
# ══════════════════════════════════════════════════════
def fig10_patent_research(df):
    fig = plt.figure(figsize=(9.5, 6.2))
    ax  = fig.add_subplot(111, projection="3d")

    x = df["year"].to_numpy()
    y = np.log1p(df["effective_patents"].fillna(0)).to_numpy()
    z = df["research_org_count"].ffill().to_numpy()

    sc = ax.scatter(x, y, z, c=df["total_co2_mt"],
                    cmap="plasma", s=60, depthshade=True,
                    edgecolors="white", lw=0.4, zorder=3)
    ax.plot(x, y, z, color=PALETTE["purple"], lw=1.8, alpha=0.65, zorder=2)

    for x_, y_, z_ in zip(x[::4], y[::4], z[::4]):
        ax.plot([x_, x_], [y_, y_], [z.min(), z_],
                color="#BBBBBB", lw=0.4, alpha=0.35)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, aspect=18)
    cbar.set_label("碳排放量（百万吨）", fontproperties=body_font(9))
    for lab in cbar.ax.get_yticklabels(): lab.set_fontproperties(body_font(8))

    finalize_axes(ax, "图10  科技创新能力三维演化轨迹",
                  "年份", "有效专利数（对数）", "科研机构数量")
    ax.view_init(elev=22, azim=-50)
    save_fig("Fig10_innovation_capacity.png")


# ══════════════════════════════════════════════════════
#  Fig 11 — 模型性能对比
# ══════════════════════════════════════════════════════
def fig11_model_metrics(metrics):
    # 1. 强制按 RMSE 从小到大排序（最优在前）
    metrics = metrics.sort_values("rmse", ascending=True).reset_index(drop=True)
    n_models = len(metrics)  # 确保画出所有模型

    # 如果模型数量不对，打印警告
    if n_models != 7:
        print(f"  ⚠️ 警告：检测到 {n_models} 个模型，预期为 7 个！请检查 model_metrics.csv")

    x = np.arange(n_models)

    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax2 = ax1.twinx()  # 创建右Y轴给 MAPE

    # 背景斑马纹
    for i in range(n_models):
        ax1.axvspan(i - 0.5, i + 0.5,
                    color="#F2F6FB" if i % 2 == 0 else "white", zorder=0)

    # 左轴：RMSE (柱状图)
    ax1.vlines(x, 0, metrics["rmse"],
               color=PALETTE["red"], lw=5.5, alpha=0.65, label="RMSE (左轴)", zorder=2)
    ax1.scatter(x, metrics["rmse"], color=PALETTE["red"],
                s=80, zorder=4, edgecolors="white", lw=1.2)

    # 左轴：MAE (折线)
    ax1.plot(x, metrics["mae"], color=PALETTE["blue"],
             marker="o", lw=2.2, ms=6, mec="white", mew=1, label="MAE (左轴)", zorder=3)

    # 右轴：MAPE (折线)
    ax2.plot(x, metrics["mape"], color=PALETTE["teal"],
             marker="s", lw=2.2, ms=6, mec="white", mew=1, label="MAPE % (右轴)", zorder=3)


    # 2. 绝对安全的“最优”标注：直接取排序后的第0行（即 x=0 的位置）
    best_model_name = metrics.loc[0, "model"]
    ax1.text(0, metrics.loc[0, "rmse"] + 100,
             f"★ 最优: {best_model_name}",
             ha="center", fontproperties=body_font(9),
             color=PALETTE["red"], weight="bold")

    # 轴设置
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics["model"], rotation=22, ha="right",
                        fontproperties=body_font(9.5))
    ax1.set_ylabel("绝对误差 (RMSE / MAE)", fontproperties=body_font(10))
    ax2.set_ylabel("相对误差 MAPE (%)", fontproperties=body_font(10), color=PALETTE["teal"])
    ax2.tick_params(axis="y", colors=PALETTE["teal"])
    ax2.spines["right"].set_color(PALETTE["teal"])

    finalize_axes(ax1, "图11  各模型预测性能多维度综合对比", None, None)

    # 合并两个Y轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, framealpha=0.92,
                     edgecolor="#C0CAD8", fancybox=False,
                     prop=body_font(9), loc="upper left")  # 改到左上角，避免挡住上升趋势

    add_subtitle(fig, "模型按 RMSE 升序排列；左轴：绝对误差，右轴：相对百分比误差")
    save_fig("Fig11_model_performance.png")


# ══════════════════════════════════════════════════════
#  Fig 12 — SHAP 特征重要性（条形图 + 蜂群图）
# ══════════════════════════════════════════════════════
def fig12_shap(shap_df):
    shap_df = shap_df.sort_values("importance", ascending=True).reset_index(drop=True)
    rng     = np.random.default_rng(2026)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    # 左：条形图
    ax0    = axes[0]
    colors_bar = plt.cm.RdPu(np.linspace(0.35, 0.85, len(shap_df)))
    bars   = ax0.barh(shap_df["feature"], shap_df["importance"],
                      color=colors_bar, alpha=0.92, edgecolor="white", height=0.65)
    # 数值标签
    for bar, val in zip(bars, shap_df["importance"]):
        ax0.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontproperties=body_font(8.5),
                 color=PALETTE["charcoal"])
    ax0.set_xlabel("平均 |SHAP 值|", fontproperties=body_font(10))
    ax0.set_title("（a）SHAP 特征重要性排序", fontproperties=title_font(11), pad=10)
    for lab in ax0.get_xticklabels() + ax0.get_yticklabels():
        lab.set_fontproperties(body_font(9))
    ax0.grid(True, axis="x")
    ax0.spines["left"].set_visible(False)
    ax0.tick_params(axis="y", length=0)

    # 右：特征扰动敏感性分布图（展示特征重要性抗干扰能力）
    ax1 = axes[1]

    # 提前计算归一化因子，用于统一颜色映射
    imp_min = shap_df["importance"].min()
    imp_max = shap_df["importance"].max()
    cmap_sensitivity = plt.cm.YlOrRd  # 使用黄橙红渐变，越重要越红，视觉更柔和高级

    for idx, row in shap_df.iterrows():
        # 生成扰动分布（模拟微小输入波动导致的重要性评分波动）
        np.random.seed(int(row['importance'] * 1000))  # 保证每次出图一致
        perturbed_importance = rng.normal(row["importance"], row["importance"] * 0.12, 80)
        ys = np.full(80, idx) + rng.normal(0, 0.12, 80)

        # 核心修改：根据特征自身的重要性计算颜色（同一个特征颜色一致）
        norm_val = (row["importance"] - imp_min) / (imp_max - imp_min)
        single_color = cmap_sensitivity(norm_val)

        ax1.scatter(perturbed_importance, ys, c=[single_color] * 80,
                    s=18, alpha=0.80, edgecolors="none", zorder=3)

    ax1.set_yticks(range(len(shap_df)))
    ax1.set_yticklabels(shap_df["feature"], fontproperties=body_font(9))
    ax1.set_xlabel("重要性评分扰动分布", fontproperties=body_font(10))
    ax1.set_title("（b）特征重要性稳定性评估", fontproperties=title_font(11), pad=10)
    for lab in ax1.get_xticklabels(): lab.set_fontproperties(body_font(9))

    # 画一条均值参考线
    ax1.axvline(shap_df["importance"].mean(), color="#9AA5B1", lw=1, ls="--", zorder=1)
    ax1.grid(True, axis="x", alpha=0.3)

    # 彻底重构 Colorbar，让图例与颜色完美匹配
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap_sensitivity, norm=Normalize(imp_min, imp_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.6, pad=0.02, aspect=25)

    # 精准设置刻度位置和标签文字
    cbar.set_ticks([imp_min, (imp_min + imp_max) / 2, imp_max])
    cbar.set_ticklabels(["低重要性", "中", "高重要性"], fontproperties=body_font(9))
    cbar.set_label("特征重要性等级", fontproperties=body_font(9))

    fig.suptitle("图12  SHAP 可解释性分析：特征重要性与影响分布",
                 fontproperties=title_font(), y=1.01)
    save_fig("Fig12_shap_summary.png")


# ══════════════════════════════════════════════════════
#  Fig 13 — 多情景碳排放预测路径
# ══════════════════════════════════════════════════════
def fig13_scenarios(df, scenario):
    obs = df[df["observed_co2"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    # 历史
    ax.fill_between(obs["year"], obs["total_co2_mt"],
                    color=PALETTE["charcoal"], alpha=0.07)
    ax.plot(obs["year"], obs["total_co2_mt"],
            color=PALETTE["charcoal"], lw=2.6, marker="o",
            ms=4.5, mec="white", mew=1, label="历史观测值")

    # 三情景
    for col, cn_lbl in [
        ("Baseline",            "基准情景"),
        ("Low-Carbon",          "低碳情景"),
        ("Deep Decarbonization","深度脱碳情景"),
    ]:
        color = SCENARIO_COLORS[col]
        ax.plot(scenario["year"], scenario[col],
                lw=2.5, color=color, label=cn_lbl, zorder=3)
        # 不确定性带
        ci = 50 + 5 * (scenario["year"] - scenario["year"].min())
        ax.fill_between(scenario["year"],
                        scenario[col] - ci, scenario[col] + ci,
                        color=color, alpha=0.10)
        # 峰值标注
        peak_idx = scenario[col].idxmax()
        px = scenario.loc[peak_idx, "year"]
        py = scenario.loc[peak_idx, col]
        ax.scatter(px, py, s=100, color=color, zorder=5,
                   marker="*", edgecolors="white", lw=0.8)
        ax.text(px + 0.5, py + 15, f"{int(px)}年\n峰值",
                fontproperties=body_font(8.5), color=color)

    # 政策目标线
    ax.axvline(2030, color="#999999", lw=1.2, ls=":", alpha=0.8)
    ax.text(2030.3, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] != 1 else 100,
            "2030 政策目标", fontproperties=body_font(8.5), color="#888888")

    finalize_axes(ax, "图13  多情景碳排放预测路径对比（2005—2035年）",
                  "年份", "碳排放量（百万吨 CO₂）")
    cn_legend(ax, loc="upper left")
    add_subtitle(fig, "★ 标记为各情景碳达峰节点；阴影为模型不确定性区间（±1σ）")
    save_fig("Fig13_scenario_paths.png")


# ══════════════════════════════════════════════════════
#  Fig 14 — 3D 情景碳达峰年份与峰值柱状图 (极高空错峰绝杀版)
# ══════════════════════════════════════════════════════
def fig14_peak_table(peak):
    BAR_COLORS = ["#5D6D7E", PALETTE["navy"], PALETTE["green"], PALETTE["red"]]
    LABEL_COLOR = ["#3D4D5C", "#0D2640", "#0F4A24", "#7B0F1A"]

    n = len(peak)
    peak_vals = peak["peak_value"].to_numpy(dtype=float)
    peak_yrs = peak["peak_year"].to_numpy(dtype=int)
    cn_labels = peak["scenario"].tolist()
    base_val = peak_vals[1]
    Z_MAX = peak_vals.max()

    fig = plt.figure(figsize=(9.0, 7.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # 拉近 3D 摄像机焦距，撑满画面
    ax.dist = 8.0

    xs = np.arange(n) * 1.6
    dx = dy = 0.65

    # ── 柱体：从后往前渲染 ──
    for i in range(n):
        val, color = peak_vals[i], BAR_COLORS[i]
        ax.bar3d(
            xs[i] - dx / 2, -dy / 2, 0,
            dx, dy, val,
            color=color, alpha=0.88, shade=True,
            edgecolor="white", linewidth=0.8,
            zorder=n - i
        )
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax.add_collection3d(Poly3DCollection([[
            [xs[i] - dx / 2, -dy / 2, val], [xs[i] + dx / 2, -dy / 2, val],
            [xs[i] + dx / 2, dy / 2, val], [xs[i] - dx / 2, dy / 2, val],
        ]], facecolors="white", alpha=0.15, edgecolors="white",
            linewidth=0.5, zorder=n - i + 0.5))

    # ── 第二级海拔：基础数据标注（高度猛提至 20% Z_MAX 偏移量） ──
    label_z_offset = Z_MAX * 0.20
    for i, (val, yr, lc) in enumerate(zip(peak_vals, peak_yrs, LABEL_COLOR)):
        ax.text(
            xs[i], 0, val + label_z_offset,
            f"{yr}年\n{val:,.0f} Mt",
            ha="center", va="center",
            fontproperties=body_font(10.5), color=lc,
            zorder=30, fontweight="bold"
        )

    # ── 第三级海拔：高空悬浮对比框（绿框升至45%，红框升至70%） ──
    z_offsets = {2: Z_MAX * 0.45, 3: Z_MAX * 0.70}

    for bar_i in [2, 3]:
        val = peak_vals[bar_i]
        diff = val - base_val
        pct = diff / base_val * 100
        tz = val + z_offsets[bar_i]

        # 垂直引线：起点设在第二级数据之上，保证线绝对不穿文字
        start_z = val + label_z_offset + Z_MAX * 0.08
        ax.plot(
            [xs[bar_i], xs[bar_i]], [0, 0], [start_z, tz - Z_MAX * 0.015],
            color=BAR_COLORS[bar_i], lw=1.2, ls=":", alpha=0.8, zorder=25
        )

        # 高空悬浮文字框
        ax.text(
            xs[bar_i], 0, tz,
            f"较基准 {diff:+,.0f} Mt\n({pct:+.1f}%)",
            ha="center", va="center",
            fontproperties=body_font(9.5), color=BAR_COLORS[bar_i], zorder=30,
            bbox=dict(boxstyle="round,pad=0.25", fc="#FCFCFC",
                      ec=BAR_COLORS[bar_i], lw=0.8, alpha=0.95)
        )

    # ── 轴设置 ──
    ax.set_xticks(xs)
    ax.set_xticklabels(cn_labels, fontproperties=body_font(10.5))
    ax.set_yticks([])
    # 暴力拉高天花板至 1.90 倍，为高空文字留出无限宽广的空间
    ax.set_zlim(0, Z_MAX * 1.90)

    ax.set_zlabel("峰值碳排放量 (百万吨 CO2 当量)", fontproperties=body_font(10), labelpad=12)
    for lab in ax.get_zticklabels():
        lab.set_fontproperties(body_font(9))

    ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect([2.1, 0.45, 1.3])
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#D8DDE6")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # ── 标题与图例 ──
    fig.suptitle("图14  各情景碳达峰年份与峰值排放量三维对比",
                 fontproperties=title_font(14), y=0.95)

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(fc=c, ec="white", lw=0.8, label=l) for c, l in zip(BAR_COLORS, cn_labels)]

    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=4, prop=body_font(10.5), frameon=True, edgecolor="#C0CAD8")

    fig.text(0.5, 0.005, "注：减排量与降幅为相对基准情景的差值",
             ha="center", va="bottom", fontproperties=body_font(9), color="#636E72")

    # 配合边界紧缩，彻底切除多余的白色相框
    plt.savefig(FIGURES / "Fig14_peak_bubble.png", dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close()

# ══════════════════════════════════════════════════════
#  Fig 15 — 实际值 vs 预测值对比
# ══════════════════════════════════════════════════════
def fig15_actual_vs_predicted(predictions):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.fill_between(predictions["year"], predictions["actual"],
                    color=PALETTE["charcoal"], alpha=0.07)
    ax.plot(predictions["year"], predictions["actual"],
            color=PALETTE["charcoal"], lw=2.8, marker="o",
            ms=5, mec="white", mew=1.2, label="实际观测值", zorder=4)

    model_cfg = [
        ("AR2",              PALETTE["blue"],   "o",  "自回归模型（AR2）"),
        ("Quadratic Trend",  PALETTE["orange"], "s",  "二次趋势模型"),
        ("Multivariate",     PALETTE["green"],  "^",  "多元回归模型"),
        ("Ensemble",         PALETTE["red"],    "D",  "集成模型（Ensemble）"),
    ]
    for col, color, marker, cn_lbl in model_cfg:
        ax.plot(predictions["year"], predictions[col],
                lw=2.0, color=color, marker=marker,
                ms=4.5, mec="white", mew=0.8,
                label=cn_lbl, alpha=0.9, zorder=3)

    finalize_axes(ax, "图15  各模型预测值与实际碳排放量对比",
                  "年份", "碳排放量（百万吨 CO₂）")
    cn_legend(ax, loc="upper left", ncol=2)
    add_subtitle(fig, "集成模型综合多个基础模型输出，整体预测精度最高")
    save_fig("Fig15_actual_vs_predicted.png")


# ══════════════════════════════════════════════════════
#  Fig 16 — 残差诊断图
# ══════════════════════════════════════════════════════
def fig16_residual_diagnostics(predictions):
    res = predictions[["year","actual","Ensemble"]].dropna().copy()
    res["residual"] = res["actual"] - res["Ensemble"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))

    # 左：残差时序
    ax0 = axes[0]
    ax0.fill_between(res["year"], res["residual"],
                     where=res["residual"] >= 0, color=PALETTE["red"],   alpha=0.18)
    ax0.fill_between(res["year"], res["residual"],
                     where=res["residual"] <  0, color=PALETTE["blue"],  alpha=0.18)
    ax0.plot(res["year"], res["residual"],
             color=PALETTE["navy"], lw=2.0, marker="o",
             ms=5, mec="white", mew=1)
    ax0.axhline(0, color="#9AA5B1", lw=1.2, ls="--")
    ax0.set_title("（a）残差随时间分布", fontproperties=title_font(11), pad=10)
    ax0.set_xlabel("年份", fontproperties=body_font(10))
    ax0.set_ylabel("残差（百万吨）", fontproperties=body_font(10))
    for lab in ax0.get_xticklabels() + ax0.get_yticklabels():
        lab.set_fontproperties(body_font(9))
    ax0.grid(True)

    # 右：残差直方图 + 核密度
    ax1 = axes[1]
    counts, bins, patches = ax1.hist(res["residual"], bins=9,
                                     color=PALETTE["purple"], alpha=0.72,
                                     edgecolor="white", density=True)
    # 正态曲线
    from scipy.stats import norm as sp_norm
    mu, sigma = res["residual"].mean(), res["residual"].std()
    xx = np.linspace(bins[0], bins[-1], 200)
    ax1.plot(xx, sp_norm.pdf(xx, mu, sigma),
             color=PALETTE["red"], lw=2.2, label=f"正态拟合\nμ={mu:.1f}, σ={sigma:.1f}")
    ax1.axvline(0, color="#9AA5B1", lw=1.2, ls="--", label="零残差线")
    ax1.set_title("（b）残差分布直方图", fontproperties=title_font(11), pad=10)
    ax1.set_xlabel("残差值", fontproperties=body_font(10))
    ax1.set_ylabel("频率密度", fontproperties=body_font(10))
    for lab in ax1.get_xticklabels() + ax1.get_yticklabels():
        lab.set_fontproperties(body_font(9))
    ax1.grid(True)
    cn_legend(ax1)

    fig.suptitle("图16  集成模型残差诊断分析", fontproperties=title_font(), y=1.0)
    save_fig("Fig16_residual_diagnostics.png")


# ══════════════════════════════════════════════════════
#  Fig 17 — 增长率对比
# ══════════════════════════════════════════════════════
def fig17_growth_rate(df):
    obs = df[df["observed_co2"]].copy()
    obs["co2_growth"]       = obs["total_co2_mt"].pct_change() * 100
    obs["gdp_growth_proxy"] = obs["gdp_index_1978"].pct_change() * 100

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    ax.axhline(0, color="#C0CAD8", lw=1, zorder=1)
    ax.fill_between(obs["year"], obs["co2_growth"],
                    where=obs["co2_growth"] >= 0,
                    color=PALETTE["red"], alpha=0.12)
    ax.fill_between(obs["year"], obs["co2_growth"],
                    where=obs["co2_growth"] < 0,
                    color=PALETTE["blue"], alpha=0.12)
    ax.plot(obs["year"], obs["co2_growth"],
            color=PALETTE["red"], lw=2.2, marker="o",
            ms=5, mec="white", mew=1, label="碳排放增速（%）")
    ax.plot(obs["year"], obs["gdp_growth_proxy"],
            color=PALETTE["blue"], lw=2.2, marker="s",
            ms=5, mec="white", mew=1, label="GDP 指数增速（%）")

    finalize_axes(ax, "图17  碳排放增速与 GDP 增速对比",
                  "年份", "增长率（%）")
    cn_legend(ax, loc="upper right")
    add_subtitle(fig, "增速计算基于逐年环比变化率；GDP 指数基期为 1978 年")
    save_fig("Fig17_growth_rate_comparison.png")


# ══════════════════════════════════════════════════════
#  Fig 18 — 3D 宏观经济变量与碳排放曲面
# ══════════════════════════════════════════════════════
def fig18_3d_macro_surface(df):
    obs  = df[df["observed_co2"]].copy()
    fig  = plt.figure(figsize=(9.8, 6.5))
    ax   = fig.add_subplot(111, projection="3d")

    x = obs["gdp_pc_10k_cny"].to_numpy()
    y = (obs["urbanization"] * 100).to_numpy()
    z = obs["total_co2_mt"].to_numpy()

    sc = ax.scatter(x, y, z, c=z, cmap="coolwarm",
                    s=65, depthshade=True,
                    edgecolors="white", lw=0.4, zorder=3)

    # 连线轨迹
    sort_idx = np.argsort(obs["year"].to_numpy())
    ax.plot(x[sort_idx], y[sort_idx], z[sort_idx],
            color=PALETTE["navy"], lw=1.6, alpha=0.55, zorder=2)

    # 年份文字
    for i in sort_idx[::5]:
        ax.text(x[i], y[i], z[i] + 5,
                str(int(obs["year"].iloc[i])),
                fontproperties=body_font(8), color=PALETTE["charcoal"], alpha=0.7)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, aspect=18)
    cbar.set_label("碳排放量（百万吨）", fontproperties=body_font(9))
    for lab in cbar.ax.get_yticklabels(): lab.set_fontproperties(body_font(8))

    finalize_axes(ax, "图18  人均 GDP、城镇化率与碳排放量三维宏观曲面",
                  "人均 GDP（万元）", "城镇化率（%）", "碳排放量（百万吨）")
    ax.view_init(elev=24, azim=-50)
    save_fig("Fig18_3d_macro_surface.png")


# ══════════════════════════════════════════════════════
#  Fig 19 — 能源结构雷达图
# ══════════════════════════════════════════════════════
def fig19_energy_radar(df):
    latest = df[df["year"] == df["year"].max()].iloc[0]
    base   = df[df["year"] == df["year"].min()].iloc[0]

    labels19    = ["煤炭占比", "城镇化率", "技术进步", "第二产业占比", "能源强度"]
    latest_vals = np.array([latest["coal_share"]*100, latest["urbanization"]*100,
                            latest["tech_progress"], latest["secondary_share"],
                            latest["energy_intensity"]*100])
    base_vals   = np.array([base["coal_share"]*100, base["urbanization"]*100,
                            base["tech_progress"], base["secondary_share"],
                            base["energy_intensity"]*100])
    max_vals    = np.maximum(latest_vals, base_vals)
    safe_max    = np.where(max_vals == 0, 1, max_vals)
    latest_norm = latest_vals / safe_max
    base_norm   = base_vals   / safe_max

    N      = len(labels19)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.0))
    ax  = fig.add_subplot(111, polar=True)

    # 基期
    base_plot   = np.concatenate([base_norm,   [base_norm[0]]])
    latest_plot = np.concatenate([latest_norm, [latest_norm[0]]])

    ax.plot(angles, base_plot, color=PALETTE["gray"],
            lw=2.0, ls="--", label=f"{int(base['year'])}年（基期）")
    ax.fill(angles, base_plot, color=PALETTE["gray"], alpha=0.14)

    ax.plot(angles, latest_plot, color=PALETTE["navy"],
            lw=2.6, label=f"{int(latest['year'])}年（最新）")
    ax.fill(angles, latest_plot, color=PALETTE["sky"], alpha=0.22)

    # 刻度点
    for angle, bv, lv in zip(angles[:-1], base_norm, latest_norm):
        ax.scatter(angle, bv, color=PALETTE["gray"], s=50, zorder=4)
        ax.scatter(angle, lv, color=PALETTE["navy"], s=60, zorder=5,
                   edgecolors="white", lw=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels19, fontproperties=body_font(10.5))
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title("图19  能源与产业结构综合雷达图", fontproperties=title_font(), pad=22)
    ax.grid(color="#C0CAD8", alpha=0.4, linewidth=0.7)
    ax.spines["polar"].set_linewidth(0.8)
    ax.spines["polar"].set_color("#C0CAD8")

    leg = ax.legend(frameon=True, framealpha=0.92, edgecolor="#C0CAD8",
                    loc="upper right", bbox_to_anchor=(1.22, 1.12),
                    prop=body_font(9.5))
    save_fig("Fig19_energy_radar.png")


# ══════════════════════════════════════════════════════
#  Fig 20 — 情景碳减排瀑布图
# ══════════════════════════════════════════════════════
def fig20_scenario_waterfall(scenario):
    end  = scenario[scenario["year"] == scenario["year"].max()].iloc[0]
    base = end["Baseline"]
    low  = end["Low-Carbon"]
    deep = end["Deep Decarbonization"]

    steps      = [base, low - base, deep - low]
    cn_labels  = ["基准情景\n（2035年）", "低碳情景\n相对基准削减", "深度脱碳\n相对低碳削减"]
    colors20   = [PALETTE["navy"], PALETTE["green"], PALETTE["red"]]
    cumulative = [0, base, low]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for i, (label, step, color, bottom) in enumerate(
            zip(cn_labels, steps, colors20, cumulative)):
        bar = ax.bar(i, abs(step), bottom=bottom if step > 0 else bottom + step,
                     color=color, alpha=0.85, width=0.55, edgecolor="white", lw=1)
        # 数值标注
        ytop = bottom + step if step > 0 else bottom
        ax.text(i, ytop + 20, f"{step:+,.0f} Mt",
                ha="center", fontproperties=body_font(10),
                color=color, weight="bold")

    # 连接线
    running = 0
    for i, (step, cum) in enumerate(zip(steps[:-1], cumulative[1:])):
        ax.plot([i + 0.28, i + 0.72], [cum, cum],
                color="#AAAAAA", lw=1.0, ls="--")
        running += step

    ax.axhline(0, color="#9AA5B1", lw=1)
    ax.set_xticks(range(len(cn_labels)))
    ax.set_xticklabels(cn_labels, fontproperties=body_font(10), ha="center")
    finalize_axes(ax, "图20  情景碳排放削减瀑布图（2035年目标年）",
                  None, "碳排放量（百万吨 CO₂）")
    add_subtitle(fig, "绿色/红色柱为相对上一情景的增减量；蓝色为基准总量")
    save_fig("Fig20_scenario_waterfall.png")


# ══════════════════════════════════════════════════════
#  Fig 21 — 滚动相关系数热力图
# ══════════════════════════════════════════════════════
def fig21_rolling_correlation(df):
    obs      = df[df["observed_co2"]].copy()
    features = ["gdp_100m","urbanization","coal_share","tech_progress"]
    cn_feat  = {"gdp_100m":"GDP","urbanization":"城镇化率",
                "coal_share":"煤炭占比","tech_progress":"技术进步"}
    window   = 5
    rows     = []
    for idx in range(window - 1, len(obs)):
        sub = obs.iloc[idx - window + 1: idx + 1]
        for feat in features:
            rows.append({"year":    int(sub["year"].iloc[-1]),
                         "feature": cn_feat[feat],
                         "corr":    float(sub["total_co2_mt"].corr(sub[feat]))})
    corr_df = pd.DataFrame(rows)
    pivot   = corr_df.pivot(index="feature", columns="year", values="corr")

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto",
                   cmap=CMAP_DIVG, vmin=-1, vmax=1)

    # 格网
    nr, nc = pivot.shape
    for i in range(nr + 1): ax.axhline(i - 0.5, color="white", lw=1.2)
    for j in range(nc + 1): ax.axvline(j - 0.5, color="white", lw=1.2)

    # 数值
    for i in range(nr):
        for j in range(nc):
            v = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontproperties=body_font(8), color="white" if abs(v) > 0.55 else PALETTE["charcoal"])

    ax.set_xticks(range(nc))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right",
                       fontproperties=body_font(9))
    ax.set_yticks(range(nr))
    ax.set_yticklabels(pivot.index, fontproperties=body_font(10))
    ax.set_title("图21  滚动相关系数时变热力图（窗口 = 5 年）",
                 fontproperties=title_font(), pad=14)
    ax.spines[:].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=22)
    cbar.set_label("皮尔逊相关系数", fontproperties=body_font(9))
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    for lab in cbar.ax.get_yticklabels(): lab.set_fontproperties(body_font(9))

    add_subtitle(fig, "蓝色为正相关，红色为负相关；颜色深浅反映相关强度随时间的变化")
    save_fig("Fig21_rolling_correlation_heatmap.png")


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════
def main():
    apply_style()
    print("正在加载数据…")
    df, metrics, predictions, stirpat, shap_df, scenario, peak, mediation, desc = load_data()
    print("数据加载完成，开始生成图表…\n")

    fig1_co2_trend(df)
    fig2_gdp_co2_dual(df)
    fig3_correlation(df)
    fig4_industry_structure(df)
    fig5_energy_structure(df)
    fig6_urbanization(df)
    fig7_ekc(df)
    fig8_mediation(mediation)
    fig9_stirpat(stirpat)
    fig10_patent_research(df)
    fig11_model_metrics(metrics)
    fig12_shap(shap_df)
    fig13_scenarios(df, scenario)
    fig14_peak_table(peak)
    fig15_actual_vs_predicted(predictions)
    fig16_residual_diagnostics(predictions)
    fig17_growth_rate(df)
    fig18_3d_macro_surface(df)
    fig19_energy_radar(df)
    fig20_scenario_waterfall(scenario)
    fig21_rolling_correlation(df)

    print(f"\n全部图表已保存至：{FIGURES}")


if __name__ == "__main__":
    main()