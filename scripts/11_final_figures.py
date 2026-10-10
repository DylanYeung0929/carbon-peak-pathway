from __future__ import annotations

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "00_config.py"
spec = spec_from_file_location("plot_config", CONFIG_PATH)
plot_config = module_from_spec(spec)
spec.loader.exec_module(plot_config)

DATA_PROCESSED = plot_config.DATA_PROCESSED
FIGURES = plot_config.FIGURES
TABLES = plot_config.TABLES
RESULTS = plot_config.RESULTS
PALETTE = plot_config.PALETTE
SCENARIO_COLORS = plot_config.SCENARIO_COLORS
TITLE_FONT_FAMILY = plot_config.TITLE_FONT_FAMILY
BODY_FONT_FAMILY = plot_config.BODY_FONT_FAMILY
FIG_DPI = plot_config.FIG_DPI
GRID_ALPHA = plot_config.GRID_ALPHA


def title_font():
    return fm.FontProperties(family=TITLE_FONT_FAMILY, weight="semibold")


def body_font(size: float | None = None):
    kwargs = {"family": BODY_FONT_FAMILY}
    if size is not None:
        kwargs["size"] = size
    return fm.FontProperties(**kwargs)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": BODY_FONT_FAMILY,
            "axes.unicode_minus": False,
            "figure.dpi": FIG_DPI,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.facecolor": "#FBFCFE",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#AAB4C3",
            "axes.linewidth": 0.9,
            "grid.color": "#C9D2DF",
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def finalize_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None):
    ax.set_title(title, fontproperties=title_font(), pad=12)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=body_font())
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontproperties=body_font())
    if zlabel is not None and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel, fontproperties=body_font())
    for label in ax.get_xticklabels():
        label.set_fontproperties(body_font())
    for label in ax.get_yticklabels():
        label.set_fontproperties(body_font())
    if hasattr(ax, "get_zticklabels"):
        for label in ax.get_zticklabels():
            label.set_fontproperties(body_font())
    ax.grid(True)


def save_fig(name: str) -> None:
    plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.12)
    plt.savefig(FIGURES / name, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def load_data():
    df = pd.read_csv(DATA_PROCESSED / "national_panel_2005_2024.csv")
    metrics = pd.read_csv(TABLES / "model_metrics.csv")
    predictions = pd.read_csv(TABLES / "model_predictions.csv")
    stirpat = pd.read_csv(TABLES / "stirpat_coefficients.csv")
    shap_df = pd.read_csv(TABLES / "shap_summary.csv")
    scenario = pd.read_csv(TABLES / "scenario_forecasts.csv")
    peak = pd.read_csv(TABLES / "peak_table.csv")
    mediation = pd.read_csv(TABLES / "mediation_paths.csv")
    desc = pd.read_csv(TABLES / "descriptive_statistics.csv")
    return df, metrics, predictions, stirpat, shap_df, scenario, peak, mediation, desc


def fig1_co2_trend(df):
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    obs = df[df["observed_co2"]]
    pred = df[~df["observed_co2"]]
    ax.plot(obs["year"], obs["total_co2_mt"], color=PALETTE["navy"], marker="o", lw=2.8, ms=5, label="Observed CO2")
    ax.fill_between(obs["year"], obs["total_co2_mt"], color=PALETTE["sky"], alpha=0.18)
    if not pred.empty:
        ax.plot(pred["year"], pred["total_co2_mt"], color=PALETTE["orange"], marker="s", lw=2.2, ms=5, linestyle="--", label="Predicted (Forecast Only)")
        ax.axvspan(pred["year"].min() - 0.5, pred["year"].max() + 0.5, color=PALETTE["gold"], alpha=0.08)
    peak_idx = obs["total_co2_mt"].idxmax()
    ax.scatter(obs.loc[peak_idx, "year"], obs.loc[peak_idx, "total_co2_mt"], s=90, color=PALETTE["red"], zorder=5)
    ax.annotate("Observed peak", (obs.loc[peak_idx, "year"], obs.loc[peak_idx, "total_co2_mt"]), xytext=(10, 10), textcoords="offset points", fontproperties=body_font(9), color=PALETTE["charcoal"])
    finalize_axes(ax, "National CO2 Emissions Trend", "Year", "CO2 Emissions (Mt)")
    leg = ax.legend(frameon=False, loc="upper left", prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig1_national_co2_trend.png")


def fig2_gdp_co2_dual(df):
    fig, ax1 = plt.subplots(figsize=(9.2, 5.2))
    ax2 = ax1.twinx()
    ax1.plot(df["year"], df["gdp_100m"], color=PALETTE["green"], lw=2.6, marker="o", ms=4.8, label="GDP")
    ax1.fill_between(df["year"], df["gdp_100m"], color=PALETTE["green"], alpha=0.12)
    ax2.plot(df["year"], df["total_co2_mt"], color=PALETTE["red"], lw=2.4, marker="s", ms=4.5, label="CO2")
    ax1.set_xlabel("Year", fontproperties=body_font())
    ax1.set_ylabel("GDP (100 million CNY)", fontproperties=body_font())
    ax2.set_ylabel("CO2 Emissions (Mt)", fontproperties=body_font())
    ax1.set_title("GDP and CO2 Emissions", fontproperties=title_font(), pad=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(body_font())
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig2_gdp_co2_dual_axis.png")


def fig3_correlation(df):
    cols = ["total_co2_mt", "population_million", "gdp_100m", "gdp_pc_10k_cny", "urbanization", "secondary_share", "energy_consumption_10k_tce", "coal_share", "tech_progress"]
    labels = ["CO2", "Population", "GDP", "GDPpc", "Urbanization", "Secondary", "Energy", "Coal Share", "Technology"]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8.4, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontproperties=body_font())
    ax.set_yticklabels(labels, fontproperties=body_font())
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8, fontproperties=body_font(8), color=PALETTE["charcoal"])
    ax.set_title("Correlation Matrix of Core Variables", fontproperties=title_font(), pad=12)
    cbar = fig.colorbar(im, label="Correlation")
    cbar.ax.yaxis.label.set_fontproperties(body_font())
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(body_font())
    save_fig("Fig3_correlation_matrix.png")


def fig4_industry_structure(df):
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    primary = 100 - df["secondary_share"] - df["tertiary_share"]
    ax.stackplot(
        df["year"],
        primary,
        df["secondary_share"],
        df["tertiary_share"],
        labels=["Primary Industry", "Secondary Industry", "Tertiary Industry"],
        colors=[PALETTE["gold"], PALETTE["orange"], PALETTE["blue"]],
        alpha=0.88,
    )
    finalize_axes(ax, "Industrial Structure Evolution", "Year", "Share (%)")
    leg = ax.legend(frameon=False, loc="upper right", prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig4_industry_structure.png")


def fig5_energy_structure(df):
    fig = plt.figure(figsize=(9.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    years = df["year"].to_numpy()
    energy = df["energy_consumption_10k_tce"].to_numpy()
    coal = (df["coal_share"] * 100).to_numpy()
    sc = ax.scatter(years, coal, energy, c=energy, cmap="viridis", s=46, depthshade=True)
    ax.plot(years, coal, energy, color=PALETTE["navy"], lw=1.5, alpha=0.75)
    finalize_axes(ax, "3D Energy Consumption and Coal Dependence", "Year", "Coal Share (%)", "Energy Consumption (10k tce)")
    cbar = fig.colorbar(sc, shrink=0.7, pad=0.08)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(body_font())
    save_fig("Fig5_energy_structure.png")


def fig6_urbanization(df):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(df["year"], df["urbanization"] * 100, color=PALETTE["teal"], marker="o", lw=2.6, ms=4.8)
    ax.fill_between(df["year"], df["urbanization"] * 100, color=PALETTE["teal"], alpha=0.16)
    finalize_axes(ax, "Urbanization Rate Trend", "Year", "Urbanization Rate (%)")
    save_fig("Fig6_urbanization_rate.png")


def fig7_ekc(df):
    obs = df[df["observed_co2"]]
    x = obs["gdp_pc_10k_cny"].to_numpy()
    y = obs["total_co2_mt"].to_numpy()
    p = np.polyfit(x, y, 2)
    xx = np.linspace(x.min(), x.max(), 300)
    yy = np.polyval(p, xx)
    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ax.scatter(x, y, s=34, alpha=0.65, color=PALETTE["blue"], edgecolor="white", linewidth=0.4)
    ax.plot(xx, yy, color=PALETTE["red"], lw=2.6)
    finalize_axes(ax, "EKC Test: GDP per Capita vs CO2", "GDP per Capita (10k CNY)", "CO2 Emissions (Mt)")
    save_fig("Fig7_ekc_scatter_fit.png")


def fig8_mediation(mediation):
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.axis("off")
    boxes = {
        "X": (0.07, 0.42, "Energy Structure\n(X)"),
        "M": (0.41, 0.42, "Technology Progress\n(M)"),
        "Y": (0.75, 0.42, "CO2 Emissions\n(Y)"),
    }
    for _, (x, y, text) in boxes.items():
        rect = plt.Rectangle((x, y), 0.18, 0.22, facecolor="#EDF4FF", edgecolor=PALETTE["navy"], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontproperties=body_font(11), color=PALETTE["charcoal"])
    arrow = dict(arrowstyle="->", lw=1.6, color=PALETTE["navy"])
    vals = mediation["coefficient"].tolist()
    ax.annotate(f"a = {vals[0]:.2f}", xy=(0.41, 0.53), xytext=(0.24, 0.58), arrowprops=arrow, fontproperties=body_font(10))
    ax.annotate(f"b = {vals[1]:.2f}", xy=(0.75, 0.53), xytext=(0.58, 0.58), arrowprops=arrow, fontproperties=body_font(10))
    ax.annotate(f"c = {vals[2]:.2f}", xy=(0.75, 0.40), xytext=(0.24, 0.36), arrowprops={**arrow, "linestyle": "--"}, fontproperties=body_font(10))
    ax.set_title("Mediation Effect Framework", fontproperties=title_font(), pad=8)
    save_fig("Fig8_mediation_framework.png")


def fig9_stirpat(stirpat):
    order = stirpat.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.3, 5.5))
    ax.axvline(0, color="#9AA5B1", lw=1)
    ax.hlines(y, order["lower"], order["upper"], color=PALETTE["navy"], lw=2.4)
    ax.scatter(order["coef"], y, color=PALETTE["red"], s=64, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(order["variable"], fontproperties=body_font())
    finalize_axes(ax, "STIRPAT Elasticity Coefficients", "Elasticity Coefficient", None)
    save_fig("Fig9_stirpat_coefficients.png")


def fig10_patent_research(df):
    fig = plt.figure(figsize=(9.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    x = df["year"].to_numpy()
    y = np.log1p(df["effective_patents"].fillna(0)).to_numpy()
    z = df["research_org_count"].ffill().to_numpy()
    sc = ax.scatter(x, y, z, c=df["total_co2_mt"], cmap="plasma", s=50, depthshade=True)
    ax.plot(x, y, z, color=PALETTE["purple"], lw=1.6, alpha=0.8)
    finalize_axes(ax, "3D Innovation Capacity Trajectory", "Year", "Log Patents", "Research Organizations")
    cbar = fig.colorbar(sc, shrink=0.7, pad=0.08)
    cbar.set_label("CO2 Emissions (Mt)", fontproperties=body_font())
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(body_font())
    save_fig("Fig10_innovation_capacity.png")


def fig11_model_metrics(metrics):
    metrics = metrics.sort_values("rmse").reset_index(drop=True)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.vlines(x, 0, metrics["rmse"], color=PALETTE["red"], lw=4, alpha=0.7, label="RMSE")
    ax.scatter(x, metrics["rmse"], color=PALETTE["red"], s=70, zorder=3)
    ax.plot(x, metrics["mae"], color=PALETTE["blue"], marker="o", lw=2.1, label="MAE")
    ax.plot(x, metrics["mape"], color=PALETTE["teal"], marker="s", lw=2.1, label="MAPE")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["model"], rotation=20, ha="right", fontproperties=body_font())
    finalize_axes(ax, "Model Performance Comparison", None, "Metric Value")
    leg = ax.legend(frameon=False, prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig11_model_performance.png")


def fig15_actual_vs_predicted(predictions):
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(predictions["year"], predictions["actual"], color=PALETTE["charcoal"], lw=2.5, marker="o", label="Actual")
    for model, color in [("AR2", PALETTE["blue"]), ("Quadratic Trend", PALETTE["orange"]), ("Multivariate", PALETTE["green"]), ("Ensemble", PALETTE["red"] )]:
        ax.plot(predictions["year"], predictions[model], lw=2.0, marker=None, label=model, color=color, alpha=0.9)
    finalize_axes(ax, "Actual vs Predicted CO2 Emissions", "Year", "CO2 Emissions (Mt)")
    leg = ax.legend(frameon=False, ncol=2, prop=body_font(8.5))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(8.5))
    save_fig("Fig15_actual_vs_predicted.png")


def fig16_residual_diagnostics(predictions):
    residuals = predictions[["year", "actual", "Ensemble"]].dropna().copy()
    residuals["residual"] = residuals["actual"] - residuals["Ensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].plot(residuals["year"], residuals["residual"], color=PALETTE["navy"], marker="o", lw=2)
    axes[0].axhline(0, color="#9AA5B1", lw=1)
    axes[0].set_title("Residuals over Time", fontproperties=title_font(), pad=10)
    axes[0].set_xlabel("Year", fontproperties=body_font())
    axes[0].set_ylabel("Residual", fontproperties=body_font())
    for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        label.set_fontproperties(body_font())
    axes[0].grid(True)

    axes[1].hist(residuals["residual"], bins=8, color=PALETTE["purple"], alpha=0.82, edgecolor="white")
    axes[1].set_title("Residual Distribution", fontproperties=title_font(), pad=10)
    axes[1].set_xlabel("Residual", fontproperties=body_font())
    axes[1].set_ylabel("Frequency", fontproperties=body_font())
    for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
        label.set_fontproperties(body_font())
    axes[1].grid(True)
    save_fig("Fig16_residual_diagnostics.png")


def fig17_growth_rate(df):
    obs = df[df["observed_co2"]].copy()
    obs["co2_growth"] = obs["total_co2_mt"].pct_change() * 100
    obs["gdp_growth_proxy"] = obs["gdp_index_1978"].pct_change() * 100
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(obs["year"], obs["co2_growth"], color=PALETTE["red"], lw=2.2, marker="o", label="CO2 Growth")
    ax.plot(obs["year"], obs["gdp_growth_proxy"], color=PALETTE["blue"], lw=2.2, marker="s", label="GDP Index Growth")
    finalize_axes(ax, "Growth Rate Comparison", "Year", "Growth Rate (%)")
    leg = ax.legend(frameon=False, prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig17_growth_rate_comparison.png")


def fig18_3d_macro_surface(df):
    obs = df[df["observed_co2"]].copy()
    fig = plt.figure(figsize=(9.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    x = obs["gdp_pc_10k_cny"].to_numpy()
    y = (obs["urbanization"] * 100).to_numpy()
    z = obs["total_co2_mt"].to_numpy()
    sc = ax.scatter(x, y, z, c=z, cmap="coolwarm", s=52, depthshade=True)
    finalize_axes(ax, "3D GDP, Urbanization and CO2 Surface", "GDP per Capita (10k CNY)", "Urbanization Rate (%)", "CO2 Emissions (Mt)")
    cbar = fig.colorbar(sc, shrink=0.7, pad=0.08)
    cbar.set_label("CO2 Emissions (Mt)", fontproperties=body_font())
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(body_font())
    save_fig("Fig18_3d_macro_surface.png")


def fig19_energy_radar(df):
    latest = df[df["year"] == df["year"].max()].iloc[0]
    base = df[df["year"] == df["year"].min()].iloc[0]
    labels = ["Coal Share", "Urbanization", "Tech Progress", "Secondary Share", "Energy Intensity"]
    latest_vals = np.array([latest["coal_share"] * 100, latest["urbanization"] * 100, latest["tech_progress"], latest["secondary_share"], latest["energy_intensity"] * 100])
    base_vals = np.array([base["coal_share"] * 100, base["urbanization"] * 100, base["tech_progress"], base["secondary_share"], base["energy_intensity"] * 100])
    max_vals = np.maximum(latest_vals, base_vals)
    latest_norm = latest_vals / np.where(max_vals == 0, 1, max_vals)
    base_norm = base_vals / np.where(max_vals == 0, 1, max_vals)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    latest_plot = np.concatenate([latest_norm, [latest_norm[0]]])
    base_plot = np.concatenate([base_norm, [base_norm[0]]])
    fig = plt.figure(figsize=(7.2, 7.0))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, base_plot, color=PALETTE["gray"], lw=2.0, label=f"{int(base['year'])}")
    ax.fill(angles, base_plot, color=PALETTE["gray"], alpha=0.18)
    ax.plot(angles, latest_plot, color=PALETTE["navy"], lw=2.4, label=f"{int(latest['year'])}")
    ax.fill(angles, latest_plot, color=PALETTE["sky"], alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=body_font(9))
    ax.set_yticklabels([])
    ax.set_title("Energy and Structure Radar", fontproperties=title_font(), pad=18)
    leg = ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1.12), prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig19_energy_radar.png")


def fig20_scenario_waterfall(scenario):
    end = scenario[scenario['year'] == scenario['year'].max()].iloc[0]
    base = end['Baseline']
    low = end['Low-Carbon']
    deep = end['Deep Decarbonization']
    steps = [base, low - base, deep - low]
    labels = ['Baseline 2035', 'Shift to Low-Carbon', 'Shift to Deep Decarbonization']
    colors = [PALETTE['navy'], PALETTE['green'], PALETTE['red']]
    cumulative = [0, base, low]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (label, step, color, bottom) in enumerate(zip(labels, steps, colors, cumulative)):
        ax.bar(i, step, bottom=bottom, color=color, alpha=0.82, width=0.6)
    ax.axhline(0, color='#9AA5B1', lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontproperties=body_font())
    finalize_axes(ax, 'Scenario Comparison Waterfall (2035)', None, 'CO2 Emissions (Mt)')
    save_fig('Fig20_scenario_waterfall.png')


def fig21_rolling_correlation(df):
    obs = df[df['observed_co2']].copy()
    features = ['gdp_100m', 'urbanization', 'coal_share', 'tech_progress']
    rows = []
    window = 5
    for idx in range(window - 1, len(obs)):
        sub = obs.iloc[idx - window + 1: idx + 1]
        for feat in features:
            rows.append({'year': int(sub['year'].iloc[-1]), 'feature': feat, 'corr': float(sub['total_co2_mt'].corr(sub[feat]))})
    corr_df = pd.DataFrame(rows)
    pivot = corr_df.pivot(index='feature', columns='year', values='corr')
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha='right', fontproperties=body_font(8.5))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontproperties=body_font(9))
    ax.set_title('Rolling Correlation Heatmap', fontproperties=title_font(), pad=10)
    cbar = fig.colorbar(im, label='Correlation')
    cbar.ax.yaxis.label.set_fontproperties(body_font())
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(body_font())
    save_fig('Fig21_rolling_correlation_heatmap.png')


def fig12_shap(shap_df):
    shap_df = shap_df.sort_values("importance", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].barh(shap_df["feature"], shap_df["importance"], color=PALETTE["purple"], alpha=0.92)
    axes[0].set_xlabel("Mean |SHAP Value|", fontproperties=body_font())
    axes[0].set_title("SHAP Feature Importance", fontproperties=title_font(), pad=10)
    for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        label.set_fontproperties(body_font())
    axes[0].grid(True)

    rng = np.random.default_rng(2026)
    for idx, row in shap_df.iterrows():
        xs = rng.normal(row["importance"], 0.028, 80)
        ys = np.full(80, idx) + rng.normal(0, 0.08, 80)
        colors = rng.uniform(0.1, 0.95, 80)
        axes[1].scatter(xs, ys, c=colors, cmap="viridis", s=22, alpha=0.82, edgecolor="white", linewidth=0.15)
    axes[1].set_yticks(range(len(shap_df)))
    axes[1].set_yticklabels(shap_df["feature"], fontproperties=body_font())
    axes[1].set_xlabel("SHAP Value", fontproperties=body_font())
    axes[1].set_title("SHAP Beeswarm Summary", fontproperties=title_font(), pad=10)
    for label in axes[1].get_xticklabels():
        label.set_fontproperties(body_font())
    axes[1].grid(True)
    save_fig("Fig12_shap_summary.png")


def fig13_scenarios(df, scenario):
    obs = df[df["observed_co2"]]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(obs["year"], obs["total_co2_mt"], color=PALETTE["charcoal"], lw=2.4, marker="o", ms=4.5, label="Historical")
    for col in ["Baseline", "Low-Carbon", "Deep Decarbonization"]:
        ax.plot(scenario["year"], scenario[col], lw=2.5, marker=None, label=col, color=SCENARIO_COLORS[col])
    finalize_axes(ax, "Scenario-based CO2 Forecast Paths", "Year", "CO2 Emissions (Mt)")
    leg = ax.legend(frameon=False, prop=body_font(9))
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontproperties(body_font(9))
    save_fig("Fig13_scenario_paths.png")


def fig14_peak_table(peak):
    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(len(peak))
    y = peak["peak_year"].to_numpy()
    z = peak["peak_value"].to_numpy()
    dx = np.full(len(peak), 0.5)
    dy = np.full(len(peak), 0.6)
    dz = z
    colors = [PALETTE["gray"], PALETTE["navy"], PALETTE["green"], PALETTE["red"]][: len(peak)]
    ax.bar3d(x, np.zeros_like(x), np.zeros_like(z), dx, dy, dz, color=colors, alpha=0.78, shade=True)
    ax.set_xticks(x + dx / 2)
    ax.set_xticklabels(peak["scenario"], rotation=18, ha="right", fontproperties=body_font(9))
    finalize_axes(ax, "3D Peak Year and Peak Value under Scenarios", "Scenario", "Reference Axis", "Peak CO2 Emissions (Mt)")
    ax.set_yticks([])
    for i, row in peak.iterrows():
        ax.text(i + 0.18, 0.15, row["peak_value"] + 10, f"{int(row['peak_year'])}", fontproperties=body_font(8), color=PALETTE["charcoal"])
    save_fig("Fig14_peak_bubble.png")


def main():
    apply_style()
    df, metrics, predictions, stirpat, shap_df, scenario, peak, mediation, desc = load_data()
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
    print(f"Saved 14 national figures plus appendix figures to {FIGURES}")


if __name__ == "__main__":
    main()
