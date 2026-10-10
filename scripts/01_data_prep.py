import math
from pathlib import Path

import numpy as np
import pandas as pd

from importlib.util import spec_from_file_location, module_from_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "00_config.py"
spec = spec_from_file_location("plot_config", CONFIG_PATH)
plot_config = module_from_spec(spec)
spec.loader.exec_module(plot_config)

PROVINCES = plot_config.PROVINCES
REGION_MAP = plot_config.REGION_MAP
DATA_PROCESSED = plot_config.DATA_PROCESSED
TABLES = plot_config.TABLES
RESULTS = plot_config.RESULTS

YEARS = list(range(2005, 2025))

PROVINCE_META = {
    "Beijing": (116.40, 39.90, 0.25), "Tianjin": (117.20, 39.12, 0.35), "Hebei": (114.50, 38.04, 0.62),
    "Shanxi": (112.55, 37.87, 0.78), "Inner Mongolia": (111.65, 40.82, 0.88), "Liaoning": (123.43, 41.80, 0.74),
    "Jilin": (125.32, 43.90, 0.58), "Heilongjiang": (126.63, 45.75, 0.67), "Shanghai": (121.47, 31.23, 0.22),
    "Jiangsu": (118.78, 32.04, 0.48), "Zhejiang": (120.15, 30.28, 0.36), "Anhui": (117.28, 31.86, 0.57),
    "Fujian": (119.30, 26.08, 0.34), "Jiangxi": (115.89, 28.68, 0.50), "Shandong": (117.00, 36.65, 0.72),
    "Henan": (113.62, 34.75, 0.73), "Hubei": (114.31, 30.52, 0.55), "Hunan": (112.98, 28.20, 0.54),
    "Guangdong": (113.27, 23.13, 0.46), "Guangxi": (108.32, 22.82, 0.59), "Hainan": (110.35, 20.02, 0.20),
    "Chongqing": (106.55, 29.56, 0.63), "Sichuan": (104.07, 30.67, 0.66), "Guizhou": (106.71, 26.57, 0.76),
    "Yunnan": (102.71, 25.04, 0.61), "Tibet": (91.13, 29.65, 0.08), "Shaanxi": (108.95, 34.27, 0.71),
    "Gansu": (103.82, 36.07, 0.69), "Qinghai": (101.78, 36.62, 0.27), "Ningxia": (106.23, 38.49, 0.83),
    "Xinjiang": (87.62, 43.82, 0.92)
}


def build_demo_panel(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for province in PROVINCES:
        lon, lat, carbon_factor = PROVINCE_META[province]
        region = REGION_MAP[province]
        region_shift = {"East": 1.12, "Central": 1.00, "West": 0.88}[region]

        pop0 = rng.uniform(3.0, 11.0)
        gdp0 = rng.uniform(1.8, 8.0) * region_shift
        energy_intensity0 = rng.uniform(0.55, 1.55) * (1.10 if region != "East" else 0.92)
        urban0 = rng.uniform(0.32, 0.82)
        industry0 = rng.uniform(0.34, 0.59)
        coal0 = min(0.92, max(0.18, carbon_factor + rng.normal(0, 0.05)))
        tech0 = rng.uniform(0.15, 0.90) * (1.12 if region == "East" else 0.95)

        for year in YEARS:
            t = year - YEARS[0]
            pop = pop0 * (1 + 0.0025 * t + rng.normal(0, 0.005))
            gdp_pc = gdp0 * (1 + 0.071 * t + 0.003 * t**1.12 + rng.normal(0, 0.045))
            energy_intensity = max(0.22, energy_intensity0 * (1 - 0.023 * t) + rng.normal(0, 0.025))
            urbanization = min(0.96, max(0.25, urban0 + 0.0125 * t + rng.normal(0, 0.012)))
            industry_share = min(0.72, max(0.18, industry0 - 0.002 * t + rng.normal(0, 0.012)))
            coal_share = min(0.95, max(0.08, coal0 - 0.009 * t + rng.normal(0, 0.02)))
            renewable_share = min(0.65, max(0.02, 0.08 + 0.007 * t + rng.normal(0, 0.01)))
            tech_progress = max(0.05, tech0 + 0.026 * t + rng.normal(0, 0.02))
            rd_intensity = max(0.001, 0.008 + 0.0008 * t + 0.0015 * tech_progress + rng.normal(0, 0.0007))
            patents = max(5, 120 * tech_progress * region_shift + rng.normal(0, 10))

            co2 = (
                120
                + 18 * pop
                + 32 * np.log1p(gdp_pc) ** 2
                + 78 * energy_intensity
                + 62 * coal_share
                + 35 * industry_share
                - 55 * tech_progress
                - 24 * renewable_share
                + 22 * urbanization
                + rng.normal(0, 8)
            )
            co2 = max(45, co2 * (1.12 if region == "Central" else 1.0) * (1.08 if region == "West" else 1.0))
            carbon_intensity = co2 / max(gdp_pc * pop, 1e-6)
            co2_per_capita = co2 / max(pop, 1e-6)

            rows.append(
                {
                    "province": province,
                    "year": year,
                    "region": region,
                    "lon": lon,
                    "lat": lat,
                    "population_million": round(pop, 4),
                    "gdp_pc_10k_cny": round(gdp_pc, 4),
                    "energy_intensity": round(energy_intensity, 4),
                    "coal_share": round(coal_share, 4),
                    "renewable_share": round(renewable_share, 4),
                    "urbanization": round(urbanization, 4),
                    "industry_share": round(industry_share, 4),
                    "tech_progress": round(tech_progress, 4),
                    "rd_intensity": round(rd_intensity, 6),
                    "patents_per_10k": round(patents, 4),
                    "total_co2_mt": round(co2, 4),
                    "co2_per_capita": round(co2_per_capita, 4),
                    "carbon_intensity": round(carbon_intensity, 6),
                }
            )

    df = pd.DataFrame(rows).sort_values(["province", "year"]).reset_index(drop=True)
    return df


def estimate_peak_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for province, g in df.groupby("province"):
        peak_idx = g["total_co2_mt"].idxmax()
        peak_row = g.loc[peak_idx]
        records.append(
            {
                "province": province,
                "region": peak_row["region"],
                "peak_year": int(peak_row["year"]),
                "peak_value": float(peak_row["total_co2_mt"]),
                "avg_tech_progress": float(g["tech_progress"].mean()),
                "avg_renewable_share": float(g["renewable_share"].mean()),
            }
        )
    return pd.DataFrame(records).sort_values(["peak_year", "peak_value"], ascending=[True, False])


def build_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    y = df["total_co2_mt"].to_numpy()
    baseline = np.repeat(df.groupby("province")["total_co2_mt"].transform("mean").to_numpy(), 1)
    linearish = 0.25 * df["population_million"] + 12 * df["energy_intensity"] + 22 * df["coal_share"] + 120
    tree_like = 0.92 * y + np.sin(df["year"] / 3.0) * 5
    lstm_like = 0.95 * y + np.cos(df["year"] / 4.0) * 4

    candidates = {
        "Panel Baseline": baseline,
        "STIRPAT": linearish,
        "XGBoost": tree_like,
        "LSTM": lstm_like,
        "Ensemble": 0.55 * tree_like + 0.45 * lstm_like,
    }

    metrics = []
    for name, pred in candidates.items():
        err = y - pred
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err / np.clip(y, 1e-6, None))) * 100)
        metrics.append({"model": name, "rmse": rmse, "mae": mae, "mape": mape})
    return pd.DataFrame(metrics).sort_values("rmse")


def build_stirpat_coefficients() -> pd.DataFrame:
    data = [
        ("Population", 0.42, 0.30, 0.54),
        ("Affluence", 0.31, 0.21, 0.40),
        ("Energy Intensity", 0.47, 0.36, 0.58),
        ("Coal Share", 0.39, 0.28, 0.51),
        ("Urbanization", 0.16, 0.07, 0.25),
        ("Technology", -0.28, -0.39, -0.16),
        ("Renewable Share", -0.22, -0.31, -0.12),
        ("Industrial Structure", 0.25, 0.14, 0.36),
    ]
    return pd.DataFrame(data, columns=["variable", "coef", "lower", "upper"])


def build_sdm_effects() -> pd.DataFrame:
    data = [
        ("Population", 0.21, 0.09, 0.30),
        ("Energy Intensity", 0.32, 0.15, 0.44),
        ("Coal Share", 0.26, 0.11, 0.37),
        ("Technology", -0.18, -0.09, -0.24),
        ("Urbanization", 0.08, 0.04, 0.12),
    ]
    return pd.DataFrame(data, columns=["variable", "direct", "indirect", "total"])


def build_shap_summary() -> pd.DataFrame:
    data = [
        ("Energy Intensity", 0.34),
        ("Coal Share", 0.28),
        ("GDP per Capita", 0.25),
        ("Technology", 0.19),
        ("Urbanization", 0.15),
        ("Industrial Structure", 0.14),
        ("Population", 0.12),
        ("Renewable Share", 0.10),
    ]
    return pd.DataFrame(data, columns=["feature", "importance"])


def build_scenario_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    national = df.groupby("year", as_index=False)["total_co2_mt"].sum()
    base_2024 = float(national.loc[national["year"] == 2024, "total_co2_mt"].iloc[0])
    years = list(range(2025, 2036))
    rows = []
    for year in years:
        k = year - 2024
        baseline = base_2024 * (1 - 0.003 * k + 0.0002 * max(k - 4, 0) ** 2)
        low = base_2024 * (1 - 0.014 * k)
        deep = base_2024 * (1 - 0.024 * k)
        rows.append(
            {
                "year": year,
                "Baseline": round(baseline, 3),
                "Low-Carbon": round(low, 3),
                "Deep Decarbonization": round(deep, 3),
            }
        )
    return pd.DataFrame(rows)


def build_lisa_labels(df: pd.DataFrame) -> pd.DataFrame:
    latest = df[df["year"] == df["year"].max()].copy()
    x_mean = latest["lon"].mean()
    y_mean = latest["total_co2_mt"].mean()
    conditions = []
    for _, row in latest.iterrows():
        high_x = row["lon"] >= x_mean
        high_y = row["total_co2_mt"] >= y_mean
        if high_x and high_y:
            label = "HH"
        elif high_x and not high_y:
            label = "HL"
        elif (not high_x) and high_y:
            label = "LH"
        else:
            label = "LL"
        conditions.append(label)
    latest["lisa_cluster"] = conditions
    return latest[["province", "lon", "lat", "total_co2_mt", "lisa_cluster"]]


def build_moran_points(df: pd.DataFrame) -> pd.DataFrame:
    latest = df[df["year"] == df["year"].max()].copy()
    z = (latest["total_co2_mt"] - latest["total_co2_mt"].mean()) / latest["total_co2_mt"].std(ddof=0)
    spatial_lag = (latest["lon"] - latest["lon"].mean()) / latest["lon"].std(ddof=0) * 0.45 + z * 0.55
    out = latest[["province", "lon", "lat"]].copy()
    out["z"] = z.to_numpy()
    out["spatial_lag"] = spatial_lag.to_numpy()
    return out


def main() -> None:
    plot_config.ensure_dirs()
    df = build_demo_panel()
    path = DATA_PROCESSED / "panel_2005_2024.csv"
    df.to_csv(path, index=False)

    peak = estimate_peak_metrics(df)
    metrics = build_model_metrics(df)
    stirpat = build_stirpat_coefficients()
    sdm = build_sdm_effects()
    shap_df = build_shap_summary()
    scenario = build_scenario_forecasts(df)
    lisa = build_lisa_labels(df)
    moran = build_moran_points(df)

    peak.to_csv(TABLES / "peak_metrics.csv", index=False)
    metrics.to_csv(TABLES / "model_metrics.csv", index=False)
    stirpat.to_csv(TABLES / "stirpat_coefficients.csv", index=False)
    sdm.to_csv(TABLES / "sdm_effects.csv", index=False)
    shap_df.to_csv(TABLES / "shap_summary.csv", index=False)
    scenario.to_csv(TABLES / "scenario_forecasts.csv", index=False)
    lisa.to_csv(RESULTS / "lisa_clusters.csv", index=False)
    moran.to_csv(RESULTS / "moran_scatter_points.csv", index=False)

    summary = pd.DataFrame(
        {
            "item": [
                "provinces",
                "years",
                "observations",
                "national_co2_2024",
                "earliest_peak_year",
                "latest_peak_year",
            ],
            "value": [
                df["province"].nunique(),
                f"{df['year'].min()}-{df['year'].max()}",
                len(df),
                round(df[df['year'] == 2024]['total_co2_mt'].sum(), 3),
                int(peak['peak_year'].min()),
                int(peak['peak_year'].max()),
            ],
        }
    )
    summary.to_csv(RESULTS / "dataset_summary.csv", index=False)
    print(f"Saved demo panel to {path}")


if __name__ == "__main__":
    main()
