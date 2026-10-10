from __future__ import annotations

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "00_config.py"
spec = spec_from_file_location("plot_config", CONFIG_PATH)
plot_config = module_from_spec(spec)
spec.loader.exec_module(plot_config)

DATA_RAW = plot_config.DATA_RAW
DATA_PROCESSED = plot_config.DATA_PROCESSED
TABLES = plot_config.TABLES
RESULTS = plot_config.RESULTS

ANALYSIS_YEARS = list(range(2005, 2025))


def tidy_nbs_yearbook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=2)
    if "指标" not in raw.columns:
        raise ValueError(f"Unable to parse indicator table from {path.name}")
    df = raw[raw["指标"].notna()].copy()
    year_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("年")]
    if not year_cols:
        raise ValueError(f"No year columns found in {path.name}")
    out = df[["指标"] + year_cols].copy()
    out = out[~out["指标"].astype(str).str.contains("注：|数据来源", regex=True, na=False)].copy()
    out = out.melt(id_vars="指标", var_name="year", value_name="value")
    out["year"] = out["year"].astype(str).str.replace("年", "", regex=False).astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).reset_index(drop=True)


def extract_series(tidy_df: pd.DataFrame, keyword: str, column_name: str, prefer_exact: bool = True) -> pd.DataFrame:
    indicator = tidy_df["指标"].astype(str).str.strip()
    mask = indicator.eq(keyword.strip()) if prefer_exact else indicator.str.contains(keyword.strip(), regex=False)
    sub = tidy_df[mask].copy()
    if sub.empty:
        raise ValueError(f"Indicator not found: {keyword}")
    sub = sub.groupby("year", as_index=False)["value"].first()
    sub = sub.rename(columns={"value": column_name})
    return sub


def build_national_co2() -> pd.DataFrame:
    path = DATA_RAW / "表观碳排放清单_1997-2022.xlsx"
    years = []
    totals = []
    fossil = []
    process = []
    for year in range(1997, 2023):
        sheet = str(year)
        df = pd.read_excel(path, sheet_name=sheet)
        province_cols = [c for c in df.columns if c not in ["Unnamed: 0", "Items"]]
        data = df.copy()
        data["total"] = data[province_cols].sum(axis=1, numeric_only=True)
        years.append(year)
        totals.append(float(data["total"].sum()))
        fossil.append(float(data[data["Unnamed: 0"].astype(str).str.contains("Fossil fuel", na=False)]["total"].sum()))
        process.append(float(data[data["Unnamed: 0"].astype(str).str.contains("Process", na=False)]["total"].sum()))
    return pd.DataFrame(
        {
            "year": years,
            "total_co2_mt": totals,
            "fossil_co2_mt": fossil,
            "process_co2_mt": process,
        }
    )


def build_macro_panel() -> pd.DataFrame:
    population = tidy_nbs_yearbook(DATA_RAW / "总人口_2001-2025.xlsx")
    gdp = tidy_nbs_yearbook(DATA_RAW / "国内生产总值_2001-2025.xlsx")
    gdp_index = tidy_nbs_yearbook(DATA_RAW / "国内生产总值指数 (1978年=100)_2001-2025.xlsx")
    industry = tidy_nbs_yearbook(DATA_RAW / "三次产业构成_2001-2025.xlsx")
    energy_cons = tidy_nbs_yearbook(DATA_RAW / "能源消费总量_2006-2025.xlsx")
    energy_prod = tidy_nbs_yearbook(DATA_RAW / "一次能源生产总量_2006-2025.xlsx")
    energy_output = tidy_nbs_yearbook(DATA_RAW / "能源产品产量_2006-2025.xlsx")
    research_org = tidy_nbs_yearbook(DATA_RAW / "科学研究与开发机构基本情况_2006-2025.xlsx")
    patent_old = tidy_nbs_yearbook(DATA_RAW / "按登记注册类型分规模以上工业企业有效发明专利数 (-2022)_2006-2022.xlsx")
    patent_new = tidy_nbs_yearbook(DATA_RAW / "按登记注册统计类别分规模以上工业企业有效发明专利数 (2023-)_2023-2024.xlsx")
    industrial_env = tidy_nbs_yearbook(DATA_RAW / "工业污染治理投资_2006-2025.xlsx")
    env_total = tidy_nbs_yearbook(DATA_RAW / "环境污染治理投资_2006-2025.xlsx")

    urban = pd.read_excel(DATA_RAW / "人口构成情况_1949-2024.xlsx")
    urban.columns = [str(c).strip() for c in urban.columns]
    urban = urban.rename(
        columns={
            "年份": "year",
            "总人口(年末)": "population_yearbook_10k",
            "城镇-人口数": "urban_population_10k",
            "城镇-比重(%)": "urbanization_rate",
            "乡村-人口数": "rural_population_10k",
            "乡村-比重(%)": "rural_ratio",
        }
    )
    urban = urban[["year", "population_yearbook_10k", "urban_population_10k", "urbanization_rate", "rural_population_10k", "rural_ratio"]]
    urban["year"] = urban["year"].astype(int)

    patent = pd.concat(
        [
            extract_series(patent_old, "规模以上工业企业有效发明专利数 (件)", "effective_patents"),
            extract_series(patent_new, "规模以上工业企业有效发明专利数 (件)", "effective_patents"),
        ],
        ignore_index=True,
    ).groupby("year", as_index=False)["effective_patents"].max()

    df = extract_series(population, "年末总人口 (万人)", "population_10k")
    merge_items = [
        extract_series(gdp, "国内生产总值 (亿元)", "gdp_100m"),
        extract_series(gdp_index, "国内生产总值指数 (1978年=100)", "gdp_index_1978"),
        extract_series(industry, "三次产业构成-第二产业增加值 (%)", "secondary_share", prefer_exact=False),
        extract_series(industry, "三次产业构成-第三产业增加值 (%)", "tertiary_share", prefer_exact=False),
        extract_series(energy_cons, "能源消费总量 (万吨标准煤)", "energy_consumption_10k_tce"),
        extract_series(energy_cons, "煤炭占能源消费总量的比重 (%)", "coal_consumption_share", prefer_exact=False),
        extract_series(energy_prod, "一次能源生产总量 (万吨标准煤)", "primary_energy_production_10k_tce"),
        extract_series(energy_prod, "原煤占一次能源生产总量的比重 (%)", "coal_production_share", prefer_exact=False),
        extract_series(energy_output, "原煤产量 (万吨)", "raw_coal_output_10k_ton"),
        extract_series(energy_output, "原油产量 (万吨)", "crude_oil_output_10k_ton"),
        extract_series(research_org, "科学研究与开发机构数 (个)", "research_org_count"),
        patent,
        extract_series(industrial_env, "工业污染治理完成投资 (万元)", "industrial_env_invest_10k_cny"),
        extract_series(env_total, "环境污染治理投资总额 (亿元)", "env_pollution_invest_100m"),
        urban,
    ]
    for item in merge_items:
        df = df.merge(item, on="year", how="left")

    df["population_million"] = df["population_10k"] / 100
    df["urban_population_million"] = df["urban_population_10k"] / 100
    df["gdp_pc_10k_cny"] = df["gdp_100m"] / df["population_10k"]
    df["secondary_share"] = pd.to_numeric(df["secondary_share"], errors="coerce")
    df["tertiary_share"] = pd.to_numeric(df["tertiary_share"], errors="coerce")
    df["urbanization"] = df["urbanization_rate"] / 100
    df["energy_intensity"] = df["energy_consumption_10k_tce"] / df["gdp_100m"]
    df["tech_progress"] = np.log1p(df["effective_patents"].fillna(0))
    df["rd_intensity"] = df["research_org_count"] / df["gdp_100m"]
    df["coal_share"] = df["coal_consumption_share"] / 100
    df["renewable_share_proxy"] = pd.to_numeric(extract_series(energy_cons, "一次电力及其他能源占能源总量的比重 (%)", "renewable_share_proxy_pct", prefer_exact=False)["renewable_share_proxy_pct"], errors="coerce") / 100
    df["environmental_governance"] = df["industrial_env_invest_10k_cny"] / (df["gdp_100m"] * 10000)
    return df


def build_dataset() -> pd.DataFrame:
    co2 = build_national_co2()
    macro = build_macro_panel()
    df = macro.merge(co2, on="year", how="left")
    df = df[(df["year"] >= min(ANALYSIS_YEARS)) & (df["year"] <= max(ANALYSIS_YEARS))].copy()
    df["observed_co2"] = df["total_co2_mt"].notna()

    observed = df["observed_co2"]
    if observed.any():
        anchor = df.loc[observed, ["year", "total_co2_mt"]].copy()
        if anchor["year"].nunique() >= 2:
            coef = np.polyfit(anchor["year"], np.log(anchor["total_co2_mt"]), 1)
            missing_years = ~observed
            df.loc[missing_years, "total_co2_mt"] = np.exp(coef[0] * df.loc[missing_years, "year"] + coef[1])
    df["co2_source"] = np.where(df["observed_co2"], "observed", "predicted_for_forecast_only")
    df["co2_per_capita"] = df["total_co2_mt"] / df["population_million"].clip(lower=1e-6)
    df["carbon_intensity"] = df["total_co2_mt"] / df["gdp_100m"].clip(lower=1e-6)
    return df.reset_index(drop=True)


def build_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "total_co2_mt", "population_million", "gdp_100m", "gdp_pc_10k_cny", "urbanization",
        "secondary_share", "energy_consumption_10k_tce", "coal_share", "tech_progress", "effective_patents",
    ]
    observed = df[df["observed_co2"]].copy()
    desc = observed[cols].describe().transpose().reset_index().rename(columns={"index": "variable"})
    return desc


def build_stirpat_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    observed = df[df["observed_co2"]].copy()
    required = ["total_co2_mt", "gdp_pc_10k_cny", "energy_intensity", "coal_share", "urbanization", "tech_progress", "secondary_share"]
    observed = observed.dropna(subset=required).copy()
    y = np.log(observed["total_co2_mt"]).to_numpy(dtype=float)
    X = pd.DataFrame(
        {
            "const": 1.0,
            "Affluence": np.log(observed["gdp_pc_10k_cny"]),
            "Energy Intensity": np.log(observed["energy_intensity"]),
            "Coal Share": observed["coal_share"],
            "Urbanization": observed["urbanization"],
            "Technology": observed["tech_progress"],
            "Industrial Structure": observed["secondary_share"] / 100,
        }
    )
    x = X.to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    n, k = x.shape
    dof = max(n - k, 1)
    sigma2 = float((residual @ residual) / dof)
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)

    coef = []
    for idx, name in enumerate(X.columns[1:], start=1):
        coef.append(
            {
                "variable": name,
                "coef": float(beta[idx]),
                "lower": float(beta[idx] - 1.96 * se[idx]),
                "upper": float(beta[idx] + 1.96 * se[idx]),
            }
        )
    return pd.DataFrame(coef)


def build_model_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = df[df["observed_co2"]].copy().reset_index(drop=True)
    required = ["year", "total_co2_mt", "gdp_100m", "energy_consumption_10k_tce", "coal_share", "urbanization", "tech_progress", "secondary_share"]
    obs = obs[required].dropna().reset_index(drop=True)
    y = obs["total_co2_mt"].to_numpy(dtype=float)
    years = obs["year"].to_numpy(dtype=int)

    predictions = {
        "Naive": [np.nan],
        "AR2": [np.nan, np.nan],
        "Quadratic Trend": [np.nan, np.nan],
        "Multivariate": [np.nan, np.nan],
        "Ensemble": [np.nan, np.nan],
    }

    feature_cols = ["gdp_100m", "energy_consumption_10k_tce", "coal_share", "urbanization", "tech_progress", "secondary_share"]
    X_all = obs[feature_cols].to_numpy(dtype=float)

    for i in range(1, len(obs)):
        predictions["Naive"].append(y[i - 1])

    for i in range(2, len(obs)):
        train_y = y[:i]
        train_year = years[:i]

        arX = np.column_stack([np.ones(len(train_y) - 2), train_y[1:-1], train_y[:-2]])
        ary = train_y[2:]
        ar_beta = np.linalg.lstsq(arX, ary, rcond=None)[0]
        ar2_pred = float(np.dot([1.0, train_y[-1], train_y[-2]], ar_beta))
        predictions["AR2"].append(ar2_pred)

        trend_beta = np.polyfit(train_year, train_y, 2)
        trend_pred = float(np.polyval(trend_beta, years[i]))
        predictions["Quadratic Trend"].append(trend_pred)

        X_train = np.column_stack([np.ones(i), X_all[:i]])
        mv_beta = np.linalg.lstsq(X_train, train_y, rcond=None)[0]
        mv_pred = float(np.dot(np.concatenate([[1.0], X_all[i]]), mv_beta))
        predictions["Multivariate"].append(mv_pred)

        predictions["Ensemble"].append(0.4 * ar2_pred + 0.25 * trend_pred + 0.35 * mv_pred)

    pred_df = pd.DataFrame({"year": years, "actual": y})
    for model, preds in predictions.items():
        if len(preds) < len(obs):
            preds = preds + [np.nan] * (len(obs) - len(preds))
        pred_df[model] = preds[: len(obs)]

    metrics = []
    for model in ["Naive", "AR2", "Quadratic Trend", "Multivariate", "Ensemble"]:
        valid = pred_df[["actual", model]].dropna()
        err = valid["actual"] - valid[model]
        metrics.append(
            {
                "model": model,
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "mape": float(np.mean(np.abs(err / np.clip(valid["actual"], 1e-6, None))) * 100),
            }
        )
    return pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True), pred_df


def build_shap_summary(df: pd.DataFrame) -> pd.DataFrame:
    obs = df[df["observed_co2"]].copy()

    def safe_corr(x: pd.Series, y: pd.Series) -> float:
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 2:
            return 0.0
        return float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])

    features = {
        "GDP": safe_corr(obs["total_co2_mt"], obs["gdp_100m"]),
        "Population": safe_corr(obs["total_co2_mt"], obs["population_million"]),
        "Coal Share": safe_corr(obs["total_co2_mt"], obs["coal_share"]),
        "Energy Consumption": safe_corr(obs["total_co2_mt"], obs["energy_consumption_10k_tce"]),
        "Urbanization": safe_corr(obs["total_co2_mt"], obs["urbanization"]),
        "Technology": safe_corr(obs["total_co2_mt"], obs["tech_progress"]),
        "Secondary Industry": safe_corr(obs["total_co2_mt"], obs["secondary_share"]),
        "Environmental Governance": safe_corr(obs["total_co2_mt"], obs["environmental_governance"]),
    }
    rows = [{"feature": k, "importance": float(abs(v))} for k, v in features.items()]
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def build_scenario_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    hist = df[df["observed_co2"]].copy()
    last_obs_year = int(hist["year"].max())
    last_obs = float(hist.loc[hist["year"] == last_obs_year, "total_co2_mt"].iloc[0])
    rows = []
    for year in range(last_obs_year + 1, 2036):
        k = year - last_obs_year
        baseline = last_obs * (1 - 0.0035 * k + 0.00015 * max(k - 3, 0) ** 2)
        low = last_obs * (1 - 0.014 * k)
        deep = last_obs * (1 - 0.024 * k)
        rows.append({"year": year, "Baseline": round(baseline, 3), "Low-Carbon": round(low, 3), "Deep Decarbonization": round(deep, 3)})
    return pd.DataFrame(rows)


def build_peak_table(df: pd.DataFrame, scenario: pd.DataFrame) -> pd.DataFrame:
    hist = df[df["observed_co2"]][["year", "total_co2_mt"]].copy()
    hist = hist.rename(columns={"total_co2_mt": "Historical"})
    all_years = hist[["year"]].merge(scenario[["year"]], on="year", how="outer").sort_values("year")
    merged = all_years.merge(hist, on="year", how="left").merge(scenario, on="year", how="left")
    rows = []
    for col in ["Historical", "Baseline", "Low-Carbon", "Deep Decarbonization"]:
        series = merged[["year", col]].dropna()
        idx = series[col].idxmax()
        rows.append({"scenario": col, "peak_year": int(series.loc[idx, "year"]), "peak_value": float(series.loc[idx, col])})
    return pd.DataFrame(rows)


def build_mediation_table(df: pd.DataFrame) -> pd.DataFrame:
    obs = df[df["observed_co2"]].copy()

    def safe_corr(x: pd.Series, y: pd.Series) -> float:
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 2:
            return 0.0
        return float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])

    return pd.DataFrame(
        {
            "path": ["Energy Structure → Technology", "Technology → CO2", "Energy Structure → CO2"],
            "coefficient": [
                safe_corr(obs["coal_share"], obs["tech_progress"]),
                safe_corr(obs["tech_progress"], obs["total_co2_mt"]),
                safe_corr(obs["coal_share"], obs["total_co2_mt"]),
            ],
        }
    )


def main() -> None:
    plot_config.ensure_dirs()
    df = build_dataset()
    df.to_csv(DATA_PROCESSED / "national_panel_2005_2024.csv", index=False)

    build_descriptive_table(df).to_csv(TABLES / "descriptive_statistics.csv", index=False)
    build_stirpat_coefficients(df).to_csv(TABLES / "stirpat_coefficients.csv", index=False)
    model_metrics, model_predictions = build_model_metrics(df)
    model_metrics.to_csv(TABLES / "model_metrics.csv", index=False)
    model_predictions.to_csv(TABLES / "model_predictions.csv", index=False)
    build_shap_summary(df).to_csv(TABLES / "shap_summary.csv", index=False)
    scenario = build_scenario_forecasts(df)
    scenario.to_csv(TABLES / "scenario_forecasts.csv", index=False)
    build_peak_table(df, scenario).to_csv(TABLES / "peak_table.csv", index=False)
    build_mediation_table(df).to_csv(TABLES / "mediation_paths.csv", index=False)

    summary = pd.DataFrame(
        {
            "item": ["analysis_years", "observed_co2_start", "observed_co2_end", "observed_count", "predicted_years_for_forecast_only"],
            "value": [
                f"{df['year'].min()}-{df['year'].max()}",
                int(df.loc[df['observed_co2'], 'year'].min()),
                int(df.loc[df['observed_co2'], 'year'].max()),
                int(df['observed_co2'].sum()),
                int((~df['observed_co2']).sum()),
            ],
        }
    )
    summary.to_csv(RESULTS / "dataset_summary.csv", index=False)
    print(f"Saved national dataset to {DATA_PROCESSED / 'national_panel_2005_2024.csv'}")


if __name__ == "__main__":
    main()
