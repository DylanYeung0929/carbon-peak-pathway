from __future__ import annotations

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap

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
    """
    真实的滚动一步预测实现（已修复数组对齐与空值过滤问题）。
    """
    obs = df[df["observed_co2"]].copy().reset_index(drop=True)
    required = ["year", "total_co2_mt", "gdp_100m", "energy_consumption_10k_tce",
                "coal_share", "urbanization", "tech_progress", "secondary_share"]
    obs = obs[required].dropna().reset_index(drop=True)

    y = obs["total_co2_mt"].to_numpy(dtype=float)
    years = obs["year"].to_numpy(dtype=int)
    feature_cols = required[2:]
    X_all = obs[feature_cols].to_numpy(dtype=float)

    # 初始化预测字典
    predictions = {
        "year": years.tolist(),
        "actual": y.tolist(),
        "Naive": [np.nan, np.nan],
        "AR2": [np.nan, np.nan],
        "Quadratic Trend": [np.nan, np.nan],
        "Multivariate": [np.nan, np.nan],
        "Random Forest": [np.nan, np.nan],
        "XGBoost": [np.nan, np.nan],
        "Ensemble": [np.nan, np.nan],
    }

    # 从第3期开始滚动预测
    for i in range(2, len(obs)):
        train_y = y[:i]
        train_X = X_all[:i]
        test_X = X_all[i].reshape(1, -1)

        # 1. Naive
        predictions["Naive"].append(y[i - 1])

        # 2. AR2 (严谨处理：必须 i>=3 才能凑齐 y_t, y_{t-1}, y_{t-2})
        if i >= 3:
            y_target = train_y[2:]  # 因变量 y_t
            lag1 = train_y[1:-1]  # y_{t-1}
            lag2 = train_y[:-2]  # y_{t-2}
            ar_X = np.column_stack([lag1, lag2])
            ar_model = LinearRegression().fit(ar_X, y_target)
            ar_pred = ar_model.predict([[train_y[-1], train_y[-2]]])
            predictions["AR2"].append(float(ar_pred[0]))
        else:
            predictions["AR2"].append(np.nan)  # i=2时样本不足，填NaN

        # 3. Quadratic Trend
        trend_X = np.column_stack([np.array(range(i)), np.array(range(i)) ** 2])
        trend_model = LinearRegression().fit(trend_X, train_y)
        predictions["Quadratic Trend"].append(float(trend_model.predict([[i, i ** 2]])[0]))

        # 4. Multivariate OLS
        mv_model = LinearRegression().fit(train_X, train_y)
        predictions["Multivariate"].append(float(mv_model.predict(test_X)[0]))

        # 5. Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=30, max_depth=2, min_samples_leaf=3, random_state=2026
        ).fit(train_X, train_y)
        predictions["Random Forest"].append(float(rf_model.predict(test_X)[0]))

        # 6. XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=30, max_depth=1, learning_rate=0.1,
            reg_alpha=5.0, reg_lambda=10.0, subsample=0.7, random_state=2026
        ).fit(train_X, train_y)
        predictions["XGBoost"].append(float(xgb_model.predict(test_X)[0]))

        # 7. Ensemble
        ens_pred = (0.4 * predictions["AR2"][i] +
                    0.25 * predictions["Quadratic Trend"][i] +
                    0.35 * predictions["Multivariate"][i])
        predictions["Ensemble"].append(ens_pred)

    pred_df = pd.DataFrame(predictions)

    # ================= 计算误差（自动过滤NaN，确保公平对比） =================
    metrics = []
    for model in ["Naive", "AR2", "Quadratic Trend", "Multivariate",
                  "Random Forest", "XGBoost", "Ensemble"]:
        valid_actual = pred_df["actual"].iloc[2:].to_numpy()
        valid_pred = pred_df[model].iloc[2:].to_numpy()

        # 核心修复：过滤掉 AR2 在 i=2 时产生的 NaN 值
        mask = ~np.isnan(valid_pred)
        valid_actual = valid_actual[mask]
        valid_pred = valid_pred[mask]

        if len(valid_actual) > 0:
            err = valid_actual - valid_pred
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            mape = float(np.mean(np.abs(err / np.clip(valid_actual, 1e-6, None))) * 100)
            metrics.append({"model": model, "rmse": rmse, "mae": mae, "mape": mape})

    metrics_df = pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True)

    print("\n--- 真实滚动预测误差 ---")
    print(metrics_df.to_string(index=False))
    print("------------------------\n")

    return metrics_df, pred_df


def build_shap_summary(df: pd.DataFrame) -> pd.DataFrame:
    import shap
    obs = df[df["observed_co2"]].copy()

    # ⚠️ 核心修改：剔除 GDP总量 和 能源总量，避免共线性吞噬结构变量
    # 保留：人均GDP(富裕度)、能源强度(效率)、煤炭占比(能源结构)、
    #       城镇化、二产占比(产业结构)、技术进步、环境治理
    feature_cols = [
        "gdp_pc_10k_cny", "energy_intensity", "coal_share",
        "urbanization", "secondary_share", "tech_progress", "environmental_governance"
    ]

    X = obs[feature_cols].fillna(0)
    y = obs["total_co2_mt"]

    # 训练全样本 XGBoost 用于解释（注意：这里仅用于提取特征重要性，不用于预测评估）
    model = xgb.XGBRegressor(n_estimators=50, max_depth=2, random_state=2026).fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 计算平均绝对 SHAP 值
    import numpy as np
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # 将英文列名映射为中文，方便后续画图和论文直接使用
    name_map = {
        "gdp_pc_10k_cny": "人均GDP",
        "energy_intensity": "能源强度",
        "coal_share": "煤炭占比",
        "urbanization": "城镇化率",
        "secondary_share": "第二产业占比",
        "tech_progress": "技术进步",
        "environmental_governance": "环境治理投入"
    }

    result_df = pd.DataFrame({
        "feature": [name_map[c] for c in feature_cols],
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=False)

    # 打印一下结果，方便你写论文
    print("\n--- 真实 SHAP 特征重要性排序 ---")
    print(result_df.to_string(index=False))
    print("----------------------------------\n")

    return result_df


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
    from sklearn.linear_model import LinearRegression
    import numpy as np

    obs = df[df["observed_co2"]].copy()
    # 确保没有空值
    obs = obs.dropna(subset=["coal_share", "tech_progress", "total_co2_mt", "gdp_pc_10k_cny"]).copy()

    X_coal = obs[["coal_share"]].values
    M_tech = obs["tech_progress"].values
    Y_co2 = np.log(obs["total_co2_mt"].values)

    # Step 1: X -> M (a 路径)
    lr1 = LinearRegression().fit(X_coal, M_tech)
    a_coef = lr1.coef_[0]

    # Step 2: X -> Y (c 路径 - 总效应)
    lr2 = LinearRegression().fit(X_coal, Y_co2)
    c_coef = lr2.coef_[0]

    # Step 3: X + M -> Y (c' 路径和 b 路径)
    # 注意：这里为了严谨，建议把人均GDP也加进去作为控制变量，防止遗漏变量偏误
    X_control = obs[["gdp_pc_10k_cny"]].values
    X_m = np.column_stack([X_coal, M_tech, X_control])
    lr3 = LinearRegression().fit(X_m, Y_co2)
    c_prime_coef = lr3.coef_[0]  # 直接效应
    b_coef = lr3.coef_[1]  # 中介效应
    control_coef = lr3.coef_[2]  # 控制变量系数

    indirect = a_coef * b_coef

    print("\n--- 真实中介效应逐步回归结果 ---")
    print(f"a 路径 (煤炭->技术): {a_coef:.4f}")
    print(f"b 路径 (技术->排放, 控制GDP后): {b_coef:.4f}")
    print(f"c' 路径 (煤炭->排放, 直接效应): {c_prime_coef:.4f}")
    print(f"间接效应 (a*b): {indirect:.4f}")
    print(f"总效应: {c_coef:.4f}")
    print("------------------------------------\n")

    return pd.DataFrame({
        "path": ["a (煤炭→技术)", "b (技术→排放)", "c' (煤炭→排放 直接)", "间接效应 (a×b)", "总效应"],
        "coefficient": [a_coef, b_coef, c_prime_coef, indirect, c_coef]
    })


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
