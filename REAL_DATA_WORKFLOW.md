# Real-data workflow

## 1. Install packages
Recommended:
```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost shap networkx openpyxl xlrd pyarrow
```
If you want to run the LSTM part later:
```bash
pip3 install tensorflow
```
If you plan to make proper GIS-style province maps instead of centroid scatter maps:
```bash
pip3 install geopandas shapely pyproj fiona mapclassify
```

## 2. Prepare raw CSV files under `data/raw/`
Required files:
- `ceads_provincial_co2.csv`
- `nbs_macro_panel.csv`
- `energy_panel.csv`
- `technology_panel.csv`

Optional:
- `province_coords.csv`

## 3. Required columns
### ceads_provincial_co2.csv
- `province`
- `year`
- `total_co2_mt`

### nbs_macro_panel.csv
- `province`
- `year`
- `population_million`
- `gdp_pc_10k_cny`
- `urbanization`
- `industry_share`

### energy_panel.csv
- `province`
- `year`
- `energy_intensity`
- `coal_share`
- `renewable_share`

### technology_panel.csv
- `province`
- `year`
- `tech_progress`
- `rd_intensity`
- `patents_per_10k`

## 4. Build the real panel
```bash
python3 scripts/02_data_sources_template.py
python3 scripts/03_build_real_panel.py
python3 scripts/11_final_figures.py
```

## 5. Notes on data search
Suggested source priority:
1. CEADs for province-level total CO2
2. National Bureau of Statistics / provincial yearbooks for macro indicators
3. China Energy Statistical Yearbook for energy intensity and energy structure
4. Wind for missing supplements or cross-checking
5. EDGAR / IEA / MEE bulletins for robustness and narrative support

## 6. Current limitation
The current `11_final_figures.py` can already consume `data/processed/panel_2005_2024.csv`. Once you replace the demo panel with the real panel, rerunning the figure script will produce real-data charts.
