import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_SOURCES = [
    {
        "id": "ceads_co2",
        "name": "CEADs provincial CO2 emissions",
        "required": True,
        "notes": "Preferred source for province-level CO2 emissions. Save cleaned file as data/raw/ceads_provincial_co2.csv.",
        "expected_columns": ["province", "year", "total_co2_mt"],
    },
    {
        "id": "nbs_macro",
        "name": "NBS / Provincial statistical yearbooks macro panel",
        "required": True,
        "notes": "Population, GDP per capita, urbanization, industrial structure. Save as data/raw/nbs_macro_panel.csv.",
        "expected_columns": ["province", "year", "population_million", "gdp_pc_10k_cny", "urbanization", "industry_share"],
    },
    {
        "id": "energy_panel",
        "name": "Energy structure and energy intensity panel",
        "required": True,
        "notes": "Coal share, renewable share, energy intensity. Save as data/raw/energy_panel.csv.",
        "expected_columns": ["province", "year", "energy_intensity", "coal_share", "renewable_share"],
    },
    {
        "id": "technology_panel",
        "name": "Technology progress panel",
        "required": True,
        "notes": "R&D intensity, patents or similar innovation proxy. Save as data/raw/technology_panel.csv.",
        "expected_columns": ["province", "year", "tech_progress", "rd_intensity", "patents_per_10k"],
    },
    {
        "id": "province_coords",
        "name": "Province centroid coordinates",
        "required": False,
        "notes": "Optional if you want to override built-in coordinates. Save as data/raw/province_coords.csv.",
        "expected_columns": ["province", "lon", "lat"],
    },
]


def main() -> None:
    out = ROOT / "data" / "raw" / "data_sources_template.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(DATA_SOURCES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved data source template to {out}")


if __name__ == "__main__":
    main()
