from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
RESULTS = OUTPUTS / "results"
PAPER = ROOT / "paper"
SECTIONS = PAPER / "sections"

PALETTE = {
    "navy": "#1D3557",
    "blue": "#4C78A8",
    "sky": "#8ECAE6",
    "green": "#2A9D8F",
    "red": "#D1495B",
    "orange": "#F4A261",
    "purple": "#7B6DCC",
    "teal": "#5AB1BB",
    "gold": "#E9C46A",
    "gray": "#8D99AE",
    "charcoal": "#3A3A3A",
    "light": "#F6F8FB"
}

SCENARIO_COLORS = {
    "Baseline": "#1D3557",
    "Low-Carbon": "#2A9D8F",
    "Deep Decarbonization": "#D1495B"
}

TITLE_FONT_FAMILY = "DejaVu Serif"
BODY_FONT_FAMILY = "Times New Roman"
FIG_DPI = 320
GRID_ALPHA = 0.22


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES, TABLES, RESULTS, PAPER, SECTIONS]:
        path.mkdir(parents=True, exist_ok=True)
