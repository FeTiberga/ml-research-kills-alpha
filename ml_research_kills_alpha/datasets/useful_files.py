# Paths to useful files used in the data processing pipeline

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = BASE_DIR / "ml_research_kills_alpha" / "datasets" / "data"
CZ_SIGNAL_DOC = DATA_DIR / "cz_signal_doc.csv"
FF_INDUSTRY_CLASSIFICATION = DATA_DIR / "ff_industry_classification.json"
GW_PREDICTOR_DATA_2024 = DATA_DIR / "gw_predictor_data_2024.csv"
