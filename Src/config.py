# Src/config.py
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Data paths
RAW_DATA       = ROOT / "Data" / "Raw" / "Case 2 Dataset.csv"
CLEANED_DATA   = ROOT / "Data" / "Cleaned" / "Case 2 Dataset.csv"
ANALYZED_DATA  = ROOT / "Data" / "Analyzed" / "mbg_analyzed.csv"
STEMMING_CACHE = ROOT / "Data" / "Cache" / "stemming_dict.json"
SLANG_DICT     = ROOT / "Data" / "Cache" / "slang_dict.json"

# Output paths
FIG_DIR        = ROOT / "Output" / "Figures"
REPORT_DIR     = ROOT / "Output" / "Reports"