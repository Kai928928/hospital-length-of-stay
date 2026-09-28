"""Data loading and cleaning for the UCI Diabetes 130-US Hospitals dataset.

The raw file contains placeholders '?' instead of NaN, an age column encoded
as bins ("[0-10)", "[10-20)", ...), and ICD-9 diagnosis codes with high
cardinality. This module normalises all of that and derives clinically
meaningful features (diagnosis categories, medication count, discharge flags).
"""
import numpy as np
import pandas as pd

from src.config import (
    CAT_COLS,
    DROP_COLS,
    ENGINEERED_CAT_COLS,
    ENGINEERED_NUM_COLS,
    MED_COLS,
    NUM_COLS,
    RAW_DATA_PATH,
    TARGET_COL,
)

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
AGE_MIDPOINTS = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}

# First digit of an ICD-9 code -> clinical category.
# Used to tame the cardinality of diag_1/2/3 (hundreds of raw codes).
DIAG_LABELS = {
    "1": "Infectious", "2": "Neoplasm", "3": "Endocrine_Metabolic",
    "4": "Blood", "5": "Mental", "6": "Nervous", "7": "Circulatory",
    "8": "Respiratory", "9": "Digestive", "10": "Genitourinary",
    "11": "Pregnancy", "12": "Skin", "13": "Musculoskeletal",
    "14": "Congenital", "15": "Perinatal", "16": "Symptoms",
    "17": "Injury_Poisoning", "18": "External_Cause", "19": "Health_Status",
}

# discharge_disposition_id values indicating death / hospice / transfer / AMA.
# These patients are not discharged home normally and their stay pattern differs.
UNUSUAL_DISCHARGE = {
    11, 13, 14, 19, 20, 21,   # death, hospice, AMA
    8, 9, 12, 17, 22, 23, 24, # transfer to other facilities / hospice
}

GLU_ORDINAL = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
A1C_ORDINAL = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Raises a clear error if the file is missing."""
    if not pd.io.common.file_exists(path):
        raise FileNotFoundError(
            f"Raw data not found at {path}.\n"
            "Run `python data/download_data.py` first to download and "
            "extract the UCI dataset."
        )
    df = pd.read_csv(path, na_values=["?"], keep_default_na=True, low_memory=False)
    return df


def _recode_icd9(code: object) -> str:
    """Map an ICD-9 code to its clinical category label."""
    if pd.isna(code):
        return "Unknown"
    s = str(code).strip()
    if not s:
        return "Unknown"
    # ICD-9 codes can start with 'E' (external cause) or 'V' (health status)
    if s[0].upper() == "E":
        return "External_Cause"
    if s[0].upper() == "V":
        return "Health_Status"
    return DIAG_LABELS.get(s[0], "Other")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning / feature-engineering pipeline on raw data."""
    data = df.copy()

    # 1. Drop identifiers and near-empty columns ----------------------------
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns])

    # 2. Replace '?' placeholders with NaN (load step keeps them as NaN only
    #    for whole-column NA; explicit here for robustness).
    data = data.replace("?", np.nan)

    # 3. Age bin -> numeric midpoint ----------------------------------------
    data["age_years"] = data["age"].map(AGE_MIDPOINTS)
    data = data.drop(columns=["age"])

    # 4. Diagnosis categories (tame cardinality of ICD-9 codes) -------------
    for i in (1, 2, 3):
        col = f"diag_{i}"
        data[f"diag_{i}_cat"] = data[col].map(_recode_icd9)
        data = data.drop(columns=[col])

    # 5. Medication count ----------------------------------------------------
    # 'No' means the drug is not part of the regimen; any other value means
    # the patient is taking it (steady / dose up / dose down).
    med_present = data[MED_COLS].notna() & (data[MED_COLS] != "No")
    data["med_count"] = med_present.sum(axis=1)
    data = data.drop(columns=MED_COLS)

    # 6. Visit history in the year before ------------------------------------
    data["visit_history_total"] = (
        data["number_outpatient"] + data["number_emergency"] + data["number_inpatient"]
    )

    # 7. Unusual discharge flag ----------------------------------------------
    data["is_discharge_unusual"] = (
        data["discharge_disposition_id"].astype(int).isin(UNUSUAL_DISCHARGE).astype(int)
    )

    # 8. Ordinal encoding for glucose / A1C ----------------------------------
    data["max_glu_serum_level"] = data["max_glu_serum"].map(GLU_ORDINAL).fillna(0).astype(int)
    data["a1c_level"] = data["A1Cresult"].map(A1C_ORDINAL).fillna(0).astype(int)
    data = data.drop(columns=["max_glu_serum", "A1Cresult"])

    # 9. Drop rows that are unusable ------------------------------------------
    # gender is required for any meaningful modelling; very few rows lack it.
    data = data.dropna(subset=["gender", "age_years"]).reset_index(drop=True)

    return data


def describe_cleaning(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Produce a small before/after summary table (useful for the report)."""
    rows = {
        "rows": [df_raw.shape[0], df_clean.shape[0]],
        "columns": [df_raw.shape[1], df_clean.shape[1]],
        "memory_MB": [
            round(df_raw.memory_usage(deep=True).sum() / 1e6, 1),
            round(df_clean.memory_usage(deep=True).sum() / 1e6, 1),
        ],
    }
    return pd.DataFrame(rows, index=["raw", "clean"])
