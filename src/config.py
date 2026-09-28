"""Central configuration: paths, target, random seed, and column lists.

The pipeline is built around the UCI "Diabetes 130-US Hospitals" dataset
(101,766 inpatient encounters), where the regression target is the number of
days spent in hospital (`time_in_hospital`).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "diabetic_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

TARGET_COL = "time_in_hospital"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Columns dropped before modelling
# ---------------------------------------------------------------------------
# encounter_id / patient_nbr : unique identifiers, carry no predictive signal.
# weight                     : ~97% missing, effectively unusable.
# payer_code                 : ~40% missing, little clinical meaning.
DROP_COLS = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
]

# ---------------------------------------------------------------------------
# Medication indicator columns (values: No / Steady / Up / Down).
# We aggregate them into a single medication-count feature.
# ---------------------------------------------------------------------------
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
NUM_COLS = [
    "num_lab_procedures",   # number of lab tests performed
    "num_procedures",       # number of non-lab procedures
    "num_medications",      # number of distinct medications
    "number_outpatient",    # outpatient visits in the year before
    "number_emergency",     # emergency visits in the year before
    "number_inpatient",     # inpatient visits in the year before
    "number_diagnoses",     # number of diagnoses entered
]

# Categorical columns that survive cleaning (age / max_glu_serum / A1Cresult
# are converted to numeric ordinals by clean_data and listed in
# ENGINEERED_NUM_COLS instead).
CAT_COLS = [
    "race", "gender", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "change", "diabetesMed",
]

# Engineered categorical features (added during cleaning / feature building)
ENGINEERED_CAT_COLS = ["diag_1_cat", "diag_2_cat", "diag_3_cat"]

ENGINEERED_NUM_COLS = [
    "age_years",          # age bin converted to a numeric midpoint
    "med_count",          # number of diabetes medications in use
    "visit_history_total",  # outpatient + emergency + inpatient in the year before
    "is_discharge_unusual", # death / hospice / transfer / AMA flag
    "max_glu_serum_level",  # ordinal: none < normal < >200 < >300
    "a1c_level",            # ordinal: none < norm < >7 < >8
]
