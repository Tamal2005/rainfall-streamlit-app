import streamlit as st
<<<<<<< HEAD

=======
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import difflib
<<<<<<< HEAD
import gdown
from datetime import datetime
from pathlib import Path

import pydeck as pdk
import matplotlib.pyplot as plt


# ============================================================
# India Rainfall & Climate Prediction
# + KNNR rainfall disaggregation
# + 3 / 5 / 7 day rainfall-pattern matching
# + 24-hour Gram-Schmidt inverse rotation
#
# FIXES IN THIS VERSION:
#   1. prepared_knnr_dataframe() was called with the knnr_library
#      DICT while its body did Path(directory_string) -> TypeError
#      on every run. Now takes a path string (hashable for
#      st.cache_data); duplicate prepare_knnr_dataframe() removed.
#   2. The Gram-Schmidt rotation was a mathematical no-op: it was
#      fed a fragment already scaled to daily_total_mm, so
#      overwriting the sum-coordinate y[-1] assigned the value it
#      already held and x' == x. It now receives the RAW analog
#      fragment so the rescale is real. R is built ONCE at module
#      level instead of ~450 times per prediction.
#   3./4. The UI claimed "3-day, 5-day, 7-day contexts used"
#      unconditionally, even when no pattern/daily-window columns
#      existed (all three windows then collapse to identical
#      day-of-year matching) or the target date wasn't in the
#      library at all. Matching method is now tracked and reported
#      honestly.
#   5. The daily-window shape distance no longer uses iterrows().
# ============================================================


=======
from datetime import datetime
from pathlib import Path
import pydeck as pdk

# ============================================================
# Rainfall & Climate Prediction — India
# ============================================================

>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
st.set_page_config(
    page_title="India Rainfall & Climate Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

<<<<<<< HEAD

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.main-title {
    font-size: 2.35rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #f8fafc, #eef6ff);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
}

.risk-box {
    border-radius: 14px;
    padding: 1rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: .5rem;
}

.small-note {
    font-size: .82rem;
    color: #64748b;
}

@media (max-width: 768px) {

    .main-title {
        font-size: 1.65rem;
    }

    .subtitle {
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    .block-container {
        padding-left: 0.6rem;
        padding-right: 0.6rem;
        padding-top: 0.8rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.15rem;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem;
    }
}

@media (max-width: 480px) {

    div[data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 0.6rem;
    }

    .main-title {
        font-size: 1.35rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.05rem;
    }

    .metric-card,
    .risk-box {
        padding: 0.7rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)

=======
# ---------- Styling (responsive) ----------
st.markdown("""
<style>
    /* ---------- Base / desktop ---------- */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1200px;
    }
    .main-title {
        font-size: clamp(1.5rem, 4vw, 2.35rem);
        font-weight: 800;
        margin-bottom: 0.2rem;
        line-height: 1.15;
    }
    .subtitle {
        color: #64748b;
        font-size: clamp(0.9rem, 2vw, 1.05rem);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #eef6ff);
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }
    .risk-box {
        border-radius: 14px;
        padding: 1rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: .5rem;
    }
    .small-note {font-size: .82rem; color: #64748b;}

    /* Make st.metric labels/values wrap and scale instead of overflowing
       on narrow columns (common once columns stack on small screens). */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
    }
    div[data-testid="stMetricLabel"] > div {
        white-space: normal;
        font-size: clamp(0.72rem, 1.6vw, 0.9rem);
    }
    div[data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 3vw, 1.6rem);
        white-space: normal;
        word-break: break-word;
    }
    div[data-testid="stMetricDelta"] {
        font-size: clamp(0.7rem, 1.6vw, 0.9rem);
    }

    /* pydeck map: keep a sane height on tall/narrow phone screens */
    iframe[title="st.pydeck_chart"] {
        min-height: 320px !important;
    }

    /* ---------- Tablet ---------- */
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1.1rem;
            padding-right: 1.1rem;
        }
    }

    /* ---------- Phone ---------- */
    @media (max-width: 600px) {
        .block-container {
            padding-top: 0.8rem;
            padding-left: 0.6rem;
            padding-right: 0.6rem;
        }
        div[data-testid="stMetric"] {
            padding: 0.5rem 0.6rem;
        }
        /* Streamlit's own columns already reflow to a single column at
           narrow widths; this just tightens the gap between the stacked
           cards so the page doesn't feel sparse on a phone. */
        div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem;
        }
        .small-note {font-size: .75rem;}
    }
</style>
""", unsafe_allow_html=True)
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03

# ============================================================
# Paths
# ============================================================
<<<<<<< HEAD

APP_DIR = Path(__file__).resolve().parent

GDRIVE_FOLDER_ID = st.secrets.get("GDRIVE_FOLDER_ID", "1H3N_Ky7kOzG6CXmchUK1ko5UVUMrVyis")
GDRIVE_CACHE_DIR = APP_DIR / "_gdrive_data"

# files that must exist for a folder to count as the project root
REQUIRED_MARKERS = [
    "prepped/district_lookup.parquet",
    "models/feature_cols.pkl",
    "disaggregation_knnr/knnr_3day.parquet",
    "disaggregation_knnr/knnr_5day.parquet",
    "disaggregation_knnr/knnr_7day.parquet",
]


def _is_project_root(d):
    return all((Path(d) / m).exists() for m in REQUIRED_MARKERS)


def _find_project_root(root):
    root = Path(root)
    if not root.exists():
        return None
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if _is_project_root(d):
            return d
    return None


@st.cache_resource(show_spinner="Downloading data from Google Drive (first run only)...")
def ensure_base_dir():
    # 1) files already next to app.py (e.g. your own machine)
    if _is_project_root(APP_DIR):
        return str(APP_DIR)

    # 2) already downloaded earlier
    found = _find_project_root(GDRIVE_CACHE_DIR)
    if found:
        return str(found)

    # 3) download the shared Drive folder
    GDRIVE_CACHE_DIR.mkdir(exist_ok=True)
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
        output=str(GDRIVE_CACHE_DIR),
        quiet=True,
        use_cookies=False,
    )

    found = _find_project_root(GDRIVE_CACHE_DIR)
    if not found:
        raise RuntimeError(
            "Download finished but the expected files were not found. "
            "The Drive folder must contain prepped/, models/ and "
            "disaggregation_knnr/, and be shared as 'Anyone with the link'."
        )
    return str(found)


try:
    BASE_DIR = Path(ensure_base_dir())
except Exception as e:
    st.error("Could not get the app data from Google Drive.")
    st.code(str(e))
    st.stop()

DATA_DIR = Path(
    st.secrets.get(
        "DATA_DIR",
        BASE_DIR / "prepped",
    )
)

MODEL_DIR = Path(
    st.secrets.get(
        "MODEL_DIR",
        BASE_DIR / "models",
    )
)

# ------------------------------------------------------------
# Compact KNNR library generated offline.
#
# project/
# ├── app.py
# ├── prepped/
# ├── models/
# └── disaggregation_knnr/
#
# Configurable through secrets:  KNNR_DIR = "/path/to/..."
# ------------------------------------------------------------

KNNR_DIR = Path(
    st.secrets.get(
        "KNNR_DIR",
        BASE_DIR / "disaggregation_knnr",
    )
)


# ============================================================
# Required model files
# ============================================================

REQUIRED_FILES = [
    DATA_DIR / "district_lookup.parquet",

    MODEL_DIR / "feature_cols.pkl",
    MODEL_DIR / "class_to_idx.pkl",
    MODEL_DIR / "rain_class_to_idx.pkl",

    MODEL_DIR / "climate_feature_cols.pkl",
    MODEL_DIR / "climate_target_cols.pkl",

    MODEL_DIR / "stage1_rain_gate.json",
    MODEL_DIR / "stage1_calibrator.pkl",
    MODEL_DIR / "stage2_calibrated.pkl",

    MODEL_DIR / "precip_bin_stats.json",
]


for target in [
    "temperature",
    "temperature_max",
    "temperature_min",
    "humidity",
    "wind_speed",
    "surface_pressure",
]:
    REQUIRED_FILES.append(
        MODEL_DIR / f"climate_{target}.json"
    )


REGIME_FEATURE_COLS = [
    "distance_to_coast_km",
    "log_distance_to_coast_km",
    "rainfall_mean",
    "rainfall_std",
    "rainfall_p95",
    "rainfall_p99",
    "rain_rate_ge20",
    "rain_rate_ge50",
    "rain_rate_ge100",
    "rain_rate_ge150",
    "rainfall_extreme_ratio",
    "tail_risk_score",
]


# ============================================================
# Main model loading
# ============================================================

@st.cache_resource(show_spinner="Loading prediction models...")
def load_assets():

    missing = [
        str(p)
        for p in REQUIRED_FILES
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing model/data files:\n\n"
            + "\n".join(missing)
        )

    district_lookup = pd.read_parquet(
        DATA_DIR / "district_lookup.parquet"
    )

    missing_regime_cols = [
        c
        for c in REGIME_FEATURE_COLS
        if c not in district_lookup.columns
    ]

    if missing_regime_cols:
        raise ValueError(
            "district_lookup.parquet is missing "
            f"regime columns: {missing_regime_cols}"
        )

    feature_cols = joblib.load(
        MODEL_DIR / "feature_cols.pkl"
    )

    class_to_idx = joblib.load(
        MODEL_DIR / "class_to_idx.pkl"
    )

    rain_class_to_idx = joblib.load(
        MODEL_DIR / "rain_class_to_idx.pkl"
    )

    climate_feature_cols = joblib.load(
        MODEL_DIR / "climate_feature_cols.pkl"
    )

    climate_target_cols = joblib.load(
        MODEL_DIR / "climate_target_cols.pkl"
    )

    with open(
        MODEL_DIR / "precip_bin_stats.json"
    ) as f:
        precip_bin_stats = json.load(f)

    stage1 = xgb.XGBClassifier()

    stage1.load_model(
        MODEL_DIR / "stage1_rain_gate.json"
    )

    stage1_calibrator = joblib.load(
        MODEL_DIR / "stage1_calibrator.pkl"
    )

    stage2_calibrated = joblib.load(
        MODEL_DIR / "stage2_calibrated.pkl"
    )

    climate_models = {}

    for target in climate_target_cols:

        model = xgb.XGBRegressor()

        model.load_model(
            MODEL_DIR / f"climate_{target}.json"
        )

=======
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "prepped"
MODEL_DIR = BASE_DIR / "models"
REGIME_PATH = BASE_DIR / "spatial_regime_features.csv"


# Expected files produced by the notebook
REQUIRED_FILES = [
    DATA_DIR / "district_lookup.parquet",
    MODEL_DIR / "feature_cols.pkl",
    MODEL_DIR / "class_to_idx.pkl",
    MODEL_DIR / "rain_class_to_idx.pkl",
    MODEL_DIR / "climate_feature_cols.pkl",
    MODEL_DIR / "climate_target_cols.pkl",
    MODEL_DIR / "stage1_rain_gate.json",
    MODEL_DIR / "stage1_calibrator.pkl",
    MODEL_DIR / "stage2_calibrated.pkl",
    MODEL_DIR / "precip_bin_stats.json",
]

for target in [
    "temperature", "temperature_max", "temperature_min",
    "humidity", "wind_speed", "surface_pressure"
]:
    REQUIRED_FILES.append(MODEL_DIR / f"climate_{target}.json")


# ============================================================
# Model loading
# ============================================================
@st.cache_resource(show_spinner="Loading prediction models...")
def load_assets():
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model/data files:\n\n" + "\n".join(missing)
        )

    district_lookup = pd.read_parquet(DATA_DIR / "district_lookup.parquet")
    regime_full = pd.read_csv(REGIME_PATH)

    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    class_to_idx = joblib.load(MODEL_DIR / "class_to_idx.pkl")
    rain_class_to_idx = joblib.load(MODEL_DIR / "rain_class_to_idx.pkl")
    climate_feature_cols = joblib.load(MODEL_DIR / "climate_feature_cols.pkl")
    climate_target_cols = joblib.load(MODEL_DIR / "climate_target_cols.pkl")

    with open(MODEL_DIR / "precip_bin_stats.json") as f:
        precip_bin_stats = json.load(f)

    stage1 = xgb.XGBClassifier()
    stage1.load_model(MODEL_DIR / "stage1_rain_gate.json")

    stage1_calibrator = joblib.load(MODEL_DIR / "stage1_calibrator.pkl")
    stage2_calibrated = joblib.load(MODEL_DIR / "stage2_calibrated.pkl")

    climate_models = {}
    for target in climate_target_cols:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_DIR / f"climate_{target}.json")
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
        climate_models[target] = model

    return {
        "district_lookup": district_lookup,
<<<<<<< HEAD
=======
        "regime_full": regime_full,
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
        "feature_cols": feature_cols,
        "class_to_idx": class_to_idx,
        "rain_class_to_idx": rain_class_to_idx,
        "climate_feature_cols": climate_feature_cols,
        "climate_target_cols": climate_target_cols,
        "precip_bin_stats": precip_bin_stats,
        "stage1": stage1,
        "stage1_calibrator": stage1_calibrator,
        "stage2_calibrated": stage2_calibrated,
        "climate_models": climate_models,
    }


<<<<<<< HEAD
# ============================================================
# KNNR LIBRARY DISCOVERY
# ============================================================

@st.cache_resource(show_spinner="Loading KNNR disaggregation library...")
def load_knnr_library():

    if not KNNR_DIR.exists():
        return None

    parquet_files = sorted(
        KNNR_DIR.rglob("*.parquet")
    )

    npy_files = sorted(
        KNNR_DIR.rglob("*.npy")
    )

    json_files = sorted(
        KNNR_DIR.rglob("*.json")
    )

    if (
        len(parquet_files) == 0
        and len(npy_files) == 0
    ):
        return None

    return {
        "directory": KNNR_DIR,
        "parquet_files": parquet_files,
        "npy_files": npy_files,
        "json_files": json_files,
    }


# ============================================================
# Load assets
# ============================================================

try:

    assets = load_assets()
    knnr_library = load_knnr_library()

except Exception as e:

    st.error(
        "The app could not load the trained model assets."
    )

    st.code(str(e))

=======
try:
    assets = load_assets()
except Exception as e:
    st.error("The app could not load the trained model assets.")
    st.code(str(e))
    st.info(
        "Put the notebook-generated `prepped/`, `models/`, and "
        "`spatial_regime_features.csv` in the app directory, or configure "
        "DATA_DIR, MODEL_DIR and REGIME_PATH in Streamlit secrets."
    )
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
    st.stop()


district_lookup = assets["district_lookup"]
<<<<<<< HEAD

FEATURE_COLS = assets["feature_cols"]
CLASS_TO_IDX = assets["class_to_idx"]
RAIN_CLASS_TO_IDX = assets["rain_class_to_idx"]

CLIMATE_FEATURE_COLS = assets["climate_feature_cols"]
CLIMATE_TARGET_COLS = assets["climate_target_cols"]

PRECIP_BIN_STATS = assets["precip_bin_stats"]

stage1 = assets["stage1"]
stage1_calibrator = assets["stage1_calibrator"]
stage2_calibrated = assets["stage2_calibrated"]

climate_models = assets["climate_models"]


IDX_TO_CLASS = {
    i: c
    for c, i in CLASS_TO_IDX.items()
}


# ============================================================
# Location lookup
# ============================================================

@st.cache_data
def location_table():

    lut = district_lookup.copy()

    lut["state_u"] = (
        lut["state_name"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    lut["district_u"] = (
        lut["district_name"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

=======
regime_full = assets["regime_full"]
FEATURE_COLS = assets["feature_cols"]
CLASS_TO_IDX = assets["class_to_idx"]
RAIN_CLASS_TO_IDX = assets["rain_class_to_idx"]
CLIMATE_FEATURE_COLS = assets["climate_feature_cols"]
CLIMATE_TARGET_COLS = assets["climate_target_cols"]
PRECIP_BIN_STATS = assets["precip_bin_stats"]
stage1 = assets["stage1"]
stage1_calibrator = assets["stage1_calibrator"]
stage2_calibrated = assets["stage2_calibrated"]
climate_models = assets["climate_models"]

IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


# ============================================================
# Inference functions — follows the notebook's Phase 4 logic
# ============================================================
def infer_is_monsoon(month: int) -> int:
    # Same assumption used in the notebook: June–September.
    return int(month in (6, 7, 8, 9))


@st.cache_data
def location_table():
    lut = district_lookup.copy()
    lut["state_u"] = lut["state_name"].astype(str).str.strip().str.upper()
    lut["district_u"] = lut["district_name"].astype(str).str.strip().str.upper()
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
    return lut


LUT = location_table()


<<<<<<< HEAD
def resolve_location(
    state_name: str,
    district_name: str,
) -> dict:

=======
def resolve_location(state_name: str, district_name: str) -> dict:
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
    state_u = state_name.strip().upper()
    district_u = district_name.strip().upper()

    match = LUT[
<<<<<<< HEAD
        (LUT["state_u"] == state_u)
        &
=======
        (LUT["state_u"] == state_u) &
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
        (LUT["district_u"] == district_u)
    ]

    if len(match) == 0:
<<<<<<< HEAD

        state_candidates = (
            LUT[
                LUT["state_u"] == state_u
            ]["district_u"]
            .unique()
        )

        suggestions = difflib.get_close_matches(
            district_u,
            state_candidates.tolist(),
            n=3,
            cutoff=0.5,
        )

        raise ValueError(
            f"No match for state='{state_name}', "
            f"district='{district_name}'. "
=======
        state_candidates = LUT[LUT["state_u"] == state_u]["district_u"].unique()
        suggestions = difflib.get_close_matches(
            district_u, state_candidates.tolist(), n=3, cutoff=0.5
        )
        raise ValueError(
            f"No match for state='{state_name}', district='{district_name}'. "
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
            f"Closest district matches: {suggestions}"
        )

    row = match.iloc[0]
<<<<<<< HEAD

    result = {
        "state_name": row["state_name"],
        "district_name": row["district_name"],
        "district_code": row["district_code"],
=======
    return {
        "state_name": row["state_name"],
        "district_name": row["district_name"],
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elevation": float(row["elevation"]),
    }

<<<<<<< HEAD
    for col in REGIME_FEATURE_COLS:

        result[col] = float(row[col])

    return result


def find_nearest_regime_row(
    latitude: float,
    longitude: float,
    elevation: float,
    distance_to_coast_km: float,
) -> dict:

    """
    For a location NOT in the district lookup: find the closest
    existing district in normalized [latitude, longitude, elevation,
    distance_to_coast_km] space, and borrow its 8 rainfall-DERIVED
    regime stats (rainfall_mean, rainfall_std, rainfall_p95/p99,
    rain_rate_ge20/50/100/150, rainfall_extreme_ratio, tail_risk_score).

    These 8 cannot be computed without a multi-year historical daily
    rainfall record for the new location -- there's no live API that
    returns a rainfall percentile or a tail-risk score -- so the
    nearest climatologically-similar KNOWN district's values are used
    as a stand-in.

    distance_to_coast_km and log_distance_to_coast_km are NOT
    borrowed. They come directly from the caller's own input (the user
    looks distance_to_coast_km up themselves; log_distance_to_coast_km
    is derived as log1p(distance_to_coast_km), the exact transform
    confirmed against spatial_regime_features.csv). Borrowing these
    two from a "nearby" district would defeat the point of including
    them in the match at all.

    Matching uses ALL FOUR of [latitude, longitude, elevation,
    distance_to_coast_km], z-score normalized against the known
    districts -- specifically so two places at a similar
    latitude/longitude but very different coastal exposure (e.g. a
    Konkan-coast district vs. an interior Deccan-plateau district at a
    similar latitude) are NOT matched to each other just because
    they're geographically close in lat/lon alone.
    """

    match_cols = [
        "latitude",
        "longitude",
        "elevation",
        "distance_to_coast_km",
    ]

    ref = LUT[match_cols].astype(float)

    means = ref.mean()
    stds = ref.std().replace(0, 1.0)

    query = pd.Series({
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "distance_to_coast_km": distance_to_coast_km,
    })

    ref_norm = (ref - means) / stds
    query_norm = (query - means) / stds

    dist = np.sqrt(
        ((ref_norm - query_norm) ** 2).sum(axis=1)
    )

    nearest_idx = dist.idxmin()
    nearest_row = LUT.loc[nearest_idx]
    nearest_distance = float(dist.loc[nearest_idx])

    result = {
        "state_name": "Custom location",
        "district_name": "Unlisted district (manual input)",
        "district_code": nearest_row["district_code"],
        "latitude": float(latitude),
        "longitude": float(longitude),
        "elevation": float(elevation),
        "distance_to_coast_km": float(distance_to_coast_km),
        "log_distance_to_coast_km": float(np.log1p(distance_to_coast_km)),
        "is_unlisted_location": True,
        "_matched_from": {
            "district_name": nearest_row["district_name"],
            "state_name": nearest_row["state_name"],
            "similarity_distance": nearest_distance,
        },
    }

    borrowed_cols = [
        c for c in REGIME_FEATURE_COLS
        if c not in ("distance_to_coast_km", "log_distance_to_coast_km")
    ]

    for col in borrowed_cols:
        result[col] = float(nearest_row[col])

    return result


# ============================================================
# Date feature construction
# ============================================================

def infer_is_monsoon(month: int) -> int:

    return int(
        month in (6, 7, 8, 9)
    )


def build_feature_row(
    date_str: str,
    loc: dict,
) -> pd.DataFrame:

    dt = datetime.strptime(
        date_str,
        "%Y-%m-%d",
    )

    day_of_year = (
        dt.timetuple().tm_yday
    )

    row = {

        "latitude": loc["latitude"],

        "longitude": loc["longitude"],

        "elevation": loc["elevation"],

        "doy_sin": np.sin(
            2
            * np.pi
            * day_of_year
            / 365.25
        ),

        "doy_cos": np.cos(
            2
            * np.pi
            * day_of_year
            / 365.25
        ),

        "is_monsoon":
            infer_is_monsoon(dt.month),
    }

    for col in REGIME_FEATURE_COLS:

        row[col] = loc[col]

    return pd.DataFrame([row])


# ============================================================
# Weather prediction
# ============================================================

def predict(
    date_str: str,
    loc: dict,
) -> dict:

    X = build_feature_row(
        date_str,
        loc,
    )

    # --------------------------------------------------------
    # Rainfall cascade
    # --------------------------------------------------------

    X_rain = X[FEATURE_COLS]

    raw_p_rain = float(
        stage1.predict_proba(
            X_rain
        )[:, 1][0]
    )

    p_rain = float(
        stage1_calibrator.predict(
            [raw_p_rain]
        )[0]
    )

    p_severity_given_rain = (
        stage2_calibrated
        .predict_proba(X_rain)[0]
    )

    severity_probs = {
        "no_rain":
            1.0 - p_rain
    }

    for cls, idx in RAIN_CLASS_TO_IDX.items():

        severity_probs[cls] = (
            p_rain
            * float(
                p_severity_given_rain[idx]
            )
        )

    expected_mean = sum(
        severity_probs[cls]
        * PRECIP_BIN_STATS[cls]["mean"]
        for cls in severity_probs
    )

    expected_median = sum(
        severity_probs[cls]
        * PRECIP_BIN_STATS[cls]["median"]
        for cls in severity_probs
    )

    most_likely_bin = max(
        severity_probs,
        key=severity_probs.get,
    )

    # --------------------------------------------------------
    # Climate
    # --------------------------------------------------------

    X_climate = X[
        CLIMATE_FEATURE_COLS
    ]

    climate_preds = {

        target:
            float(
                climate_models[target]
                .predict(X_climate)[0]
            )

        for target
        in CLIMATE_TARGET_COLS
    }

    return {

        "date": date_str,

        "location": {

            "state_name":
                loc["state_name"],

            "district_name":
                loc["district_name"],

            "district_code":
                loc["district_code"],

            "latitude":
                loc["latitude"],

            "longitude":
                loc["longitude"],

            "elevation":
                loc["elevation"],
        },

        "rain_probability":
            p_rain,

        "rain_severity_probabilities":
            severity_probs,

        "expected_precipitation_mm": {

            "weighted_mean":
                expected_mean,

            "weighted_median":
                expected_median,

            "most_likely_scenario": {

                "bin":
                    most_likely_bin,

                "probability":
                    severity_probs[
                        most_likely_bin
                    ],

                "p25":
                    PRECIP_BIN_STATS[
                        most_likely_bin
                    ]["p25"],

                "p75":
                    PRECIP_BIN_STATS[
                        most_likely_bin
                    ]["p75"],
            },
        },

        "climate_predictions":
            climate_preds,
=======

def build_feature_row(date_str: str, lat: float, lon: float, elevation: float):
    # Match regime row by coordinates, with a small numeric fallback.
    regime_match = regime_full[
        (np.isclose(regime_full["latitude"], lat, atol=1e-7)) &
        (np.isclose(regime_full["longitude"], lon, atol=1e-7))
    ]

    if len(regime_match) == 0:
        raise ValueError(
            f"No regime data found for lat={lat}, lon={lon}. "
            "This district may not be one of the modeled locations."
        )

    regime_row = regime_match.iloc[0]
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = dt.timetuple().tm_yday

    row = {
        "latitude": lat,
        "longitude": lon,
        "elevation": elevation,
        "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "is_monsoon": infer_is_monsoon(dt.month),
        "distance_to_coast_km": regime_row["distance_to_coast_km"],
        "log_distance_to_coast_km": regime_row["log_distance_to_coast_km"],
        "rainfall_mean": regime_row["rainfall_mean"],
        "rainfall_std": regime_row["rainfall_std"],
        "rainfall_p95": regime_row["rainfall_p95"],
        "rainfall_p99": regime_row["rainfall_p99"],
        "rain_rate_ge20": regime_row["rain_rate_ge20"],
        "rain_rate_ge50": regime_row["rain_rate_ge50"],
        "rain_rate_ge100": regime_row["rain_rate_ge100"],
        "rain_rate_ge150": regime_row["rain_rate_ge150"],
        "rainfall_extreme_ratio": regime_row["rainfall_extreme_ratio"],
        "tail_risk_score": regime_row["tail_risk_score"],
    }

    return pd.DataFrame([row])


def predict(date_str: str, state_name: str, district_name: str) -> dict:
    loc = resolve_location(state_name, district_name)
    X = build_feature_row(
        date_str,
        loc["latitude"],
        loc["longitude"],
        loc["elevation"]
    )

    # Rainfall severity: calibrated Stage 1 + Stage 2 cascade.
    X_rain = X[FEATURE_COLS]
    raw_p_rain = float(stage1.predict_proba(X_rain)[:, 1][0])
    p_rain = float(stage1_calibrator.predict([raw_p_rain])[0])
    p_severity_given_rain = stage2_calibrated.predict_proba(X_rain)[0]

    severity_probs = {"no_rain": 1.0 - p_rain}
    for cls, idx in RAIN_CLASS_TO_IDX.items():
        severity_probs[cls] = p_rain * float(p_severity_given_rain[idx])

    # Expected rainfall, exactly following the notebook's bin-stat approach.
    expected_mean = sum(
        severity_probs[cls] * PRECIP_BIN_STATS[cls]["mean"]
        for cls in severity_probs
    )
    expected_median = sum(
        severity_probs[cls] * PRECIP_BIN_STATS[cls]["median"]
        for cls in severity_probs
    )

    most_likely_bin = max(severity_probs, key=severity_probs.get)

    # Climate regression models.
    X_climate = X[CLIMATE_FEATURE_COLS]
    climate_preds = {
        target: float(climate_models[target].predict(X_climate)[0])
        for target in CLIMATE_TARGET_COLS
    }

    return {
        "date": date_str,
        "location": loc,
        "rain_probability": p_rain,
        "rain_severity_probabilities": severity_probs,
        "expected_precipitation_mm": {
            "weighted_mean": expected_mean,
            "weighted_median": expected_median,
            "most_likely_scenario": {
                "bin": most_likely_bin,
                "probability": severity_probs[most_likely_bin],
                "p25": PRECIP_BIN_STATS[most_likely_bin]["p25"],
                "p75": PRECIP_BIN_STATS[most_likely_bin]["p75"],
            },
        },
        "climate_predictions": climate_preds,
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
    }


# ============================================================
<<<<<<< HEAD
# KNNR utilities
# ============================================================

def circular_day_distance(a, b):

    diff = abs(
        float(a) - float(b)
    )

    return min(
        diff,
        365.25 - diff,
    )


# ============================================================
# KNNR loading + matching — REWRITTEN for your actual library layout
#
# Your `disaggregation_knnr/` folder is NOT one shared file scored to
# find "the best" — it's three genuinely separate, purpose-built files:
#   knnr_3day.parquet, knnr_5day.parquet, knnr_7day.parquet
# each with its own pre-computed `pattern` label and window-specific
# `rain_-k` / `rain_+k` antecedent columns, plus hourly columns named
# `y00`-`y23` (not `h00`-`h23`), and a `target_date` column (not `date`).
#
# This replaces: get_hour_columns, detect_pattern_column,
# detect_daily_window_columns, prepared_knnr_dataframe,
# knnr_disaggregate, generate_knnr_ensemble.
# Keep: circular_day_distance, ROTATION_24 / apply_inverse_rotation
# (unchanged, still correct).
# ============================================================

KNNR_WINDOW_FILES = {
    3: "knnr_3day.parquet",
    5: "knnr_5day.parquet",
    7: "knnr_7day.parquet",
}


def get_hour_columns(df):
    """Your library uses y00-y23, not h00-h23 — accept both prefixes
    so this still works if you ever rebuild with different naming."""
    hour_columns = []
    for c in df.columns:
        name = str(c).lower()
        if name and name[0] in ("h", "y"):
            suffix = name[1:]
            if suffix.isdigit():
                h = int(suffix)
                if 0 <= h <= 23:
                    hour_columns.append((h, c))
    hour_columns.sort(key=lambda x: x[0])
    if len(hour_columns) == 24:
        return [c for _, c in hour_columns]
    return []


def _date_column(df):
    """Your window files use target_date; daily_totals.parquet uses date.
    Return whichever is present, or None."""
    if "date" in df.columns:
        return "date"
    if "target_date" in df.columns:
        return "target_date"
    return None

def _build_gram_schmidt_rotation(dimension: int):

    """
    Orthogonal rotation whose FINAL basis vector is

        [1, 1, ..., 1] / sqrt(dimension)

    so that

        y[-1] = sum(x) / sqrt(dimension)

    i.e. the last rotated coordinate carries the daily
    total, and the remaining coordinates carry the
    within-day shape.
    """

    if dimension < 2:

        raise ValueError(
            "dimension must be >= 2"
        )

    rainfall_direction = np.ones(
        dimension,
        dtype=float,
    )

    rainfall_direction /= np.linalg.norm(
        rainfall_direction
    )

    basis = []

    for i in range(dimension - 1):

        v = np.zeros(
            dimension,
            dtype=float,
        )

        v[i] = 1.0

        for q in basis:

            v -= np.dot(v, q) * q

        v -= (
            np.dot(v, rainfall_direction)
            * rainfall_direction
        )

        norm = np.linalg.norm(v)

        if norm < 1e-12:

            found = False

            for j in range(dimension):

                candidate = np.zeros(
                    dimension,
                    dtype=float,
                )

                candidate[j] = 1.0

                for q in basis:

                    candidate -= (
                        np.dot(candidate, q)
                        * q
                    )

                candidate -= (
                    np.dot(
                        candidate,
                        rainfall_direction,
                    )
                    * rainfall_direction
                )

                candidate_norm = np.linalg.norm(
                    candidate
                )

                if candidate_norm > 1e-12:

                    v = candidate
                    norm = candidate_norm
                    found = True

                    break

            if not found:

                raise RuntimeError(
                    "Could not construct "
                    "orthogonal basis."
                )

        v /= norm

        basis.append(v)

    basis.append(rainfall_direction)

    R = np.vstack(basis)

    # Numerical cleanup on the shape subspace.
    Q, _ = np.linalg.qr(R[:-1].T)

    R_clean = np.vstack(
        [
            Q[:, :dimension - 1].T,
            rainfall_direction,
        ]
    )

    if not np.allclose(
        R_clean @ R_clean.T,
        np.eye(dimension),
        atol=1e-8,
    ):

        raise RuntimeError(
            "Rotation matrix is not orthogonal."
        )

    return R_clean


ROTATION_24 = _build_gram_schmidt_rotation(24)


def apply_inverse_rotation(
    stored_y_fragment,
    target_total,
):

    """
    Rescale an analogue's ALREADY Gram-Schmidt-rotated hourly
    fragment (as stored in knnr_Xday.parquet's y00-y23 columns
    by build_knnr_library.py's add_rotated_coordinates()) to
    target_total, in rotated space:

        y[-1] = target_total/sqrt(24)  (swap the total, keep shape)
        x' = R.T @ y

    IMPORTANT: the builder script's prepare_deployment_columns()
    drops the raw h00-h23 hourly columns entirely and stores only
    the rotated y00-y23 coordinates. The input here is therefore
    ALREADY y, not x -- do NOT apply the forward rotation
    (y = R @ x) to it. Doing so would rotate already-rotated data
    a second time and produce a meaningless hourly shape. This was
    the original no-op bug's mirror-image: rotating twice instead
    of not rotating at all.

    For the same reason, do not clip the incoming components to
    non-negative -- unlike raw hourly rainfall, the first 23
    rotated "shape" coordinates are legitimately signed. Only the
    final RECONSTRUCTED real-hourly values (after R.T @ y) must be
    non-negative.

    Replacing the sum-coordinate adds a UNIFORM
    (target_total - sum(x)) / 24 to every hour, so this is
    an ADDITIVE rescale preserving absolute hour-to-hour
    differences, unlike multiplicative scaling which
    preserves ratios. Negatives are clipped and the result
    renormalized so the daily total is conserved exactly.
    """

    y = np.asarray(
        stored_y_fragment,
        dtype=float,
    )

    if len(y) != 24:

        raise ValueError(
            "Rotated hourly fragment must contain "
            "exactly 24 values."
        )

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    y[-1] = (
        target_total
        / np.sqrt(24.0)
    )

    reconstructed = ROTATION_24.T @ y

    # Clipping breaks exact conservation, so renormalize.
    reconstructed[reconstructed < 0] = 0.0

    total = reconstructed.sum()

    if total <= 0:

        reconstructed[:] = (
            target_total
            / 24.0
        )

    else:

        reconstructed *= (
            target_total
            / total
        )

    return reconstructed

@st.cache_data(show_spinner=False)
def load_knnr_window_file(directory_string: str, window: int):
    """Load and normalize ONE window's dedicated file — not a shared
    'best file' across all windows, which was the structural bug."""
    filename = KNNR_WINDOW_FILES.get(window)
    if filename is None:
        return None

    path = Path(directory_string) / filename
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    date_col = _date_column(df)
    if date_col is not None:
        df["_date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")

    if "district_code" in df.columns:
        df["_district_code_str"] = (
            df["district_code"].astype(str).str.replace(r"\.0$", "", regex=True)
        )

    return df


def knnr_disaggregate(
    library_df,
    district_code,
    target_date,
    daily_total_mm,
    pattern_window,
    n_draws=50,
    top_k=10,
    random_seed=42,
):
    """Returns (matrix, analog_dates, match_info) or None.

    Matching strategy for this library:
      Tier 1 (preferred): find the target (district, date) row IN THIS
        WINDOW'S OWN FILE, read its pre-computed `pattern` label, and
        restrict candidates to rows sharing that label — this is real
        antecedent-pattern conditioning, using the library's own
        classification rather than a re-derived one.
      Fallback: target date not in the library (true for any future
        date) -> seasonal (day-of-year) matching only, reported
        honestly rather than silently.
    """
    match_info = {
        "pattern_column": False,
        "daily_shape": False,
        "seasonal_only": True,
        "target_in_library": False,
    }

    if library_df is None:
        return None
    if daily_total_mm <= 0:
        return np.zeros((n_draws, 24)), [], match_info

    df = library_df
    hour_cols = get_hour_columns(df)
    if len(hour_cols) != 24:
        return None

    if "_district_code_str" in df.columns:
        target_code = str(district_code)
        if target_code.endswith(".0"):
            target_code = target_code[:-2]
        work = df[df["_district_code_str"] == target_code].copy()
    else:
        work = df.copy()

    if len(work) == 0:
        return None

    target_dt = pd.to_datetime(target_date)
    target_doy = target_dt.dayofyear

    # --- Find the target row using WHICHEVER date column this file has ---
    target_row = None
    if "_date_parsed" in work.columns:
        target_rows = work[work["_date_parsed"] == target_dt]
        if len(target_rows) > 0:
            target_row = target_rows.iloc[0]
            match_info["target_in_library"] = True

    # --- Seasonal narrowing (always applied) ---
    if "day_of_year" in work.columns:
        work["_doy_distance"] = work["day_of_year"].apply(
            lambda x: circular_day_distance(x, target_doy)
        )
        seasonal = work[work["_doy_distance"] <= 45]
        if len(seasonal) >= top_k:
            work = seasonal.copy()
        work["_distance"] = work["_doy_distance"].astype(float)
    else:
        work["_distance"] = 0.0

    # --- Tier 1: this library's own pre-computed `pattern` column ---
    if "pattern" in work.columns and target_row is not None:
        target_pattern = target_row.get("pattern")
        if target_pattern is not None and pd.notna(target_pattern):
            same_pattern = work[work["pattern"] == target_pattern]
            # Exclude the target row itself from its own candidate pool,
            # if it's in there (avoid the trivial "matched to itself").
            if "_date_parsed" in same_pattern.columns:
                same_pattern = same_pattern[same_pattern["_date_parsed"] != target_dt]
            if len(same_pattern) >= min(top_k, 3):
                work = same_pattern.copy()
                match_info["pattern_column"] = True
                match_info["seasonal_only"] = False

    # --- Always exclude the target day's own row from the candidate pool.
    #     Tier 1 already does this for the pattern-matched branch above,
    #     but the seasonal-only fallback did not, which meant a target
    #     date that IS in the library (used during backtesting/validation,
    #     or if a "future" prediction date happens to already have real
    #     hourly data) could be matched to itself -- true leave-one-out
    #     requires this exclusion unconditionally. ---
    if "_date_parsed" in work.columns:
        work = work[work["_date_parsed"] != target_dt]

    if len(work) == 0:
        return None

    # --- Select K nearest by remaining distance (mostly seasonal at this point) ---
    work = work.sort_values("_distance")
    nearest = work.head(min(top_k, len(work))).copy()
    if len(nearest) == 0:
        return None

    ranks = np.arange(1, len(nearest) + 1, dtype=float)
    weights = 1.0 / ranks
    weights /= weights.sum()

    rng = np.random.RandomState(random_seed)
    matrix = np.zeros((n_draws, 24), dtype=float)
    analog_dates = []

    fragments = nearest[hour_cols].to_numpy(dtype=float)
    date_col_for_output = "_date_parsed" if "_date_parsed" in nearest.columns else None
    date_values = nearest[date_col_for_output].to_numpy() if date_col_for_output else None

    for draw in range(n_draws):
        idx = rng.choice(len(nearest), p=weights)
        y_fragment = fragments[idx].copy()
        y_fragment[~np.isfinite(y_fragment)] = 0.0

        matrix[draw, :] = apply_inverse_rotation(y_fragment, daily_total_mm)

        analog_dates.append(date_values[idx] if date_values is not None else None)

    return matrix, analog_dates, match_info


def generate_knnr_ensemble(
    district_code,
    target_date,
    daily_total_mm,
    n_draws=50,
    top_k=10,
):
    if knnr_library is None:
        return None

    results = {}
    for window in [3, 5, 7]:
        # Load THIS window's own dedicated file, not a shared "best" file.
        df = load_knnr_window_file(str(KNNR_DIR), window)
        if df is None:
            continue
        try:
            output = knnr_disaggregate(
                df, district_code, target_date, daily_total_mm,
                pattern_window=window, n_draws=n_draws, top_k=top_k,
                random_seed=1000 + window,
            )
            if output is not None:
                matrix, analogs, match_info = output
                results[window] = {"matrix": matrix, "analogs": analogs, "match_info": match_info}
        except Exception:
            continue

    if not results:
        return None

    matrices, analogs = [], []
    for window in [3, 5, 7]:
        if window not in results:
            continue
        matrices.append(results[window]["matrix"])
        analogs.extend(results[window]["analogs"])

    if not matrices:
        return None

    combined = np.vstack(matrices)
    for i in range(len(combined)):
        combined[i] = np.maximum(combined[i], 0.0)
        total = combined[i].sum()
        combined[i] = combined[i] * (daily_total_mm / total) if total > 0 else np.full(24, daily_total_mm / 24.0)

    any_pattern = any(not results[w]["match_info"]["seasonal_only"] for w in results)
    target_in_library = any(results[w]["match_info"]["target_in_library"] for w in results)

    return {
        "matrix": combined,
        "analogs": analogs,
        "windows": list(results.keys()),
        "match_info": {w: results[w]["match_info"] for w in results},
        "any_pattern_matching": any_pattern,
        "target_in_library": target_in_library,
    }


# ============================================================
# KNNR VALIDATION (paper-style leave-one-out backtest)
#
# Reproduces the evaluation approach in Park & Chung (2020),
# "A Nonparametric Stochastic Approach for Disaggregation of
# Daily to Hourly Rainfall Using 3-day Rainfall Patterns",
# Water 12(8):2306 -- and the notebook's own Phase E, Part 3
# validation cell.
#
# For a sample of historical wet days with REAL hourly data,
# disaggregate using only antecedent conditions (the target
# day's own row is excluded from its own candidate pool --
# true leave-one-out, enforced inside knnr_disaggregate), then
# compare the disaggregated hourly curve against what actually
# happened.
# ============================================================

@st.cache_data(show_spinner=False)
def _validation_candidate_pool(directory_string: str, window: int, district_code=None):
    """All library rows for `window` that have a real, complete 24-hour
    fragment and a positive daily total -- i.e. every row we can use as
    a held-out 'ground truth' validation day."""

    df = load_knnr_window_file(directory_string, window)

    if df is None:
        return None

    hour_cols = get_hour_columns(df)

    if len(hour_cols) != 24:
        return None

    work = df.copy()

    if district_code is not None and "_district_code_str" in work.columns:
        target_code = str(district_code)
        if target_code.endswith(".0"):
            target_code = target_code[:-2]
        work = work[work["_district_code_str"] == target_code]

    if "_date_parsed" not in work.columns:
        return None

    work = work.dropna(subset=["_date_parsed"])

    totals = work[hour_cols].sum(axis=1)
    work = work[totals > 0].copy()
    work["_daily_total"] = totals[totals > 0]

    return work.reset_index(drop=True)


def run_knnr_validation(
    window: int,
    district_code,
    n_validation_days: int,
    n_draws: int,
    top_k: int,
    random_seed: int = 42,
):
    """Leave-one-out backtest for ONE window (3, 5, or 7 day). Returns
    (results_df, per_day_detail) or (None, None) if no library / pool."""

    if knnr_library is None:
        return None, None

    pool = _validation_candidate_pool(str(KNNR_DIR), window, district_code)

    if pool is None or len(pool) == 0:
        return None, None

    hour_cols = get_hour_columns(pool)

    n = min(n_validation_days, len(pool))
    sample = pool.sample(n, random_state=random_seed)

    library_df = load_knnr_window_file(str(KNNR_DIR), window)

    rows = []
    detail = []

    for _, target_row in sample.iterrows():

        target_date = target_row["_date_parsed"].strftime("%Y-%m-%d")
        daily_total = float(target_row["_daily_total"])
        actual_hourly = target_row[hour_cols].to_numpy(dtype=float)

        out = knnr_disaggregate(
            library_df,
            district_code=target_row.get("district_code", district_code),
            target_date=target_date,
            daily_total_mm=daily_total,
            pattern_window=window,
            n_draws=n_draws,
            top_k=top_k,
            random_seed=random_seed,
        )

        if out is None:
            continue

        matrix, analogs, match_info = out
        predicted_hourly = matrix.mean(axis=0)

        # Naive baseline: spread the (true) daily total evenly across 24h.
        naive_hourly = np.full(24, daily_total / 24.0)

        def _score(pred):
            mae = float(np.mean(np.abs(actual_hourly - pred)))
            actual_peak = int(np.argmax(actual_hourly))
            pred_peak = int(np.argmax(pred))
            peak_hit = abs(actual_peak - pred_peak) <= 1
            ss_res = float(np.sum((actual_hourly - pred) ** 2))
            ss_tot = float(np.sum((actual_hourly - actual_hourly.mean()) ** 2))
            nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            return mae, peak_hit, nse

        mae, peak_hit, nse = _score(predicted_hourly)
        naive_mae, naive_peak_hit, naive_nse = _score(naive_hourly)

        rows.append({
            "date": target_date,
            "district_code": target_row.get("district_code", district_code),
            "daily_total_mm": daily_total,
            "mae": mae,
            "peak_hour_hit": peak_hit,
            "nse": nse,
            "naive_mae": naive_mae,
            "naive_peak_hour_hit": naive_peak_hit,
            "naive_nse": naive_nse,
            "match_type": (
                "pattern" if not match_info.get("seasonal_only", True) else "seasonal-only"
            ),
        })

        detail.append({
            "date": target_date,
            "district_code": target_row.get("district_code", district_code),
            "daily_total_mm": daily_total,
            "actual_hourly": actual_hourly,
            "predicted_hourly": predicted_hourly,
            "predicted_p10": np.percentile(matrix, 10, axis=0),
            "predicted_p90": np.percentile(matrix, 90, axis=0),
            "naive_hourly": naive_hourly,
        })

    if not rows:
        return None, None

    return pd.DataFrame(rows), detail


def render_knnr_validation_tab():

    st.markdown(
        "## 📊 KNNR validation — paper-style backtest"
    )

    st.caption(
        "Leave-one-out evaluation methodology follows Park & Chung "
        "(2020), *A Nonparametric Stochastic Approach for "
        "Disaggregation of Daily to Hourly Rainfall Using 3-day "
        "Rainfall Patterns*, Water 12(8):2306. For each held-out "
        "historical wet day with real hourly data, the day's own "
        "record is excluded from its own analogue pool, then the "
        "disaggregated hourly curve is compared against what "
        "actually happened."
    )

    if knnr_library is None:
        st.warning(
            "KNNR library was not found. Place the disaggregation_knnr "
            "folder beside this Streamlit script."
        )
        return

    v1, v2, v3 = st.columns(3)

    with v1:
        window = st.selectbox(
            "Antecedent window",
            [3, 5, 7],
            format_func=lambda w: f"{w}-day pattern",
            key="val_window",
        )

    with v2:
        district_scope = st.selectbox(
            "District scope",
            ["Selected district (sidebar)", "All districts in library"],
            key="val_scope",
        )

    with v3:
        n_days = st.slider(
            "Validation days to sample",
            min_value=20,
            max_value=1000,
            value=200,
            step=20,
            key="val_n_days",
        )

    v4, v5 = st.columns(2)

    with v4:
        val_draws = st.slider(
            "Ensemble simulations per day",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="val_draws",
        )

    with v5:
        val_top_k = st.slider(
            "Nearest neighbours (K)",
            min_value=3,
            max_value=50,
            value=10,
            step=1,
            key="val_top_k",
        )

    district_code_filter = (
        selected_location["district_code"]
        if district_scope.startswith("Selected")
        else None
    )

    run_val = st.button(
        "📊 Run validation",
        type="primary",
        use_container_width=True,
        key="run_validation",
    )

    state_key = "_knnr_validation_results"

    # IMPORTANT: st.button() only returns True on the single rerun
    # immediately after it's clicked -- Streamlit reruns this whole
    # function on EVERY widget interaction, including just browsing
    # the "Validation day" selectbox further down. Without storing
    # the computed results somewhere that survives a rerun, that
    # selectbox interaction would hit the "not run_val" branch below
    # and wipe out (or never redisplay) anything except whatever
    # happened to render on the run that just clicked the button --
    # which is exactly why only one day (e.g. 2004) ever seemed to
    # plot, regardless of which one you picked afterward.
    if run_val:
        with st.spinner("Running leave-one-out backtest..."):
            results_df, detail = run_knnr_validation(
                window=window,
                district_code=district_code_filter,
                n_validation_days=n_days,
                n_draws=val_draws,
                top_k=val_top_k,
            )

        if results_df is None:
            st.session_state.pop(state_key, None)
            st.warning(
                "No valid historical wet days with complete hourly data "
                "were found for this window / district scope."
            )
            return

        st.session_state[state_key] = {
            "results_df": results_df,
            "detail": detail,
            "window": window,
        }

    if state_key not in st.session_state:
        st.info(
            "Choose a window and sample size, then click "
            "**Run validation**. Larger samples take longer."
        )
        return

    results_df = st.session_state[state_key]["results_df"]
    detail = st.session_state[state_key]["detail"]
    window = st.session_state[state_key]["window"]

    # --------------------------------------------------------
    # Summary: KNNR vs naive uniform-spread baseline
    # --------------------------------------------------------

    st.markdown("### Summary — KNNR vs. naive uniform-spread baseline")

    summary = pd.DataFrame({
        "Metric": [
            "Mean hourly MAE (mm)",
            "Median hourly MAE (mm)",
            "Peak-hour match rate (±1h)",
            "Mean NSE",
            "Median NSE",
        ],
        "KNNR (this app)": [
            f'{results_df["mae"].mean():.3f}',
            f'{results_df["mae"].median():.3f}',
            f'{results_df["peak_hour_hit"].mean() * 100:.1f}%',
            f'{results_df["nse"].mean():.3f}',
            f'{results_df["nse"].median():.3f}',
        ],
        "Naive (uniform 24h spread)": [
            f'{results_df["naive_mae"].mean():.3f}',
            f'{results_df["naive_mae"].median():.3f}',
            f'{results_df["naive_peak_hour_hit"].mean() * 100:.1f}%',
            f'{results_df["naive_nse"].mean():.3f}',
            f'{results_df["naive_nse"].median():.3f}',
        ],
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    beats_naive = results_df["mae"].mean() < results_df["naive_mae"].mean()

    if beats_naive:
        st.success(
            "KNNR disaggregation beats the naive baseline on mean MAE — "
            "the analogue matching is adding real value over assuming "
            "rain falls evenly through the day."
        )
    else:
        st.error(
            "KNNR disaggregation does **not** beat the naive baseline "
            "on mean MAE for this sample — the analogue matching isn't "
            "currently adding value here. Consider a larger K, a "
            "different window, or checking the candidate pool size."
        )

    match_counts = results_df["match_type"].value_counts()

    st.caption(
        "Match type across sampled days: "
        + ", ".join(f"{k} = {v}" for k, v in match_counts.items())
    )

    # --------------------------------------------------------
    # Download full results
    # --------------------------------------------------------

    csv_data = results_df.round(4).to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download validation results CSV",
        data=csv_data,
        file_name=f"knnr_validation_{window}day.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Browse individual validation days — observed vs disaggregated
    # hyetograph, the classic figure from this literature.
    # --------------------------------------------------------

    st.markdown("### Observed vs. disaggregated hyetograph")

    day_options = [d["date"] for d in detail]

    chosen_date = st.selectbox(
        "Validation day",
        day_options,
        key="val_chosen_day",
    )

    chosen = next(d for d in detail if d["date"] == chosen_date)

    hours = np.arange(24)
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4.2))

    ax.bar(
        hours - width / 2,
        chosen["actual_hourly"],
        width=width,
        label="Observed",
        color="#1d4ed8",
        alpha=0.85,
    )

    ax.bar(
        hours + width / 2,
        chosen["predicted_hourly"],
        width=width,
        label="KNNR disaggregated (ensemble mean)",
        color="#dc2626",
        alpha=0.75,
    )

    ax.fill_between(
        hours,
        chosen["predicted_p10"],
        chosen["predicted_p90"],
        alpha=0.15,
        color="#dc2626",
        label="KNNR 10–90 percentile",
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(
        f"{chosen_date} — daily total {chosen['daily_total_mm']:.1f} mm"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    day_row = results_df[results_df["date"] == chosen_date].iloc[0]

    d1, d2, d3 = st.columns(3)
    d1.metric("Hourly MAE", f'{day_row["mae"]:.3f} mm')
    d2.metric("Peak-hour match", "Yes" if day_row["peak_hour_hit"] else "No")
    d3.metric("NSE", f'{day_row["nse"]:.3f}')


# ============================================================
# UI
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🌧️ India Rainfall & Climate Prediction'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'District-level rainfall and climate prediction '
    'with KNNR temporal rainfall disaggregation.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "📍 Prediction inputs"
    )

    location_mode = st.radio(
        "Location",
        ["Known district", "New / unlisted district"],
        help=(
            "If your district isn't in the list below (a newly "
            "formed district, or one just missing from our lookup), "
            "choose 'New / unlisted district' and enter its "
            "coordinates directly. The app will borrow rainfall "
            "regime statistics from the closest climatologically "
            "similar KNOWN district -- these can't be computed live, "
            "since they require years of historical rainfall records."
        ),
    )

    if location_mode == "Known district":

        states = sorted(
            LUT["state_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        state = st.selectbox(
            "State / Union Territory",
            states,
        )

        districts = sorted(
            LUT.loc[
                LUT["state_name"]
                .astype(str)
                == state,
                "district_name",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        district = st.selectbox(
            "District",
            districts,
        )

    else:

        st.caption(
            "Enter the new location's coordinates. "
            "Distance to coast can be measured with "
            "Google Maps' ruler tool (straight-line "
            "distance to the nearest coastline point, in km)."
        )

        manual_latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=20.0,
            format="%.4f",
        )

        manual_longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.0,
            format="%.4f",
        )

        manual_elevation = st.number_input(
            "Elevation (m)",
            min_value=-500.0,
            max_value=9000.0,
            value=200.0,
            format="%.1f",
        )

        manual_distance_to_coast_km = st.number_input(
            "Distance to coast (km)",
            min_value=0.0,
            max_value=3000.0,
            value=100.0,
            format="%.1f",
        )
=======
# UI
# ============================================================
st.markdown('<div class="main-title">🌧️ India Rainfall & Climate Prediction</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">District-level prediction using the trained '
    'XGBoost rainfall cascade and climate regression models.</div>',
    unsafe_allow_html=True
)

# Sidebar inputs
with st.sidebar:
    st.header("📍 Prediction inputs")

    states = sorted(LUT["state_name"].dropna().astype(str).unique().tolist())
    state = st.selectbox("State / Union Territory", states)

    districts = sorted(
        LUT.loc[LUT["state_name"].astype(str) == state, "district_name"]
        .dropna().astype(str).unique().tolist()
    )
    district = st.selectbox("District", districts)
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03

    date = st.date_input(
        "Prediction date",
        value=datetime.now().date(),
<<<<<<< HEAD
        min_value=datetime(
            2001,
            1,
            1,
        ).date(),
        max_value=datetime(
            2100,
            12,
            31,
        ).date(),
    )

    st.divider()

    st.subheader(
        "🕐 KNNR disaggregation"
    )

    enable_knnr = st.checkbox(
        "Generate hourly rainfall",
        value=True,
    )

    knnr_draws = st.slider(
        "Ensemble simulations",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    knnr_top_k = st.slider(
        "Nearest neighbours (K)",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )

    st.caption(
        "The compact KNNR library is used instead "
        "of the original large historical CSV."
    )

    st.divider()

    run = st.button(
        "🔮 Predict weather",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "Monsoon flag assumption: June–September."
    )


# ============================================================
# Selected location
# ============================================================

try:

    if location_mode == "Known district":

        selected_location = resolve_location(
            state,
            district,
        )

    else:

        selected_location = find_nearest_regime_row(
            manual_latitude,
            manual_longitude,
            manual_elevation,
            manual_distance_to_coast_km,
        )

except Exception as e:

    st.error(str(e))

    st.stop()



# ============================================================
# Tabs: Predict vs KNNR Validation
# ============================================================

tab_predict, tab_validate = st.tabs(["🔮 Predict", "📊 KNNR Validation"])

with tab_predict:

    info1, info2, info3, info4 = st.columns(4)

    info1.metric(
        "State",
        selected_location["state_name"],
    )

    info2.metric(
        "District",
        selected_location["district_name"],
    )

    info3.metric(
        "Latitude",
        f'{selected_location["latitude"]:.4f}°',
    )

    info4.metric(
        "Longitude",
        f'{selected_location["longitude"]:.4f}°',
    )

    if selected_location.get("is_unlisted_location"):

        matched = selected_location["_matched_from"]

        st.info(
            "This location isn't in the district lookup. Rainfall "
            "regime statistics (mean, percentiles, tail risk, etc.) "
            f"are borrowed from the closest climatologically similar "
            f"known district: **{matched['district_name']}, "
            f"{matched['state_name']}** (normalized similarity "
            f"distance: {matched['similarity_distance']:.2f}). "
            "Hourly KNNR disaggregation will also draw its analogue "
            "patterns from that district's historical fragment "
            "library, since this location has no hourly rainfall "
            "history of its own."
        )


    # ============================================================
    # Map
    # ============================================================

    st.markdown(
        "### 🗺️ Selected location"
    )

    map_df = pd.DataFrame(
        [selected_location]
    )

    view_state = pdk.ViewState(
        latitude=selected_location["latitude"],
        longitude=selected_location["longitude"],
        zoom=5.0,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_radius=14000,
        get_fill_color="[220, 38, 38, 220]",
        get_line_color="[255, 255, 255, 255]",
        line_width_min_pixels=2,
        pickable=True,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer],
            tooltip={
                "text":
                    "{district_name}, "
                    "{state_name}\n"
                    "Lat: {latitude}\n"
                    "Lon: {longitude}\n"
                    "Elevation: {elevation} m"
            },
        ),
        use_container_width=True,
    )


    if not run:

        st.info(
            "Select a state, district and date, "
            "then click **Predict weather**."
        )

        if knnr_library is None:

            st.warning(
                "KNNR library was not found. "
                "Place the disaggregation_knnr folder "
                "beside this Streamlit script."
            )

        else:

            st.success(
                "KNNR disaggregation library detected."
            )

    else:



        # ============================================================
        # Prediction
        # ============================================================

        date_str = date.strftime("%Y-%m-%d")

        try:

            with st.spinner(
                "Running rainfall and climate prediction..."
            ):

                result = predict(
                    date_str,
                    selected_location,
                )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)

            st.stop()


        # ============================================================
        # Prediction summary
        # ============================================================

        st.markdown(
            "## 📊 Prediction result"
        )

        climate = result["climate_predictions"]

        rain = result["rain_severity_probabilities"]

        expected = result["expected_precipitation_mm"]

        scenario = expected["most_likely_scenario"]


        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rain probability",
            f'{result["rain_probability"] * 100:.1f}%',
        )

        c2.metric(
            "Expected rainfall",
            f'{expected["weighted_mean"]:.1f} mm',
        )

        c3.metric(
            "Typical median",
            f'{expected["weighted_median"]:.1f} mm',
        )

        c4.metric(
            "Most likely severity",
            scenario["bin"]
            .replace("_", " ")
            .title(),
            f'{scenario["probability"] * 100:.1f}%',
        )


        # ============================================================
        # Rainfall probabilities
        # ============================================================

        st.markdown(
            "### 🌧️ Rainfall severity probabilities"
        )

        severity_order = [
            "no_rain",
            "light",
            "medium",
            "moderate",
            "heavy",
            "extreme",
        ]

        for cls in severity_order:

            prob = rain.get(cls, 0.0)

            st.write(
                f"**{cls.replace('_', ' ').title()}** "
                f"— {prob * 100:.2f}%"
            )

            st.progress(
                min(
                    max(prob, 0.0),
                    1.0,
                )
            )


        # ============================================================
        # KNNR hourly disaggregation
        # ============================================================

        st.markdown(
            "## 🕐 Hourly rainfall disaggregation"
        )


        daily_total = float(
            expected["weighted_mean"]
        )


        if not enable_knnr:

            st.info(
                "Hourly KNNR disaggregation is disabled."
            )

        elif knnr_library is None:

            st.warning(
                f"KNNR library not found at:\n"
                f"`{KNNR_DIR}`\n\n"
                "The large historical CSV is NOT required "
                "by the deployed application."
            )

        elif daily_total < 0.01:

            st.info(
                "Expected rainfall is effectively zero, "
                "so hourly disaggregation is skipped."
            )

        else:

            with st.spinner(
                "Generating 3-day / 5-day / 7-day "
                "KNNR rainfall ensemble..."
            ):

                knnr_result = generate_knnr_ensemble(
                    district_code=selected_location["district_code"],
                    target_date=date_str,
                    daily_total_mm=daily_total,
                    n_draws=knnr_draws,
                    top_k=knnr_top_k,
                )

            if knnr_result is None:

                st.warning(
                    "No suitable historical KNNR "
                    "analogue was found for this "
                    "district/date."
                )

            else:

                matrix = knnr_result["matrix"]

                analogs = knnr_result["analogs"]

                windows = knnr_result["windows"]

                # ----------------------------------------------------
                # Ensemble statistics
                # ----------------------------------------------------

                hourly_mean = matrix.mean(axis=0)

                hourly_p10 = np.percentile(
                    matrix,
                    10,
                    axis=0,
                )

                hourly_p25 = np.percentile(
                    matrix,
                    25,
                    axis=0,
                )

                hourly_p75 = np.percentile(
                    matrix,
                    75,
                    axis=0,
                )

                hourly_p90 = np.percentile(
                    matrix,
                    90,
                    axis=0,
                )

                hours = np.arange(24)

                # ----------------------------------------------------
                # Chart
                # ----------------------------------------------------

                fig, ax = plt.subplots(
                    figsize=(10, 4.2)
                )

                ax.bar(
                    hours,
                    hourly_mean,
                    alpha=0.75,
                    label="KNNR ensemble mean",
                )

                ax.fill_between(
                    hours,
                    hourly_p10,
                    hourly_p90,
                    alpha=0.20,
                    label="10–90 percentile",
                )

                ax.plot(
                    hours,
                    hourly_p25,
                    linestyle="--",
                    linewidth=1,
                    label="25th percentile",
                )

                ax.plot(
                    hours,
                    hourly_p75,
                    linestyle="--",
                    linewidth=1,
                    label="75th percentile",
                )

                ax.set_xlabel("Hour of day")

                ax.set_ylabel("Rainfall (mm)")

                ax.set_xticks(
                    range(0, 24, 2)
                )

                ax.set_title(
                    "KNNR disaggregated hourly rainfall"
                )

                ax.legend(fontsize=8)

                ax.grid(alpha=0.15)

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

                # ----------------------------------------------------
                # Metrics
                # ----------------------------------------------------

                h1, h2, h3, h4 = st.columns(4)

                peak_hour = int(
                    np.argmax(hourly_mean)
                )

                peak_value = float(
                    hourly_mean[peak_hour]
                )

                h1.metric(
                    "Daily total",
                    f"{hourly_mean.sum():.2f} mm",
                )

                h2.metric(
                    "Peak hour",
                    f"{peak_hour:02d}:00",
                )

                h3.metric(
                    "Peak rainfall",
                    f"{peak_value:.2f} mm",
                )

                h4.metric(
                    "Simulations",
                    f"{len(matrix)}",
                )

                # ----------------------------------------------------
                # How matching ACTUALLY happened
                #
                # Previously this block unconditionally printed
                # "KNNR temporal contexts used: 3-day, 5-day, 7-day",
                # which is false whenever the library lacks pattern /
                # daily-window columns or the target date isn't in the
                # library — in which case all three windows collapse to
                # identical seasonal-only matching.
                # ----------------------------------------------------

                match_info = knnr_result.get(
                    "match_info",
                    {},
                )

                any_pattern = knnr_result.get(
                    "any_pattern_matching",
                    False,
                )

                target_in_library = knnr_result.get(
                    "target_in_library",
                    False,
                )

                if any_pattern:

                    detail_rows = []

                    for w in sorted(match_info.keys()):

                        info = match_info[w]

                        if info["pattern_column"]:

                            how = "pattern-label column"

                        elif info["daily_shape"]:

                            how = "daily-window shape distance"

                        else:

                            how = "seasonal (day-of-year) only"

                        detail_rows.append(
                            {
                                "Window": f"{w}-day",
                                "Matched on": how,
                            }
                        )

                    st.success(
                        "KNNR antecedent-pattern matching active "
                        "for at least one window."
                    )

                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    if not target_in_library:

                        st.warning(
                            f"**Seasonal matching only** — {date_str} is not in "
                            "the KNNR library, so there is no known antecedent "
                            "rainfall sequence to match against. Analogues were "
                            "selected by day-of-year proximity within this "
                            "district alone. The 3/5/7-day windows are therefore "
                            "equivalent here, and the ensemble spread reflects "
                            "seasonal variability, not antecedent-pattern "
                            "similarity."
                        )

                    else:

                        st.warning(
                            "**Seasonal matching only** — the KNNR library does "
                            "not contain pattern-label columns (`pattern_3d` / "
                            "`pattern_5d` / `pattern_7d`) or daily-window columns "
                            "(e.g. `rain_m1`, `day_m1`), so antecedent-pattern "
                            "conditioning could not be applied. Analogues were "
                            "selected by day-of-year proximity alone and the "
                            "3/5/7-day windows are equivalent here. Rebuild the "
                            "library with those columns to enable pattern matching."
                        )

                st.caption(
                    "The hourly sequence is generated from historical analogues, "
                    "rescaled in Gram-Schmidt rotated space so the 24 hourly "
                    "values conserve the predicted daily rainfall total. "
                    "Percentile bands reflect disagreement across sampled "
                    "analogues — they are not a calibrated confidence interval "
                    "on the true hourly values."
                )

                # ----------------------------------------------------
                # Hourly table
                # ----------------------------------------------------

                hourly_table = pd.DataFrame(
                    {
                        "Hour": [
                            f"{h:02d}:00"
                            for h in hours
                        ],

                        "Mean rainfall (mm)": hourly_mean,

                        "P10 (mm)": hourly_p10,

                        "P25 (mm)": hourly_p25,

                        "P75 (mm)": hourly_p75,

                        "P90 (mm)": hourly_p90,
                    }
                )

                hourly_table = hourly_table.round(3)

                st.markdown(
                    "### 📋 Hourly rainfall table"
                )

                st.dataframe(
                    hourly_table,
                    use_container_width=True,
                    hide_index=True,
                )

                # ----------------------------------------------------
                # Download
                # ----------------------------------------------------

                csv_data = (
                    hourly_table
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    "⬇️ Download hourly rainfall CSV",
                    data=csv_data,
                    file_name=(
                        f"hourly_rainfall_"
                        f"{selected_location['district_name']}_"
                        f"{date_str}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                )

                # ----------------------------------------------------
                # Analogue information
                # ----------------------------------------------------

                with st.expander(
                    "🔎 View KNNR analogue information"
                ):

                    analogue_rows = []

                    for d in analogs:

                        if d is None:
                            continue

                        analogue_rows.append(
                            {
                                "Historical analogue date": str(d)
                            }
                        )

                    if analogue_rows:

                        st.dataframe(
                            pd.DataFrame(analogue_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:

                        st.write(
                            "Historical analogue dates were not "
                            "available in the library output."
                        )


        # ============================================================
        # Climate predictions
        # ============================================================

        st.markdown(
            "### 🌡️ Climate predictions"
        )

        labels = {

            "temperature":
                ("Temperature", "°C"),

            "temperature_max":
                ("Maximum temperature", "°C"),

            "temperature_min":
                ("Minimum temperature", "°C"),

            "humidity":
                ("Humidity", "%"),

            "wind_speed":
                ("Wind speed", "m/s"),

            "surface_pressure":
                ("Surface pressure", "hPa"),

            "root_zone_soil_moisture":
                ("Root-zone soil moisture", "m³/m³"),

            "surface_soil_moisture":
                ("Surface soil moisture", "m³/m³"),

            "solar_radiation":
                ("Solar radiation", "W/m²"),
        }


        climate_cols = st.columns(3)

        for i, target in enumerate(CLIMATE_TARGET_COLS):

            label, unit = labels.get(
                target,
                (
                    target.replace("_", " ").title(),
                    "",
                ),
            )

            with climate_cols[i % 3]:

                st.metric(
                    label,
                    f'{climate[target]:.2f} {unit}',
                )


        # ============================================================
        # Expected rainfall scenario
        # ============================================================

        st.markdown(
            "### ☔ Expected rainfall scenario"
        )

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Most likely class",
            scenario["bin"]
            .replace("_", " ")
            .title(),
        )

        s2.metric(
            "Typical probability",
            f'{scenario["probability"] * 100:.1f}%',
        )

        s3.metric(
            "Training-set typical range",
            f'{scenario["p25"]:.1f}'
            f'–'
            f'{scenario["p75"]:.1f} mm',
        )


        # ============================================================
        # Model input location
        # ============================================================

        st.markdown(
            "### 📍 Model input location"
        )

        location_df = pd.DataFrame(
            [
                {
                    "State":
                        result["location"]["state_name"],

                    "District":
                        result["location"]["district_name"],

                    "Latitude":
                        result["location"]["latitude"],

                    "Longitude":
                        result["location"]["longitude"],

                    "Elevation (m)":
                        result["location"]["elevation"],

                    "Date":
                        result["date"],
                }
            ]
        )

        st.dataframe(
            location_df,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            "Expected rainfall is derived from the "
            "predicted severity probabilities and "
            "training-set precipitation statistics. "
            "KNNR is then used to estimate the "
            "within-day hourly rainfall distribution."
        )


        # ============================================================
        # Raw output
        # ============================================================

        with st.expander(
            "🔎 View raw prediction output"
        ):

            st.json(result)
with tab_validate:

    render_knnr_validation_tab()
=======
        min_value=datetime(2001, 1, 1).date(),
        max_value=datetime(2100, 12, 31).date(),
    )

    run = st.button(
        "🔮 Predict weather",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.caption(
        "The model uses latitude, longitude, elevation, date-derived "
        "features and spatial rainfall-regime features."
    )
    st.caption(
        "Monsoon flag assumption: June–September, matching the supplied "
        "inference code."
    )


# Resolve location immediately so the map can show the selected point.
try:
    selected_location = resolve_location(state, district)
except Exception as e:
    st.error(str(e))
    st.stop()


# Top location information
# On phones (<600px) Streamlit stacks these into a single column
# automatically; the CSS above keeps each card readable at that width.
info1, info2, info3, info4 = st.columns(4)
info1.metric("State", selected_location["state_name"])
info2.metric("District", selected_location["district_name"])
info3.metric("Latitude", f'{selected_location["latitude"]:.4f}°')
info4.metric("Longitude", f'{selected_location["longitude"]:.4f}°')

st.markdown("### 🗺️ Selected location")

# India-focused map. pydeck's default basemap is used; no external API key is required.
map_df = pd.DataFrame([selected_location])

view_state = pdk.ViewState(
    latitude=selected_location["latitude"],
    longitude=selected_location["longitude"],
    zoom=5.0,
    pitch=0,
)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[longitude, latitude]",
    get_radius=14000,
    get_fill_color="[220, 38, 38, 220]",
    get_line_color="[255, 255, 255, 255]",
    line_width_min_pixels=2,
    pickable=True,
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={
            "text": (
                "{district_name}, {state_name}\n"
                "Lat: {latitude}\nLon: {longitude}\n"
                "Elevation: {elevation} m"
            )
        },
    ),
    use_container_width=True,
    height=380,
)

if not run:
    st.info(
        "Select a state, district and date, then click **Predict weather** "
        "to run the trained models."
    )
    st.stop()


# Run prediction
date_str = date.strftime("%Y-%m-%d")

try:
    with st.spinner("Running rainfall and climate prediction..."):
        result = predict(date_str, state, district)
except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()


# ============================================================
# Prediction summary
# ============================================================
st.markdown("## 📊 Prediction result")

climate = result["climate_predictions"]
rain = result["rain_severity_probabilities"]
expected = result["expected_precipitation_mm"]
scenario = expected["most_likely_scenario"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rain probability", f'{result["rain_probability"] * 100:.1f}%')
c2.metric("Expected rainfall", f'{expected["weighted_mean"]:.1f} mm')
c3.metric("Typical (median-weighted)", f'{expected["weighted_median"]:.1f} mm')
c4.metric(
    "Most likely severity",
    scenario["bin"].replace("_", " ").title(),
    f'{scenario["probability"] * 100:.1f}%'
)

st.markdown("### 🌧️ Rainfall severity probabilities")

severity_order = ["no_rain", "light", "medium", "moderate", "heavy", "extreme"]

for cls in severity_order:
    prob = rain.get(cls, 0.0)
    st.write(f"**{cls.replace('_', ' ').title()}** — {prob * 100:.2f}%")
    st.progress(min(max(prob, 0.0), 1.0))

st.markdown("### 🌡️ Climate predictions")

labels = {
    "temperature": ("Temperature", "°C"),
    "temperature_max": ("Maximum temperature", "°C"),
    "temperature_min": ("Minimum temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "wind_speed": ("Wind speed", "m/s"),
    "surface_pressure": ("Surface pressure", "hPa"),
}

# 3 columns on desktop; Streamlit reflows these to 1-2 per row on
# narrower viewports, and the metric CSS above keeps the text legible.
climate_cols = st.columns(3)
for i, target in enumerate(CLIMATE_TARGET_COLS):
    label, unit = labels.get(target, (target.replace("_", " ").title(), ""))
    with climate_cols[i % 3]:
        st.metric(label, f'{climate[target]:.2f} {unit}')

st.markdown("### ☔ Expected rainfall scenario")
s1, s2, s3 = st.columns(3)
s1.metric("Most likely class", scenario["bin"].replace("_", " ").title())
s2.metric("Typical probability", f'{scenario["probability"] * 100:.1f}%')
s3.metric(
    "Training-set typical range",
    f'{scenario["p25"]:.1f}–{scenario["p75"]:.1f} mm'
)

st.markdown("### 📍 Model input location")
location_df = pd.DataFrame([{
    "State": result["location"]["state_name"],
    "District": result["location"]["district_name"],
    "Latitude": result["location"]["latitude"],
    "Longitude": result["location"]["longitude"],
    "Elevation (m)": result["location"]["elevation"],
    "Date": result["date"],
}])
st.dataframe(location_df, use_container_width=True, hide_index=True)

st.caption(
    "Expected rainfall is derived from the predicted severity probabilities "
    "and training-set precipitation statistics; it is not a separate "
    "direct precipitation regression."
)

with st.expander("🔎 View raw prediction output"):
    st.json(result)
>>>>>>> bd40be851e8e483cc35bff30a2f824cd0f5e0b03
