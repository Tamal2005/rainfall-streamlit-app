import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import difflib
from datetime import datetime
from pathlib import Path
import pydeck as pdk

# ============================================================
# Rainfall & Climate Prediction — India
# ============================================================

st.set_page_config(
    page_title="India Rainfall & Climate Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# ============================================================
# Paths
# ============================================================
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
        climate_models[target] = model

    return {
        "district_lookup": district_lookup,
        "regime_full": regime_full,
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
    st.stop()


district_lookup = assets["district_lookup"]
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
    return lut


LUT = location_table()


def resolve_location(state_name: str, district_name: str) -> dict:
    state_u = state_name.strip().upper()
    district_u = district_name.strip().upper()

    match = LUT[
        (LUT["state_u"] == state_u) &
        (LUT["district_u"] == district_u)
    ]

    if len(match) == 0:
        state_candidates = LUT[LUT["state_u"] == state_u]["district_u"].unique()
        suggestions = difflib.get_close_matches(
            district_u, state_candidates.tolist(), n=3, cutoff=0.5
        )
        raise ValueError(
            f"No match for state='{state_name}', district='{district_name}'. "
            f"Closest district matches: {suggestions}"
        )

    row = match.iloc[0]
    return {
        "state_name": row["state_name"],
        "district_name": row["district_name"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elevation": float(row["elevation"]),
    }


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
    }


# ============================================================
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

    date = st.date_input(
        "Prediction date",
        value=datetime.now().date(),
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