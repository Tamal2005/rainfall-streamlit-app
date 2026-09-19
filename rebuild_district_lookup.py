"""
One-time fix: rebuild district_lookup.parquet to include ALL regime
features, sourced from the EXACT regime CSV used during Phase 0 training.

Run this ONCE, from anywhere — it resolves paths relative to this file's
own location (or to explicit CLI args), not to your current working
directory, so it can't silently write to / read from the wrong copy.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from app import BASE_DIR

SCRIPT_DIR = Path(__file__).resolve().parent

REGIME_COLS = [
    "latitude", "longitude",
    "distance_to_coast_km", "log_distance_to_coast_km",
    "rainfall_mean", "rainfall_std", "rainfall_p95", "rainfall_p99",
    "rain_rate_ge20", "rain_rate_ge50", "rain_rate_ge100", "rain_rate_ge150",
    "rainfall_extreme_ratio", "tail_risk_score",
]

# Decimal places to round lat/lon to before joining. Two files that store
# the "same" coordinate with different float precision (e.g. 21.15 vs
# 21.150000000000002, or 6 vs 4 decimal places) will fail an exact merge
# with zero warning other than a NaN count you might not scrutinize.
COORD_ROUND_DP = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "prepped",
        help="Directory containing district_lookup.parquet (default: "
             "prepped next to this script — must match the Streamlit "
             "app's DATA_DIR, including any st.secrets override)",
    )
    parser.add_argument(
        "--regime-path",
        type=Path,
        default=SCRIPT_DIR / "spatial_regime_features.csv",
        help="Path to the EXACT regime CSV used in phase0_pipeline.py",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    regime_path = args.regime_path.resolve()
    lookup_path = data_dir / "district_lookup.parquet"

    print(f"Resolved paths:\n  district_lookup: {lookup_path}\n  regime CSV: {regime_path}\n")

    if not lookup_path.exists():
        sys.exit(f"ERROR: {lookup_path} does not exist. Check --data-dir.")
    if not regime_path.exists():
        sys.exit(f"ERROR: {regime_path} does not exist. Check --regime-path.")

    print("Loading existing district_lookup.parquet...")
    lookup = pd.read_parquet(lookup_path)
    print(f"  {len(lookup)} districts")

    print("Loading the regime CSV...")
    regime = pd.read_csv(regime_path, usecols=REGIME_COLS)
    print(f"  {len(regime)} locations")

    # Round coordinates on both sides before merging so float-precision
    # differences between the two source files don't produce silent
    # non-matches.
    lookup = lookup.copy()
    regime = regime.copy()
    lookup["_lat_r"] = lookup["latitude"].round(COORD_ROUND_DP)
    lookup["_lon_r"] = lookup["longitude"].round(COORD_ROUND_DP)
    regime["_lat_r"] = regime["latitude"].round(COORD_ROUND_DP)
    regime["_lon_r"] = regime["longitude"].round(COORD_ROUND_DP)

    dup_regime_coords = regime.duplicated(subset=["_lat_r", "_lon_r"]).sum()
    if dup_regime_coords > 0:
        sys.exit(
            f"ERROR: {dup_regime_coords} duplicate (lat, lon) pairs in the "
            f"regime CSV after rounding to {COORD_ROUND_DP} dp. Merging "
            f"would silently fan out rows. Lower COORD_ROUND_DP precision "
            f"or fix the source CSV."
        )

    print("Merging on rounded lat/lon...")
    lookup_base = lookup.drop(columns=["distance_to_coast_km"], errors="ignore")
    regime_slim = regime.drop(columns=["latitude", "longitude"])
    enriched = lookup_base.merge(
        regime_slim, on=["_lat_r", "_lon_r"], how="left"
    ).drop(columns=["_lat_r", "_lon_r"])

    missing_mask = enriched["distance_to_coast_km"].isna()
    missing = int(missing_mask.sum())
    print(f"Rows with no regime match: {missing} / {len(enriched)}")

    if missing > 0:
        bad = enriched.loc[missing_mask, ["state_name", "district_name", "latitude", "longitude"]]
        print("\nUnmatched districts (first 20):")
        print(bad.head(20).to_string(index=False))
        sys.exit(
            "\nERROR: refusing to write a district_lookup.parquet with "
            "unmatched regime rows — this is exactly the corruption bug "
            "this script exists to prevent. Fix the coordinate mismatch "
            "(check COORD_ROUND_DP, or confirm --regime-path is really "
            "the file used in phase0_pipeline.py) and re-run."
        )

    missing_regime_cols = [c for c in REGIME_COLS if c not in enriched.columns and c not in ("latitude", "longitude")]
    if missing_regime_cols:
        sys.exit(f"ERROR: merge did not produce expected columns: {missing_regime_cols}")

    enriched.to_parquet(lookup_path, index=False)
    print(f"\nRebuilt {lookup_path} with {len(enriched.columns)} columns:")
    print(list(enriched.columns))
    print(
        "\nIf a Streamlit session is already running against this "
        "DATA_DIR, restart it (or clear the resource cache) — "
        "@st.cache_resource will otherwise keep serving the old copy."
    )


if __name__ == "__main__":
    main()