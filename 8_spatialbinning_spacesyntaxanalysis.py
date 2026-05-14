#!/usr/bin/env python3
"""
6d_prepare_spatialbins_for_space_syntax.py

Purpose:
    Prepare participant-level spatial-bin data across ALL terrain types,
    without filtering to cobblestone.

Output:
    C:/LocoGaze/data/group/mixedmodel_spatialbins_allterrain.csv

This file is compatible with:
    - existing mixed-model workflows
    - added street-network / space-syntax analyses

Key features:
    - walking-only samples
    - valid GPS only
    - spatial bins based on rounded latitude/longitude
    - modal terrain label per bin
    - binary social presence: people_bin = 0 vs 1plus
    - continuous social density: total_people mean per bin
    - gaze allocation proportions
    - gaze geometry measures
    - gait measures
    - stride variability CV
    - bin centre latitude/longitude retained for network assignment
"""

import os
import warnings
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR   = r"C:/LocoGaze/data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_CSV = os.path.join(GROUP_DIR, "mixedmodel_spatialbins_allterrain.csv")

os.makedirs(GROUP_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Spatial binning precision
# EPS = 3 gives bins on the order of tens of metres
EPS = 4

# Scene geometry
X_MAX    = 1920.0
Y_MAX    = 1080.0
X_CENTER = 960.0

GAIT_COLS = [
    "mean_rms_LEFT",
    "stride_duration_LEFT",
    "stride_length_LEFT",
    "cadence_LEFT",
    "pace_LEFT",
]

GAZE_CONT_COLS = [
    "radius_of_gyration",
    "spatial_entropy",
    "depth_d_s",
    "depth_head_pitch_deg",
    "depth_d_vergence",
    "depth_calib_mm",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_total_people(df: pd.DataFrame) -> pd.Series:
    """
    Build total visible people count from available columns.
    Uses number_people if available, otherwise reconstructs from pedestrian
    and cyclist counts when possible.
    """
    if "number_people" in df.columns:
        total = pd.to_numeric(df["number_people"], errors="coerce").fillna(0).clip(lower=0)

    elif ("number_people_nocyclist" in df.columns) and ("number_cyclist" in df.columns):
        total = (
            pd.to_numeric(df["number_people_nocyclist"], errors="coerce").fillna(0) +
            pd.to_numeric(df["number_cyclist"], errors="coerce").fillna(0)
        ).clip(lower=0)
        warnings.warn("number_people missing; reconstructed total_people from nocyclist + cyclist.")

    elif "number_people_nocyclist" in df.columns:
        total = pd.to_numeric(df["number_people_nocyclist"], errors="coerce").fillna(0).clip(lower=0)
        warnings.warn("Only number_people_nocyclist present; using this as total_people.")

    elif "number_cyclist" in df.columns:
        total = pd.to_numeric(df["number_cyclist"], errors="coerce").fillna(0).clip(lower=0)
        warnings.warn("Only number_cyclist present; using this as total_people.")

    else:
        total = pd.Series(0, index=df.index, dtype=float)
        warnings.warn("No people-count columns found; assuming 0 total people everywhere.")

    return total


def majority_category(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return np.nan
    return s.value_counts().idxmax()


def stride_cv_combined(group: pd.DataFrame):
    """
    Combined stride variability index:
        mean of CV(stride length) and CV(stride duration)
    """
    if not {"stride_length_LEFT", "stride_duration_LEFT"}.issubset(group.columns):
        return np.nan

    len_vals = pd.to_numeric(group["stride_length_LEFT"], errors="coerce").dropna()
    dur_vals = pd.to_numeric(group["stride_duration_LEFT"], errors="coerce").dropna()

    if len_vals.size < 2 or dur_vals.size < 2:
        return np.nan

    mean_len = len_vals.mean()
    mean_dur = dur_vals.mean()

    if mean_len <= 0 or mean_dur <= 0:
        return np.nan

    cv_len = len_vals.std(ddof=1) / mean_len
    cv_dur = dur_vals.std(ddof=1) / mean_dur

    return (cv_len + cv_dur) / 2.0


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        meta = pd.read_csv(META_CSV)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: cannot find {META_CSV}")

    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column missing in metadata_all.csv")

    participants = (
        meta["participant"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not participants:
        raise SystemExit("ERROR: no participants found in metadata_all.csv")

    all_bins = []
    missing = []

    for pid in participants:
        vis_path = os.path.join(BASE_DIR, pid, "output", "visual_events.csv")

        if not os.path.exists(vis_path):
            missing.append(pid)
            continue

        print(f"Processing participant {pid}...")

        df = pd.read_csv(vis_path)

        needed = {
            "is_walking",
            "latitude",
            "longitude",
            "fixated",
            "depth_looking_floor",
            "area_label",
        }

        if not needed.issubset(df.columns):
            warnings.warn(f"{vis_path} missing some of {needed}; skipping this participant.")
            continue

        # Walking-only samples
        df = df[df["is_walking"] == True].copy()
        if df.empty:
            continue

        # Valid GPS only
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        if df.empty:
            continue

        # Clean terrain labels, but DO NOT filter terrain
        df["area_label"] = (
            df["area_label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Social density at sample level
        df["total_people"] = build_total_people(df)

        total_int = df["total_people"].astype(int).clip(lower=0)
        df["people_bin"] = np.where(total_int == 0, "0", "1plus")
        df["people_bin"] = pd.Categorical(
            df["people_bin"],
            categories=["0", "1plus"],
            ordered=True,
        )

        # Gaze allocation indicators
        df["fixated"] = df["fixated"].astype(str).str.strip().str.lower()
        df["depth_looking_floor"] = df["depth_looking_floor"].astype(bool)

        df["looking_people"] = df["fixated"].eq("person").astype(float)

        df["looking_floor"] = (
            df["fixated"].eq("other") &
            df["depth_looking_floor"]
        ).astype(float)

        df["looking_environment"] = (
            df["fixated"].eq("other") &
            (~df["depth_looking_floor"])
        ).astype(float)

        other_obj_labels = {"car", "cars", "bicycle", "bike", "bicycles"}
        df["looking_other_objects"] = df["fixated"].isin(other_obj_labels).astype(float)

        # Gaze verticality and horizontal eccentricity
        if "gaze_x" in df.columns:
            df["gaze_x"] = pd.to_numeric(df["gaze_x"], errors="coerce")
        else:
            df["gaze_x"] = np.nan

        if "gaze_y" in df.columns:
            df["gaze_y"] = pd.to_numeric(df["gaze_y"], errors="coerce")
            df["gaze_y"] = Y_MAX - df["gaze_y"]
        else:
            df["gaze_y"] = np.nan

        df["horizontal_ecc"] = np.abs(df["gaze_x"] - X_CENTER)

        # Spatial binning
        df["latitude_bin"] = df["latitude"].round(EPS)
        df["longitude_bin"] = df["longitude"].round(EPS)

        group_cols = ["latitude_bin", "longitude_bin"]

        gait_cols_exist = [c for c in GAIT_COLS if c in df.columns]
        gaze_cols_exist = [c for c in GAZE_CONT_COLS if c in df.columns]

        # Aggregation dictionary
        agg_dict = {
            # bin centre / mean raw coordinates for network assignment
            "latitude": "mean",
            "longitude": "mean",

            # context labels
            "area_label": majority_category,
            "people_bin": majority_category,
            "total_people": "mean",

            # gaze allocation
            "looking_people": "mean",
            "looking_floor": "mean",
            "looking_environment": "mean",
            "looking_other_objects": "mean",

            # gaze geometry
            "horizontal_ecc": "mean",
            "gaze_y": "mean",
        }

        for c in gait_cols_exist + gaze_cols_exist:
            agg_dict[c] = "mean"

        df_bins = (
            df.groupby(group_cols)
              .agg(agg_dict)
              .reset_index()
        )

        # Stride variability per bin
        if {"stride_length_LEFT", "stride_duration_LEFT"}.issubset(df.columns):
            stride_var = (
                df.groupby(group_cols)
                  .apply(stride_cv_combined)
                  .rename("stride_var_CV")
                  .reset_index()
            )
            df_bins = df_bins.merge(stride_var, on=group_cols, how="left")
        else:
            df_bins["stride_var_CV"] = np.nan

        # Add participant
        df_bins.insert(0, "participant", pid)

        # Tidy people_bin
        df_bins["people_bin"] = df_bins["people_bin"].astype(str).str.strip()
        df_bins["people_bin"] = pd.Categorical(
            df_bins["people_bin"],
            categories=["0", "1plus"],
            ordered=True,
        )

        all_bins.append(df_bins)

    if not all_bins:
        raise SystemExit("ERROR: no spatial-bin data collected from any participant.")

    group_df = pd.concat(all_bins, ignore_index=True)

    # Add aliases expected by the space-syntax analysis script
    group_df["social_density"] = group_df["total_people"]
    group_df["look_people_prop"] = group_df["looking_people"]
    group_df["floor_looking_frac"] = group_df["looking_floor"]

    # Helpful ordering
    preferred_cols = [
        "participant",
        "latitude_bin",
        "longitude_bin",
        "latitude",
        "longitude",
        "area_label",
        "people_bin",
        "total_people",
        "social_density",
        "looking_people",
        "look_people_prop",
        "looking_floor",
        "floor_looking_frac",
        "looking_environment",
        "looking_other_objects",
        "horizontal_ecc",
        "gaze_y",
        "radius_of_gyration",
        "spatial_entropy",
        "depth_d_s",
        "depth_head_pitch_deg",
        "depth_d_vergence",
        "depth_calib_mm",
        "mean_rms_LEFT",
        "stride_duration_LEFT",
        "stride_length_LEFT",
        "cadence_LEFT",
        "pace_LEFT",
        "stride_var_CV",
    ]

    existing_preferred = [c for c in preferred_cols if c in group_df.columns]
    remaining = [c for c in group_df.columns if c not in existing_preferred]
    group_df = group_df[existing_preferred + remaining]

    group_df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved all-terrain spatial-bin dataset to:")
    print(f"  {OUTPUT_CSV}")

    print("\nDataset summary:")
    print(f"  Participants: {group_df['participant'].nunique()}")
    print(f"  Spatial bins: {len(group_df)}")

    if "area_label" in group_df.columns:
        print("\nBins by terrain:")
        print(group_df["area_label"].value_counts(dropna=False))

    if missing:
        print("\nMissing visual_events.csv for participants:")
        for pid in missing:
            print(f"  - {pid}")


if __name__ == "__main__":
    main()