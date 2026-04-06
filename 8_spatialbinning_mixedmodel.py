#!/usr/bin/env python3
"""
6d_prepare_spatialbins_for_mixedmodels.py  (cobblestone ONLY)

Same as original, but:
- Filters to area_label == "cobblestone" BEFORE spatial binning
- Output file gets _cobblestone suffix
"""

import os
import warnings
import numpy as np
import pandas as pd

BASE_DIR   = r"C:/LocoGaze/data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_CSV = os.path.join(GROUP_DIR, "mixedmodel_spatialbins_cobblestone.csv")

# --------- FILTER ---------
TARGET_TERRAIN = "cobblestone"

# Spatial binning precision
EPS = 3

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

def build_total_people(df: pd.DataFrame) -> pd.Series:
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
    s = series.dropna().astype(str)
    if s.empty:
        return np.nan
    return s.value_counts().idxmax()

def main():
    os.makedirs(GROUP_DIR, exist_ok=True)

    try:
        meta = pd.read_csv(META_CSV)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: cannot find {META_CSV}")

    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column missing in metadata_all.csv")

    participants = meta["participant"].dropna().astype(str).str.strip().tolist()
    if not participants:
        raise SystemExit("ERROR: no participants found in metadata_all.csv")

    all_bins = []
    missing  = []

    for pid in participants:
        vis_path = os.path.join(BASE_DIR, pid, "output", "visual_events.csv")
        if not os.path.exists(vis_path):
            missing.append(pid)
            continue

        df = pd.read_csv(vis_path)

        needed = {"is_walking", "latitude", "longitude", "fixated", "depth_looking_floor", "area_label"}
        if not needed.issubset(df.columns):
            warnings.warn(f"{vis_path} missing some of {needed}; skipping this participant.")
            continue

        # Walking-only
        df = df[df["is_walking"] == True].copy()
        if df.empty:
            continue

        # Drop rows without GPS
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        if df.empty:
            continue

        # ---- NEW: restrict to cobblestone samples only ----
        df["area_label"] = df["area_label"].astype(str).str.strip().str.lower()
        df = df[df["area_label"] == TARGET_TERRAIN].copy()
        if df.empty:
            continue

        # ---- Social density at sample level ----
        total_people = build_total_people(df)
        df["total_people"] = total_people

        total_int = total_people.astype(int).clip(lower=0)
        bins = np.where(total_int == 0, "0", "1plus")
        df["people_bin"] = pd.Categorical(bins, categories=["0", "1plus"], ordered=True)

        # ---- Gaze allocation indicators ----
        df["fixated"] = df["fixated"].astype(str).str.strip().str.lower()
        df["depth_looking_floor"] = df["depth_looking_floor"].astype(bool)

        df["looking_people"] = df["fixated"].eq("person").astype(float)
        df["looking_floor"] = (df["fixated"].eq("other") & df["depth_looking_floor"]).astype(float)
        df["looking_environment"] = (df["fixated"].eq("other") & (~df["depth_looking_floor"])).astype(float)

        other_obj_labels = {"car", "cars", "bicycle", "bike", "bicycles"}
        df["looking_other_objects"] = df["fixated"].isin(other_obj_labels).astype(float)

        # ---- Gaze verticality and horizontal eccentricity ----
        df["gaze_x"] = pd.to_numeric(df["gaze_x"], errors="coerce") if "gaze_x" in df.columns else np.nan
        if "gaze_y" in df.columns:
            df["gaze_y"] = pd.to_numeric(df["gaze_y"], errors="coerce")
            df["gaze_y"] = Y_MAX - df["gaze_y"]
        else:
            df["gaze_y"] = np.nan

        df["horizontal_ecc"] = np.abs(df["gaze_x"] - X_CENTER)

        # ---- Spatial binning ----
        df["lat_bin"] = df["latitude"].round(EPS)
        df["lon_bin"] = df["longitude"].round(EPS)

        group_cols = ["lat_bin", "lon_bin"]

        gait_cols_exist = [c for c in GAIT_COLS if c in df.columns]
        gaze_cols_exist = [c for c in GAZE_CONT_COLS if c in df.columns]

        agg_dict = {
            # keep area_label for traceability (constant now)
            "area_label": "first",
            "people_bin": majority_category,
            "total_people": "mean",
            "looking_people": "mean",
            "looking_floor": "mean",
            "looking_environment": "mean",
            "looking_other_objects": "mean",
            "horizontal_ecc": "mean",
            "gaze_y": "mean",
        }

        for c in gait_cols_exist + gaze_cols_exist:
            agg_dict[c] = "mean"

        df_bins = df.groupby(group_cols).agg(agg_dict).reset_index()

        # ----- Stride variability (combined CV) -----
        if {"stride_length_LEFT", "stride_duration_LEFT"}.issubset(df.columns):
            def stride_cv_combined(group):
                len_vals = group["stride_length_LEFT"].dropna()
                dur_vals = group["stride_duration_LEFT"].dropna()
                if len_vals.size < 2 or dur_vals.size < 2:
                    return np.nan
                mean_len = len_vals.mean()
                sd_len   = len_vals.std(ddof=1)
                mean_dur = dur_vals.mean()
                sd_dur   = dur_vals.std(ddof=1)
                if mean_len <= 0 or mean_dur <= 0:
                    return np.nan
                cv_len = sd_len / mean_len
                cv_dur = sd_dur / mean_dur
                return (cv_len + cv_dur) / 2.0

            stride_var = (
                df.groupby(group_cols)
                  .apply(stride_cv_combined)
                  .rename("stride_var_CV")
                  .reset_index()
            )
            df_bins = df_bins.merge(stride_var, on=group_cols, how="left")
        else:
            df_bins["stride_var_CV"] = np.nan

        df_bins = df_bins.rename(columns={"lat_bin": "latitude_bin", "lon_bin": "longitude_bin"})
        df_bins.insert(0, "participant", pid)

        # tidy factors
        df_bins["people_bin"] = df_bins["people_bin"].astype(str).str.strip()
        df_bins["people_bin"] = pd.Categorical(df_bins["people_bin"], categories=["0", "1plus"], ordered=True)
        df_bins["area_label"] = TARGET_TERRAIN  # enforce constant

        all_bins.append(df_bins)

    if not all_bins:
        raise SystemExit("ERROR: no spatial-bin data collected from any participant.")

    group_df = pd.concat(all_bins, ignore_index=True)

    group_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved bin-level dataset (cobblestone only) to: {OUTPUT_CSV}")

    if missing:
        print("Missing visual_events.csv for participants:")
        for pid in missing:
            print(f"  - {pid}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
6f_mixedmodels_followups_spatialbins.py (COBBLESTONE ONLY)

UPDATED (requested):
- Uses input mixedmodel_spatialbins_cobblestone.csv
- No terrain factor anymore
- Model: DV ~ C(people_bin) + (1|participant)
- Outputs get _cobblestone suffix

NEW:
- Extracts beta (β) and SE for the people_bin effect (1plus vs 0) from MixedLM:
    param: C(people_bin)[T.1plus]
- Computes participant-level paired t-test and Cohen's dz using per-participant means
  across spatial bins (avoids pseudo-replication).
- Saves an extra CSV with paired t + dz results.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR   = r"C:/LocoGaze/data"
GROUP_DIR  = os.path.join(BASE_DIR, "group")

INPUT_CSV   = os.path.join(GROUP_DIR, "mixedmodel_spatialbins_cobblestone.csv")

OUT_MAIN    = os.path.join(GROUP_DIR, "mixedmodel_main_APA_cobblestone.csv")
OUT_MMARG   = os.path.join(GROUP_DIR, "mixedmodel_marginal_means_cobblestone.csv")
OUT_PERPART = os.path.join(GROUP_DIR, "mixedmodel_participant_social_means_cobblestone.csv")
OUT_PAIRED  = os.path.join(GROUP_DIR, "mixedmodel_paired_t_dz_cobblestone.csv")

PLOT_DIR   = os.path.join(GROUP_DIR, "mixedmodel_plots_cobblestone")
os.makedirs(PLOT_DIR, exist_ok=True)

DVS = [
    "looking_people",
    "looking_floor",
    "looking_environment",
    "looking_other_objects",

    "radius_of_gyration",
    "spatial_entropy",
    "depth_d_s",
    "depth_head_pitch_deg",
    "depth_d_vergence",
    "depth_calib_mm",
    "horizontal_ecc",
    "gaze_y",

    "mean_rms_LEFT",
    "stride_duration_LEFT",
    "stride_length_LEFT",
    "cadence_LEFT",
    "pace_LEFT",
    "stride_var_CV",
]

def format_p(p):
    if not np.isfinite(p):
        return "p = NA"
    if p < 0.001:
        return "p < .001"
    else:
        return f"p = {p:.3f}".replace("0.", ".")

def format_chi2(chi2, df, p):
    return f"χ²({df}) = {chi2:.2f}, {format_p(p)}"

def wald_joint(mdf, param_indices):
    """
    Joint Wald test for a set of parameters (by index) from statsmodels MixedLMResults.
    Returns (chi2, df, p) or (None, None, None) on failure.
    """
    idx = sorted(set(int(i) for i in param_indices))
    if len(idx) == 0:
        return None, None, None
    p = len(mdf.params)
    C = np.zeros((len(idx), p))
    for row_i, param_i in enumerate(idx):
        C[row_i, param_i] = 1.0
    try:
        wres = mdf.wald_test(C, scalar=False)
        chi2 = float(np.atleast_1d(wres.statistic).ravel()[0])
        df_w = len(idx)
        pval = float(wres.pvalue)
    except Exception:
        return None, None, None
    if not np.isfinite(chi2) or not np.isfinite(pval):
        return None, None, None
    return chi2, df_w, pval

def compute_marginal_means_by_peoplebin(df, dv, people_order):
    g = (
        df.groupby("people_bin")[dv]
          .agg(["mean", "std", "count"])
          .reindex(people_order)
    )
    g["se"] = g["std"] / np.sqrt(g["count"])
    g = g.reset_index()
    return g

def paired_t_and_dz_from_bins(sub_df: pd.DataFrame, dv: str, people_order=("0", "1plus")):
    """
    Participant-level paired t-test + Cohen's dz using per-participant means
    across spatial bins.

    dz = mean(diff) / sd(diff), where diff = mean_dv(1plus) - mean_dv(0) per participant.
    """
    tmp = sub_df[["participant", "people_bin", dv]].dropna().copy()
    if tmp.empty:
        return None

    wide = (tmp.groupby(["participant", "people_bin"])[dv]
              .mean()
              .unstack("people_bin"))

    if not set(people_order).issubset(wide.columns):
        return None

    wide = wide.dropna(subset=list(people_order))
    if wide.shape[0] < 3:
        return None

    d = wide[people_order[1]] - wide[people_order[0]]  # 1plus - 0

    t, p = stats.ttest_rel(wide[people_order[1]], wide[people_order[0]])

    sd = d.std(ddof=1)
    dz = (d.mean() / sd) if (sd is not None and np.isfinite(sd) and sd > 0) else np.nan

    return {
        "n": int(wide.shape[0]),
        "t": float(t),
        "df": int(wide.shape[0] - 1),
        "p": float(p),
        "dz": float(dz),
        "mean_diff": float(d.mean()),
        "sd_diff": float(sd) if np.isfinite(sd) else np.nan,
    }

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"ERROR: cannot find {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = ["participant", "people_bin"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: bin-level file missing required columns: {missing}")

    df["participant"] = df["participant"].astype(str).str.strip()
    df["people_bin"]  = df["people_bin"].astype(str).str.strip()

    people_order = ["0", "1plus"]
    df["people_bin"] = pd.Categorical(df["people_bin"], categories=people_order, ordered=True)

    main_rows = []
    marg_rows = []

    for dv in DVS:
        if dv not in df.columns:
            print(f"Skipping {dv}: not in bin-level dataset.")
            continue

        sub = df.dropna(subset=[dv]).copy()
        if sub.empty:
            print(f"Skipping {dv}: all values are NaN.")
            continue
        if sub["participant"].nunique() < 3:
            print(f"Skipping {dv}: fewer than 3 participants with data.")
            continue

        print(f"Fitting mixed model for DV = {dv} ...")
        formula = f"{dv} ~ C(people_bin)"

        try:
            md = smf.mixedlm(formula, data=sub, groups=sub["participant"])
            mdf = md.fit(reml=False)
        except Exception as e:
            print(f"  -> FAILED for {dv}: {e}")
            continue

        # --- Extract beta & SE for 1plus vs 0 ---
        beta = np.nan
        se_beta = np.nan
        z_beta = np.nan
        p_beta = np.nan
        param_name = "C(people_bin)[T.1plus]"
        if param_name in mdf.params.index:
            beta = float(mdf.params[param_name])
            se_beta = float(mdf.bse[param_name])
            z_beta = float(beta / se_beta) if (np.isfinite(se_beta) and se_beta != 0) else np.nan
            p_beta = float(mdf.pvalues[param_name]) if param_name in mdf.pvalues.index else np.nan

        param_names = list(mdf.params.index)

        # Joint Wald test for people_bin (should be a single parameter: T.1plus)
        idx_people = [i for i, name in enumerate(param_names) if name.startswith("C(people_bin)[T.")]
        chi2, df_w, pval = wald_joint(mdf, idx_people)

        # Paired t-test + dz on participant-level means (bins -> participant means)
        paired = paired_t_and_dz_from_bins(sub, dv, people_order=tuple(people_order))

        if chi2 is not None:
            row = {
                "dv": dv,
                "effect": "people_bin",
                "term": "C(people_bin)",
                "chi2": chi2,
                "df": df_w,
                "p": pval,
                "apa": format_chi2(chi2, df_w, pval),

                # β and SE from MixedLM
                "beta_1plus_vs_0": beta,
                "se_beta": se_beta,
                "z_beta": z_beta,
                "p_beta": p_beta,
            }

            # Paired t and dz (participant-level)
            if paired is not None:
                row.update({
                    "n_paired": paired["n"],
                    "t_paired": paired["t"],
                    "df_paired": paired["df"],
                    "p_paired": paired["p"],
                    "cohens_dz": paired["dz"],
                    "mean_diff_1plus_minus_0": paired["mean_diff"],
                })
            else:
                row.update({
                    "n_paired": np.nan,
                    "t_paired": np.nan,
                    "df_paired": np.nan,
                    "p_paired": np.nan,
                    "cohens_dz": np.nan,
                    "mean_diff_1plus_minus_0": np.nan,
                })

            main_rows.append(row)

        # Marginal means
        mm = compute_marginal_means_by_peoplebin(sub, dv, people_order)
        for _, r in mm.iterrows():
            marg_rows.append({
                "dv": dv,
                "factor": "people_bin",
                "level": r["people_bin"],
                "mean": r["mean"],
                "se": r["se"],
                "n": r["count"],
            })

        # Plot (0 vs 1+)
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            mm2 = mm.set_index("people_bin").reindex(people_order)
            x = np.arange(len(people_order))
            ax.errorbar(x, mm2["mean"].values, yerr=mm2["se"].values, marker="o", linestyle="-", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(["0", "1+"], fontsize=10)
            ax.set_xlabel("Social density (0 vs 1+ people around)")
            ax.set_ylabel(dv)
            ax.set_title(f"{dv} by social density (cobblestone bins)")
            plt.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, f"{dv}_peoplebin_cobblestone.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"  -> Plotting failed for {dv}: {e}")

    # Save main results (χ² + β/SE + paired dz)
    if main_rows:
        pd.DataFrame(main_rows).to_csv(OUT_MAIN, index=False)
        print(f"Saved main APA-style results to: {OUT_MAIN}")
    else:
        print("No main results were produced.")

    # Save marginal means
    if marg_rows:
        marg_df = pd.DataFrame(marg_rows)[["dv", "factor", "level", "mean", "se", "n"]]
        marg_df.to_csv(OUT_MMARG, index=False)
        print(f"Saved marginal means to: {OUT_MMARG}")

    # Save paired t-test + dz table (clean, one row per DV)
    if main_rows:
        paired_cols = [
            "dv",
            "n_paired",
            "t_paired",
            "df_paired",
            "p_paired",
            "cohens_dz",
            "mean_diff_1plus_minus_0",
        ]
        paired_df = pd.DataFrame(main_rows)
        paired_df = paired_df[[c for c in paired_cols if c in paired_df.columns]]
        paired_df.to_csv(OUT_PAIRED, index=False)
        print(f"Saved paired t-test + dz results to: {OUT_PAIRED}")

    # Per-participant means by people_bin (wide)
    dvs_exist = [dv for dv in DVS if dv in df.columns]
    rows = []
    for pid, df_p in df.groupby("participant"):
        row = {"participant": pid}
        for pb in people_order:
            df_pb = df_p[df_p["people_bin"] == pb]
            for dv in dvs_exist:
                row[f"{dv}_{pb}"] = df_pb[dv].mean(skipna=True) if not df_pb.empty else np.nan
        rows.append(row)

    if rows:
        perpart_df = pd.DataFrame(rows)
        cols = ["participant"] + [f"{dv}_{pb}" for dv in dvs_exist for pb in people_order]
        for c in cols:
            if c not in perpart_df.columns:
                perpart_df[c] = np.nan
        perpart_df = perpart_df[cols]
        perpart_df.to_csv(OUT_PERPART, index=False)
        print(f"Saved per-participant social means to: {OUT_PERPART}")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
6g_mixedmodels_posthoc_pairwise.py (cobblestone ONLY)

UPDATED:
- Uses mixedmodel_spatialbins_cobblestone.csv
- Only tests people_bin (0 vs 1plus)
- Output gets _cobblestone suffix
"""

import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE_DIR   = r"C:/LocoGaze/data"
GROUP_DIR  = os.path.join(BASE_DIR, "group")
INPUT_CSV  = os.path.join(GROUP_DIR, "mixedmodel_spatialbins_cobblestone.csv")
OUT_POSTHOC = os.path.join(GROUP_DIR, "mixedmodel_posthoc_pairwise_cobblestone.csv")

DVS = [
    "looking_people",
    "looking_floor",
    "looking_environment",
    "looking_other_objects",

    "radius_of_gyration",
    "spatial_entropy",
    "depth_d_s",
    "depth_head_pitch_deg",
    "depth_d_vergence",
    "depth_calib_mm",
    "horizontal_ecc",
    "gaze_y",

    "mean_rms_LEFT",
    "stride_duration_LEFT",
    "stride_length_LEFT",
    "cadence_LEFT",
    "pace_LEFT",
    "stride_var_CV",
]

def format_p(p):
    if p < 0.001:
        return "p < .001"
    else:
        return f"p = {p:.3f}".replace("0.", ".")

def format_chi2(chi2, df, p):
    return f"χ²({df}) = {chi2:.2f}, {format_p(p)}"

def run_pairwise_mixedlm(sub_df, dv, factor, level1, level2):
    df2 = sub_df[sub_df[factor].isin([level1, level2])].copy()
    if df2[factor].nunique() < 2:
        return None
    if df2["participant"].nunique() < 3:
        return None

    df2[factor] = pd.Categorical(df2[factor], categories=[level1, level2], ordered=True)
    formula = f"{dv} ~ C({factor})"

    try:
        md = smf.mixedlm(formula, data=df2, groups=df2["participant"])
        mdf = md.fit(reml=False)
    except Exception as e:
        print(f"  -> Pairwise model FAILED for {dv}: {e}")
        return None

    param_name = f"C({factor})[T.{level2}]"
    if param_name not in mdf.params.index:
        return None

    est = float(mdf.params[param_name])
    se  = float(mdf.bse[param_name])
    z   = float(est / se) if (se is not None and se != 0 and np.isfinite(se)) else np.nan
    p   = float(mdf.pvalues[param_name])
    chi2 = float(z**2) if np.isfinite(z) else np.nan

    if not np.isfinite(chi2) or not np.isfinite(p):
        return None

    return {"estimate": est, "se": se, "z": z, "chi2": chi2, "df": 1, "p_raw": p}

def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"ERROR: cannot find {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = ["participant", "people_bin"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: bin-level file missing required columns: {missing}")

    df["participant"] = df["participant"].astype(str).str.strip()
    df["people_bin"]  = df["people_bin"].astype(str).str.strip()

    people_order = ["0", "1plus"]
    df["people_bin"] = pd.Categorical(df["people_bin"], categories=people_order, ordered=True)

    results_rows = []

    for dv in DVS:
        if dv not in df.columns:
            print(f"Skipping {dv}: not in dataset.")
            continue

        sub = df.dropna(subset=[dv]).copy()
        if sub.empty or sub["participant"].nunique() < 3:
            continue

        res = run_pairwise_mixedlm(sub, dv, "people_bin", "0", "1plus")
        if res is None:
            continue

        direction = ">" if res["estimate"] > 0 else "<" if res["estimate"] < 0 else "="
        results_rows.append({
            "dv": dv,
            "factor": "people_bin",
            "level1": "0",
            "level2": "1plus",
            "estimate": res["estimate"],
            "se": res["se"],
            "z": res["z"],
            "chi2": res["chi2"],
            "df": res["df"],
            "p_raw": res["p_raw"],
            "direction": direction,
            "apa_chi2": format_chi2(res["chi2"], 1, res["p_raw"]),
        })

    if not results_rows:
        print("No post-hoc results were produced.")
        return

    res_df = pd.DataFrame(results_rows)
    res_df.to_csv(OUT_POSTHOC, index=False)
    print(f"Saved post-hoc pairwise results to: {OUT_POSTHOC}")

if __name__ == "__main__":
    main()





