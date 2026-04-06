#!/usr/bin/env python3
"""
load_visual_events.py — environment object proportions (people now INCLUDE cyclists)

UPDATED:
- Filters analysis to ONE terrain only via TARGET_TERRAIN (flat or cobblestone)
"""

import os
import warnings
import numpy as np
import pandas as pd

# ============================================================
# TERRAIN SELECTION (edit this one line)
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

VALID_TERRAINS = {"flat", "cobblestone", "green"}

def filter_to_terrain(df: pd.DataFrame, col: str = "area_label") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' for terrain filter.")
    if TARGET_TERRAIN not in VALID_TERRAINS:
        raise ValueError(f"TARGET_TERRAIN must be one of {sorted(VALID_TERRAINS)}; got: {TARGET_TERRAIN}")

    lab = df[col].astype(str).str.strip().str.lower()
    out = df.loc[lab == TARGET_TERRAIN].copy()
    out[col] = TARGET_TERRAIN
    return out

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df  = pd.read_csv(META_CSV, nrows=1)
reldir   = meta_df.at[0, 'reldir']

# ---- Paths ----
BASE_DIR       = r'C:/LocoGaze/data/'
INPUT_DIR      = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR     = os.path.join(INPUT_DIR, 'stats')
VISUAL_CSV     = os.path.join(INPUT_DIR, 'visual_events.csv')
PROPORTION_CSV = os.path.join(OUTPUT_DIR, f'environment_objects_proportion_{TARGET_TERRAIN}.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load & pre-filter ----
vis_df = pd.read_csv(VISUAL_CSV)

# Keep only the first row of each continuous event (where event_duration changes)
vis_df = vis_df.loc[vis_df['event_duration'].ne(vis_df['event_duration'].shift())].reset_index(drop=True)

# Only walking + fixation samples
event_df = vis_df[(vis_df['is_walking'] == True) & (vis_df['event_label'] == 'FIXA')].copy()

# NEW: terrain filter
event_df = filter_to_terrain(event_df, col="area_label")

# Duration for weighting
event_df['duration'] = pd.to_numeric(event_df['event_duration'], errors='coerce').fillna(0.0)
if event_df['duration'].sum() <= 0:
    warnings.warn("Total duration is zero after filtering; outputs will be empty/zero.")

# ---- Required columns (presence + labels) ----
required_bool = [
    'people_present',
    'cyclist_present',
    'bicycle_present',
    'car_present',
    'depth_looking_floor',
    'looking_cyclist',
]
required_any = required_bool + ['fixated', 'area_label', 'duration']
missing = [c for c in required_any if c not in event_df.columns]
if missing:
    raise KeyError(f"Missing columns in visual_events.csv: {missing}")

# Optional count columns (for averages)
has_people_total      = 'number_people' in event_df.columns
has_people_nocyc_cnt  = 'number_people_nocyclist' in event_df.columns
has_cyclist_cnt       = 'number_cyclist' in event_df.columns
has_bike_standalone   = 'number_bicycle_standalone' in event_df.columns

# sanitize dtypes
for c in required_bool:
    event_df[c] = event_df[c].astype(bool)

# ---- People-inclusive masks/counters ----
event_df['people_any_present'] = event_df['people_present'] | event_df['cyclist_present']

fix_people_any   = (event_df['fixated'] == 'person')
fix_cyclist      = event_df['looking_cyclist']
fix_bicycle_st   = (event_df['fixated'] == 'bicycle') & (event_df['bicycle_present'])
fix_car          = (event_df['fixated'] == 'car')
fix_floor        = (event_df['fixated'] == 'other') & (event_df['depth_looking_floor'])
fix_bg           = (event_df['fixated'] == 'other') & (~event_df['depth_looking_floor'])

# Total people count (prefer explicit total; else sum components if available)
num_people_total = None
if has_people_total:
    num_people_total = pd.to_numeric(event_df['number_people'], errors='coerce').fillna(0.0)
elif has_people_nocyc_cnt and has_cyclist_cnt:
    num_people_total = (
        pd.to_numeric(event_df['number_people_nocyclist'], errors='coerce').fillna(0.0) +
        pd.to_numeric(event_df['number_cyclist'], errors='coerce').fillna(0.0)
    )

def wmean(mask: pd.Series, dur: pd.Series) -> float:
    denom = float(dur.sum())
    if denom <= 0:
        return 0.0
    return float(dur[mask].sum()) / denom

def conditional_wmean(mask_present: pd.Series, mask_fix: pd.Series, dur: pd.Series) -> float:
    if not mask_present.any():
        return 0.0
    dsel = dur[mask_present]
    if dsel.sum() <= 0:
        return 0.0
    fix_sel = mask_fix & mask_present
    return float(dur[fix_sel].sum()) / float(dsel.sum())

def duration_weighted_average_count(count_series: pd.Series | None,
                                    present_mask: pd.Series,
                                    dur: pd.Series):
    if (count_series is None) or (not present_mask.any()):
        return None
    dsel = dur[present_mask]
    if dsel.sum() <= 0:
        return None
    cnt = pd.to_numeric(count_series, errors='coerce').fillna(0.0)
    cnt_sel = cnt[present_mask]
    return float((cnt_sel * dsel).sum() / dsel.sum())

def compute_environment_object_proportions(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    def push(obj, area, pres, fix, cond_fix, avg_num):
        out.append({
            'object': obj,
            'area_label': area,
            'proportion_present': pres,
            'proportion_fixated': fix,
            'proportion_fixated_when_present': cond_fix,
            'avg_number_when_present': avg_num
        })

    dur = df['duration']

    # ---------- Overall (area = TARGET_TERRAIN) ----------
    pres_people  = wmean(df['people_any_present'], dur)
    pres_cyc     = wmean(df['cyclist_present'],    dur)
    pres_bike    = wmean(df['bicycle_present'],    dur)
    pres_car     = wmean(df['car_present'],        dur)
    pres_floor   = 1.0
    pres_bg      = 1.0

    fix_people_o = wmean(fix_people_any,  dur)
    fix_cyc_o    = wmean(fix_cyclist,     dur)
    fix_bike_o   = wmean(fix_bicycle_st,  dur)
    fix_car_o    = wmean(fix_car,         dur)
    fix_floor_o  = wmean(fix_floor,       dur)
    fix_bg_o     = wmean(fix_bg,          dur)

    cond_fix_people = conditional_wmean(df['people_any_present'], fix_people_any, dur)
    cond_fix_cyc    = conditional_wmean(df['cyclist_present'],    fix_cyclist,   dur)
    cond_fix_bike   = conditional_wmean(df['bicycle_present'],    fix_bicycle_st, dur)
    cond_fix_car    = conditional_wmean(df['car_present'],        fix_car,       dur)

    avg_people = duration_weighted_average_count(num_people_total, df['people_any_present'], dur)
    avg_bike   = duration_weighted_average_count(
        df['number_bicycle_standalone'] if has_bike_standalone else None,
        df['bicycle_present'], dur
    )

    push('people',     TARGET_TERRAIN, pres_people, fix_people_o, cond_fix_people, avg_people)
    push('cyclist',    TARGET_TERRAIN, pres_cyc,    fix_cyc_o,    cond_fix_cyc,    None)
    push('bicycle',    TARGET_TERRAIN, pres_bike,   fix_bike_o,   cond_fix_bike,   avg_bike)
    push('car',        TARGET_TERRAIN, pres_car,    fix_car_o,    cond_fix_car,    None)
    push('floor',      TARGET_TERRAIN, pres_floor,  fix_floor_o,  None,            None)
    push('background', TARGET_TERRAIN, pres_bg,     fix_bg_o,     None,            None)

    return pd.DataFrame(out)

if __name__ == '__main__':
    prop_df = compute_environment_object_proportions(event_df)
    prop_df.to_csv(PROPORTION_CSV, index=False)
    print(f"Saved environment object proportions to {PROPORTION_CSV}")


#!/usr/bin/env python3
"""
compute_rog_entropy_social.py

UPDATED:
- Filters analysis to ONE terrain only via TARGET_TERRAIN
- Social = people_present OR cyclist_present
"""

import os
import pandas as pd

# ============================================================
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

VALID_TERRAINS = {"flat", "cobblestone", "green"}

def filter_to_terrain(df: pd.DataFrame, col: str = "area_label") -> pd.DataFrame:
    if TARGET_TERRAIN not in VALID_TERRAINS:
        raise ValueError(f"TARGET_TERRAIN must be one of {sorted(VALID_TERRAINS)}; got: {TARGET_TERRAIN}")
    lab = df[col].astype(str).str.strip().str.lower()
    out = df.loc[lab == TARGET_TERRAIN].copy()
    out[col] = TARGET_TERRAIN
    return out

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df  = pd.read_csv(META_CSV, nrows=1)
reldir   = meta_df.at[0, 'reldir']

# ---- Paths ----
BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'stats')
INPUT_CSV  = os.path.join(INPUT_DIR, 'visual_events.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f'rog_entropy_stats_social_{TARGET_TERRAIN}.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_COLS = ['radius_of_gyration', 'spatial_entropy']

df = pd.read_csv(INPUT_CSV)

required = ['is_walking', 'people_present', 'cyclist_present', 'area_label'] + METRIC_COLS
missing  = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"visual_events.csv missing required columns: {missing}")

df_walk = df[df['is_walking'] == True].copy()

# NEW: terrain filter
df_walk = filter_to_terrain(df_walk, col="area_label")

# SOCIAL = any people (non-cyclists OR cyclists)
df_walk['social'] = df_walk['people_present'].astype(bool) | df_walk['cyclist_present'].astype(bool)

grouped = (
    df_walk.groupby(['social'])[METRIC_COLS]
           .mean()
           .reset_index()
)
grouped['area_label'] = TARGET_TERRAIN

grouped_both = df_walk[METRIC_COLS].mean().to_frame().T
grouped_both['area_label'] = TARGET_TERRAIN
grouped_both['social'] = 'both'

result = pd.concat([grouped, grouped_both], ignore_index=True)

cols = ['area_label', 'social'] + METRIC_COLS
result = result[cols]
result.to_csv(OUTPUT_CSV, index=False)
print(f"Saved RoG & entropy stats (terrain={TARGET_TERRAIN}) to {OUTPUT_CSV}")


#!/usr/bin/env python3
"""
compute_depth_stats_social.py

UPDATED:
- Filters analysis to ONE terrain only via TARGET_TERRAIN
- Social = people_present OR cyclist_present
- Filters to walking AND depth_looking_floor == True
"""

import os
import pandas as pd

# ============================================================
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

VALID_TERRAINS = {"flat", "cobblestone", "green"}

def filter_to_terrain(df: pd.DataFrame, col: str = "area_label") -> pd.DataFrame:
    if TARGET_TERRAIN not in VALID_TERRAINS:
        raise ValueError(f"TARGET_TERRAIN must be one of {sorted(VALID_TERRAINS)}; got: {TARGET_TERRAIN}")
    lab = df[col].astype(str).str.strip().str.lower()
    out = df.loc[lab == TARGET_TERRAIN].copy()
    out[col] = TARGET_TERRAIN
    return out

# ---- Load metadata ----
META_CSV = r'C:/LocoGaze/data/metadata.csv'
meta_df  = pd.read_csv(META_CSV, nrows=1)
reldir   = meta_df.at[0, 'reldir']

# ---- Paths ----
BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'stats')
INPUT_CSV  = os.path.join(INPUT_DIR, 'visual_events.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f'gazedepth_stats_social_{TARGET_TERRAIN}.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRIC_COLS = [
    'depth_d_s',
    'depth_head_pitch_deg',
    'depth_d_vergence',
    'depth_calib_mm',
]

df = pd.read_csv(INPUT_CSV)

required = ['is_walking', 'depth_looking_floor', 'people_present', 'cyclist_present', 'area_label'] + METRIC_COLS
missing  = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"visual_events.csv missing required columns: {missing}")

df_walk = df[(df['is_walking'] == True) & (df['depth_looking_floor'] == True)].copy()

# NEW: terrain filter
df_walk = filter_to_terrain(df_walk, col="area_label")

df_walk['social'] = df_walk['people_present'].astype(bool) | df_walk['cyclist_present'].astype(bool)

grouped = (
    df_walk.groupby(['social'])[METRIC_COLS]
           .mean()
           .reset_index()
)
grouped['area_label'] = TARGET_TERRAIN

grouped_both = df_walk[METRIC_COLS].mean().to_frame().T
grouped_both['area_label'] = TARGET_TERRAIN
grouped_both['social'] = 'both'

result = pd.concat([grouped, grouped_both], ignore_index=True)
cols = ['area_label', 'social'] + METRIC_COLS
result = result[cols]

result.to_csv(OUTPUT_CSV, index=False)
print(f"Saved gaze depth & head pitch stats (terrain={TARGET_TERRAIN}) to {OUTPUT_CSV}")

#!/usr/bin/env python3
"""
compute_gait_stats_social.py (refined, total-people bins)

UPDATED:
- Filters analysis to ONE terrain only via TARGET_TERRAIN
- Part 1: means/variances by social (present/absent)
- Part 2: means/variances by total-people bins (0/1/2plus)
"""

import os
import warnings
import pandas as pd
import numpy as np

# ============================================================
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

VALID_TERRAINS = {"flat", "cobblestone", "green"}

def filter_to_terrain(df: pd.DataFrame, col: str = "area_label") -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' for terrain filter.")
    if TARGET_TERRAIN not in VALID_TERRAINS:
        raise ValueError(f"TARGET_TERRAIN must be one of {sorted(VALID_TERRAINS)}; got: {TARGET_TERRAIN}")
    lab = df[col].astype(str).str.strip().str.lower()
    out = df.loc[lab == TARGET_TERRAIN].copy()
    out[col] = TARGET_TERRAIN
    return out

# ---- PATHS ----
META_CSV   = r'C:/LocoGaze/data/metadata.csv'
meta_df    = pd.read_csv(META_CSV, nrows=1)
reldir     = meta_df.at[0, 'reldir']

BASE_DIR   = r'C:/LocoGaze/data/'
INPUT_DIR  = os.path.join(BASE_DIR, reldir, 'output')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'stats')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV          = os.path.join(INPUT_DIR, 'visual_events.csv')
OUTPUT_CSV_SOCIAL  = os.path.join(OUTPUT_DIR, f'gait_stats_social_{TARGET_TERRAIN}.csv')
OUTPUT_CSV_BINS    = os.path.join(OUTPUT_DIR, f'gait_stats_peoplebins_{TARGET_TERRAIN}.csv')

GAIT_COLS = [
    'mean_rms_LEFT',
    'stride_duration_LEFT',
    'stride_length_LEFT',
    'cadence_LEFT',
    'pace_LEFT'
]
VAR_BASE_COLS = ['stride_length_LEFT', 'stride_duration_LEFT']

df = pd.read_csv(INPUT_CSV)

if 'is_walking' not in df.columns:
    raise KeyError("visual_events.csv missing required column 'is_walking'")
if 'area_label' not in df.columns:
    raise KeyError("visual_events.csv missing required column 'area_label'")

df_walk = df[df['is_walking'] == True].copy()

# NEW: terrain filter
df_walk = filter_to_terrain(df_walk, col="area_label")

# ---- SOCIAL FLAG (any person present: pedestrians + cyclists) ----
if 'people_present' in df_walk.columns and 'cyclist_present' in df_walk.columns:
    df_walk['social'] = df_walk['people_present'].astype(bool) | df_walk['cyclist_present'].astype(bool)
elif 'people_present' in df_walk.columns:
    df_walk['social'] = df_walk['people_present'].astype(bool)
    warnings.warn("cyclist_present missing; social = people_present only.")
else:
    has_np  = 'number_people' in df_walk.columns
    has_npn = 'number_people_nocyclist' in df_walk.columns
    has_nc  = 'number_cyclist' in df_walk.columns

    if has_np:
        df_walk['social'] = pd.to_numeric(df_walk['number_people'], errors='coerce').fillna(0) > 0
        warnings.warn("people_present missing: using number_people > 0 for social flag.")
    elif has_npn and has_nc:
        total_people = (
            pd.to_numeric(df_walk['number_people_nocyclist'], errors='coerce').fillna(0) +
            pd.to_numeric(df_walk['number_cyclist'],        errors='coerce').fillna(0)
        )
        df_walk['social'] = total_people > 0
        warnings.warn("people_present missing: reconstructed total people from nocyclist + cyclist.")
    else:
        df_walk['social'] = False
        warnings.warn("No people_present/number_people columns found; social set to False for all rows.")

# ---- Ensure GAIT_COLS exist ----
existing_gait_cols = [c for c in GAIT_COLS if c in df_walk.columns]
missing_gait_cols  = [c for c in GAIT_COLS if c not in df_walk.columns]
if missing_gait_cols:
    warnings.warn(f"Missing gait columns (will be skipped): {missing_gait_cols}")
GAIT_COLS = existing_gait_cols

# --------------------------
# PART 1: by social (means + variances)
# --------------------------
grouped_mean = (
    df_walk
    .groupby(['social'])[GAIT_COLS]
    .mean()
)

var_cols = [c for c in VAR_BASE_COLS if c in df_walk.columns]
if var_cols:
    grouped_var = (
        df_walk
        .groupby(['social'])[var_cols]
        .var(ddof=1)
        .rename(columns={
            'stride_length_LEFT': 'stride_length_LEFT_var',
            'stride_duration_LEFT': 'stride_duration_LEFT_var'
        })
    )
    grouped = grouped_mean.join(grouped_var, how='left').reset_index()
else:
    grouped = grouped_mean.reset_index()

grouped['area_label'] = TARGET_TERRAIN

out_cols = ['area_label', 'social'] + GAIT_COLS
for extra in ['stride_length_LEFT_var', 'stride_duration_LEFT_var']:
    if extra in grouped.columns:
        out_cols.append(extra)
grouped = grouped[out_cols]

grouped.to_csv(OUTPUT_CSV_SOCIAL, index=False)
print(f"Saved gait stats by social (terrain={TARGET_TERRAIN}) to {OUTPUT_CSV_SOCIAL}")

# --------------------------
# PART 2: by people_bin (0/1/2plus)
# --------------------------
# Build total_people_count (pedestrians + cyclists)
if 'number_people' in df_walk.columns:
    total_people_count = pd.to_numeric(df_walk['number_people'], errors='coerce').fillna(0).clip(lower=0)
elif ('number_people_nocyclist' in df_walk.columns) and ('number_cyclist' in df_walk.columns):
    total_people_count = (
        pd.to_numeric(df_walk['number_people_nocyclist'], errors='coerce').fillna(0) +
        pd.to_numeric(df_walk['number_cyclist'], errors='coerce').fillna(0)
    ).clip(lower=0)
    warnings.warn("number_people missing; reconstructed total people as nocyclist + cyclist.")
elif 'number_people_nocyclist' in df_walk.columns:
    total_people_count = pd.to_numeric(df_walk['number_people_nocyclist'], errors='coerce').fillna(0).clip(lower=0)
    warnings.warn("Only number_people_nocyclist present; using this as total people (cyclists may be undercounted).")
elif 'number_cyclist' in df_walk.columns:
    total_people_count = pd.to_numeric(df_walk['number_cyclist'], errors='coerce').fillna(0).clip(lower=0)
    warnings.warn("Only number_cyclist present; using cyclists as total people (pedestrians may be undercounted).")
else:
    total_people_count = pd.Series(0, index=df_walk.index, dtype=float)
    warnings.warn("No people count columns found; assuming 0 total people for binning.")

total_people_int = total_people_count.astype(int).clip(lower=0)
bins = np.where(
    total_people_int == 0, '0',
    np.where(total_people_int == 1, '1', '2plus')
)
df_walk['people_bin'] = pd.Categorical(bins, categories=['0', '1', '2plus'], ordered=True)

grouped_pb_mean = (
    df_walk
    .groupby(['people_bin'])[GAIT_COLS]
    .mean()
)

if var_cols:
    grouped_pb_var = (
        df_walk
        .groupby(['people_bin'])[var_cols]
        .var(ddof=1)
        .rename(columns={
            'stride_length_LEFT': 'stride_length_LEFT_var',
            'stride_duration_LEFT': 'stride_duration_LEFT_var'
        })
    )
    grouped_pb = grouped_pb_mean.join(grouped_pb_var, how='left').reset_index()
else:
    grouped_pb = grouped_pb_mean.reset_index()

grouped_pb['area_label'] = TARGET_TERRAIN

out_cols_bins = ['area_label', 'people_bin'] + GAIT_COLS
for extra in ['stride_length_LEFT_var', 'stride_duration_LEFT_var']:
    if extra in grouped_pb.columns:
        out_cols_bins.append(extra)
grouped_pb = grouped_pb[out_cols_bins]

grouped_pb.to_csv(OUTPUT_CSV_BINS, index=False)
print(f"Saved gait stats by TOTAL-people bins (terrain={TARGET_TERRAIN}) to {OUTPUT_CSV_BINS}")

#!/usr/bin/env python3
"""
collect_group_gaze_peoplebins.py

UPDATED:
- Reads only the row for TARGET_TERRAIN (flat or cobblestone)
- Produces group wide CSV with bin0_/bin1_/bin2plus_ prefixes
"""

import os
import warnings
import pandas as pd

# ============================================================
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

BASE_DIR   = r"C:/LocoGaze/data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_CSV = os.path.join(GROUP_DIR, f"group_gaze_peoplebins_{TARGET_TERRAIN}.csv")

def collect_gaze_peoplebins_wide():
    os.makedirs(GROUP_DIR, exist_ok=True)

    meta = pd.read_csv(META_CSV)
    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column not found in metadata_all.csv")

    participants = meta["participant"].dropna().astype(str).str.strip().tolist()
    if not participants:
        raise SystemExit("ERROR: No participants found in 'participant' column.")

    collected_rows = []
    missing_files = []
    seen_metrics, all_metrics = set(), []

    ORDERED_BINS = ["0", "1", "2plus"]

    for pid in participants:
        csv_path = os.path.join(BASE_DIR, pid, "output", "stats", f"gaze_stats_peoplebins_{TARGET_TERRAIN}.csv")
        if not os.path.exists(csv_path):
            # fallback (if your single-subject script still uses old filename)
            csv_path = os.path.join(BASE_DIR, pid, "output", "stats", "gaze_stats_peoplebins.csv")

        if not os.path.exists(csv_path):
            missing_files.append(pid)
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python")

        if "people_bin" not in df.columns:
            warnings.warn(f"{csv_path} missing required column 'people_bin'; skipping.")
            continue
        if "area_label" not in df.columns:
            warnings.warn(f"{csv_path} missing required column 'area_label'; skipping.")
            continue

        # Keep only TARGET_TERRAIN
        sub = df[df["area_label"].astype(str).str.strip().str.lower() == TARGET_TERRAIN].copy()
        if sub.empty:
            continue

        sub["people_bin"] = sub["people_bin"].astype(str).str.strip()
        sub = sub[sub["people_bin"].isin(ORDERED_BINS)].copy()
        if sub.empty:
            continue

        exclude = {"area_label", "people_bin"}
        candidate_cols = [c for c in sub.columns if c not in exclude]
        tmp = sub[candidate_cols].apply(pd.to_numeric, errors="coerce")
        numeric_metrics = [c for c in tmp.columns if tmp[c].notna().any()]
        if not numeric_metrics:
            continue

        tmp["people_bin"] = sub["people_bin"].values
        agg = tmp.groupby("people_bin", as_index=False).mean(numeric_only=True)

        row = {"participant": pid}
        for _, r in agg.iterrows():
            bin_tag = str(r["people_bin"])
            prefix = "bin0_" if bin_tag == "0" else ("bin1_" if bin_tag == "1" else "bin2plus_")
            for m in numeric_metrics:
                row[f"{prefix}{m}"] = r[m]

        for m in numeric_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_metrics.append(m)

        collected_rows.append(row)

    if not collected_rows:
        raise SystemExit("ERROR: No data collected for gaze peoplebins.")

    group_df = pd.DataFrame(collected_rows)

    desired_cols = ["participant"]
    for bin_tag in ["0", "1", "2plus"]:
        prefix = "bin0_" if bin_tag == "0" else ("bin1_" if bin_tag == "1" else "bin2plus_")
        for m in all_metrics:
            desired_cols.append(f"{prefix}{m}")

    for c in desired_cols:
        if c not in group_df.columns:
            group_df[c] = pd.NA
    group_df = group_df[desired_cols].sort_values("participant", ignore_index=True)

    group_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved group gaze peoplebins (terrain={TARGET_TERRAIN}) to: {OUTPUT_CSV}")

    if missing_files:
        print("Missing per-participant gaze peoplebins file for:")
        for pid in missing_files:
            print(f"  - {pid}")

if __name__ == "__main__":
    collect_gaze_peoplebins_wide()

#!/usr/bin/env python3
"""
collect_group_pace_social.py (wide)

UPDATED:
- Uses TARGET_TERRAIN and keeps only that terrain instead of area_label == 'all'
- Writes terrain-specific group CSVs
"""

import os
import re
import warnings
import pandas as pd

# ============================================================
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

BASE_DIR   = r"C:\LocoGaze\data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_SOCIAL_CSV = os.path.join(GROUP_DIR, f"group_pace_social_{TARGET_TERRAIN}.csv")
OUTPUT_BINS_CSV   = os.path.join(GROUP_DIR, f"group_gait_peoplebins_{TARGET_TERRAIN}.csv")

def coerce_social_to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str).str.strip().str.lower()
         .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
         .astype("boolean").fillna(False).astype(bool)
    )

def rename_drop_left_suffix(col: str) -> str:
    if col == "participant":
        return col
    return re.sub(r"_LEFT$", "", col)

def collect_social_wide():
    os.makedirs(GROUP_DIR, exist_ok=True)
    meta = pd.read_csv(META_CSV)
    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column not found in metadata_all.csv")
    participants = meta["participant"].dropna().astype(str).str.strip().tolist()

    collected_rows, missing_files = [], []
    all_metrics_list, seen_metrics = [], set()

    for pid in participants:
        # prefer new terrain-specific filename, fallback to old
        csv_path = os.path.join(BASE_DIR, pid, "output", "stats", f"gait_stats_social_{TARGET_TERRAIN}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(BASE_DIR, pid, "output", "stats", "gait_stats_social.csv")

        if not os.path.exists(csv_path):
            missing_files.append(pid)
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python")

        if "area_label" not in df.columns and "areal_label" in df.columns:
            df = df.rename(columns={"areal_label": "area_label"})

        if "social" not in df.columns or "area_label" not in df.columns:
            warnings.warn(f"{csv_path} missing required columns; skipping.")
            continue

        # Keep only TARGET_TERRAIN
        sub = df[df["area_label"].astype(str).str.strip().str.lower() == TARGET_TERRAIN].copy()
        if sub.empty:
            continue

        sub["social"] = coerce_social_to_bool(sub["social"])

        exclude = {"area_label", "areal_label", "social"}
        candidate_cols = [c for c in sub.columns if c not in exclude]
        tmp = sub[candidate_cols].apply(pd.to_numeric, errors="coerce")
        numeric_metrics = [c for c in tmp.columns if tmp[c].notna().any()]
        if not numeric_metrics:
            continue

        tmp["social"] = sub["social"].values
        agg = tmp.groupby("social", as_index=False).mean(numeric_only=True)

        row = {"participant": pid}
        for _, r in agg.iterrows():
            lvl_prefix = "present_" if bool(r["social"]) else "absent_"
            for m in numeric_metrics:
                row[f"{lvl_prefix}{m}"] = r[m]

        for m in numeric_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_metrics_list.append(m)

        collected_rows.append(row)

    if not collected_rows:
        raise SystemExit("ERROR: No data collected for Part A (social).")

    group_df = pd.DataFrame(collected_rows)

    desired_cols = ["participant"] + \
        [f"absent_{m}" for m in all_metrics_list] + \
        [f"present_{m}" for m in all_metrics_list]

    for c in desired_cols:
        if c not in group_df.columns:
            group_df[c] = pd.NA
    group_df = group_df[desired_cols].sort_values("participant", ignore_index=True)

    group_df = group_df.rename(columns={c: rename_drop_left_suffix(c) for c in group_df.columns})

    group_df.to_csv(OUTPUT_SOCIAL_CSV, index=False)
    print(f"Saved group social (terrain={TARGET_TERRAIN}) to: {OUTPUT_SOCIAL_CSV}")

    if missing_files:
        print("Missing gait_stats_social file for:")
        for pid in missing_files:
            print(f"  - {pid}")

def collect_peoplebins_wide():
    os.makedirs(GROUP_DIR, exist_ok=True)
    meta = pd.read_csv(META_CSV)
    if "participant" not in meta.columns:
        raise SystemExit("ERROR: 'participant' column not found in metadata_all.csv")
    participants = meta["participant"].dropna().astype(str).str.strip().tolist()

    collected_rows, missing_files = [], []
    seen_metrics, all_metrics = set(), []
    ORDERED_BINS = ["0", "1", "2plus"]

    for pid in participants:
        csv_path = os.path.join(BASE_DIR, pid, "output", "stats", f"gait_stats_peoplebins_{TARGET_TERRAIN}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(BASE_DIR, pid, "output", "stats", "gait_stats_peoplebins.csv")

        if not os.path.exists(csv_path):
            missing_files.append(pid)
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python")

        if "area_label" not in df.columns and "areal_label" in df.columns:
            df = df.rename(columns={"areal_label": "area_label"})

        if "people_bin" not in df.columns or "area_label" not in df.columns:
            warnings.warn(f"{csv_path} missing required columns; skipping.")
            continue

        sub = df[df["area_label"].astype(str).str.strip().str.lower() == TARGET_TERRAIN].copy()
        if sub.empty:
            continue

        sub["people_bin"] = sub["people_bin"].astype(str).str.strip()
        sub = sub[sub["people_bin"].isin(ORDERED_BINS)].copy()
        if sub.empty:
            continue

        exclude = {"area_label", "areal_label", "people_bin"}
        candidate_cols = [c for c in sub.columns if c not in exclude]
        tmp = sub[candidate_cols].apply(pd.to_numeric, errors="coerce")
        numeric_metrics = [c for c in tmp.columns if tmp[c].notna().any()]
        if not numeric_metrics:
            continue

        tmp["people_bin"] = sub["people_bin"].values
        agg = tmp.groupby("people_bin", as_index=False).mean(numeric_only=True)

        row = {"participant": pid}
        for _, r in agg.iterrows():
            b = str(r["people_bin"])
            prefix = "bin0_" if b == "0" else ("bin1_" if b == "1" else "bin2plus_")
            for m in numeric_metrics:
                row[f"{prefix}{m}"] = r[m]

        for m in numeric_metrics:
            if m not in seen_metrics:
                seen_metrics.add(m)
                all_metrics.append(m)

        collected_rows.append(row)

    if not collected_rows:
        raise SystemExit("ERROR: No data collected for Part B (people bins).")

    group_df = pd.DataFrame(collected_rows)

    desired_cols = ["participant"]
    for b in ["0", "1", "2plus"]:
        prefix = "bin0_" if b == "0" else ("bin1_" if b == "1" else "bin2plus_")
        for m in all_metrics:
            desired_cols.append(f"{prefix}{m}")

    for c in desired_cols:
        if c not in group_df.columns:
            group_df[c] = pd.NA
    group_df = group_df[desired_cols].sort_values("participant", ignore_index=True)

    group_df = group_df.rename(columns={c: rename_drop_left_suffix(c) for c in group_df.columns})

    group_df.to_csv(OUTPUT_BINS_CSV, index=False)
    print(f"Saved group people bins (terrain={TARGET_TERRAIN}) to: {OUTPUT_BINS_CSV}")

    if missing_files:
        print("Missing gait_stats_peoplebins file for:")
        for pid in missing_files:
            print(f"  - {pid}")

if __name__ == "__main__":
    collect_social_wide()
    collect_peoplebins_wide()


#!/usr/bin/env python3
"""
collect_group_distance_motion_fix_proportions_by_class.py

UPDATED:
- Filters to ONE terrain only via TARGET_TERRAIN ("flat" or "cobblestone")
- Writes a terrain-specific group file:
    group_distance_motion_fix_proportions_by_class_{TARGET_TERRAIN}.csv
"""

import os
import pandas as pd

# ============================================================
TARGET_TERRAIN = "cobblestone"   # "flat" or "cobblestone"
# ============================================================

VALID_TERRAINS = {"flat", "cobblestone", "green"}

BASE_DIR   = r"C:\LocoGaze\data"
META_CSV   = os.path.join(BASE_DIR, "metadata_all.csv")
GROUP_DIR  = os.path.join(BASE_DIR, "group")
OUTPUT_CSV = os.path.join(
    GROUP_DIR,
    f"group_distance_motion_fix_proportions_by_class_{TARGET_TERRAIN}.csv"
)

REQUIRED = {
    "area_label",
    "object_class",
    "distance",
    "motion",
    "proportion_present",
    "proportion_looked",
    "proportion_looked_when_present",
    "total_time_s",
}

# Expected canonical categories (matching single-subject script)
OBJECT_CLASSES = ["person", "cyclist", "bicycle", "car"]
DISTANCES      = ["close", "far"]
MOTIONS        = ["approaching", "going away", "stable"]

def main():
    if TARGET_TERRAIN not in VALID_TERRAINS:
        raise SystemExit(f"ERROR: TARGET_TERRAIN must be one of {sorted(VALID_TERRAINS)}")

    os.makedirs(GROUP_DIR, exist_ok=True)

    # Load participant list
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
        raise SystemExit("ERROR: no participants found.")

    collected, missing = [], []

    for pid in participants:
        csv_path = os.path.join(
            BASE_DIR, pid, "output", "stats", "distance_motion_fix_proportions_by_class.csv"
        )
        if not os.path.exists(csv_path):
            missing.append(pid)
            continue

        df = pd.read_csv(csv_path, sep=None, engine="python")

        # Tolerate occasional typos in 'area_label'
        if "area_label" not in df.columns:
            if "areal_label" in df.columns:
                df = df.rename(columns={"areal_label": "area_label"})
            elif "area_labe" in df.columns:
                df = df.rename(columns={"area_labe": "area_label"})

        # Tolerate 'class'/'object' instead of 'object_class'
        if "object_class" not in df.columns:
            if "class" in df.columns:
                df = df.rename(columns={"class": "object_class"})
            elif "object" in df.columns:
                df = df.rename(columns={"object": "object_class"})

        missing_cols = REQUIRED - set(df.columns)
        if missing_cols:
            raise SystemExit(
                f"ERROR: {csv_path} missing columns: {sorted(missing_cols)}\n"
                f"Found: {list(df.columns)}"
            )

        out = df[list(REQUIRED)].copy()
        out.insert(0, "participant", pid)

        # Normalise string columns
        out["area_label"]   = out["area_label"].astype(str).str.strip().str.lower()
        out["object_class"] = out["object_class"].astype(str).str.strip().str.lower()
        out["distance"]     = out["distance"].astype(str).str.strip().str.lower()
        out["motion"]       = out["motion"].astype(str).str.strip().str.lower()

        # NEW: keep only TARGET_TERRAIN
        out = out[out["area_label"] == TARGET_TERRAIN].copy()
        if out.empty:
            continue

        # Filter to canonical categories
        out = out[
            out["object_class"].isin(OBJECT_CLASSES)
            & out["distance"].isin(DISTANCES)
            & out["motion"].isin(MOTIONS)
        ].copy()

        if out.empty:
            continue

        # Set categorical ordering for tidy sorting
        out["object_class"] = pd.Categorical(out["object_class"], categories=OBJECT_CLASSES, ordered=True)
        out["distance"]     = pd.Categorical(out["distance"], categories=DISTANCES, ordered=True)
        out["motion"]       = pd.Categorical(out["motion"], categories=MOTIONS, ordered=True)

        collected.append(out)

    if not collected:
        raise SystemExit(f"ERROR: no data collected from any participant for terrain='{TARGET_TERRAIN}'.")

    group_df = pd.concat(collected, ignore_index=True)

    group_df = group_df.sort_values(
        ["participant", "area_label", "object_class", "distance", "motion"],
        ignore_index=True
    )

    group_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    if missing:
        print("Missing distance_motion_fix_proportions_by_class.csv for participants:")
        for pid in missing:
            print(f"  - {pid}")

if __name__ == "__main__":
    main()

