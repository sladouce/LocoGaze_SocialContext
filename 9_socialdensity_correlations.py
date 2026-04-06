#!/usr/bin/env python3
"""
compute_spatialbin_correlations_socialdensity.py  (COBBLESTONE ONLY)

Same computations, but:
- Restrict to area_label == TARGET_TERRAIN before spatial binning
- Outputs go to corr_socialdensity_cobblestone
- Figures only reflect bins that were included (because bins_df is already filtered)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, colors

# ---- Optional stats ----
try:
    from scipy.stats import spearmanr, pearsonr
    _SCIPY_OK = True
except Exception:
    print("[WARN] scipy not found; per-subject p-value masking disabled. Install: pip install scipy")
    _SCIPY_OK = False

# ---------------- CONFIG ----------------
BASE_DIR   = r"C:/LocoGaze/data/"
META_ALL   = os.path.join(BASE_DIR, "metadata_all.csv")

GROUP_DIR  = os.path.join(BASE_DIR, "group")

# ---------- NEW ----------
TARGET_TERRAIN = "cobblestone"
OUT_DIR    = os.path.join(GROUP_DIR, f"corr_socialdensity_{TARGET_TERRAIN}")
os.makedirs(OUT_DIR, exist_ok=True)

# Column name for number of people per sample
PEOPLE_COL = "number_people"

# Bin precision / inclusion
EPS = 3
MIN_SAMPLES_PER_BIN       = 0
MIN_FLOOR_SAMPLES_PER_BIN = 0

# Correlations
CORR_METHOD   = 'spearman'  # or 'pearson'
ALPHA_SUBJECT = 0.01

# Group permutations
N_PERM      = 5000
RNG_SEED    = 2025

# Multiple comparisons (FDR) on group permutation p-values
FDR_Q_ALPHA      = 0.05
USE_FDR_FOR_MASK = True

# Colormap bins
BIN_EDGES     = np.arange(-1.0, 1.0 + 0.25, 0.25)
MAX_CELL_FILL = 0.99

# Scene geometry
X_MAX    = 1920.0
Y_MAX    = 1080.0
X_CENTER = 960.0

FEAT_MEAN = {
    'pace_LEFT'            : 'pace',
    'cadence_LEFT'         : 'cadence',
    'stride_length_LEFT'   : 'stride_length',
    'stride_duration_LEFT' : 'stride_duration',
    'horizontal_ecc'       : 'horizontal_ecc',
    'gaze_y'               : 'gaze_y',
    'depth_head_pitch_deg' : 'head_pitch_deg',
    'depth_looking_floor'  : 'floor_looking_frac',
    PEOPLE_COL             : 'people_mean',
    'radius_of_gyration'   : 'gaze_rG',
    'spatial_entropy'      : 'gaze_entropy',
    'mean_rms_LEFT'        : 'acc_rms',
}

VAR_TARGETS = {
    'stride_length_LEFT'   : 'stride_length_var',
    'stride_duration_LEFT' : 'stride_duration_var',
}

FEATURE_ORDER = [
    'pace', 'cadence', 'stride_length', 'stride_duration',
    'stride_length_var', 'stride_duration_var', 'stride_var_CV',
    'gaze_depth_m', 'horizontal_ecc', 'gaze_y',
    'floor_looking_frac', 'head_pitch_deg',
    'gaze_rG', 'gaze_entropy',
    'acc_rms',
    'people_mean',
]

# ---------------- HELPERS ----------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None

def fisher_mean(rhos):
    rhos = np.array(rhos, dtype=float)
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return np.nan
    rhos = np.clip(rhos, -0.999999, 0.999999)
    z = np.arctanh(rhos)
    return np.tanh(np.nanmean(z))

def _corr_pair(x, y, method='spearman'):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, 0
    if method == 'pearson':
        if _SCIPY_OK:
            r, _ = pearsonr(x[mask], y[mask])
        else:
            r = np.corrcoef(x[mask], y[mask])[0, 1]
    else:
        if _SCIPY_OK:
            r, _ = spearmanr(x[mask], y[mask])
        else:
            xr = pd.Series(x[mask]).rank().values
            yr = pd.Series(y[mask]).rank().values
            r = np.corrcoef(xr, yr)[0, 1]
    return r, mask.sum()

def compute_per_bin_features(vis):
    """Return per-bin features for TARGET_TERRAIN only (walking already filtered upstream)."""
    if not {'latitude', 'longitude'}.issubset(vis.columns):
        return pd.DataFrame()

    tmp = vis.dropna(subset=['latitude', 'longitude']).copy()
    if tmp.empty:
        return pd.DataFrame()

    # ---- NEW: terrain filter (sample-level) ----
    if "area_label" not in tmp.columns:
        return pd.DataFrame()
    tmp["area_label"] = tmp["area_label"].astype(str).str.strip().str.lower()
    tmp = tmp[tmp["area_label"] == TARGET_TERRAIN].copy()
    if tmp.empty:
        return pd.DataFrame()

    # Gaze numeric + invert vertical
    tmp['gaze_x'] = pd.to_numeric(tmp.get('gaze_x', np.nan), errors='coerce')
    tmp['gaze_y'] = pd.to_numeric(tmp.get('gaze_y', np.nan), errors='coerce')
    tmp['gaze_y'] = Y_MAX - tmp['gaze_y']

    # Horizontal eccentricity
    tmp['horizontal_ecc'] = np.abs(tmp['gaze_x'] - X_CENTER)

    # Floor helpers
    tmp['depth_looking_floor'] = tmp.get('depth_looking_floor', False).astype(bool)
    tmp['floor_int'] = tmp['depth_looking_floor'].astype(int)

    # Floor-only depth
    if 'depth_d_s' in tmp.columns:
        tmp['depth_d_s'] = pd.to_numeric(tmp['depth_d_s'], errors='coerce')
        tmp['depth_m_flooronly'] = np.where(tmp['floor_int'] == 1, tmp['depth_d_s'] / 1000.0, np.nan)
    else:
        tmp['depth_m_flooronly'] = np.nan

    # Social density
    if PEOPLE_COL in tmp.columns:
        tmp[PEOPLE_COL] = pd.to_numeric(tmp[PEOPLE_COL], errors='coerce')
    else:
        print(f"[WARN] PEOPLE_COL='{PEOPLE_COL}' not found; people_mean will be NaN.")
        tmp[PEOPLE_COL] = np.nan

    # Spatial bins
    tmp['lat_bin'] = tmp['latitude'].round(EPS)
    tmp['lon_bin'] = tmp['longitude'].round(EPS)
    gb = tmp.groupby(['lat_bin', 'lon_bin'], sort=False)

    out = gb['latitude'].size().to_frame(name='n_samples_bin').reset_index()
    out = out.rename(columns={'lat_bin': 'latitude', 'lon_bin': 'longitude'})

    for src_col, out_name in FEAT_MEAN.items():
        if src_col == 'depth_looking_floor':
            out[out_name] = gb['floor_int'].mean().values
        elif src_col in tmp.columns:
            out[out_name] = gb[src_col].mean().values
        else:
            out[out_name] = np.nan

    for src_col, out_name in VAR_TARGETS.items():
        if src_col in tmp.columns:
            out[out_name] = gb[src_col].var(ddof=1).values
        else:
            out[out_name] = np.nan

    out['gaze_depth_m'] = gb['depth_m_flooronly'].mean().values
    out['n_floor_samples_bin'] = gb['floor_int'].sum().values.astype(int)

    # ---- Composite stride variability ----
    needed_cols = {'stride_length', 'stride_duration', 'stride_length_var', 'stride_duration_var'}
    if needed_cols.issubset(out.columns):
        mean_len = out['stride_length']
        var_len  = out['stride_length_var']
        mean_dur = out['stride_duration']
        var_dur  = out['stride_duration_var']
        with np.errstate(divide='ignore', invalid='ignore'):
            sd_len = np.sqrt(var_len)
            sd_dur = np.sqrt(var_dur)
            cv_len = sd_len / mean_len
            cv_dur = sd_dur / mean_dur
            stride_var_CV = (cv_len + cv_dur) / 2.0
        stride_var_CV[~np.isfinite(stride_var_CV)] = np.nan
        out['stride_var_CV'] = stride_var_CV
    else:
        out['stride_var_CV'] = np.nan

    # Inclusion filters (these define what bins are "included")
    out = out[out['n_samples_bin'] >= MIN_SAMPLES_PER_BIN].reset_index(drop=True)
    if MIN_FLOOR_SAMPLES_PER_BIN > 0:
        out = out[out['n_floor_samples_bin'] >= MIN_FLOOR_SAMPLES_PER_BIN].reset_index(drop=True)

    return out

def compute_corr_and_pvals(df_bins, method='spearman'):
    present = [c for c in FEATURE_ORDER if c in df_bins.columns]
    if len(present) < 2:
        return pd.DataFrame(), pd.DataFrame(), present

    X = df_bins[present].copy()
    nunique = X.nunique(dropna=True)
    present = [c for c in present if nunique.get(c, 0) > 1]
    X = X[present]
    if X.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame(), present

    n = len(present)
    R = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)

    if not _SCIPY_OK:
        for i in range(n):
            for j in range(n):
                r, _ = _corr_pair(X[present[i]].values, X[present[j]].values, method=method)
                R[i, j] = r
        return pd.DataFrame(R, index=present, columns=present), pd.DataFrame(P, index=present, columns=present), present

    for i in range(n):
        for j in range(n):
            xi, xj = X[present[i]].values, X[present[j]].values
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() >= 3:
                if method == 'pearson':
                    r, p = pearsonr(xi[mask], xj[mask])
                else:
                    r, p = spearmanr(xi[mask], xj[mask])
                R[i, j], P[i, j] = r, p

    return pd.DataFrame(R, index=present, columns=present), pd.DataFrame(P, index=present, columns=present), present

def compute_group_perm_pvals(per_subject_bins, vars_order, per_subject_corrs,
                             method='spearman', n_perm=5, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n = len(vars_order)
    P = np.full((n, n), np.nan)
    Zobs = np.full((n, n), np.nan)

    subj_C = {s: C.reindex(index=vars_order, columns=vars_order) for s, C, _ in per_subject_corrs}

    subj_arrays = {}
    for s, bins_df in per_subject_bins.items():
        subj_arrays[s] = {v: (bins_df[v].values if v in bins_df.columns else None) for v in vars_order}

    for i in range(1, n):
        for j in range(0, i):
            r_list = []
            valid_subjects = []
            for s, C in subj_C.items():
                rij = C.iat[i, j] if C is not None else np.nan
                if np.isfinite(rij):
                    r_list.append(rij)
                    valid_subjects.append(s)
            if not r_list:
                continue

            z_obs = np.arctanh(np.clip(r_list, -0.999999, 0.999999))
            z_mean = np.nanmean(z_obs)
            Zobs[i, j] = z_mean

            z_null = []
            for _ in range(n_perm):
                r_perm = []
                for s in valid_subjects:
                    xi_full = subj_arrays[s][vars_order[i]]
                    xj_full = subj_arrays[s][vars_order[j]]
                    if xi_full is None or xj_full is None:
                        continue
                    mask = np.isfinite(xi_full) & np.isfinite(xj_full)
                    if mask.sum() < 3:
                        continue
                    xi = xi_full[mask]
                    xj = xj_full[mask].copy()
                    rng.shuffle(xj)
                    r, _ = _corr_pair(xi, xj, method=method)
                    if np.isfinite(r):
                        r_perm.append(r)
                if r_perm:
                    z_null.append(np.nanmean(np.arctanh(np.clip(r_perm, -0.999999, 0.999999))))

            z_null = np.array(z_null, float)
            z_null = z_null[np.isfinite(z_null)]
            if z_null.size > 0:
                P[i, j] = np.mean(np.abs(z_null) >= np.abs(z_mean))

    return pd.DataFrame(P, index=vars_order, columns=vars_order), pd.DataFrame(Zobs, index=vars_order, columns=vars_order)

def fdr_bh_on_lower_triangle(P):
    qdf = P.copy()
    vals = P.values
    n = vals.shape[0]

    flats, coords = [], []
    for i in range(1, n):
        for j in range(0, i):
            if np.isfinite(vals[i, j]):
                flats.append(vals[i, j])
                coords.append((i, j))
    if not flats:
        return qdf

    p = np.asarray(flats, float)
    m = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    q = p * m / ranks
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty_like(q)
    q_adj[order] = np.minimum(q_sorted, 1.0)

    for (i, j), qv in zip(coords, q_adj):
        qdf.iat[i, j] = qv
        qdf.iat[j, i] = qv
    return qdf

def plot_corr_square_triangle(C, vars_order, title, outfile, pvals=None, alpha=None,
                              cmap_name='coolwarm', bin_edges=BIN_EDGES,
                              max_cell_fill=MAX_CELL_FILL, annotate=False,
                              annot_fmt=".2f", fs=0.7):
    if C.empty:
        return

    C = C.reindex(index=vars_order, columns=vars_order)
    P = pvals.reindex(index=vars_order, columns=vars_order) if (pvals is not None and not pvals.empty) else None

    vals = C.values
    n = len(vars_order)

    fig, ax = plt.subplots(figsize=(max(6, fs*n), max(6, fs*n)))

    Nbins = len(bin_edges) - 1
    cmap_discrete = plt.get_cmap(cmap_name, Nbins)
    norm_discrete = colors.BoundaryNorm(bin_edges, Nbins, clip=True)

    for i in range(n):
        ax.axhline(i - 0.5, color='lightgray', lw=0.5, zorder=0)
        ax.axvline(i - 0.5, color='lightgray', lw=0.5, zorder=0)

    d = max_cell_fill
    for i in range(n):
        ax.add_patch(
            patches.Rectangle(
                (i - d/2, i - d/2), d, d,
                facecolor='#d0d0d0', edgecolor='white', lw=0.8, zorder=2
            )
        )

    for i in range(1, n):
        for j in range(0, i):
            r = vals[i, j]
            if not np.isfinite(r):
                continue
            if (P is not None) and np.isfinite(P.values[i, j]) and (alpha is not None):
                if P.values[i, j] >= alpha:
                    continue
            side = (np.abs(r)) * max_cell_fill
            if side <= 0:
                continue
            ax.add_patch(
                patches.Rectangle(
                    (j - side/2, i - side/2), side, side,
                    facecolor=cmap_discrete(norm_discrete(r)),
                    edgecolor='white', lw=0.8, zorder=3
                )
            )
            if annotate:
                ax.text(j, i, format(r, annot_fmt), ha='center', va='center', fontsize=8, color='black', zorder=4)

    ax.set_xticks(range(n))
    ax.set_xticklabels(vars_order, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(vars_order)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, pad=14)

    ax.add_patch(
        patches.Polygon(
            [(-0.5, -0.5), (n - 0.5, -0.5), (n - 0.5, n - 0.5)],
            closed=True, facecolor='white', edgecolor='none', zorder=1
        )
    )

    sm = plt.cm.ScalarMappable(cmap=cmap_discrete, norm=norm_discrete)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=bin_edges)
    cb.set_label('Correlation (r)')

    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

def main():
    np.random.seed(RNG_SEED)
    meta = safe_read_csv(META_ALL)
    if meta is None or meta.empty or 'reldir' not in meta.columns:
        print(f"[ERROR] Could not read metadata_all.csv or missing 'reldir' at {META_ALL}")
        return

    per_subject_corrs = []
    per_subject_bins  = {}
    any_vars          = set()
    used_subjects     = []

    for _, row in meta.iterrows():
        subj = str(row['reldir'])
        vis_path = os.path.join(BASE_DIR, subj, 'output', 'visual_events.csv')
        vis = safe_read_csv(vis_path)
        if vis is None or vis.empty:
            continue

        # walking only
        if 'is_walking' in vis.columns:
            vis = vis[vis['is_walking'] == True].copy()
        if vis.empty:
            continue

        bins = compute_per_bin_features(vis)  # already cobblestone-filtered + inclusion-filtered
        if bins.empty:
            continue

        per_subject_bins[subj] = bins
        C, P, cols = compute_corr_and_pvals(bins, method=CORR_METHOD)
        if C.empty:
            continue

        C.to_csv(os.path.join(OUT_DIR, f"{subj}_corr.csv"))
        P.to_csv(os.path.join(OUT_DIR, f"{subj}_corr_pvals.csv"))

        per_subject_corrs.append((subj, C, P))
        any_vars.update(list(C.columns))
        used_subjects.append(subj)

    if not per_subject_corrs:
        print("[WARN] No subject correlation matrices were produced.")
        return

    vars_order = [v for v in FEATURE_ORDER if v in any_vars]
    with open(os.path.join(OUT_DIR, "vars_order.json"), "w") as f:
        json.dump(vars_order, f, indent=2)
    with open(os.path.join(OUT_DIR, "participants_used.txt"), "w") as f:
        f.write("\n".join(used_subjects))

    aligned = [C.reindex(index=vars_order, columns=vars_order).values for _, C, _ in per_subject_corrs]
    stack = np.stack(aligned, axis=0)
    n = stack.shape[1]
    group = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            group[i, j] = fisher_mean(stack[:, i, j])

    group_df = pd.DataFrame(group, index=vars_order, columns=vars_order)
    group_df.to_csv(os.path.join(OUT_DIR, f"group_corr_mean_{TARGET_TERRAIN}.csv"))

    print(f"[INFO] Running group permutations ({TARGET_TERRAIN}) with N_PERM={N_PERM} ...")
    gP, gZ = compute_group_perm_pvals(
        per_subject_bins=per_subject_bins,
        vars_order=vars_order,
        per_subject_corrs=per_subject_corrs,
        method=CORR_METHOD,
        n_perm=N_PERM,
        rng_seed=RNG_SEED
    )
    gP.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_pvals_{TARGET_TERRAIN}.csv"))
    gZ.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_zobs_{TARGET_TERRAIN}.csv"))

    gQ = fdr_bh_on_lower_triangle(gP)
    gQ.to_csv(os.path.join(OUT_DIR, f"group_corr_perm_qvals_{TARGET_TERRAIN}.csv"))

    if USE_FDR_FOR_MASK:
        mask_df = gQ
        mask_alpha = FDR_Q_ALPHA
        title_suffix = f" (FDR q<{FDR_Q_ALPHA})"
    else:
        mask_df = gP
        mask_alpha = 0.01
        title_suffix = f" (p<0.01)"

    plot_corr_square_triangle(
        group_df, vars_order,
        f"Group ({TARGET_TERRAIN}) — {CORR_METHOD.capitalize()} r (Fisher z-mean){title_suffix}",
        os.path.join(OUT_DIR, f"group_corr_mean_heatmap_{TARGET_TERRAIN}.png"),
        pvals=mask_df, alpha=mask_alpha, annotate=False
    )
    plot_corr_square_triangle(
        group_df, vars_order,
        f"Group ({TARGET_TERRAIN}) — {CORR_METHOD.capitalize()} r (Fisher z-mean) — annotated{title_suffix}",
        os.path.join(OUT_DIR, f"group_corr_mean_heatmap_annotated_{TARGET_TERRAIN}.png"),
        pvals=mask_df, alpha=mask_alpha, annotate=True
    )

    if 'people_mean' in vars_order:
        idx = vars_order.index('people_mean')
        group_df.iloc[idx, :].to_frame(name='r_people_mean').to_csv(
            os.path.join(OUT_DIR, f"group_corr_vs_people_mean_{TARGET_TERRAIN}.csv")
        )
        gP.iloc[idx, :].to_frame(name='p_perm').to_csv(
            os.path.join(OUT_DIR, f"group_p_vs_people_mean_{TARGET_TERRAIN}.csv")
        )
        gQ.iloc[idx, :].to_frame(name='q_fdr').to_csv(
            os.path.join(OUT_DIR, f"group_q_vs_people_mean_{TARGET_TERRAIN}.csv")
        )

    manifest = {
        "TARGET_TERRAIN": TARGET_TERRAIN,
        "BASE_DIR": BASE_DIR,
        "OUT_DIR": OUT_DIR,
        "EPS": EPS,
        "MIN_SAMPLES_PER_BIN": MIN_SAMPLES_PER_BIN,
        "MIN_FLOOR_SAMPLES_PER_BIN": MIN_FLOOR_SAMPLES_PER_BIN,
        "CORR_METHOD": CORR_METHOD,
        "N_PERM": N_PERM,
        "RNG_SEED": RNG_SEED,
        "BIN_EDGES": BIN_EDGES.tolist(),
        "participants_used": used_subjects,
        "FDR_Q_ALPHA": FDR_Q_ALPHA,
        "USE_FDR_FOR_MASK": USE_FDR_FOR_MASK,
        "PEOPLE_COL": PEOPLE_COL,
        "FEATURE_ORDER": FEATURE_ORDER,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved all outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
group_spatial_covariation_maps_socialdensity.py

Creates group-level spatial co-variation maps between
SOCIAL DENSITY (people_mean) and each dependent variable across spatial bins.

Key features:
- Subject-balanced binning: first per-subject bin means, then average across subjects.
- Social density per bin: mean number of people per sample in that bin (people_mean).
- Dependent variables per bin (means / variances):
    * pace, cadence, stride_length, stride_duration
    * stride_length_var, stride_duration_var
    * gaze_depth_m (floor-only), horizontal_ecc, gaze_y (inverted, top high)
    * floor_looking_frac, head_pitch_deg
    * gaze dispersion: gaze_rG (radius_of_gyration), gaze_entropy (spatial_entropy)
    * acceleration RMS: acc_rms (from mean_rms_LEFT)
- For each DV, generates a bivariate map:
    * x-axis: people_mean (social density)
    * y-axis: DV
    * Color of each bin encodes the (people_mean, DV) combination using a 4×4 bivariate palette.
- Optional minimum subjects per bin filter for reliability.

Inputs:
  C:/LocoGaze/data/metadata_all.csv
  Per-subject visual_events.csv under: C:/LocoGaze/data/<reldir>/output/

Outputs:
  C:/LocoGaze/data/group/maps_cov_socialdensity/
    covar__people_mean__pace__bivariate.html
    covar__people_mean__gaze_depth_m__bivariate.html
    ...
    bivariate_4x4_palette.png (legend image)
"""

import os
import json
import numpy as np
import pandas as pd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca
from branca.element import MacroElement, Template
from PIL import Image, ImageDraw  # for the palette PNG

# ---------------- CONFIG ----------------
BASE_DIR   = r"C:/LocoGaze/data/"
META_ALL   = os.path.join(BASE_DIR, "metadata_all.csv")

GROUP_DIR  = os.path.join(BASE_DIR, "group")

# ---------- NEW ----------
TARGET_TERRAIN = "cobblestone"
OUT_DIR    = os.path.join(GROUP_DIR, f"maps_cov_socialdensity_{TARGET_TERRAIN}")
os.makedirs(OUT_DIR, exist_ok=True)


# Column name in visual_events.csv that holds the number of people per sample
# ADAPT THIS to your actual column name (e.g. "total_people", "n_people", etc.)
PEOPLE_COL = "number_people"

# The social density feature name at bin level
SOCIAL_FEATURE = "people_mean"

# Dependent variables to pair with social density
DV_FEATURES = [
    "pace",
    "cadence",
    "stride_length",
    "stride_duration",
    "stride_length_var",
    "stride_duration_var",
    "gaze_depth_m",
    "horizontal_ecc",
    "gaze_y",
    "floor_looking_frac",
    "head_pitch_deg",
    "gaze_rG",
    "gaze_entropy",
    "acc_rms",
]

# Spatial bin precision
EPS = 4

# Consistent map view across all outputs
MAP_CENTER = [50.877, 4.704]   # set to None to auto-center from data
MAP_ZOOM   = 17

# Tile provider (we'll resolve to a URL so we can apply CSS dimming)
TILES_PROVIDER = 'CartoDB.Positron'

# Dimming controls (darken light tiles for contrast)
USE_CSS_DIMMING = True
DIM_BRIGHTNESS  = 0.72
DIM_CONTRAST    = 1.25
DIM_SATURATE    = 0.92

# Reliability filter
MIN_SUBJECTS_PER_BIN = 10   # set to 1 to keep all

# Scene geometry (for engineered gaze features)
X_MAX = 1920.0
Y_MAX = 1080.0
X_CENTER = 960.0

# Tiles
#TILES_LIGHT = 'CartoDB.VoyagerLabelsUnder'
TILES_LIGHT = 'CartoDB.Positron'
TILES_DARK  = 'CartoDB.Positron'

# ------------------------------------------------------------------
# Display ranges (min,max) per feature for consistent scaling across maps
# You can adjust these for your actual data. If a feature is not listed
# here, its range is inferred from the data.
# ------------------------------------------------------------------
FEATURE_RANGES = {
    # Social density (requested 0–5)
    "people_mean":        (0.0, 5.0),       # mean number of people per sample

    # Gait means
    "pace":               (4.5, 6.0),       # km/h
    "cadence":            (105.0, 115.0),   # steps/min
    "stride_length":      (1.0, 2.0),       # m
    "stride_duration":    (1.05, 1.15),    # s

    # Gait variances
    "stride_length_var":   (0.0, 0.08),     # m^2
    "stride_duration_var": (0.0, 0.04),     # s^2

    # Gaze & floor depth
    "gaze_depth_m":       (1.0, 2.5),       # m (floor-only)
    "horizontal_ecc":     (75.0, 140.0),     # px from center
    "gaze_y":             (550.0, 650.0),   # px (inverted; top high)
    "floor_looking_frac": (0.2, 0.3),       # proportion 0–1
    "head_pitch_deg":     (-8.0, -4.0),     # deg (neg = down)

    # Gaze dispersion (ballpark ranges; adjust to your stats if needed)
    "gaze_rG":            (9.0, 12.0),     # px radius of gyration
    "gaze_entropy":       (0.0, 6.0),       # arbitrary entropy units

    # Acceleration RMS (ballpark range; adjust as needed)
    "acc_rms":            (16.0, 18.0),       # e.g. m/s^2 or a.u.
}

# Colormaps per feature (used mainly when that feature is on the color axis)
FEATURE_CMAPS = {
    "pace":               plt.get_cmap("viridis"),
    "cadence":            plt.get_cmap("viridis"),
    "stride_length":      plt.get_cmap("viridis"),
    "stride_duration":    plt.get_cmap("viridis"),
    "stride_length_var":  plt.get_cmap("magma"),
    "stride_duration_var":plt.get_cmap("magma"),
    "gaze_depth_m":       plt.get_cmap("bwr"),
    "horizontal_ecc":     plt.get_cmap("plasma"),
    "gaze_y":             plt.get_cmap("coolwarm"),
    "floor_looking_frac": plt.get_cmap("Reds"),
    "head_pitch_deg":     plt.get_cmap("PuOr_r"),
    "people_mean":        plt.get_cmap("Reds"),
    "gaze_rG":            plt.get_cmap("viridis"),
    "gaze_entropy":       plt.get_cmap("viridis"),
    "acc_rms":            plt.get_cmap("viridis"),
}

# Marker size scaling (only used in color_size style; we use bivariate here, but keep it)
RADIUS_MIN = 6
RADIUS_MAX = 24

# --- 4×4 BIVARIATE COLORMAP ---
BIVAR_RES = 4

# TOP row colors, left→right  (dark/saturated)
BIVAR_TOP_ROW    = ["#5B9ED6", "#2E5F7F", "#0E2E45", "#000000"]
# BOTTOM row colors, left→right (warm/pale)
BIVAR_BOTTOM_ROW = ["#F7F3D1", "#E8D7A8", "#E9C57E", "#F39A1E"]

def _lerp_rgb(c0, c1, t):
    c0 = np.array(mcolors.to_rgb(c0)); c1 = np.array(mcolors.to_rgb(c1))
    return (1.0 - t) * c0 + t * c1

def bivariate_4x4_table(top_row, bottom_row):
    """Return a (4,4,3) array for the bivariate palette."""
    assert len(top_row) == 4 and len(bottom_row) == 4
    table = np.zeros((4, 4, 3), float)
    for x in range(4):
        for y in range(4):
            t = y / 3.0  # 0, 1/3, 2/3, 1
            table[3 - y, x, :] = _lerp_rgb(bottom_row[x], top_row[x], t)
    return np.clip(table, 0, 1)

def save_bivariate_discrete_png(table_4x4, out_path,
                                px_per_cell=200,
                                grid=True, grid_color="#FFFFFF", grid_width=6,
                                border=20, border_color="#FFFFFF", dpi=300):
    """Save the 4×4 palette as a high-res PNG (no axes/labels)."""
    h, w, _ = table_4x4.shape
    assert (h, w) == (4, 4), "table_4x4 must be 4×4×3"

    sep_x = grid_width if grid else 0
    sep_y = grid_width if grid else 0
    W = w*px_per_cell + (w-1)*sep_x + 2*border
    H = h*px_per_cell + (h-1)*sep_y + 2*border

    img = Image.new("RGB", (W, H), border_color)
    draw = ImageDraw.Draw(img)

    y = border
    for i in range(h):  # i = 0..3 (top..bottom)
        x = border
        for j in range(w):  # j = 0..3 (left..right)
            rgb = tuple((np.clip(table_4x4[i, j, :], 0, 1) * 255).astype(np.uint8))
            draw.rectangle([x, y, x+px_per_cell-1, y+px_per_cell-1], fill=rgb)
            x += px_per_cell
            if grid and j < w-1:
                draw.rectangle([x, y, x+grid_width-1, y+px_per_cell-1], fill=grid_color)
                x += grid_width
        y += px_per_cell
        if grid and i < h-1:
            draw.rectangle([border, y, W-border-1, y+grid_width-1], fill=grid_color)
            y += grid_width

    img.save(out_path, dpi=(dpi, dpi))
    print(f"[OK] Saved bivariate discrete colormap: {out_path}  ({W}×{H}px)")

# Build bivariate palette once
TABLE_4X4 = bivariate_4x4_table(BIVAR_TOP_ROW, BIVAR_BOTTOM_ROW)
save_bivariate_discrete_png(
    TABLE_4X4,
    out_path=os.path.join(OUT_DIR, "bivariate_4x4_palette.png"),
    px_per_cell=240, grid=True, grid_width=8, border=24, dpi=300
)

# ---------------- HELPERS ----------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Skipping (cannot read): {path} — {e}")
        return None

def _resolve_tile_url(provider_name: str) -> tuple[str, str]:
    """Return (tiles_url, attribution) for a given provider alias."""
    if provider_name == 'CartoDB.Positron':
        url = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
        attr = '&copy; OpenStreetMap contributors &copy; CARTO'
        return url, attr

    # Default fallback
    url = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
    attr = '&copy; OpenStreetMap contributors'
    return url, attr


def _add_dimmed_tiles(m: folium.Map, provider_name: str):
    """Add tiles with a CSS class and inject a CSS filter to dim them."""
    tiles_url, attr = _resolve_tile_url(provider_name)
    folium.TileLayer(
        tiles=tiles_url,
        attr=attr,
        name='Basemap (dimmed)',
        control=False,
        overlay=False,
        show=True,
        **{'className': 'dim-tiles'}
    ).add_to(m)

    css = (
        "<style>\n"
        "  .leaflet-tile-pane .dim-tiles img.leaflet-tile {"
        f"filter: brightness({DIM_BRIGHTNESS}) contrast({DIM_CONTRAST}) saturate({DIM_SATURATE});"
        "}\n"
        "</style>\n"
    )
    style = MacroElement()
    style._template = Template("{% macro html(this, kwargs) %}\n" + css + "\n{% endmacro %}\n")
    m.get_root().add_child(style)

def _add_dim_overlay(m: folium.Map):
    """Alternative to CSS filters: semi-transparent overlay pane below markers."""
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-pane.leaflet-dim-pane { z-index: 300; }</style>"
    ))
    m.add_child(folium.map.Pane('leaflet-dim-pane', m))
    folium.Rectangle(
        bounds=[[-85, -180], [85, 180]],
        color=None,
        fill=True,
        fill_color='#000000',
        fill_opacity=0.20,
        interactive=False,
        pane='leaflet-dim-pane'
    ).add_to(m)

def engineer_features(vis: pd.DataFrame) -> pd.DataFrame:
    """Create engineered columns that match your correlation pipeline."""
    tmp = vis.copy()

    # Ensure numeric for relevant columns
    for c in [
        "gaze_x", "gaze_y", "depth_d_s",
        "pace_LEFT", "cadence_LEFT",
        "stride_length_LEFT", "stride_duration_LEFT",
        "depth_head_pitch_deg",
        "radius_of_gyration", "spatial_entropy",
        "mean_rms_LEFT", PEOPLE_COL
    ]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # Invert vertical gaze so top has higher values
    if "gaze_y" in tmp.columns:
        tmp["gaze_y"] = Y_MAX - tmp["gaze_y"]

    # Horizontal eccentricity from center
    if "gaze_x" in tmp.columns:
        tmp["horizontal_ecc"] = np.abs(tmp["gaze_x"] - X_CENTER)

    # Floor-looking (0/1)
    if "depth_looking_floor" in tmp.columns:
        tmp["floor_int"] = tmp["depth_looking_floor"].astype(bool).astype(int)
    else:
        tmp["floor_int"] = 0

    # Floor-only depth in meters (depth_d_s is mm)
    if "depth_d_s" in tmp.columns:
        tmp["depth_m_flooronly"] = np.where(tmp["floor_int"] == 1, tmp["depth_d_s"]/1000.0, np.nan)
    else:
        tmp["depth_m_flooronly"] = np.nan

    # Rename to standardized feature names
    rename = {
        "pace_LEFT":            "pace",
        "cadence_LEFT":         "cadence",
        "stride_length_LEFT":   "stride_length",
        "stride_duration_LEFT": "stride_duration",
        "depth_head_pitch_deg": "head_pitch_deg",
        "mean_rms_LEFT":        "acc_rms",
    }
    tmp = tmp.rename(columns=rename)

    return tmp

def per_subject_bin_means_multi(vis: pd.DataFrame, subject: str, eps: int) -> pd.DataFrame:
    """
    Compute per-subject, per-bin means for all supported features,
    but only for TARGET_TERRAIN samples.
    """
    if vis is None or vis.empty or not {'latitude','longitude'}.issubset(vis.columns):
        return pd.DataFrame()

    # ---- NEW: terrain filter (sample-level) ----
    if "area_label" not in vis.columns:
        return pd.DataFrame()
    v2 = vis.copy()
    v2["area_label"] = v2["area_label"].astype(str).str.strip().str.lower()
    v2 = v2[v2["area_label"] == TARGET_TERRAIN].copy()
    if v2.empty:
        return pd.DataFrame()

    df = engineer_features(v2)

    df = df.dropna(subset=['latitude','longitude']).copy()
    if df.empty:
        return pd.DataFrame()

    df['lat_bin'] = df['latitude'].round(eps)
    df['lon_bin'] = df['longitude'].round(eps)

    gb = df.groupby(['lat_bin','lon_bin'])

    means = gb.agg({
        'pace':               'mean',
        'cadence':            'mean',
        'stride_length':      'mean',
        'stride_duration':    'mean',
        'horizontal_ecc':     'mean',
        'gaze_y':             'mean',
        'head_pitch_deg':     'mean',
        'floor_int':          'mean',
        'depth_m_flooronly':  'mean',
        'acc_rms':            'mean',
        'radius_of_gyration': 'mean',
        'spatial_entropy':    'mean',
        PEOPLE_COL:           'mean',
    })

    means = means.rename(columns={
        'floor_int':          'floor_looking_frac',
        'depth_m_flooronly':  'gaze_depth_m',
        'radius_of_gyration': 'gaze_rG',
        'spatial_entropy':    'gaze_entropy',
        PEOPLE_COL:           'people_mean',
    })

    vars_ = gb.agg({
        'stride_length':   lambda x: np.nanvar(x, ddof=1),
        'stride_duration': lambda x: np.nanvar(x, ddof=1)
    }).rename(columns={
        'stride_length':   'stride_length_var',
        'stride_duration': 'stride_duration_var'
    })

    sizes = gb.size().rename('n_samples_subj')

    out = pd.concat([means, vars_, sizes], axis=1).reset_index()
    out = out.rename(columns={'lat_bin':'latitude','lon_bin':'longitude'})
    out['subject'] = subject
    out['area_label'] = TARGET_TERRAIN  # optional traceability
    return out


def aggregate_subject_balanced(per_sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average per-subject bin means across subjects (subject-balanced).
    Compute n_subjects and total n_samples as well.
    """
    if per_sub_df is None or per_sub_df.empty:
        return pd.DataFrame()

    group_keys = ['latitude','longitude']

    # n_subjects contributing to a bin
    n_subj = per_sub_df.groupby(group_keys)['subject'].nunique().rename('n_subjects')

    # sum of raw samples across subjects
    n_samp = per_sub_df.groupby(group_keys)['n_samples_subj'].sum().rename('n_samples')

    # --- FIX: only average numeric feature columns (exclude strings like area_label) ---
    exclude = set(group_keys + ['subject', 'n_samples_subj'])
    feature_cols = [c for c in per_sub_df.columns if c not in exclude]

    # keep only numeric columns for mean aggregation
    num_cols = per_sub_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    means_across_subj = per_sub_df.groupby(group_keys)[num_cols].mean()

    out = pd.concat([means_across_subj, n_subj, n_samp], axis=1).reset_index()

    # Optional: re-attach area_label as constant (now safe)
    out["area_label"] = TARGET_TERRAIN

    return out


def center_from(dfs):
    lat, lon = [], []
    for d in dfs:
        if d is not None and not d.empty:
            lat.extend(d['latitude'].values.tolist())
            lon.extend(d['longitude'].values.tolist())
    if lat:
        return [float(np.mean(lat)), float(np.mean(lon))]
    # Leuven fallback
    return [50.879, 4.701]

def scale_to_radius(val, vmin, vmax, rmin=RADIUS_MIN, rmax=RADIUS_MAX):
    if not np.isfinite(val):
        return rmin
    if vmax <= vmin:
        return (rmin + rmax) / 2.0
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return rmin + t * (rmax - rmin)

def color_from_bivariate_discrete(val_a, vmin_a, vmax_a, val_b, vmin_b, vmax_b, table_4x4):
    """
    Map (A,B) to a 4×4 color using floor binning.
    A controls columns (→), B controls rows (↑).
    """
    if not np.isfinite(val_a) or not np.isfinite(val_b):
        return "#888888"
    eps = 1e-12
    ta = 0.0 if vmax_a <= vmin_a else (val_a - vmin_a) / (vmax_a - vmin_a + eps)
    tb = 0.0 if vmax_b <= vmin_b else (val_b - vmin_b) / (vmax_b - vmin_b + eps)
    ta = float(np.clip(ta, 0, 0.999999))
    tb = float(np.clip(tb, 0, 0.999999))
    xi = int(np.floor(ta * 4))
    yi = int(np.floor(tb * 4))
    rgb = table_4x4[3 - yi, xi, :]
    return mcolors.rgb2hex(rgb)

def folium_map_bivariate(df, featA, featB, vminA, vmaxA, vminB, vmaxB,
                         table4x4, outfile, center, zoom=14, provider_name=TILES_PROVIDER):
    """Bivariate map: color encodes (featA, featB) via 4×4 palette; marker size is fixed."""
    if df is None or df.empty:
        print(f"[WARN] Nothing to plot for {outfile}")
        return

    m = folium.Map(location=center, tiles=None, zoom_start=zoom)
    if USE_CSS_DIMMING:
        _add_dimmed_tiles(m, provider_name)
    else:
        tiles_url, attr = _resolve_tile_url(provider_name)
        folium.TileLayer(tiles=tiles_url, attr=attr, control=False, overlay=False, show=True).add_to(m)
        _add_dim_overlay(m)

    for r in df.itertuples():
        a = float(getattr(r, featA))
        b = float(getattr(r, featB))
        color = color_from_bivariate_discrete(a, vminA, vmaxA, b, vminB, vmaxB, table4x4)
        popup = f"{featA}: {a:.3f} | {featB}: {b:.3f} | n_subj={r.n_subjects} | n={r.n_samples}"
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=12,
            color=color,
            opacity=0.00,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup
        ).add_to(m)

    # SVG legend for the 4×4 palette with min/max labels for A (→) and B (↑)
    cell = 22
    pad  = 10
    w = cell*4 + pad*2 + 60
    h = cell*4 + pad*2 + 40

    rects = []
    for yi in range(4):              # 0 top .. 3 bottom in SVG
        for xi in range(4):          # 0 left .. 3 right
            rgb = (table4x4[yi, xi, :] * 255).astype(np.uint8)
            hexc = '#%02x%02x%02x' % tuple(rgb.tolist())
            x = pad + xi*cell
            y = pad + yi*cell
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{hexc}" stroke="#333" stroke-width="0.5"/>'
            )
    rects_svg = "\n".join(rects)

    legend_svg = f"""
    <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{w}" height="{h}" rx="6" ry="6" fill="white" opacity="0.95"/>
      <g font-family="sans-serif" font-size="11">
        {rects_svg}
        <!-- A (→) axis -->
        <text x="{pad}" y="{pad+cell*4+16}" text-anchor="start">{featA}: {vminA:.2f}</text>
        <text x="{pad+cell*4}" y="{pad+cell*4+16}" text-anchor="end">{vmaxA:.2f}</text>
        <!-- B (↑) axis -->
        <text x="{pad+cell*4+8}" y="{pad+cell*4}" transform="rotate(-90 {pad+cell*4+8},{pad+cell*4})"
              text-anchor="start">{featB}: {vminB:.2f}</text>
        <text x="{pad+cell*4+8}" y="{pad}" transform="rotate(-90 {pad+cell*4+8},{pad})"
              text-anchor="end">{vmaxB:.2f}</text>
        <!-- Axis arrows -->
        <line x1="{pad}" y1="{pad+cell*4+6}" x2="{pad+cell*4}" y2="{pad+cell*4+6}" stroke="#333" stroke-width="1.2"/>
        <polygon points="{pad+cell*4},{pad+cell*4+6} {pad+cell*4-5},{pad+cell*4+2} {pad+cell*4-5},{pad+cell*4+10}" fill="#333"/>
        <line x1="{pad+cell*4+6}" y1="{pad+cell*4}" x2="{pad+cell*4+6}" y2="{pad}" stroke="#333" stroke-width="1.2"/>
        <polygon points="{pad+cell*4+6},{pad} {pad+cell*4+2},{pad+5} {pad+cell*4+10},{pad+5}" fill="#333"/>
      </g>
    </svg>
    """

    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 10px; z-index:9999;
                padding: 8px; border-radius: 8px; box-shadow: 0 0 6px rgba(0,0,0,0.18);">
      {legend_svg}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(outfile)
    print(f"[OK] Saved {outfile}")

# ---------------- MAIN ----------------
def main():
    meta = safe_read_csv(META_ALL)
    if meta is None or meta.empty or 'reldir' not in meta.columns:
        print(f"[ERROR] Could not read metadata_all.csv or missing 'reldir' at {META_ALL}")
        return

    per_sub_bins = []

    # ---- Build per-subject binned features ----
    for _, row in meta.iterrows():
        reldir = str(row['reldir'])
        vis_path = os.path.join(BASE_DIR, reldir, 'output', 'visual_events.csv')
        vis = safe_read_csv(vis_path)
        if vis is None or vis.empty:
            continue

        # Walking-only (if available) to match other scripts
        if 'is_walking' in vis.columns:
            vis = vis[vis['is_walking'] == True].copy()
            if vis.empty:
                continue

        subj_bins = per_subject_bin_means_multi(vis, subject=reldir, eps=EPS)
        if not subj_bins.empty:
            per_sub_bins.append(subj_bins)

    if not per_sub_bins:
        print("[WARN] No per-subject bin data assembled.")
        return

    per_sub_bins = pd.concat(per_sub_bins, ignore_index=True)
    group_df = aggregate_subject_balanced(per_sub_bins)

    # Reliability filter
    if MIN_SUBJECTS_PER_BIN > 1:
        group_df = group_df[group_df['n_subjects'] >= MIN_SUBJECTS_PER_BIN].reset_index(drop=True)

    if group_df.empty:
        print("[WARN] All bins filtered out; nothing to plot.")
        return

    # ---- Map center/zoom ----
    center_auto = center_from([group_df])
    center = MAP_CENTER if MAP_CENTER else center_auto
    zoom   = MAP_ZOOM

    # ---- Bivariate color table ----
    table4x4 = TABLE_4X4

    if SOCIAL_FEATURE not in group_df.columns:
        print(f"[ERROR] Social density feature '{SOCIAL_FEATURE}' not found in group_df.")
        return

    # ---- Generate bivariate maps: people_mean × each DV ----
    for dv in DV_FEATURES:
        if dv not in group_df.columns:
            print(f"[WARN] Skipping DV '{dv}' — missing column in group_df.")
            continue

        featA = SOCIAL_FEATURE
        featB = dv

        # Ranges: from config if present, else from data
        vminA, vmaxA = FEATURE_RANGES.get(
            featA,
            (float(np.nanmin(group_df[featA].values)), float(np.nanmax(group_df[featA].values)))
        )
        vminB, vmaxB = FEATURE_RANGES.get(
            featB,
            (float(np.nanmin(group_df[featB].values)), float(np.nanmax(group_df[featB].values)))
        )

        outfile = os.path.join(OUT_DIR, f"covar__{featA}__{featB}__bivariate_{TARGET_TERRAIN}.html")
        folium_map_bivariate(
            df=group_df,
            featA=featA, featB=featB,
            vminA=vminA, vmaxA=vmaxA,
            vminB=vminB, vmaxB=vmaxB,
            table4x4=table4x4,
            outfile=outfile,
            center=center, zoom=zoom,
            provider_name=TILES_PROVIDER
        )

    # Save a small manifest
    manifest = {
        "BASE_DIR": BASE_DIR,
        "OUT_DIR": OUT_DIR,
        "EPS": EPS,
        "MIN_SUBJECTS_PER_BIN": MIN_SUBJECTS_PER_BIN,
        "SOCIAL_FEATURE": SOCIAL_FEATURE,
        "DV_FEATURES": DV_FEATURES,
        "MAP_CENTER": MAP_CENTER,
        "MAP_ZOOM": MAP_ZOOM,
        "PEOPLE_COL": PEOPLE_COL,
        "FEATURE_RANGES": FEATURE_RANGES,
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved all outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
