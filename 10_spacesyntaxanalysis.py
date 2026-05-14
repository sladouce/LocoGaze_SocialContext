#!/usr/bin/env python3
"""
10_spacesyntax_sequential_analysis.py

Sequential street-network / space-syntax-inspired analysis.

Main analyses:
    1. social_density_z ~ syntax_z + participant fixed effects
    2. DV_z ~ syntax_z + participant fixed effects

Optional exploratory moderation:
    3. DV_z ~ social_density_z * syntax_z + participant fixed effects

Additional spatial/contextual measures:
    - OSM street-network metrics
    - local street-context metrics
    - approximate isovist / visibility proxies from OSM building footprints

Terrain labels are retained but NOT included in models.
"""

from pathlib import Path
import warnings
import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf
from scipy import stats
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

import folium
from branca.element import MacroElement, Template
from PIL import Image, ImageDraw

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:/LocoGaze/data")
GROUP_DIR = BASE_DIR / "group"

INPUT_CSV = GROUP_DIR / "mixedmodel_spatialbins_allterrain.csv"
OUTPUT_DIR = GROUP_DIR / "space_syntax_sequential"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLUMN SETTINGS
# ============================================================

SUBJECT_COL = "participant"
LAT_COL = "latitude"
LON_COL = "longitude"
TERRAIN_COL = "area_label"
SOCIAL_DENSITY_COL = "social_density"

DVS = [
    "look_people_prop",
    "floor_looking_frac",
    "looking_environment",
    "looking_other_objects",
    "gaze_y",
    "horizontal_ecc",
    "radius_of_gyration",
    "spatial_entropy",
    "depth_head_pitch_deg",
    "mean_rms_LEFT",
    "pace_LEFT",
    "cadence_LEFT",
    "stride_duration_LEFT",
    "stride_length_LEFT",
    "stride_var_CV",
]


# ============================================================
# NETWORK SETTINGS
# ============================================================

NETWORK_TYPE = "walk"
LOCAL_RADIUS_METERS = 300
BBOX_MARGIN_DEGREES = 0.003

# Set True once when recomputing new metrics.
# After successful run, set back to False.
FORCE_RECOMPUTE_NETWORK = True

BETWEENNESS_K = 150


# ============================================================
# ADDITIONAL URBAN CONFIGURATION SETTINGS
# ============================================================

COMPUTE_VISIBILITY_PROXIES = True
COMPUTE_LOCAL_STREET_CONTEXT = True

ISOVIST_MAX_RADIUS_M = 80
ISOVIST_N_RAYS = 72

LOCAL_CONTEXT_RADIUS_M = 50
INTERSECTION_DEGREE_THRESHOLD = 3


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

RUN_EXPLORATORY_MODERATION = True

N_PRED = 75
HEXBIN_GRIDSIZE = 50


# ============================================================
# BIVARIATE MAP SETTINGS
# ============================================================

MAKE_BIVARIATE_MAPS = True

BIVAR_MAP_DIR = OUTPUT_DIR / "bivariate_maps_syntax_gazegait"
BIVAR_MAP_DIR.mkdir(parents=True, exist_ok=True)

MAP_CENTER = [50.877, 4.704]
MAP_ZOOM = 17
TILES_PROVIDER = "CartoDB.Positron"

MIN_SUBJECTS_PER_BIN = 5
MARKER_RADIUS = 11

USE_QUANTILE_RANGES = True
Q_LOW = 0.05
Q_HIGH = 0.95

BIVAR_TOP_ROW = ["#5B9ED6", "#2E5F7F", "#0E2E45", "#000000"]
BIVAR_BOTTOM_ROW = ["#F7F3D1", "#E8D7A8", "#E9C57E", "#F39A1E"]


# ============================================================
# BASIC HELPERS
# ============================================================

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def safe_filename(name):
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


# ============================================================
# BIVARIATE MAP HELPERS
# ============================================================

def _lerp_rgb(c0, c1, t):
    c0 = np.array(mcolors.to_rgb(c0))
    c1 = np.array(mcolors.to_rgb(c1))
    return (1.0 - t) * c0 + t * c1


def bivariate_4x4_table(top_row, bottom_row):
    table = np.zeros((4, 4, 3), float)
    for x in range(4):
        for y in range(4):
            t = y / 3.0
            table[3 - y, x, :] = _lerp_rgb(bottom_row[x], top_row[x], t)
    return np.clip(table, 0, 1)


TABLE_4X4 = bivariate_4x4_table(BIVAR_TOP_ROW, BIVAR_BOTTOM_ROW)


def save_bivariate_palette(out_path):
    px = 220
    grid = 8
    border = 24
    h, w, _ = TABLE_4X4.shape

    W = w * px + (w - 1) * grid + 2 * border
    H = h * px + (h - 1) * grid + 2 * border

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    y = border
    for i in range(h):
        x = border
        for j in range(w):
            rgb = tuple((TABLE_4X4[i, j, :] * 255).astype(np.uint8))
            draw.rectangle([x, y, x + px - 1, y + px - 1], fill=rgb)
            x += px + grid
        y += px + grid

    img.save(out_path, dpi=(300, 300))


def _resolve_tile_url(provider_name):
    if provider_name == "CartoDB.Positron":
        return (
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "&copy; OpenStreetMap contributors &copy; CARTO"
        )

    return (
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "&copy; OpenStreetMap contributors"
    )


def _add_dimmed_tiles(m, provider_name):
    tiles_url, attr = _resolve_tile_url(provider_name)

    folium.TileLayer(
        tiles=tiles_url,
        attr=attr,
        name="Basemap",
        control=False,
        overlay=False,
        show=True,
        **{"className": "dim-tiles"}
    ).add_to(m)

    css = """
    <style>
      .leaflet-tile-pane .dim-tiles img.leaflet-tile {
        filter: brightness(0.74) contrast(1.20) saturate(0.90);
      }
    </style>
    """

    style = MacroElement()
    style._template = Template(
        "{% macro html(this, kwargs) %}\n" + css + "\n{% endmacro %}\n"
    )
    m.get_root().add_child(style)


def get_range(df, col):
    vals = pd.to_numeric(df[col], errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        return np.nan, np.nan

    if USE_QUANTILE_RANGES:
        return float(vals.quantile(Q_LOW)), float(vals.quantile(Q_HIGH))

    return float(vals.min()), float(vals.max())


def color_from_bivariate(val_a, vmin_a, vmax_a, val_b, vmin_b, vmax_b):
    if not np.isfinite(val_a) or not np.isfinite(val_b):
        return "#888888"

    ta = 0.0 if vmax_a <= vmin_a else (val_a - vmin_a) / (vmax_a - vmin_a)
    tb = 0.0 if vmax_b <= vmin_b else (val_b - vmin_b) / (vmax_b - vmin_b)

    ta = float(np.clip(ta, 0, 0.999999))
    tb = float(np.clip(tb, 0, 0.999999))

    xi = int(np.floor(ta * 4))
    yi = int(np.floor(tb * 4))

    rgb = TABLE_4X4[3 - yi, xi, :]
    return mcolors.rgb2hex(rgb)


def aggregate_subject_balanced_for_maps(df, syntax_metrics, dv_features):
    """
    First average within participant × shared spatial bin, then average across participants.

    Uses latitude_bin / longitude_bin as shared bin identifiers, so the same
    rounded spatial bins are used across participants.
    """

    bin_lat_col = "latitude_bin"
    bin_lon_col = "longitude_bin"

    if bin_lat_col not in df.columns or bin_lon_col not in df.columns:
        raise ValueError(
            "Missing latitude_bin / longitude_bin. "
            "Run the spatial-binning preparation script first."
        )

    feature_cols = [c for c in syntax_metrics + dv_features if c in df.columns]

    cols = [
        SUBJECT_COL,
        bin_lat_col,
        bin_lon_col,
        LAT_COL,
        LON_COL,
    ] + feature_cols

    tmp = df[cols].copy()

    for c in feature_cols + [LAT_COL, LON_COL, bin_lat_col, bin_lon_col]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.dropna(subset=[bin_lat_col, bin_lon_col])

    # Average within participant × shared spatial bin
    per_subject = (
        tmp.groupby([SUBJECT_COL, bin_lat_col, bin_lon_col], as_index=False)
        .agg({
            LAT_COL: "mean",
            LON_COL: "mean",
            **{c: "mean" for c in feature_cols},
        })
    )

    # Then average across participants within the same shared bin
    n_subjects = (
        per_subject
        .groupby([bin_lat_col, bin_lon_col])[SUBJECT_COL]
        .nunique()
        .rename("n_subjects")
    )

    group_mean = (
        per_subject
        .groupby([bin_lat_col, bin_lon_col])
        .agg({
            LAT_COL: "mean",
            LON_COL: "mean",
            **{c: "mean" for c in feature_cols},
        })
    )

    out = pd.concat([group_mean, n_subjects], axis=1).reset_index()

    if MIN_SUBJECTS_PER_BIN > 1:
        out = out[out["n_subjects"] >= MIN_SUBJECTS_PER_BIN].copy()

    return out


def add_bivariate_legend(m, feat_a, feat_b, vmin_a, vmax_a, vmin_b, vmax_b):
    cell = 22
    pad = 10
    w = cell * 4 + pad * 2 + 70
    h = cell * 4 + pad * 2 + 45

    rects = []
    for yi in range(4):
        for xi in range(4):
            rgb = (TABLE_4X4[yi, xi, :] * 255).astype(np.uint8)
            hexc = "#%02x%02x%02x" % tuple(rgb.tolist())
            x = pad + xi * cell
            y = pad + yi * cell
            rects.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{hexc}" stroke="#333" stroke-width="0.5"/>'
            )

    rects_svg = "\n".join(rects)

    legend_svg = f"""
    <svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{w}" height="{h}" rx="6" ry="6"
            fill="white" opacity="0.95"/>
      <g font-family="sans-serif" font-size="10">
        {rects_svg}
        <text x="{pad}" y="{pad + cell*4 + 16}" text-anchor="start">
            {feat_a}: {vmin_a:.2f}
        </text>
        <text x="{pad + cell*4}" y="{pad + cell*4 + 16}" text-anchor="end">
            {vmax_a:.2f}
        </text>
        <text x="{pad + cell*4 + 8}" y="{pad + cell*4}"
              transform="rotate(-90 {pad + cell*4 + 8},{pad + cell*4})"
              text-anchor="start">
            {feat_b}: {vmin_b:.2f}
        </text>
        <text x="{pad + cell*4 + 8}" y="{pad}"
              transform="rotate(-90 {pad + cell*4 + 8},{pad})"
              text-anchor="end">
            {vmax_b:.2f}
        </text>
      </g>
    </svg>
    """

    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 10px; z-index:9999;
                padding: 8px; border-radius: 8px;
                box-shadow: 0 0 6px rgba(0,0,0,0.18);">
      {legend_svg}
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))


def folium_bivariate_map(df, feat_a, feat_b, outfile):
    vmin_a, vmax_a = get_range(df, feat_a)
    vmin_b, vmax_b = get_range(df, feat_b)

    if not np.isfinite(vmin_a) or not np.isfinite(vmin_b):
        return

    m = folium.Map(location=MAP_CENTER, tiles=None, zoom_start=MAP_ZOOM)
    _add_dimmed_tiles(m, TILES_PROVIDER)

    plot_df = df.dropna(subset=[feat_a, feat_b, LAT_COL, LON_COL]).copy()

    for r in plot_df.itertuples():
        a = float(getattr(r, feat_a))
        b = float(getattr(r, feat_b))

        color = color_from_bivariate(
            a, vmin_a, vmax_a,
            b, vmin_b, vmax_b,
        )

        popup = (
            f"{feat_a}: {a:.4f}<br>"
            f"{feat_b}: {b:.4f}<br>"
            f"n_subjects: {int(r.n_subjects)}"
        )

        folium.CircleMarker(
            location=[getattr(r, LAT_COL), getattr(r, LON_COL)],
            radius=MARKER_RADIUS,
            color=color,
            opacity=0.0,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(m)

    add_bivariate_legend(
        m,
        feat_a,
        feat_b,
        vmin_a,
        vmax_a,
        vmin_b,
        vmax_b,
    )

    m.save(str(outfile))
    print(f"[OK] Saved {outfile}")


def make_bivariate_covariation_maps(enriched_df, syntax_metrics, dv_features):
    available_syntax = [c for c in syntax_metrics if c in enriched_df.columns]
    available_dvs = [c for c in dv_features if c in enriched_df.columns]

    group_map_df = aggregate_subject_balanced_for_maps(
        enriched_df,
        syntax_metrics=available_syntax,
        dv_features=available_dvs,
    )

    if group_map_df.empty:
        print("[WARN] No data available for bivariate maps.")
        return

    save_bivariate_palette(BIVAR_MAP_DIR / "bivariate_4x4_palette.png")

    group_map_df.to_csv(
        BIVAR_MAP_DIR / "group_subject_balanced_bins_for_bivariate_maps.csv",
        index=False,
    )

    manifest = {
        "syntax_features": available_syntax,
        "dv_features": available_dvs,
        "min_subjects_per_bin": MIN_SUBJECTS_PER_BIN,
        "use_quantile_ranges": USE_QUANTILE_RANGES,
        "q_low": Q_LOW,
        "q_high": Q_HIGH,
        "map_center": MAP_CENTER,
        "map_zoom": MAP_ZOOM,
    }

    with open(BIVAR_MAP_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for syntax_feat in available_syntax:
        for dv in available_dvs:
            outfile = BIVAR_MAP_DIR / f"covar__{syntax_feat}__{dv}__bivariate.html"
            folium_bivariate_map(
                df=group_map_df,
                feat_a=syntax_feat,
                feat_b=dv,
                outfile=outfile,
            )


# ============================================================
# NETWORK METRICS
# ============================================================

def compute_local_closeness(G, radius_m=300):
    values = {}

    for node in G.nodes:
        try:
            ego = nx.ego_graph(G, node, radius=radius_m, distance="length")
            if len(ego.nodes) <= 1:
                values[node] = np.nan
            else:
                values[node] = nx.closeness_centrality(
                    ego,
                    u=node,
                    distance="length"
                )
        except Exception:
            values[node] = np.nan

    return values


def build_osm_network_and_metrics(df):
    north = df[LAT_COL].max() + BBOX_MARGIN_DEGREES
    south = df[LAT_COL].min() - BBOX_MARGIN_DEGREES
    east = df[LON_COL].max() + BBOX_MARGIN_DEGREES
    west = df[LON_COL].min() - BBOX_MARGIN_DEGREES

    print("Downloading walking network from OpenStreetMap...")
    bbox = (west, south, east, north)
    print(f"OSM bbox = {bbox}")

    G = ox.graph_from_bbox(
        bbox,
        network_type=NETWORK_TYPE,
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )

    G_proj = ox.project_graph(G)
    nodes_proj, edges_proj = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)

    print(f"Network: {len(G_proj.nodes)} nodes, {len(G_proj.edges)} edges.")
    print("Computing network metrics...")

    degree_dict = dict(G_proj.degree())

    closeness_dict = nx.closeness_centrality(
        G_proj,
        distance="length"
    )

    local_closeness_dict = compute_local_closeness(
        G_proj,
        radius_m=LOCAL_RADIUS_METERS
    )

    betweenness_dict = nx.betweenness_centrality(
        G_proj,
        k=min(BETWEENNESS_K, len(G_proj.nodes)),
        weight="length",
        normalized=True,
        seed=42,
    )

    nodes_proj["connectivity"] = nodes_proj.index.map(degree_dict)
    nodes_proj["integration_global"] = nodes_proj.index.map(closeness_dict)
    nodes_proj[f"integration_local_{LOCAL_RADIUS_METERS}m"] = nodes_proj.index.map(local_closeness_dict)
    nodes_proj["betweenness"] = nodes_proj.index.map(betweenness_dict)

    nodes_path = OUTPUT_DIR / "street_network_nodes_with_metrics.gpkg"
    edges_path = OUTPUT_DIR / "street_network_edges.gpkg"

    nodes_proj.to_file(nodes_path, driver="GPKG")
    edges_proj.to_file(edges_path, driver="GPKG")

    print(f"Saved network nodes to: {nodes_path}")
    print(f"Saved network edges to: {edges_path}")

    return G_proj, nodes_proj, edges_proj


def load_existing_network():
    nodes_path = OUTPUT_DIR / "street_network_nodes_with_metrics.gpkg"
    edges_path = OUTPUT_DIR / "street_network_edges.gpkg"

    if not nodes_path.exists() or not edges_path.exists():
        return None, None

    nodes_proj = gpd.read_file(nodes_path)
    edges_proj = gpd.read_file(edges_path)

    return nodes_proj, edges_proj


def assign_nearest_network_metrics(df, G_proj, nodes_proj, metrics_cols):
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
        crs="EPSG:4326",
    )

    gdf_proj = gdf.to_crs(nodes_proj.crs)

    nearest_nodes = ox.distance.nearest_nodes(
        G_proj,
        X=gdf_proj.geometry.x.to_numpy(),
        Y=gdf_proj.geometry.y.to_numpy(),
    )

    gdf_proj["nearest_node"] = nearest_nodes

    if "osmid" in nodes_proj.columns:
        nodes_indexed = nodes_proj.set_index("osmid")
    else:
        nodes_indexed = nodes_proj

    for col in metrics_cols:
        if col in nodes_indexed.columns:
            gdf_proj[col] = gdf_proj["nearest_node"].map(nodes_indexed[col].to_dict())

    return gdf_proj.drop(columns="geometry")


# ============================================================
# ADDITIONAL STREET CONTEXT AND VISIBILITY PROXIES
# ============================================================

def download_building_footprints(df):
    north = df[LAT_COL].max() + BBOX_MARGIN_DEGREES
    south = df[LAT_COL].min() - BBOX_MARGIN_DEGREES
    east = df[LON_COL].max() + BBOX_MARGIN_DEGREES
    west = df[LON_COL].min() - BBOX_MARGIN_DEGREES

    bbox = (west, south, east, north)
    tags = {"building": True}

    print("Downloading OSM building footprints...")

    try:
        buildings = ox.features_from_bbox(bbox, tags=tags)
    except Exception:
        buildings = ox.geometries_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            tags=tags,
        )

    if buildings.empty:
        print("[WARN] No building footprints found.")
        return None

    buildings = buildings.reset_index()
    buildings = buildings[
        buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if buildings.empty:
        print("[WARN] No polygonal building footprints found.")
        return None

    buildings = buildings.set_crs("EPSG:4326", allow_override=True)
    buildings_proj = ox.projection.project_gdf(buildings)

    out_path = OUTPUT_DIR / "osm_building_footprints.gpkg"
    buildings_proj.to_file(out_path, driver="GPKG")
    print(f"Saved building footprints to: {out_path}")

    return buildings_proj


def _nearest_distance_to_obstacle(point, ray, obstacle_union, max_radius):
    if obstacle_union is None or obstacle_union.is_empty:
        return max_radius

    inter = ray.intersection(obstacle_union)

    if inter.is_empty:
        return max_radius

    try:
        d = point.distance(inter)
        if not np.isfinite(d):
            return max_radius
        return float(np.clip(d, 0, max_radius))
    except Exception:
        return max_radius


def compute_isovist_proxy_for_point(point, obstacle_union, max_radius=80, n_rays=72):
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    distances = []
    endpoints = []

    for a in angles:
        x2 = point.x + max_radius * np.cos(a)
        y2 = point.y + max_radius * np.sin(a)
        ray = LineString([(point.x, point.y), (x2, y2)])

        d = _nearest_distance_to_obstacle(
            point=point,
            ray=ray,
            obstacle_union=obstacle_union,
            max_radius=max_radius,
        )

        distances.append(d)
        endpoints.append((
            point.x + d * np.cos(a),
            point.y + d * np.sin(a),
        ))

    distances = np.asarray(distances, dtype=float)

    try:
        poly = Polygon(endpoints)
        area = float(poly.area) if poly.is_valid else np.nan
        perimeter = float(poly.length) if poly.is_valid else np.nan
    except Exception:
        area = np.nan
        perimeter = np.nan

    mean_sightline = float(np.nanmean(distances))
    max_sightline = float(np.nanmax(distances))
    min_sightline = float(np.nanmin(distances))

    openness = mean_sightline / max_radius if max_radius > 0 else np.nan
    occlusivity = 1.0 - openness if np.isfinite(openness) else np.nan

    if np.isfinite(area) and np.isfinite(perimeter) and perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter ** 2)
    else:
        compactness = np.nan

    return {
        "isovist_area_proxy": area,
        "isovist_perimeter_proxy": perimeter,
        "isovist_mean_sightline": mean_sightline,
        "isovist_max_sightline": max_sightline,
        "isovist_min_sightline": min_sightline,
        "isovist_openness": openness,
        "isovist_occlusivity_proxy": occlusivity,
        "isovist_compactness_proxy": compactness,
    }


def add_visibility_proxies(df, buildings_proj):
    if buildings_proj is None or buildings_proj.empty:
        print("[WARN] No buildings available; skipping visibility proxies.")
        return df

    print("Computing isovist-like visibility proxies...")

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
        crs="EPSG:4326",
    )

    gdf_proj = gdf.to_crs(buildings_proj.crs)
    obstacle_union = unary_union(buildings_proj.geometry)

    rows = []

    for pt in gdf_proj.geometry:
        rows.append(
            compute_isovist_proxy_for_point(
                point=pt,
                obstacle_union=obstacle_union,
                max_radius=ISOVIST_MAX_RADIUS_M,
                n_rays=ISOVIST_N_RAYS,
            )
        )

    vis_df = pd.DataFrame(rows, index=gdf_proj.index)

    out = pd.concat(
        [
            gdf_proj.drop(columns="geometry").reset_index(drop=True),
            vis_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return out


def add_local_street_context(df, G_proj, nodes_proj, edges_proj):
    print("Computing local street-context metrics...")

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
        crs="EPSG:4326",
    )

    gdf_proj = gdf.to_crs(edges_proj.crs)

    nearest_edges = ox.distance.nearest_edges(
        G_proj,
        X=gdf_proj.geometry.x.to_numpy(),
        Y=gdf_proj.geometry.y.to_numpy(),
    )

    edges_indexed = edges_proj.copy()

    if not isinstance(edges_indexed.index, pd.MultiIndex):
        if {"u", "v", "key"}.issubset(edges_indexed.columns):
            edges_indexed = edges_indexed.set_index(["u", "v", "key"], drop=False)

    edge_lengths = []
    for e in nearest_edges:
        try:
            edge_lengths.append(float(edges_indexed.loc[e, "length"]))
        except Exception:
            edge_lengths.append(np.nan)

    gdf_proj["nearest_street_segment_length_m"] = edge_lengths

    degree_dict = dict(G_proj.degree())
    nodes_proj = nodes_proj.copy()
    nodes_proj["degree"] = nodes_proj.index.map(degree_dict)

    intersection_nodes = nodes_proj[
        nodes_proj["degree"] >= INTERSECTION_DEGREE_THRESHOLD
    ].copy()

    if not intersection_nodes.empty:
        inter_xy = np.vstack([
            intersection_nodes.geometry.x.to_numpy(),
            intersection_nodes.geometry.y.to_numpy(),
        ]).T

        tree = cKDTree(inter_xy)

        pt_xy = np.vstack([
            gdf_proj.geometry.x.to_numpy(),
            gdf_proj.geometry.y.to_numpy(),
        ]).T

        dists, _ = tree.query(pt_xy, k=1)
        gdf_proj["distance_to_nearest_intersection_m"] = dists

        counts = tree.query_ball_point(pt_xy, r=LOCAL_CONTEXT_RADIUS_M)
        gdf_proj[f"local_intersection_count_{LOCAL_CONTEXT_RADIUS_M}m"] = [len(c) for c in counts]
    else:
        gdf_proj["distance_to_nearest_intersection_m"] = np.nan
        gdf_proj[f"local_intersection_count_{LOCAL_CONTEXT_RADIUS_M}m"] = np.nan

    edges_sindex = edges_proj.sindex
    local_lengths = []

    for pt in gdf_proj.geometry:
        buf = pt.buffer(LOCAL_CONTEXT_RADIUS_M)
        possible_idx = list(edges_sindex.intersection(buf.bounds))

        if len(possible_idx) == 0:
            local_lengths.append(np.nan)
            continue

        cand = edges_proj.iloc[possible_idx]
        total_len = 0.0

        for geom in cand.geometry:
            try:
                total_len += geom.intersection(buf).length
            except Exception:
                continue

        local_lengths.append(total_len)

    length_col = f"local_street_length_{LOCAL_CONTEXT_RADIUS_M}m"
    density_col = f"local_street_density_{LOCAL_CONTEXT_RADIUS_M}m"

    gdf_proj[length_col] = local_lengths

    buffer_area = np.pi * (LOCAL_CONTEXT_RADIUS_M ** 2)
    gdf_proj[density_col] = gdf_proj[length_col] / buffer_area

    return gdf_proj.drop(columns="geometry")


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def prepare_model_df(df, outcome_col, predictor_cols):
    cols = [SUBJECT_COL, outcome_col] + predictor_cols
    model_df = df[cols].copy()

    model_df[SUBJECT_COL] = model_df[SUBJECT_COL].astype(str).astype(object)

    for c in [outcome_col] + predictor_cols:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce").astype(float)

    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if model_df.empty:
        return None

    if model_df[SUBJECT_COL].nunique() < 3:
        return None

    if model_df[outcome_col].nunique() < 5:
        return None

    model_df["outcome_z"] = zscore(model_df[outcome_col]).astype(float)

    for c in predictor_cols:
        model_df[f"{c}_z"] = zscore(model_df[c]).astype(float)

    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(model_df) < 30:
        return None

    return model_df


def fit_ols_participant_fe(df, outcome_col, syntax_metric, analysis_name):
    model_df = prepare_model_df(
        df=df,
        outcome_col=outcome_col,
        predictor_cols=[syntax_metric],
    )

    if model_df is None:
        return None

    syntax_z_col = f"{syntax_metric}_z"
    formula = f"outcome_z ~ {syntax_z_col} + C({SUBJECT_COL})"

    try:
        res = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    except Exception as e:
        warnings.warn(f"OLS failed for {outcome_col} ~ {syntax_metric}: {e}")
        return None

    term = syntax_z_col

    if term not in res.params.index:
        return None

    row = {
        "analysis": analysis_name,
        "outcome": outcome_col,
        "syntax_metric": syntax_metric,
        "term": term,
        "beta": float(res.params[term]),
        "se": float(res.bse[term]),
        "t_or_z": float(res.tvalues[term]),
        "p": float(res.pvalues[term]),
        "n_bins": int(len(model_df)),
        "n_participants": int(model_df[SUBJECT_COL].nunique()),
        "model_type": "OLS_participant_fixed_effects_HC3_no_terrain",
        "aic": float(res.aic) if np.isfinite(res.aic) else np.nan,
        "bic": float(res.bic) if np.isfinite(res.bic) else np.nan,
    }

    return row, res, model_df


def fit_exploratory_moderation(df, dv, syntax_metric):
    model_df = prepare_model_df(
        df=df,
        outcome_col=dv,
        predictor_cols=[SOCIAL_DENSITY_COL, syntax_metric],
    )

    if model_df is None:
        return None

    social_z = f"{SOCIAL_DENSITY_COL}_z"
    syntax_z = f"{syntax_metric}_z"

    formula = f"outcome_z ~ {social_z} * {syntax_z} + C({SUBJECT_COL})"

    try:
        res = smf.ols(formula, data=model_df).fit(cov_type="HC3")
    except Exception as e:
        warnings.warn(f"Moderation OLS failed for {dv} × {syntax_metric}: {e}")
        return None

    rows = []
    terms_of_interest = [
        social_z,
        syntax_z,
        f"{social_z}:{syntax_z}",
    ]

    for term in terms_of_interest:
        if term not in res.params.index:
            continue

        rows.append({
            "analysis": "exploratory_social_density_x_syntax",
            "outcome": dv,
            "syntax_metric": syntax_metric,
            "term": term,
            "beta": float(res.params[term]),
            "se": float(res.bse[term]),
            "t_or_z": float(res.tvalues[term]),
            "p": float(res.pvalues[term]),
            "n_bins": int(len(model_df)),
            "n_participants": int(model_df[SUBJECT_COL].nunique()),
            "model_type": "OLS_participant_fixed_effects_HC3_no_terrain",
            "aic": float(res.aic) if np.isfinite(res.aic) else np.nan,
            "bic": float(res.bic) if np.isfinite(res.bic) else np.nan,
        })

    return pd.DataFrame(rows), res, model_df


# ============================================================
# STATIC PLOTS
# ============================================================

def plot_simple_syntax_effect(df, outcome_col, syntax_metric, out_path):
    plot_df = prepare_model_df(
        df=df,
        outcome_col=outcome_col,
        predictor_cols=[syntax_metric],
    )

    if plot_df is None:
        return

    syntax_z_col = f"{syntax_metric}_z"

    try:
        ols = smf.ols(
            f"outcome_z ~ {syntax_z_col}",
            data=plot_df,
        ).fit()
    except Exception:
        return

    xgrid = np.linspace(
        plot_df[syntax_z_col].quantile(0.02),
        plot_df[syntax_z_col].quantile(0.98),
        N_PRED,
    )

    pred = pd.DataFrame({syntax_z_col: xgrid})
    yhat = ols.predict(pred)

    plt.figure(figsize=(5, 4))
    plt.scatter(
        plot_df[syntax_z_col],
        plot_df["outcome_z"],
        alpha=0.15,
        s=12,
    )
    plt.plot(xgrid, yhat)
    plt.axhline(0, linewidth=0.8)
    plt.xlabel(f"{syntax_metric}, z-scored")
    plt.ylabel(f"{outcome_col}, z-scored")
    plt.title(f"{outcome_col} ~ {syntax_metric}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_hexbin_map(df, value_col, edges_gdf=None, gridsize=25):
    tmp = df.dropna(subset=[LAT_COL, LON_COL, value_col]).copy()
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    if edges_gdf is not None:
        edges_wgs = edges_gdf.to_crs("EPSG:4326")
        edges_wgs.plot(ax=ax, linewidth=0.5, alpha=0.25)

    hb = ax.hexbin(
        tmp[LON_COL],
        tmp[LAT_COL],
        C=tmp[value_col],
        reduce_C_function=np.mean,
        gridsize=gridsize,
        mincnt=1,
    )

    plt.colorbar(hb, ax=ax, label=f"Mean {value_col}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Hexbin map: mean {value_col}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"map_hexbin_{safe_filename(value_col)}.png", dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"ERROR: cannot find input CSV:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = [SUBJECT_COL, LAT_COL, LON_COL, SOCIAL_DENSITY_COL]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise SystemExit(f"ERROR: input file missing required columns: {missing}")

    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str).str.strip().astype(object)

    if TERRAIN_COL in df.columns:
        df[TERRAIN_COL] = (
            df[TERRAIN_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .astype(object)
        )

    df = df.dropna(subset=[LAT_COL, LON_COL]).copy()

    print(f"Loaded {len(df)} spatial bins from {df[SUBJECT_COL].nunique()} participants.")

    if TERRAIN_COL in df.columns:
        print("\nBins by terrain:")
        print(df[TERRAIN_COL].value_counts(dropna=False))

    local_count_col = f"local_intersection_count_{LOCAL_CONTEXT_RADIUS_M}m"
    local_length_col = f"local_street_length_{LOCAL_CONTEXT_RADIUS_M}m"
    local_density_col = f"local_street_density_{LOCAL_CONTEXT_RADIUS_M}m"
    local_integration_col = f"integration_local_{LOCAL_RADIUS_METERS}m"

    metrics_to_compute = [
        "connectivity",
        "integration_global",
        local_integration_col,
        "betweenness",
    ]

    syntax_metrics = [
        "connectivity",
        "integration_global",
        local_integration_col,
        "betweenness",
        "nearest_street_segment_length_m",
        "distance_to_nearest_intersection_m",
        local_count_col,
        local_length_col,
        local_density_col,
        "isovist_area_proxy",
        "isovist_perimeter_proxy",
        "isovist_mean_sightline",
        "isovist_max_sightline",
        "isovist_min_sightline",
        "isovist_openness",
        "isovist_occlusivity_proxy",
        "isovist_compactness_proxy",
    ]

    enriched_csv = OUTPUT_DIR / "spatial_bins_with_space_syntax_metrics.csv"

    edges_gdf = None

    if enriched_csv.exists() and not FORCE_RECOMPUTE_NETWORK:
        print(f"Loading existing enriched file: {enriched_csv}")
        enriched_df = pd.read_csv(enriched_csv)

        _, edges_existing = load_existing_network()
        if edges_existing is not None:
            edges_gdf = edges_existing

    else:
        G_proj, nodes_proj, edges_gdf = build_osm_network_and_metrics(df)

        enriched_df = assign_nearest_network_metrics(
            df,
            G_proj,
            nodes_proj,
            metrics_to_compute,
        )

        if COMPUTE_LOCAL_STREET_CONTEXT:
            enriched_df = add_local_street_context(
                enriched_df,
                G_proj=G_proj,
                nodes_proj=nodes_proj,
                edges_proj=edges_gdf,
            )

        if COMPUTE_VISIBILITY_PROXIES:
            buildings_proj = download_building_footprints(df)
            enriched_df = add_visibility_proxies(
                enriched_df,
                buildings_proj=buildings_proj,
            )

        enriched_df.to_csv(enriched_csv, index=False)
        print(f"Saved enriched spatial-bin file to: {enriched_csv}")

    syntax_metrics = [m for m in syntax_metrics if m in enriched_df.columns]

    # ========================================================
    # 1. Syntax metrics -> social density
    # ========================================================

    social_rows = []

    for metric in syntax_metrics:
        fit = fit_ols_participant_fe(
            enriched_df,
            outcome_col=SOCIAL_DENSITY_COL,
            syntax_metric=metric,
            analysis_name="syntax_predicts_social_density",
        )

        if fit is None:
            continue

        row, _, _ = fit
        social_rows.append(row)

        plot_simple_syntax_effect(
            enriched_df,
            outcome_col=SOCIAL_DENSITY_COL,
            syntax_metric=metric,
            out_path=OUTPUT_DIR / f"plot_social_density_by_{safe_filename(metric)}.png",
        )

    social_df = pd.DataFrame(social_rows)

    if len(social_df) > 0:
        social_df["p_fdr"] = multipletests(
            social_df["p"],
            method="fdr_bh",
        )[1]

    social_out = OUTPUT_DIR / "analysis1_syntax_predicts_social_density.csv"
    social_df.to_csv(social_out, index=False)
    print(f"Saved Analysis 1 to: {social_out}")

    # Descriptive Spearman correlations
    corr_rows = []

    for metric in syntax_metrics:
        tmp = enriched_df[[SOCIAL_DENSITY_COL, metric]].dropna()

        if len(tmp) < 10:
            continue

        r, p = stats.spearmanr(tmp[SOCIAL_DENSITY_COL], tmp[metric])

        corr_rows.append({
            "x": SOCIAL_DENSITY_COL,
            "y": metric,
            "spearman_r": float(r),
            "p_uncorrected": float(p),
            "n_bins": int(len(tmp)),
        })

    corr_df = pd.DataFrame(corr_rows)

    if len(corr_df) > 0:
        corr_df["p_fdr"] = multipletests(
            corr_df["p_uncorrected"],
            method="fdr_bh",
        )[1]

    corr_out = OUTPUT_DIR / "descriptive_correlations_social_density_syntax.csv"
    corr_df.to_csv(corr_out, index=False)
    print(f"Saved descriptive correlations to: {corr_out}")

    # ========================================================
    # 2. Syntax metrics -> gaze/gait behaviour
    # ========================================================

    behaviour_rows = []
    available_dvs = [dv for dv in DVS if dv in enriched_df.columns]

    for dv in available_dvs:
        for metric in syntax_metrics:
            print(f"Fitting behaviour model: {dv} ~ {metric}")

            fit = fit_ols_participant_fe(
                enriched_df,
                outcome_col=dv,
                syntax_metric=metric,
                analysis_name="syntax_predicts_gaze_gait",
            )

            if fit is None:
                continue

            row, _, _ = fit
            behaviour_rows.append(row)

            plot_simple_syntax_effect(
                enriched_df,
                outcome_col=dv,
                syntax_metric=metric,
                out_path=OUTPUT_DIR / f"plot_{safe_filename(dv)}_by_{safe_filename(metric)}.png",
            )

    behaviour_df = pd.DataFrame(behaviour_rows)

    if len(behaviour_df) > 0:
        behaviour_df["p_fdr"] = multipletests(
            behaviour_df["p"],
            method="fdr_bh",
        )[1]

    behaviour_out = OUTPUT_DIR / "analysis2_syntax_predicts_gaze_gait.csv"
    behaviour_df.to_csv(behaviour_out, index=False)
    print(f"Saved Analysis 2 to: {behaviour_out}")

    # ========================================================
    # 3. Optional exploratory moderation
    # ========================================================

    if RUN_EXPLORATORY_MODERATION:
        moderation_rows = []

        for dv in available_dvs:
            for metric in syntax_metrics:
                fit = fit_exploratory_moderation(
                    enriched_df,
                    dv=dv,
                    syntax_metric=metric,
                )

                if fit is None:
                    continue

                mod_df, _, _ = fit
                moderation_rows.append(mod_df)

        if moderation_rows:
            moderation_df = pd.concat(moderation_rows, ignore_index=True)

            interaction_mask = moderation_df["term"].str.contains(
                f"{SOCIAL_DENSITY_COL}_z:",
                regex=False,
                na=False,
            )

            moderation_df["p_fdr_interaction"] = np.nan

            if interaction_mask.sum() > 0:
                moderation_df.loc[interaction_mask, "p_fdr_interaction"] = multipletests(
                    moderation_df.loc[interaction_mask, "p"],
                    method="fdr_bh",
                )[1]

            moderation_out = OUTPUT_DIR / "analysis3_exploratory_social_density_x_syntax.csv"
            moderation_df.to_csv(moderation_out, index=False)
            print(f"Saved exploratory moderation analysis to: {moderation_out}")

            moderation_interactions_out = OUTPUT_DIR / "analysis3_interaction_terms_only.csv"
            moderation_df[interaction_mask].to_csv(moderation_interactions_out, index=False)
            print(f"Saved exploratory interaction terms to: {moderation_interactions_out}")

    # ========================================================
    # Static maps
    # ========================================================

    map_cols = [
        SOCIAL_DENSITY_COL,
        "look_people_prop",
        "floor_looking_frac",
        "gaze_y",
        "horizontal_ecc",
    ] + syntax_metrics

    for col in map_cols:
        if col in enriched_df.columns:
            plot_hexbin_map(
                enriched_df,
                value_col=col,
                edges_gdf=edges_gdf,
                gridsize=HEXBIN_GRIDSIZE,
            )

    # ========================================================
    # Bivariate covariation maps:
    # syntax metric × gaze/gait feature
    # ========================================================

    if MAKE_BIVARIATE_MAPS:
        make_bivariate_covariation_maps(
            enriched_df=enriched_df,
            syntax_metrics=syntax_metrics,
            dv_features=DVS,
        )

    manifest = {
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUTPUT_DIR),
        "network_type": NETWORK_TYPE,
        "local_radius_m": LOCAL_RADIUS_METERS,
        "local_context_radius_m": LOCAL_CONTEXT_RADIUS_M,
        "isovist_max_radius_m": ISOVIST_MAX_RADIUS_M,
        "isovist_n_rays": ISOVIST_N_RAYS,
        "syntax_metrics_used": syntax_metrics,
        "dvs_used": available_dvs,
        "force_recompute_network": FORCE_RECOMPUTE_NETWORK,
    }

    with open(OUTPUT_DIR / "manifest_full_analysis.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()