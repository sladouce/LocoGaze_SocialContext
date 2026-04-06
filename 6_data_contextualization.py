#!/usr/bin/env python3
"""
annotate_distance_and_motion.py

Combines:
- Distance classification (close vs far) based on smoothed bbox area
- Motion classification (approaching / going away / stable / unknown)
  based on linear trend of bbox area over time.

Input:
    {reldir}/output/tracking_gaze_deepsort.json
    {reldir}/scenevideo.mp4   (for FPS to define smoothing window)

Output:
    {reldir}/output/tracking_with_distance_motion.json

Each object in each frame gets:
    obj['distance'] ∈ {'close', 'far'}
    obj['motion']   ∈ {'approaching', 'going away', 'stable', 'unknown'}
"""

import json
import numpy as np
import cv2
from collections import defaultdict
import os
import pandas as pd

# ----------------- Helpers (robust parsing) -----------------
def _to_float_or_nan(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _parse_bbox(bbox):
    """Accepts [x1,y1,w,h] or dict; returns (x1,y1,w,h) as floats or None if unusable."""
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1, y1, w, h = bbox[:4]
    elif isinstance(bbox, dict):
        x1 = bbox.get('x1', bbox.get('left', bbox.get(0)))
        y1 = bbox.get('y1', bbox.get('top',  bbox.get(1)))
        w  = bbox.get('w',  bbox.get('width', bbox.get(2)))
        h  = bbox.get('h',  bbox.get('height',bbox.get(3)))
    else:
        return None
    x1, y1, w, h = _to_float_or_nan(x1), _to_float_or_nan(y1), _to_float_or_nan(w), _to_float_or_nan(h)
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(w) and np.isfinite(h)):
        return None
    if w <= 0 or h <= 0:
        return None
    return x1, y1, w, h

# ----------------- Config -----------------
# Load metadata for relative folder
META_CSV  = r'C:/LocoGaze/data/metadata.csv'
meta      = pd.read_csv(META_CSV, nrows=1)
reldir    = meta.at[0, 'reldir']

BASE_DIR  = r'C:/LocoGaze/data/'
input_dir = os.path.join(BASE_DIR, reldir)
output_dir = os.path.join(input_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Filepaths
INPUT_JSON  = os.path.join(output_dir, 'tracking_gaze_deepsort.json')
ORIG_VIDEO  = os.path.join(input_dir, 'scenevideo.mp4')
OUTPUT_JSON = os.path.join(output_dir, 'tracking_with_distance_motion.json')

# --- Distance classification parameters ---
VIDEO_ONSET_SHIFT_S = 0.5   # not used here, but kept for compatibility if needed later
AREA_CLOSE_THRESH   = 80000  # px²; adjust to your scene
SMOOTH_WINDOW_SEC   = 1.0    # seconds

# --- Motion classification parameters ---
WINDOW_SIZE     = 25      # frames for trend estimation
MIN_POINTS      = 5       # minimum points to compute trend
SLOPE_THRESHOLD = 5000    # area change/sec threshold

# ----------------- 1) Load JSON -----------------
with open(INPUT_JSON, 'r') as f:
    frames = json.load(f)

if not isinstance(frames, list):
    raise ValueError("Expected frames to be a list in tracking_gaze_deepsort.json")

# ----------------- 2) Get FPS and smoothing window in frames -----------------
cap = cv2.VideoCapture(ORIG_VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open original video: {ORIG_VIDEO}")
FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()

SMOOTH_WINDOW_FRAMES = max(1, int(round(SMOOTH_WINDOW_SEC * FPS)))

# ----------------- 3) Build per-track history: (frame_idx, time_s, area) -----------------
# track_history[track_id] = list of (frame_idx, time_s, area)
track_history = defaultdict(list)

for idx, fr in enumerate(frames):
    t = _to_float_or_nan(fr.get('time_s', np.nan))
    objs = fr.get('objects', [])
    if not isinstance(objs, (list, tuple)):
        continue
    for obj in objs:
        tid = obj.get('id', None)
        if tid is None:
            continue
        bb = _parse_bbox(obj.get('bbox'))
        if bb is None:
            continue
        x, y, w, h = bb
        area = float(w * h)
        track_history[tid].append((idx, t, area))

# Ensure histories are sorted by frame index
for tid in track_history:
    track_history[tid].sort(key=lambda x: x[0])

# ----------------- 4) Distance labels (close/far) via smoothed area -----------------
distance_labels = defaultdict(dict)  # distance_labels[tid][frame_idx] = 'close'/'far'

for tid, hist in track_history.items():
    if not hist:
        continue
    frame_idxs = [h[0] for h in hist]
    areas      = [h[2] for h in hist]
    n = len(areas)
    for j, fi in enumerate(frame_idxs):
        # trailing-window median over SMOOTH_WINDOW_FRAMES *frames*
        start_j = max(0, j - SMOOTH_WINDOW_FRAMES + 1)
        window_areas = areas[start_j : j+1]
        smooth_area = float(np.median(window_areas)) if window_areas else 0.0
        label = 'close' if smooth_area >= AREA_CLOSE_THRESH else 'far'
        distance_labels[tid][fi] = label

# ----------------- 5) Motion labels via linear regression on area over time -----------------
# slopes[tid][frame_idx] = slope (area change per second)
slopes = defaultdict(dict)

for tid, hist in track_history.items():
    m_hist = len(hist)
    if m_hist < MIN_POINTS:
        continue

    for i in range(m_hist):
        # Need at least MIN_POINTS in current window
        if i + 1 < MIN_POINTS:
            continue
        start_i = max(0, i - WINDOW_SIZE + 1)
        pts = hist[start_i:i+1]
        if len(pts) < MIN_POINTS:
            continue

        times = np.array([pt[1] for pt in pts], dtype=float)
        areas = np.array([pt[2] for pt in pts], dtype=float)

        # Exclude NaN times if any
        valid = np.isfinite(times)
        if valid.sum() < MIN_POINTS:
            continue

        t_fit = times[valid]
        a_fit = areas[valid]

        if np.allclose(t_fit, t_fit[0]):
            # Degenerate (no time variation); skip
            continue

        m, _ = np.polyfit(t_fit, a_fit, 1)  # slope = area change per second
        frame_idx_last = pts[-1][0]
        slopes[tid][frame_idx_last] = m

# Convert slopes into motion labels
motion_labels = defaultdict(dict)  # motion_labels[tid][frame_idx] = 'approaching'/'going away'/'stable'/'unknown'

for tid, hist in track_history.items():
    for (fi, t, area) in hist:
        m = slopes.get(tid, {}).get(fi, None)
        if m is None:
            label = 'unknown'
        elif m > SLOPE_THRESHOLD:
            label = 'approaching'
        elif m < -SLOPE_THRESHOLD:
            label = 'going away'
        else:
            label = 'stable'
        motion_labels[tid][fi] = label

# ----------------- 6) Annotate each frame/object with distance + motion labels -----------------
for idx, fr in enumerate(frames):
    objs = fr.get('objects', [])
    if not isinstance(objs, (list, tuple)):
        continue
    for obj in objs:
        tid = obj.get('id', None)
        # Distance: default 'far' if no info
        obj['distance'] = distance_labels.get(tid, {}).get(idx, 'far')
        # Motion: default 'unknown' if no info
        obj['motion']   = motion_labels.get(tid, {}).get(idx, 'unknown')

# ----------------- 7) Save combined annotated JSON -----------------
with open(OUTPUT_JSON, 'w') as f:
    json.dump(frames, f, indent=2)

print(f"Saved annotated JSON with distance + motion labels: {OUTPUT_JSON}")
