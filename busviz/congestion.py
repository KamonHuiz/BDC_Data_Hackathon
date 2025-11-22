from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from .data_loader import load_gps

# ----------------------------------------------------------------------------
# Congestion hotspot computation (grid clustering of low-speed points)
# ----------------------------------------------------------------------------

def compute_congestion_hotspots(start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                hour_range: Optional[Tuple[int, int]] = None,
                                speed_threshold_kmh: float = 10.0,
                                min_points_cell: int = 15,
                                min_points_cluster: int = 30,
                                grid_size_m: float = 80.0,
                                sample_n: Optional[int] = None,
                                vehicle_ids: Optional[list[str]] = None,
                                minute_start: Optional[int] = None,
                                minute_end: Optional[int] = None,
                                return_debug: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute congestion hotspots by clustering low-speed GPS points.

    Added minute_start/minute_end to allow sub-hour (e.g. 07:15-07:29) windows when hour_range spans one hour.
    """
    debug: Dict[str, Any] = {
        "total_points": 0,
        "after_hour_filter_points": 0,
        "after_minute_filter_points": 0,
        "slow_points": 0,
        "candidate_cells": 0,
        "clusters_found": 0,
        "clusters_passing_threshold": 0,
        "params": {
            "hour_range": hour_range,
            "minute_start": minute_start,
            "minute_end": minute_end,
            "speed_threshold_kmh": speed_threshold_kmh,
            "min_points_cell": min_points_cell,
            "min_points_cluster": min_points_cluster,
            "grid_size_m": grid_size_m,
            "sample_n": sample_n,
        }
    }
    gps = load_gps(start_date=start_date, end_date=end_date, vehicles=vehicle_ids, n_rows=sample_n)
    debug["total_points"] = len(gps)
    if gps.empty:
        empty_df = pd.DataFrame(columns=["cluster_id","n_points","mean_speed_kmh","median_speed_kmh","p25_speed_kmh","p75_speed_kmh","min_speed_kmh","max_speed_kmh","slow_ratio","center_lat","center_lng","radius_m"])
        return (empty_df, debug) if return_debug else empty_df
    if not np.issubdtype(gps["datetime"].dtype, np.datetime64):
        gps["datetime"] = pd.to_datetime(gps["datetime"], errors="coerce")
    gps = gps.dropna(subset=["datetime","lat","lng","speed"])  # minimal required columns
    if hour_range:
        h0, h1 = hour_range
        gps = gps[(gps["datetime"].dt.hour >= h0) & (gps["datetime"].dt.hour <= h1)]
    debug["after_hour_filter_points"] = len(gps)
    # Minute-level filtering only if provided
    if minute_start is not None and minute_end is not None:
        # supports minute_end < minute_start (wrap) though not needed here
        if minute_start <= minute_end:
            gps = gps[(gps["datetime"].dt.minute >= minute_start) & (gps["datetime"].dt.minute <= minute_end)]
        else:
            gps = gps[(gps["datetime"].dt.minute >= minute_start) | (gps["datetime"].dt.minute <= minute_end)]
    debug["after_minute_filter_points"] = len(gps)
    if gps.empty:
        empty_df = pd.DataFrame(columns=["cluster_id","n_points","mean_speed_kmh","median_speed_kmh","p25_speed_kmh","p75_speed_kmh","min_speed_kmh","max_speed_kmh","slow_ratio","center_lat","center_lng","radius_m"])
        return (empty_df, debug) if return_debug else empty_df
    gps["is_slow"] = gps["speed"].astype(float) <= speed_threshold_kmh
    slow = gps[gps["is_slow"]].copy()
    debug["slow_points"] = len(slow)
    if slow.empty:
        empty_df = pd.DataFrame(columns=["cluster_id","n_points","mean_speed_kmh","median_speed_kmh","p25_speed_kmh","p75_speed_kmh","min_speed_kmh","max_speed_kmh","slow_ratio","center_lat","center_lng","radius_m"])
        return (empty_df, debug) if return_debug else empty_df
    lat0 = slow["lat"].median()
    lng0 = slow["lng"].median()
    lat_rad = math.radians(lat0)
    m_per_deg_lat = 111320.0
    m_per_deg_lng = 111320.0 * math.cos(lat_rad)
    slow["x_m"] = (slow["lng"] - lng0) * m_per_deg_lng
    slow["y_m"] = (slow["lat"] - lat0) * m_per_deg_lat
    slow["gx"] = np.floor_divide(slow["x_m"].astype(float), grid_size_m).astype(int)
    slow["gy"] = np.floor_divide(slow["y_m"].astype(float), grid_size_m).astype(int)
    cell_grp = slow.groupby(["gx","gy"], as_index=False).agg(
        n_points=("speed","size"),
        mean_speed_kmh=("speed","mean"),
        median_speed_kmh=("speed","median"),
        p25_speed_kmh=("speed", lambda s: float(np.percentile(s,25))),
        p75_speed_kmh=("speed", lambda s: float(np.percentile(s,75))),
        min_speed_kmh=("speed","min"),
        max_speed_kmh=("speed","max"),
    )
    cell_grp = cell_grp[cell_grp["n_points"] >= min_points_cell].copy()
    debug["candidate_cells"] = len(cell_grp)
    if cell_grp.empty:
        empty_df = pd.DataFrame(columns=["cluster_id","n_points","mean_speed_kmh","median_speed_kmh","p25_speed_kmh","p75_speed_kmh","min_speed_kmh","max_speed_kmh","slow_ratio","center_lat","center_lng","radius_m"])
        return (empty_df, debug) if return_debug else empty_df
    occupied = {(int(r.gx), int(r.gy)) for r in cell_grp.itertuples()}
    visited: set[Tuple[int,int]] = set()
    clusters: list[list[Tuple[int,int]]] = []
    for cell in occupied:
        if cell in visited:
            continue
        stack = [cell]
        comp: list[Tuple[int,int]] = []
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            comp.append((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nb = (cx+dx, cy+dy)
                if nb in occupied and nb not in visited:
                    stack.append(nb)
        clusters.append(comp)
    debug["clusters_found"] = len(clusters)
    rows: list[Dict[str, Any]] = []
    gps["x_m"] = (gps["lng"] - lng0) * m_per_deg_lng
    gps["y_m"] = (gps["lat"] - lat0) * m_per_deg_lat
    gps["gx"] = np.floor_divide(gps["x_m"].astype(float), grid_size_m).astype(int)
    gps["gy"] = np.floor_divide(gps["y_m"].astype(float), grid_size_m).astype(int)
    passing_clusters = 0
    for cid, comp in enumerate(clusters):
        comp_df = cell_grp.merge(pd.DataFrame(comp, columns=["gx","gy"]), on=["gx","gy"], how="inner")
        total_points_cluster = gps.merge(pd.DataFrame(comp, columns=["gx","gy"]), on=["gx","gy"], how="inner")
        n_points = int(comp_df["n_points"].sum())
        if n_points < min_points_cluster:
            continue
        passing_clusters += 1
        speeds = slow.merge(pd.DataFrame(comp, columns=["gx","gy"]), on=["gx","gy"], how="inner")["speed"].astype(float)
        mean_speed = float(speeds.mean())
        median_speed = float(speeds.median())
        p25_speed = float(np.percentile(speeds,25))
        p75_speed = float(np.percentile(speeds,75))
        min_speed = float(speeds.min())
        max_speed = float(speeds.max())
        cluster_points = slow.merge(pd.DataFrame(comp, columns=["gx","gy"]), on=["gx","gy"], how="inner")
        center_lat = float(cluster_points["lat"].mean())
        center_lng = float(cluster_points["lng"].mean())
        xs = [c[0] for c in comp]; ys = [c[1] for c in comp]
        span_x = (max(xs) - min(xs) + 1) * grid_size_m
        span_y = (max(ys) - min(ys) + 1) * grid_size_m
        radius_m = 0.5 * math.sqrt(span_x**2 + span_y**2)
        slow_ratio = n_points / max(len(total_points_cluster), 1)
        rows.append({
            "cluster_id": cid,
            "n_points": n_points,
            "mean_speed_kmh": mean_speed,
            "median_speed_kmh": median_speed,
            "p25_speed_kmh": p25_speed,
            "p75_speed_kmh": p75_speed,
            "min_speed_kmh": min_speed,
            "max_speed_kmh": max_speed,
            "slow_ratio": float(slow_ratio),
            "center_lat": center_lat,
            "center_lng": center_lng,
            "radius_m": radius_m,
        })
    debug["clusters_passing_threshold"] = passing_clusters
    if not rows:
        empty_df = pd.DataFrame(columns=["cluster_id","n_points","mean_speed_kmh","median_speed_kmh","p25_speed_kmh","p75_speed_kmh","min_speed_kmh","max_speed_kmh","slow_ratio","center_lat","center_lng","radius_m"])
        return (empty_df, debug) if return_debug else empty_df
    out = pd.DataFrame(rows).sort_values("n_points", ascending=False).reset_index(drop=True)
    return (out, debug) if return_debug else out
