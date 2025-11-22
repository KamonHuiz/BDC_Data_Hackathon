from __future__ import annotations
from typing import Optional, List, Dict, Any, Iterable
import pandas as pd
import numpy as np
from .data_loader import load_stops, load_gps, load_all_stops

# ----------------------------- Geo helpers -----------------------------

EARTH_R = 6371000.0  # meters


def _to_rad(v: np.ndarray) -> np.ndarray:
    return np.deg2rad(v.astype(float))


def haversine_np(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in meters between two equal-length arrays."""
    lon1r, lat1r, lon2r, lat2r = _to_rad(lon1), _to_rad(lat1), _to_rad(lon2), _to_rad(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_R * c

# ----------------------------- Matching core -----------------------------

def find_stop_proximity(gps_df: pd.DataFrame, stops_df: pd.DataFrame, radius_m: float = 80.0) -> pd.DataFrame:
    """Assign nearest stop to each GPS point and filter by radius.

    Returns a subset of gps_df with added columns: nearest_StopId, nearest_StopIdx, nearest_dist_m.
    """
    if gps_df.empty or stops_df.empty:
        return gps_df.head(0).copy()
    # Prepare stop arrays
    stop_lng = stops_df["Lng"].to_numpy(float)
    stop_lat = stops_df["Lat"].to_numpy(float)
    # Pre-compute an approximate bounding box filter to speed up
    min_lng, max_lng = stop_lng.min() - 0.02, stop_lng.max() + 0.02
    min_lat, max_lat = stop_lat.min() - 0.02, stop_lat.max() + 0.02
    in_box = (gps_df["lng"] >= min_lng) & (gps_df["lng"] <= max_lng) & (gps_df["lat"] >= min_lat) & (gps_df["lat"] <= max_lat)
    cand = gps_df.loc[in_box].copy()
    if cand.empty:
        return cand
    # Compute distances to all stops by broadcasting (N x M)
    glon = cand["lng"].to_numpy(float)[:, None]
    glat = cand["lat"].to_numpy(float)[:, None]
    dists = haversine_np(glon.repeat(stop_lng.shape[0], axis=1), glat.repeat(stop_lat.shape[0], axis=1), stop_lng[None, :], stop_lat[None, :])
    nearest_idx = dists.argmin(axis=1)
    nearest_dist = dists[np.arange(dists.shape[0]), nearest_idx]
    # Filter by radius
    mask = nearest_dist <= radius_m
    cand = cand.loc[mask].copy()
    if cand.empty:
        return cand
    cand["nearest_StopIdx"] = nearest_idx[mask]
    # Robust mapping to StopId by position
    cand["nearest_StopId"] = stops_df["StopId"].iloc[cand["nearest_StopIdx"].astype(int)].values
    cand["nearest_dist_m"] = nearest_dist[mask]
    return cand


def detect_door_open(gps_df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean column 'door_open' marking transitions when either door_up or door_down goes True."""
    if gps_df.empty:
        return gps_df
    df = gps_df.sort_values(["anonymized_vehicle", "datetime"]).copy()
    for col in ("door_up", "door_down"):
        if col not in df.columns:
            df[col] = False
    # Convert possibly empty strings to False
    for col in ("door_up", "door_down"):
        if df[col].dtype != bool:
            df[col] = df[col].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    df["door_any"] = df["door_up"] | df["door_down"]
    df["door_any_prev"] = df.groupby("anonymized_vehicle")["door_any"].shift(1).fillna(False)
    df["door_open"] = (~df["door_any_prev"]) & (df["door_any"])  # rising edge
    return df


def build_actual_trips_for_route(gps_df: pd.DataFrame, stops_df: pd.DataFrame, radius_m: float = 80.0, max_gap_min: int = 20) -> List[Dict[str, Any]]:
    """Construct actual trips by snapping door-open events to stops and threading monotonic sequences.

    Returns a list of trip dicts: {vehicle, start_time, end_time, stop_ids [..], stop_times [..]}
    """
    if gps_df.empty or stops_df.empty:
        return []
    df = detect_door_open(gps_df)
    prox = find_stop_proximity(df[df["door_open"]], stops_df, radius_m=radius_m)
    if prox.empty:
        return []
    # Attach stop sequence index order from stops_df index
    # nearest_StopIdx already refers to positional index in stops_df
    prox = prox.sort_values(["anonymized_vehicle", "datetime"]).copy()
    trips: List[Dict[str, Any]] = []
    for vid, g in prox.groupby("anonymized_vehicle"):
        g = g.reset_index(drop=True)
        current: Dict[str, Any] = {"vehicle": vid, "stop_ids": [], "stop_times": [], "stop_idx": []}
        last_time = None
        last_idx = -1
        for _, row in g.iterrows():
            ts = row["datetime"]
            idx = int(row["nearest_StopIdx"])
            # new segment if large time gap or index regression too big
            new_seg = False
            if last_time is not None:
                dt = (ts - last_time).total_seconds() / 60.0
                if dt > max_gap_min:
                    new_seg = True
            if last_idx >= 0 and idx + 1 < last_idx:  # large backward jump
                new_seg = True
            if new_seg and len(current["stop_ids"]) >= 2:
                trips.append({
                    "vehicle": current["vehicle"],
                    "start_time": current["stop_times"][0],
                    "end_time": current["stop_times"][-1],
                    "stop_ids": current["stop_ids"],
                    "stop_times": current["stop_times"],
                    "stop_idx": current["stop_idx"],
                })
                current = {"vehicle": vid, "stop_ids": [], "stop_times": [], "stop_idx": []}
            # append only when index strictly increases (dedupe same stop)
            if len(current["stop_idx"]) == 0 or idx > current["stop_idx"][-1]:
                current["stop_ids"].append(row["nearest_StopId"])
                current["stop_times"].append(ts)
                current["stop_idx"].append(idx)
                last_idx = idx
                last_time = ts
        if len(current["stop_ids"]) >= 2:
            trips.append({
                "vehicle": current["vehicle"],
                "start_time": current["stop_times"][0],
                "end_time": current["stop_times"][-1],
                "stop_ids": current["stop_ids"],
                "stop_times": current["stop_times"],
                "stop_idx": current["stop_idx"],
            })
    return trips


def map_trips_to_schedule(actual_trips: List[Dict[str, Any]], schedule_df: pd.DataFrame, start_time_col: str = "StartTime", route_no: Optional[str] = None, max_diff_min: int = 10) -> pd.DataFrame:
    """Map actual trips to timetable by nearest scheduled start time.

    Expected schedule_df columns (flexible, at minimum):
      - StartTime: datetime or time-like string for trip start
      - RouteNo: route number (optional if route_no is provided)
      - TripId or equivalent identifier (optional)

    Returns a DataFrame linking each actual trip to the best matching scheduled row with time difference.
    """
    if not actual_trips or schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=["vehicle", "start_time", "sched_idx", "sched_time", "diff_min"])
    sched = schedule_df.copy()
    if route_no and "RouteNo" in sched.columns:
        sched = sched[sched["RouteNo"].astype(str) == str(route_no)]
    # Normalize times to pandas datetime (today's date if only time provided)
    if not np.issubdtype(sched[start_time_col].dtype, np.datetime64):
        sched[start_time_col] = pd.to_datetime(sched[start_time_col], errors="coerce")
    matches = []
    for trip in actual_trips:
        st = pd.to_datetime(trip["start_time"]) \
            if not isinstance(trip["start_time"], pd.Timestamp) else trip["start_time"]
        # Find nearest scheduled start within threshold
        diffs = (sched[start_time_col] - st).abs().dt.total_seconds() / 60.0
        if diffs.isna().all():
            continue
        best_idx = diffs.idxmin()
        best_diff = diffs.loc[best_idx]
        if np.isnan(best_diff) or best_diff > max_diff_min:
            continue
        matches.append({
            "vehicle": trip["vehicle"],
            "start_time": st,
            "sched_idx": best_idx,
            "sched_time": sched.loc[best_idx, start_time_col],
            "diff_min": float(best_diff),
        })
    return pd.DataFrame(matches)

# ----------------------------- Convenience runner -----------------------------

def build_and_match_route_day(route_id: str, outbound: bool = True, date: Optional[str] = None, n_rows_per_file: Optional[int] = 50000, radius_m: float = 80.0) -> Dict[str, Any]:
    """Convenience: load stops and a day's GPS sample, build actual trips. schedule matching left to caller.

    Returns dict with keys: stops_df, gps_df, trips (list)
    """
    stops_df = load_stops(route_id, outbound=outbound)
    # Load just the day's file if date provided
    if date:
        gps_df = load_gps(start_date=date, end_date=date, n_rows=n_rows_per_file)
    else:
        gps_df = load_gps(n_rows=n_rows_per_file)
    trips = build_actual_trips_for_route(gps_df, stops_df, radius_m=radius_m)
    return {"stops_df": stops_df, "gps_df": gps_df, "trips": trips}


def infer_vehicle_routes(start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         outbound: Optional[bool] = None,
                         vehicles: Optional[Iterable[str]] = None,
                         n_rows_per_file: Optional[int] = 50000) -> pd.DataFrame:
    """Infer route number per vehicle by snapping GPS points to nearest stops across all routes.

    Returns a DataFrame with columns:
      anonymized_vehicle, RouteNo, RouteId, Outbound, hits, total_points, hit_ratio, mean_dist_m
    """
    stops_all = load_all_stops(outbound=outbound)
    if stops_all.empty:
        return pd.DataFrame(columns=["anonymized_vehicle","RouteNo","RouteId","Outbound","hits","total_points","hit_ratio","mean_dist_m"])
    gps = load_gps(start_date=start_date, end_date=end_date, vehicles=vehicles, n_rows=n_rows_per_file)
    if gps.empty:
        return pd.DataFrame(columns=["anonymized_vehicle","RouteNo","RouteId","Outbound","hits","total_points","hit_ratio","mean_dist_m"])

    # Snap all GPS points (not only door events) to nearest stops to maximize signal
    prox = find_stop_proximity(gps, stops_all, radius_m=120.0)
    if prox.empty:
        # No points near stops
        res = gps.groupby('anonymized_vehicle', as_index=False).size().rename(columns={'size':'total_points'})
        res["hits"] = 0
        res["hit_ratio"] = 0.0
        res["mean_dist_m"] = np.nan
        res["RouteNo"] = None
        res["RouteId"] = None
        res["Outbound"] = None
        cols = ["anonymized_vehicle","RouteNo","RouteId","Outbound","hits","total_points","hit_ratio","mean_dist_m"]
        return res.reindex(columns=cols)

    # Map nearest_StopIdx to route metadata
    meta_map = stops_all[["RouteNo","RouteId","Outbound"]].reset_index(drop=True)
    sel_meta = meta_map.iloc[prox["nearest_StopIdx"].astype(int)].reset_index(drop=True)
    prox = prox.reset_index(drop=True)
    prox = pd.concat([prox, sel_meta], axis=1)

    # Aggregate hits per vehicle-route-direction
    grp = prox.groupby(["anonymized_vehicle","RouteNo","RouteId","Outbound"], as_index=False).agg(
        hits=("nearest_dist_m","size"),
        mean_dist_m=("nearest_dist_m","mean")
    )

    totals = gps.groupby("anonymized_vehicle", as_index=False).size().rename(columns={"size":"total_points"})
    out = grp.merge(totals, on="anonymized_vehicle", how="left")
    out["hit_ratio"] = out["hits"] / out["total_points"].replace(0, np.nan)

    # Pick best route per vehicle by highest hit_ratio, tie-break by smallest mean_dist_m
    out_sorted = out.sort_values(["anonymized_vehicle","hit_ratio","hits","mean_dist_m"], ascending=[True, False, False, True])
    best = out_sorted.groupby("anonymized_vehicle").head(1).reset_index(drop=True)
    cols = ["anonymized_vehicle","RouteNo","RouteId","Outbound","hits","total_points","hit_ratio","mean_dist_m"]
    return best.reindex(columns=cols)
