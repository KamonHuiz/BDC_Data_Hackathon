from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any

# Robust import handling: allow running as standalone script inside package directory
try:
    from .data_loader import load_stops, load_gps
    from .gps_matching import build_actual_trips_for_route, haversine_np, find_stop_proximity
except ImportError:
    import sys
    _CUR_DIR = Path(__file__).resolve().parent
    _ROOT_DIR = _CUR_DIR.parent
    # Ensure root and current dir are on sys.path for direct script execution
    for p in (str(_ROOT_DIR), str(_CUR_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        # Try absolute package path if available
        from busviz.data_loader import load_stops, load_gps  # type: ignore
        from busviz.gps_matching import build_actual_trips_for_route, haversine_np, find_stop_proximity  # type: ignore
    except ImportError:
        # Final fallback: local module imports
        from data_loader import load_stops, load_gps  # type: ignore
        from gps_matching import build_actual_trips_for_route, haversine_np, find_stop_proximity  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
AGG_VEHICLE_ROUTES = ROOT / "gps_to_routes" / "vehicle_routes" / "aggregated_vehicle_routes.csv"

SEGMENT_COLUMNS = [
    "date","route_id","outbound","vehicle","trip_index","segment_index",
    "from_stop_id","from_code","to_stop_id","to_code",
    "depart_time","arrive_time","travel_min","distance_m"
]


def _load_route_assigned_vehicles(route_id: str, outbound: Optional[bool], start_date: Optional[str], end_date: Optional[str]) -> List[str]:
    """Load vehicles inferred to be on a specific route/direction within date range from aggregated_vehicle_routes.csv.
    Falls back to empty list if file missing or no matches (caller may still attempt matching all vehicles).
    """
    if not AGG_VEHICLE_ROUTES.exists():
        return []
    try:
        df = pd.read_csv(AGG_VEHICLE_ROUTES, encoding="utf-8")
    except Exception:
        return []
    # Normalize types
    df["RouteId"] = df["RouteId"].astype(str)
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    df = df[df["RouteId"] == str(route_id)]
    if outbound is not None and "Outbound" in df.columns:
        df = df[df["Outbound"].astype(bool) == bool(outbound)]
    return sorted(df["anonymized_vehicle"].unique().tolist())


def compute_route_stop_segments(route_id: str,
                                outbound: bool = True,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                radius_m: float = 80.0,
                                max_gap_min: int = 20,
                                n_rows_per_file: Optional[int] = None,
                                filter_by_inferred: bool = True,
                                mode: str = "door",  # 'door' uses door events; 'all' uses all GPS points
                                strict_adjacent: bool = True) -> pd.DataFrame:
    """Compute per-day stop-to-stop travel segments for a route direction.

    Parameters:
        route_id: Directory name under Bus_route_data/HCMC_bus_routes (e.g. '4').
        outbound: True => use stops_by_var.csv; False => use rev_stops_by_var.csv.
        start_date/end_date: Inclusive YYYY-MM-DD bounds. None => all available.
        radius_m: Proximity radius for snapping door-open events to stops.
        max_gap_min: Gap threshold separating distinct trips.
        n_rows_per_file: Limit GPS rows per file (for sampling/performance). None => full file.
        filter_by_inferred: If True, restrict GPS to vehicles previously inferred as on this route/direction.
        mode: 'door' uses door events; 'all' uses all GPS points.
        strict_adjacent: If True, enforces strict stop adjacency (no skipping stops).

    Returns DataFrame with columns:
        date, route_id, outbound, vehicle, trip_index, segment_index,
        from_stop_id, from_code, to_stop_id, to_code,
        depart_time, arrive_time, travel_min, distance_m (straight-line haversine meters)
    """
    stops_df = load_stops(route_id, outbound=outbound).copy()
    if stops_df.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    vehicles: Optional[Iterable[str]] = None
    if filter_by_inferred:
        vehicles = _load_route_assigned_vehicles(route_id, outbound, start_date, end_date)
        # If no inferred vehicles, fall back to all
        if not vehicles:
            vehicles = None

    gps_df = load_gps(start_date=start_date, end_date=end_date, vehicles=vehicles, n_rows=n_rows_per_file)
    if gps_df.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    if mode == "door":
        trips = build_actual_trips_for_route(gps_df, stops_df, radius_m=radius_m, max_gap_min=max_gap_min)
        seq_trips = []
        for tr in trips:
            # Each trip already contains stop_idx list
            seq_trips.append(tr)
    elif mode == "all":
        # Use proximity of all points, then build monotonic sequences per vehicle
        prox = find_stop_proximity(gps_df, stops_df, radius_m=radius_m)
        if prox.empty:
            return pd.DataFrame(columns=SEGMENT_COLUMNS)
        prox = prox.sort_values(["anonymized_vehicle","datetime"]).copy()
        seq_trips = []
        for vid, g in prox.groupby("anonymized_vehicle"):
            # Deduplicate consecutive same stop, keep first timestamp
            g = g.reset_index(drop=True)
            last_idx = None
            stops_seq: List[int] = []
            times_seq: List[pd.Timestamp] = []
            for _, row in g.iterrows():
                idx = int(row["nearest_StopIdx"])
                ts = row["datetime"]
                if last_idx is None or idx != last_idx:
                    # enforce monotonic increasing order (route direction) else treat as new trip
                    if last_idx is not None and idx <= last_idx:
                        # close current trip if length >=2
                        if len(stops_seq) >= 2:
                            seq_trips.append({
                                "vehicle": vid,
                                "stop_ids": [stops_df.iloc[i]["StopId"] for i in stops_seq],
                                "stop_times": times_seq,
                                "stop_idx": stops_seq,
                            })
                        stops_seq = []
                        times_seq = []
                    stops_seq.append(idx)
                    times_seq.append(ts)
                    last_idx = idx
            if len(stops_seq) >= 2:
                seq_trips.append({
                    "vehicle": vid,
                    "stop_ids": [stops_df.iloc[i]["StopId"] for i in stops_seq],
                    "stop_times": times_seq,
                    "stop_idx": stops_seq,
                })
    else:
        raise ValueError("mode must be 'door' or 'all'")

    if not seq_trips:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)

    stop_meta = stops_df.set_index("StopId")["Code"].to_dict()
    lng_arr = stops_df["Lng"].to_numpy(float)
    lat_arr = stops_df["Lat"].to_numpy(float)

    rows: List[Dict[str, Any]] = []
    for t_i, trip in enumerate(seq_trips):
        stop_ids = trip.get("stop_ids", [])
        stop_times = trip.get("stop_times", [])
        vehicle = trip.get("vehicle")
        idx_seq = trip.get("stop_idx", [])
        if len(stop_ids) < 2:
            continue
        for s_i in range(len(stop_ids) - 1):
            idx_current = idx_seq[s_i]
            idx_next = idx_seq[s_i + 1]
            if strict_adjacent and idx_next != idx_current + 1:
                continue  # skip non-adjacent jump
            from_id = stop_ids[s_i]
            to_id = stop_ids[s_i + 1]
            depart_time = pd.to_datetime(stop_times[s_i])
            arrive_time = pd.to_datetime(stop_times[s_i + 1])
            travel_min = (arrive_time - depart_time).total_seconds() / 60.0
            dist_m = haversine_np(
                np.array([lng_arr[idx_current]]),
                np.array([lat_arr[idx_current]]),
                np.array([lng_arr[idx_next]]),
                np.array([lat_arr[idx_next]]),
            )[0]
            rows.append({
                "date": depart_time.date().isoformat(),
                "route_id": str(route_id),
                "outbound": bool(outbound),
                "vehicle": vehicle,
                "trip_index": t_i,
                "segment_index": s_i,
                "from_stop_id": from_id,
                "from_code": stop_meta.get(from_id),
                "to_stop_id": to_id,
                "to_code": stop_meta.get(to_id),
                "depart_time": depart_time,
                "arrive_time": arrive_time,
                "travel_min": travel_min,
                "distance_m": float(dist_m),
            })

    if not rows:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    out_df = pd.DataFrame(rows).sort_values(["date","vehicle","trip_index","segment_index"]).reset_index(drop=True)
    return out_df[SEGMENT_COLUMNS]


def aggregate_segment_stats(segments_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per unique stop-to-stop pair across all days.

    Returns columns:
      route_id, outbound, from_stop_id, from_code, to_stop_id, to_code,
      n_segments, mean_travel_min, median_travel_min, std_travel_min,
      min_travel_min, max_travel_min, mean_distance_m
    """
    if segments_df.empty:
        return pd.DataFrame(columns=["route_id","outbound","from_stop_id","from_code","to_stop_id","to_code","n_segments","mean_travel_min","median_travel_min","std_travel_min","min_travel_min","max_travel_min","mean_distance_m"])
    grp = segments_df.groupby(["route_id","outbound","from_stop_id","from_code","to_stop_id","to_code"], as_index=False).agg(
        n_segments=("travel_min","size"),
        mean_travel_min=("travel_min","mean"),
        median_travel_min=("travel_min","median"),
        std_travel_min=("travel_min","std"),
        min_travel_min=("travel_min","min"),
        max_travel_min=("travel_min","max"),
        mean_distance_m=("distance_m","mean"),
    )
    return grp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute stop-to-stop segments for a route")
    ap.add_argument("route_id", type=str, help="Route id directory (e.g. 4)")
    ap.add_argument("--inbound", action="store_true", help="Use inbound (rev_stops_by_var.csv) instead of outbound")
    ap.add_argument("--start_date", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end_date", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--radius_m", type=float, default=80.0)
    ap.add_argument("--max_gap_min", type=int, default=20)
    ap.add_argument("--limit_rows", type=int, default=None, help="Limit rows per GPS file for sampling")
    ap.add_argument("--no_filter_inferred", action="store_true", help="Do not restrict to vehicles inferred for the route")
    ap.add_argument("--mode", type=str, choices=["door","all"], default="door", help="Sequence construction mode")
    ap.add_argument("--no_strict_adjacent", action="store_true", help="Allow non-adjacent stop jumps")
    ap.add_argument("--output", type=str, default=None, help="Optional output CSV path for raw segments")
    ap.add_argument("--agg_output", type=str, default=None, help="Optional output CSV path for aggregated stats")
    args = ap.parse_args()

    df_segments = compute_route_stop_segments(
        route_id=args.route_id,
        outbound=not args.inbound,
        start_date=args.start_date,
        end_date=args.end_date,
        radius_m=args.radius_m,
        max_gap_min=args.max_gap_min,
        n_rows_per_file=args.limit_rows,
        filter_by_inferred=not args.no_filter_inferred,
        mode=args.mode,
        strict_adjacent=not args.no_strict_adjacent,
    )

    if args.output:
        df_segments.to_csv(args.output, index=False)
        print(f"Wrote {len(df_segments)} segments to {args.output}")
    else:
        print(df_segments.head())
        print(f"Total segments: {len(df_segments)}")

    if args.agg_output:
        df_agg = aggregate_segment_stats(df_segments)
        df_agg.to_csv(args.agg_output, index=False)
        print(f"Wrote aggregated stats ({len(df_agg)} rows) to {args.agg_output}")
