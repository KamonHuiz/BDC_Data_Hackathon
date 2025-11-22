from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Reuse logic from route_segments with robust import fallbacks
try:
    from .route_segments import compute_route_stop_segments, aggregate_segment_stats
    from .data_loader import load_gps, load_stops
    from .gps_matching import find_stop_proximity
except ImportError:
    from busviz.route_segments import compute_route_stop_segments, aggregate_segment_stats  # type: ignore
    from busviz.data_loader import load_gps, load_stops  # type: ignore
    from busviz.gps_matching import find_stop_proximity  # type: ignore

DEF_MODE = "all"


def _daily_diagnostics(route_id: str,
                       outbound: bool,
                       date_str: str,
                       radius_m: float,
                       vehicles_used: List[str],
                       filter_by_inferred: bool) -> Dict[str, any]:
    stops_df = load_stops(route_id, outbound=outbound)
    gps_df = load_gps(start_date=date_str, end_date=date_str)
    n_gps = len(gps_df)
    n_unique_vehicles_all = gps_df['anonymized_vehicle'].nunique() if not gps_df.empty else 0
    prox_df = find_stop_proximity(gps_df, stops_df, radius_m=radius_m)
    n_prox = len(prox_df)
    vehicles_near = sorted(prox_df['anonymized_vehicle'].unique().tolist()) if not prox_df.empty else []
    vehicles_all = sorted(gps_df['anonymized_vehicle'].unique().tolist()) if not gps_df.empty else []
    vehicles_missing = [v for v in vehicles_all if v not in vehicles_used]
    return {
        'date': date_str,
        'route_id': route_id,
        'outbound': outbound,
        'gps_points': n_gps,
        'gps_points_near_stops': n_prox,
        'vehicles_all': ';'.join(vehicles_all),
        'vehicles_near_stops': ';'.join(vehicles_near),
        'vehicles_used_for_segments': ';'.join(vehicles_used),
        'n_vehicles_all': n_unique_vehicles_all,
        'n_vehicles_near_stops': len(vehicles_near),
        'n_vehicles_used': len(vehicles_used),
        'vehicles_excluded': ';'.join(vehicles_missing),
        'filter_by_inferred': filter_by_inferred,
        'mode': DEF_MODE,
        'radius_m': radius_m,
    }


def generate_daily_segments(route_id: str,
                             outbound: bool,
                             start_date: str,
                             end_date: str,
                             output_dir: Path,
                             mode: str = DEF_MODE,
                             radius_m: float = 80.0,
                             max_gap_min: int = 20,
                             filter_by_inferred: bool = True,
                             strict_adjacent: bool = True,
                             diagnostics_output: Optional[Path] = None) -> Path:
    """Generate one CSV per day with adjacent stop-to-stop segments and an aggregated summary.

    Output structure:
      output_dir/
        segments_<routeId>_<dir>_<YYYY-MM-DD>.csv  (daily raw segments)
        aggregated_segments_<routeId>_<dir>_<start>_<end>.csv (summary across all days)

    Returns path to aggregated summary file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    all_rows = []
    diag_rows: List[Dict] = []
    for d in dates:
        date_str = d.date().isoformat()
        df = compute_route_stop_segments(
            route_id=route_id,
            outbound=outbound,
            start_date=date_str,
            end_date=date_str,
            mode=mode,
            radius_m=radius_m,
            max_gap_min=max_gap_min,
            filter_by_inferred=filter_by_inferred,
            strict_adjacent=strict_adjacent,
        )
        daily_fname = f"segments_{route_id}_{'outbound' if outbound else 'inbound'}_{date_str}.csv"
        daily_path = output_dir / daily_fname
        df.to_csv(daily_path, index=False)
        if not df.empty:
            all_rows.append(df)
        # Diagnostics gather vehicles used
        vehicles_used = sorted(df['vehicle'].unique().tolist()) if not df.empty else []
        diag_rows.append(_daily_diagnostics(route_id, outbound, date_str, radius_m, vehicles_used, filter_by_inferred))
        print(f"Wrote {len(df)} rows -> {daily_path}")
    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
    else:
        # Empty aggregated file
        full_df = pd.DataFrame(columns=[
            "date","route_id","outbound","vehicle","trip_index","segment_index",
            "from_stop_id","from_code","to_stop_id","to_code","depart_time","arrive_time","travel_min","distance_m"
        ])
    agg = aggregate_segment_stats(full_df)
    agg_fname = f"aggregated_segments_{route_id}_{'outbound' if outbound else 'inbound'}_{start_date}_{end_date}.csv"
    agg_path = output_dir / agg_fname
    agg.to_csv(agg_path, index=False)
    print(f"Wrote aggregated summary ({len(agg)} rows) -> {agg_path}")
    if diagnostics_output:
        diag_df = pd.DataFrame(diag_rows)
        diag_df.to_csv(diagnostics_output, index=False)
        print(f"Wrote diagnostics ({len(diag_df)} days) -> {diagnostics_output}")
    return agg_path


def _valid_date(s: str) -> str:
    datetime.strptime(s, "%Y-%m-%d")  # raises if invalid
    return s

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate per-day route segments and aggregated summary")
    ap.add_argument("route_id", type=str)
    ap.add_argument("--inbound", action="store_true", help="Use inbound direction (rev_stops_by_var.csv)")
    ap.add_argument("--start_date", type=str, required=True)
    ap.add_argument("--end_date", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="processed_data/route_segments")
    ap.add_argument("--mode", choices=["door","all"], default= DEF_MODE)
    ap.add_argument("--radius_m", type=float, default=80.0)
    ap.add_argument("--max_gap_min", type=int, default=20)
    ap.add_argument("--no_filter_inferred", action="store_true")
    ap.add_argument("--no_strict_adjacent", action="store_true")
    ap.add_argument("--diagnostics", type=str, default=None, help="Optional diagnostics CSV path")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) / f"{args.route_id}_{'outbound' if not args.inbound else 'inbound'}"
    diag_path = Path(args.diagnostics) if args.diagnostics else None
    generate_daily_segments(
        route_id=args.route_id,
        outbound=not args.inbound,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=out_dir,
        mode=args.mode,
        radius_m=args.radius_m,
        max_gap_min=args.max_gap_min,
        filter_by_inferred=not args.no_filter_inferred,
        strict_adjacent=not args.no_strict_adjacent,
        diagnostics_output=diag_path,
    )
