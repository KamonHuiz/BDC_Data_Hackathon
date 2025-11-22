from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from .visualize import congestion_hotspots_map
from .congestion import compute_congestion_hotspots

WINDOWS = [

    (5, 0, 14),
    (5, 15, 29),
    (5, 30, 44),
    (5, 45, 59),

    (6, 0, 14),
    (6, 15, 29),
    (6, 30, 44),
    (6, 45, 59),

    (7, 0, 14),
    (7, 15, 29),
    (7, 30, 44),
    (7, 45, 59),

    (8, 0, 14),
    (8, 15, 29),
    (8, 30, 44),
    (8, 45, 59),

    (9, 0, 14),
    (9, 15, 29),
    (9, 30, 44),
    (9, 45, 59),

    (10, 0, 14),
    (10, 15, 29),
    (10, 30, 44),
    (10, 45, 59),

    (11, 0, 14),
    (11, 15, 29),
    (11, 30, 44),
    (11, 45, 59),

    (12, 0, 14),
    (12, 15, 29),
    (12, 30, 44),
    (12, 45, 59),

    (13, 0, 14),
    (13, 15, 29),
    (13, 30, 44),
    (13, 45, 59),

    (14, 0, 14),
    (14, 15, 29),
    (14, 30, 44),
    (14, 45, 59),

    (15, 0, 14),
    (15, 15, 29),
    (15, 30, 44),
    (15, 45, 59),

    (16, 0, 14),
    (16, 15, 29),
    (16, 30, 44),
    (16, 45, 59),

    (17, 0, 14),
    (17, 15, 29),
    (17, 30, 44),
    (17, 45, 59),

    (18, 0, 14),
    (18, 15, 29),
    (18, 30, 44),
    (18, 45, 59),

    (19, 0, 14),
    (19, 15, 29),
    (19, 30, 44),
    (19, 45, 59),

    (20, 0, 14),
    (20, 15, 29),
    (20, 30, 44),
    (20, 45, 59),

    (21, 0, 14),
    (21, 15, 29),
    (21, 30, 44),
    (21, 45, 59),

    (22, 0, 14),
    (22, 15, 29),
    (22, 30, 44),
    (22, 45, 59),

    (23, 0, 14),
    (23, 15, 29),
    (23, 30, 44),
    (23, 45, 59),

]


def generate_maps(date: str,
                  output_dir: Path,
                  speed_threshold_kmh: float = 12.0,
                  grid_size_m: float = 100.0,
                  sample_n: Optional[int] = None,
                  min_points_cell: int = 10,
                  min_points_cluster: int = 20) -> Tuple[List[Path], pd.DataFrame, pd.DataFrame]:
    """Generate quarter-hour hotspot maps and return list of HTML paths plus aggregated hotspot table and debug table.

    Hotspot table columns:
        date, window_start, window_end, cluster_id, n_points, median_speed_kmh, slow_ratio, center_lat, center_lng, radius_m
    Debug table columns mirror hotspot_debug script.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    table_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    for hour, m_start, m_end in WINDOWS:
        hour_range = (hour, hour)
        df_hot, dbg = compute_congestion_hotspots(
            start_date=date,
            end_date=date,
            hour_range=hour_range,
            speed_threshold_kmh=speed_threshold_kmh,
            min_points_cell=min_points_cell,
            min_points_cluster=min_points_cluster,
            grid_size_m=grid_size_m,
            sample_n=sample_n,
            minute_start=m_start,
            minute_end=m_end,
            return_debug=True,
        )
        debug_rows.append({
            'date': date,
            'window_start': f"{hour:02d}:{m_start:02d}",
            'window_end': f"{hour:02d}:{m_end:02d}",
            'total_points': dbg.get('total_points'),
            'after_hour_filter_points': dbg.get('after_hour_filter_points'),
            'after_minute_filter_points': dbg.get('after_minute_filter_points'),
            'slow_points': dbg.get('slow_points'),
            'candidate_cells': dbg.get('candidate_cells'),
            'clusters_found': dbg.get('clusters_found'),
            'clusters_passing_threshold': dbg.get('clusters_passing_threshold'),
        })
        if df_hot.empty:
            table_rows.append({
                "date": date,
                "window_start": f"{hour:02d}:{m_start:02d}",
                "window_end": f"{hour:02d}:{m_end:02d}",
                "cluster_id": None,
                "n_points": 0,
                "median_speed_kmh": None,
                "slow_ratio": None,
                "center_lat": None,
                "center_lng": None,
                "radius_m": None,
            })
        else:
            for _, r in df_hot.iterrows():
                table_rows.append({
                    "date": date,
                    "window_start": f"{hour:02d}:{m_start:02d}",
                    "window_end": f"{hour:02d}:{m_end:02d}",
                    "cluster_id": int(r["cluster_id"]),
                    "n_points": int(r["n_points"]),
                    "median_speed_kmh": float(r["median_speed_kmh"]),
                    "slow_ratio": float(r["slow_ratio"]),
                    "center_lat": float(r["center_lat"]),
                    "center_lng": float(r["center_lng"]),
                    "radius_m": float(r["radius_m"]),
                })
        # Map
        map_obj = congestion_hotspots_map(
            start_date=date,
            end_date=date,
            hour_range=hour_range,
            speed_threshold_kmh=speed_threshold_kmh,
            grid_size_m=grid_size_m,
            sample_n=sample_n,
            min_points_cell=min_points_cell,
            min_points_cluster=min_points_cluster,
            minute_start=m_start,
            minute_end=m_end,
        )
        fname = f"hotspots_{date}_{hour:02d}{m_start:02d}-{hour:02d}{m_end:02d}.html"
        out_path = output_dir / fname
        map_obj.save(str(out_path))
        created.append(out_path)
    table_df = pd.DataFrame(table_rows)[[
        "date","window_start","window_end","cluster_id","n_points","median_speed_kmh","slow_ratio","center_lat","center_lng","radius_m"
    ]]
    debug_df = pd.DataFrame(debug_rows)[[
        'date','window_start','window_end','total_points','after_hour_filter_points','after_minute_filter_points','slow_points','candidate_cells','clusters_found','clusters_passing_threshold'
    ]]
    return created, table_df, debug_df


def main():
    ap = argparse.ArgumentParser(description="Generate 8 quarter-hour congestion hotspot maps between 07:00 and 09:00 and output hotspot + debug tables")
    ap.add_argument("date", type=str, help="Date YYYY-MM-DD")
    ap.add_argument("--out", type=str, default="quarter_hour_hotspots", help="Output directory for HTML maps")
    ap.add_argument("--table_csv", type=str, default=None, help="Optional path to write aggregated hotspot table CSV")
    ap.add_argument("--debug_csv", type=str, default=None, help="Optional path to write debug metrics CSV")
    ap.add_argument("--speed_threshold", type=float, default=12.0, help="Speed threshold (km/h) for slow points")
    ap.add_argument("--grid_size_m", type=float, default=100.0, help="Grid size meters")
    ap.add_argument("--sample_n", type=str, default='none', help="Rows per GPS file to sample: number or 'none' for full")
    ap.add_argument("--min_points_cell", type=int, default=10, help="Minimum slow points to keep cell")
    ap.add_argument("--min_points_cluster", type=int, default=20, help="Minimum total slow points for cluster")
    args = ap.parse_args()

    sample_n: Optional[int] = None if args.sample_n.lower() == 'none' else int(args.sample_n)

    created, table_df, debug_df = generate_maps(
        date=args.date,
        output_dir=Path(args.out),
        speed_threshold_kmh=args.speed_threshold,
        grid_size_m=args.grid_size_m,
        sample_n=sample_n,
        min_points_cell=args.min_points_cell,
        min_points_cluster=args.min_points_cluster,
    )

    # Print hotspot table
    print("Hotspot table (aggregated):")
    print(table_df.to_string(index=False))
    print("\nDebug metrics:")
    print(debug_df.to_string(index=False))

    if args.table_csv:
        Path(args.table_csv).parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(args.table_csv, index=False)
        print(f"Hotspot table written to {args.table_csv}")
    if args.debug_csv:
        Path(args.debug_csv).parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(args.debug_csv, index=False)
        print(f"Debug metrics written to {args.debug_csv}")

    # Warning if sample_n used and windows empty
    if sample_n is not None and (table_df['n_points'] == 0).all():
        print("WARNING: All windows empty with a row limit. Likely the limited sample did not include the 07-09 time range. Re-run with --sample_n none.")

    print("Generated HTML maps:")
    for p in created:
        print(p)

if __name__ == "__main__":
    main()
