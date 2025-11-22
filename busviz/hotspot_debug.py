from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from .congestion import compute_congestion_hotspots

WINDOWS = [
    (7, 0, 14),
    (7, 15, 29),
    (7, 30, 44),
    (7, 45, 59),
    (8, 0, 14),
    (8, 15, 29),
    (8, 30, 44),
    (8, 45, 59),
]

def run_debug(date: str,
              speed_threshold_kmh: float = 12.0,
              grid_size_m: float = 100.0,
              sample_n: Optional[int] = None,
              min_points_cell: int = 10,
              min_points_cluster: int = 20) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for hour, m_start, m_end in WINDOWS:
        hour_range = (hour, hour)
        df_clusters, dbg = compute_congestion_hotspots(
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
        if df_clusters is not None and not df_clusters.empty:
            # Aggregate cluster speeds for the window
            mean_speed = float(df_clusters['mean_speed_kmh'].mean())
            min_speed = float(df_clusters['min_speed_kmh'].min())
            max_speed = float(df_clusters['max_speed_kmh'].max())
        else:
            mean_speed = None
            min_speed = None
            max_speed = None
        rows.append({
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
            'mean_speed_window_kmh': mean_speed,
            'min_speed_window_kmh': min_speed,
            'max_speed_window_kmh': max_speed,
            'speed_threshold_kmh': speed_threshold_kmh,
            'grid_size_m': grid_size_m,
            'min_points_cell': min_points_cell,
            'min_points_cluster': min_points_cluster,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Debug congestion hotspot generation for 07:00-09:00 quarter-hour windows.')
    ap.add_argument('date', type=str, help='Date YYYY-MM-DD')
    ap.add_argument('--speed_threshold', type=float, default=12.0)
    ap.add_argument('--grid_size_m', type=float, default=100.0)
    ap.add_argument('--sample_n', type=str, default=None, help='Rows per file (None for all)')
    ap.add_argument('--min_points_cell', type=int, default=10)
    ap.add_argument('--min_points_cluster', type=int, default=20)
    ap.add_argument('--csv', type=str, default=None, help='Optional CSV output for debug table')
    args = ap.parse_args()
    sample_n: Optional[int] = None if args.sample_n is None or args.sample_n.lower() == 'none' else int(args.sample_n)

    df = run_debug(
        date=args.date,
        speed_threshold_kmh=args.speed_threshold,
        grid_size_m=args.grid_size_m,
        sample_n=sample_n,
        min_points_cell=args.min_points_cell,
        min_points_cluster=args.min_points_cluster,
    )
    if df.empty:
        print('No data loaded for given date.')
        return
    print('Hotspot debug table:')
    print(df.to_string(index=False))
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f'Debug table written to {args.csv}')

if __name__ == '__main__':
    main()
