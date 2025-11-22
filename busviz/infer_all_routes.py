from __future__ import annotations
"""Batch route inference for all raw GPS files.

This script infers the most likely (RouteNo, RouteId, Outbound) per vehicle per day by
snapping GPS points to the nearest stops (both directions) and aggregating hit statistics.

Output:
  processed_data_2/vehicle_routes/<filename>.csv  (per-day vehicle best route)
  processed_data_2/vehicle_routes/aggregated_vehicle_routes.csv (best route across all days)

Usage:
  python -m busviz.infer_all_routes \
      --input-dir Bus_route_data/raw_GPS \
      --output-dir processed_data_2/vehicle_routes \
      --radius-m 120 \
      --chunksize 50000 \
      --stop-batch-size 400

Options:
  --limit-rows N   Sample first N rows of each file (for quick tests)
  --include-zero   Include vehicles with zero hits (route columns blank)

Methodology:
  For each chunk, compute batched distance matrix (chunk_size x stop_batch_size) to reduce memory.
  Track per (vehicle, routeId, outbound) hits and cumulative distance of hits.
  After file processed: derive mean_dist_m, hit_ratio = hits / total_points.
  Select best route per vehicle by descending hit_ratio, then hits, then ascending mean_dist_m.
"""
import argparse
from pathlib import Path
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from .data_loader import load_all_stops

EARTH_R = 6371000.0
DEF_RADIUS = 120.0
DEF_CHUNKSIZE = 50000
DEF_STOP_BATCH = 400

# --------------------------------------------------------------------------------------
# Distance helpers
# --------------------------------------------------------------------------------------

def _haversine_batch(glon: np.ndarray, glat: np.ndarray, stop_lng_batch: np.ndarray, stop_lat_batch: np.ndarray) -> np.ndarray:
    lon1 = np.deg2rad(glon)[:, None]
    lat1 = np.deg2rad(glat)[:, None]
    lon2 = np.deg2rad(stop_lng_batch)[None, :]
    lat2 = np.deg2rad(stop_lat_batch)[None, :]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return (EARTH_R * c).astype(np.float32)

# --------------------------------------------------------------------------------------
# Core per-file inference
# --------------------------------------------------------------------------------------

def infer_routes_for_file(
    file_path: Path,
    stops: pd.DataFrame,
    radius_m: float,
    chunksize: int,
    stop_batch_size: int,
    limit_rows: Optional[int] = None,
    include_zero: bool = False,
) -> pd.DataFrame:
    """Infer best route per vehicle for a single GPS CSV file.

    Returns columns: date, anonymized_vehicle, RouteNo, RouteId, Outbound, hits, total_points, hit_ratio, mean_dist_m
    """
    # Extract date from filename (last piece before extension)
    date_str = file_path.stem.split('_')[-1]
    total_points: Dict[str, int] = {}
    # key: (vehicle, routeId, outbound, routeNo) -> [hits, dist_sum]
    route_stats: Dict[Tuple[str, str, bool, str], list] = {}

    # Pre-extract stop arrays
    stop_lng = stops['Lng'].to_numpy(dtype=float)
    stop_lat = stops['Lat'].to_numpy(dtype=float)

    reader = pd.read_csv(file_path, parse_dates=['datetime'], encoding='utf-8', chunksize=chunksize)
    processed_rows = 0
    for chunk in reader:
        if limit_rows is not None and processed_rows >= limit_rows:
            break
        if limit_rows is not None:
            remaining = limit_rows - processed_rows
            if remaining <= 0:
                break
            if remaining < len(chunk):
                chunk = chunk.iloc[:remaining]
        processed_rows += len(chunk)

        glon = chunk['lng'].to_numpy(dtype=float)
        glat = chunk['lat'].to_numpy(dtype=float)
        n_pts = glon.shape[0]
        # Update total points per vehicle
        vehs = chunk['anonymized_vehicle'].astype(str).tolist()
        for v in vehs:
            total_points[v] = total_points.get(v, 0) + 1

        if n_pts == 0:
            continue

        best_dist = np.full(n_pts, np.inf, dtype=np.float32)
        best_idx = np.full(n_pts, -1, dtype=np.int32)

        for start in range(0, len(stop_lng), stop_batch_size):
            end = start + stop_batch_size
            batch_lng = stop_lng[start:end]
            batch_lat = stop_lat[start:end]
            dists = _haversine_batch(glon, glat, batch_lng, batch_lat)
            local_min = dists.min(axis=1)
            local_idx = dists.argmin(axis=1) + start
            mask = local_min < best_dist
            if mask.any():
                best_dist[mask] = local_min[mask]
                best_idx[mask] = local_idx[mask]

        valid = (best_idx >= 0) & (best_dist <= radius_m)
        if not valid.any():
            continue
        # Slice stops for valid indices
        sel = stops.iloc[best_idx[valid]]
        sel_vehicle = chunk['anonymized_vehicle'].iloc[valid].astype(str).to_numpy()
        sel_dist = best_dist[valid]
        for veh, (rno, rid, outb), dist in zip(
            sel_vehicle,
            sel[['RouteNo','RouteId','Outbound']].to_numpy(),
            sel_dist,
        ):
            key = (veh, str(rid), bool(outb), str(rno))
            if key not in route_stats:
                route_stats[key] = [0, 0.0]
            route_stats[key][0] += 1
            route_stats[key][1] += float(dist)

    # Build candidate DataFrame
    records = []
    for (veh, rid, outb, rno), (hits, dist_sum) in route_stats.items():
        total = total_points.get(veh, 0)
        hit_ratio = hits / total if total else 0.0
        mean_dist = dist_sum / hits if hits else np.nan
        records.append({
            'date': date_str,
            'anonymized_vehicle': veh,
            'RouteNo': rno,
            'RouteId': rid,
            'Outbound': outb,
            'hits': hits,
            'total_points': total,
            'hit_ratio': hit_ratio,
            'mean_dist_m': mean_dist,
        })

    # Include zero-hit vehicles if requested
    if include_zero:
        for veh, total in total_points.items():
            if not any(rec['anonymized_vehicle'] == veh for rec in records):
                records.append({
                    'date': date_str,
                    'anonymized_vehicle': veh,
                    'RouteNo': '',
                    'RouteId': '',
                    'Outbound': '',
                    'hits': 0,
                    'total_points': total,
                    'hit_ratio': 0.0,
                    'mean_dist_m': np.nan,
                })

    if not records:
        return pd.DataFrame(columns=['date','anonymized_vehicle','RouteNo','RouteId','Outbound','hits','total_points','hit_ratio','mean_dist_m'])

    candidates = pd.DataFrame(records)
    # Select best per vehicle
    candidates_sorted = candidates.sort_values(['anonymized_vehicle','hit_ratio','hits','mean_dist_m'], ascending=[True, False, False, True])
    best = candidates_sorted.groupby('anonymized_vehicle').head(1).reset_index(drop=True)
    return best

# --------------------------------------------------------------------------------------
# Aggregated computation across all files
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Batch infer vehicle routes for all GPS files')
    parser.add_argument('--input-dir', required=True, help='Directory containing raw GPS CSV files')
    parser.add_argument('--output-dir', required=True, help='Directory to write per-day inference CSVs')
    parser.add_argument('--radius-m', type=float, default=DEF_RADIUS)
    parser.add_argument('--chunksize', type=int, default=DEF_CHUNKSIZE)
    parser.add_argument('--stop-batch-size', type=int, default=DEF_STOP_BATCH)
    parser.add_argument('--limit-rows', type=int, default=None, help='Sample first N rows of each file')
    parser.add_argument('--include-zero', action='store_true', help='Include vehicles with zero valid hits')
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob('anonymized_raw_*.csv'))
    if not files:
        raise FileNotFoundError(f'No GPS files in {in_dir}')

    stops = load_all_stops(outbound=None)
    if stops.empty:
        raise RuntimeError('No stops loaded for inference')

    all_records = []
    for f in files:
        print(f'Inferring routes for {f.name} ...')
        day_df = infer_routes_for_file(
            f, stops, args.radius_m, args.chunksize, args.stop_batch_size, args.limit_rows, args.include_zero
        )
        out_file = out_dir / f'{f.stem}_vehicle_routes.csv'
        day_df.to_csv(out_file, index=False)
        print(f'  -> wrote {out_file} ({len(day_df)} vehicles)')
        all_records.append(day_df)

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        # Best overall per vehicle across all days
        overall_sorted = combined.sort_values(['anonymized_vehicle','hit_ratio','hits','mean_dist_m'], ascending=[True, False, False, True])
        overall_best = overall_sorted.groupby('anonymized_vehicle').head(1).reset_index(drop=True)
        overall_best.to_csv(out_dir / 'aggregated_vehicle_routes.csv', index=False)
        print(f'Aggregated best routes written to {out_dir / "aggregated_vehicle_routes.csv"}')
    print('Done.')

if __name__ == '__main__':
    main()

