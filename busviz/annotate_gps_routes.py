from __future__ import annotations
"""Annotate raw GPS files with a route_id column based on nearest stop.

Usage (from project root):
    python -m busviz.annotate_gps_routes \
        --input-dir Bus_route_data/raw_GPS \
        --output-dir processed_data/raw_GPS_with_route_ids \
        --radius-m 120 \
        --chunksize 50000

Creates per-day CSV files with added column `route_id`.
Processes large files in chunks to reduce memory usage, using a streaming nearest-stop search (no large distance matrix).
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

from .data_loader import load_all_stops

DEF_RADIUS = 120.0
DEF_CHUNKSIZE = 50000
DEF_STOP_BATCH = 400  # number of stops per batch for distance matrix
REQUIRED_COLUMNS = ["datetime","lng","lat","speed","door_up","door_down","anonymized_vehicle","anonymized_driver"]
EARTH_R = 6371000.0

# ----------------------------- Helpers -----------------------------

def _validate_input_file(path: Path) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"File {path} missing required columns: {missing}")


def _prepare_stops() -> pd.DataFrame:
    stops = load_all_stops(outbound=None)  # both directions
    if stops.empty:
        raise RuntimeError("No stops loaded; cannot annotate route IDs.")
    return stops[["StopId","Lng","Lat","RouteId"]].reset_index(drop=True)


def _haversine_vec(lon: np.ndarray, lat: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
    """Vector haversine distance from arrays (lon,lat) to single point (lon2,lat2)."""
    lon1r = np.deg2rad(lon)
    lat1r = np.deg2rad(lat)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_R * c


def _haversine_batch(glon: np.ndarray, glat: np.ndarray, stop_lng_batch: np.ndarray, stop_lat_batch: np.ndarray) -> np.ndarray:
    """Vectorized haversine for a batch of stops.
    glon, glat: shape (N,)
    stop_lng_batch, stop_lat_batch: shape (M,)
    Returns distances shape (N, M) in meters using float32 to reduce memory.
    """
    lon1 = np.deg2rad(glon)[:, None]
    lat1 = np.deg2rad(glat)[:, None]
    lon2 = np.deg2rad(stop_lng_batch)[None, :]
    lat2 = np.deg2rad(stop_lat_batch)[None, :]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2, dtype=np.float64)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2, dtype=np.float64)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return (EARTH_R * c).astype(np.float32)


def _annotate_chunk(chunk: pd.DataFrame, stops: pd.DataFrame, radius_m: float, stop_batch_size: int) -> pd.Series:
    """Nearest-stop search using batched distance matrices (faster than per-stop loop; lower memory than full matrix)."""
    if chunk.empty:
        return pd.Series([], dtype=str)
    glon = chunk['lng'].to_numpy(dtype=float)
    glat = chunk['lat'].to_numpy(dtype=float)
    stop_lng = stops['Lng'].to_numpy(dtype=float)
    stop_lat = stops['Lat'].to_numpy(dtype=float)
    n_pts = glon.shape[0]
    best_dist = np.full(n_pts, np.inf, dtype=np.float32)
    best_idx = np.full(n_pts, -1, dtype=np.int32)

    for start in range(0, len(stop_lng), stop_batch_size):
        end = start + stop_batch_size
        batch_lng = stop_lng[start:end]
        batch_lat = stop_lat[start:end]
        dists = _haversine_batch(glon, glat, batch_lng, batch_lat)  # shape (N, batch)
        # argmin per row within batch
        local_min = dists.min(axis=1)
        local_idx = dists.argmin(axis=1) + start
        mask = local_min < best_dist
        if mask.any():
            best_dist[mask] = local_min[mask]
            best_idx[mask] = local_idx[mask]

    route_ids_arr = np.empty(n_pts, dtype=object)
    route_ids_arr[:] = ""
    valid = (best_idx >= 0) & (best_dist <= radius_m)
    if valid.any():
        route_ids_arr[valid] = stops['RouteId'].to_numpy()[best_idx[valid]].astype(str)
    return pd.Series(route_ids_arr, index=chunk.index, dtype='object')

# ----------------------------- Main annotation loop -----------------------------

def annotate_file(input_path: Path, output_path: Path, stops: pd.DataFrame, radius_m: float, chunksize: int, stop_batch_size: int, limit_rows: Optional[int]=None) -> None:
    total_written = 0
    header_written = False
    for chunk in pd.read_csv(input_path, parse_dates=['datetime'], encoding='utf-8', chunksize=chunksize):
        if limit_rows is not None and total_written >= limit_rows:
            break
        remaining = None
        if limit_rows is not None:
            remaining = limit_rows - total_written
            if remaining <= 0:
                break
        if remaining is not None and remaining < len(chunk):
            chunk = chunk.iloc[:remaining]
        route_ids = _annotate_chunk(chunk, stops, radius_m, stop_batch_size)
        chunk['route_id'] = route_ids.values
        os.makedirs(output_path.parent, exist_ok=True)
        mode = 'a' if header_written else 'w'
        chunk.to_csv(output_path, index=False, mode=mode, header=not header_written)
        header_written = True
        total_written += len(chunk)


def main():
    parser = argparse.ArgumentParser(description='Annotate raw GPS with route_id')
    parser.add_argument('--input-dir', required=True, help='Directory containing raw GPS CSV files')
    parser.add_argument('--output-dir', required=True, help='Directory to write annotated CSV files')
    parser.add_argument('--radius-m', type=float, default=DEF_RADIUS, help='Radius (meters) within which a stop assigns a route_id')
    parser.add_argument('--chunksize', type=int, default=DEF_CHUNKSIZE, help='Rows per chunk for processing')
    parser.add_argument('--stop-batch-size', type=int, default=DEF_STOP_BATCH, help='Number of stops per distance batch')
    parser.add_argument('--limit-rows', type=int, default=None, help='Optional limit of rows per file (sampling)')
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stops = _prepare_stops()

    files = sorted(in_dir.glob('anonymized_raw_*.csv'))
    if not files:
        raise FileNotFoundError(f"No raw GPS files found in {in_dir}")

    for f in files:
        print(f"Annotating {f.name} ...")
        _validate_input_file(f)
        out_path = out_dir / f.name
        annotate_file(f, out_path, stops, radius_m=args.radius_m, chunksize=args.chunksize, stop_batch_size=args.stop_batch_size, limit_rows=args.limit_rows)
        print(f" -> wrote {out_path}")

    print("Done.")

if __name__ == '__main__':
    main()
