from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from io import BytesIO
import urllib.request

from .congestion import compute_congestion_hotspots
from .data_loader import load_gps

# Full-day quarter-hour windows (24 * 4 = 96 windows)
WINDOWS = []
for h in range(24):
    for start in (0,15,30,45):
        WINDOWS.append((h, start, start+14))

_FALLBACK_EXTENT: Tuple[float, float, float, float] = (106.60, 106.78, 10.70, 10.84)


def _compute_extent(date: str, sample_n: Optional[int]) -> Tuple[float, float, float, float]:
    gps = load_gps(start_date=date, end_date=date, n_rows=sample_n)
    if gps is None or gps.empty or not {'lat','lng'}.issubset(gps.columns):
        return _FALLBACK_EXTENT
    gps = gps.dropna(subset=['lat','lng'])
    if gps.empty:
        return _FALLBACK_EXTENT
    min_lng = float(gps['lng'].min()); max_lng = float(gps['lng'].max())
    min_lat = float(gps['lat'].min()); max_lat = float(gps['lat'].max())
    pad_lng = (max_lng - min_lng) * 0.03 if max_lng > min_lng else 0.01
    pad_lat = (max_lat - min_lat) * 0.03 if max_lat > min_lat else 0.01
    return (min_lng - pad_lng, max_lng + pad_lng, min_lat - pad_lat, max_lat + pad_lat)

# ---------------- Tiles ----------------

def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int,int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def _fetch_tile(z: int, x: int, y: int) -> Optional[Any]:
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        return plt.imread(BytesIO(data), format='png')
    except Exception:
        return None

def _build_tile_mosaic(extent: Tuple[float,float,float,float], zoom: int) -> Tuple[Optional[Any], Tuple[float,float,float,float]]:
    min_lng, max_lng, min_lat, max_lat = extent
    # Choose representative lat for y tile range; approximate using corners
    x_min, y_max = _deg2num(min_lat, min_lng, zoom)
    x_max, y_min = _deg2num(max_lat, max_lng, zoom)
    # Ensure proper ordering
    if x_max < x_min: x_min, x_max = x_max, x_min
    if y_max < y_min: y_min, y_max = y_max, y_min
    tiles: List[List[Optional[Any]]] = []
    for y in range(y_min, y_max+1):
        row: List[Optional[Any]] = []
        for x in range(x_min, x_max+1):
            row.append(_fetch_tile(zoom, x, y))
        tiles.append(row)
    if not tiles:
        return None, extent
    # Determine tile shape
    h_tiles = len(tiles); w_tiles = len(tiles[0])
    # Validate no None entire row/col
    import numpy as np
    tile_h, tile_w = 256, 256
    mosaic = np.zeros((h_tiles*tile_h, w_tiles*tile_w, 4), dtype=float)
    for yi, row in enumerate(tiles):
        for xi, img in enumerate(row):
            if img is None:
                continue
            # Normalize shape to RGBA
            if img.shape[2] == 3:  # add alpha
                alpha = np.ones((img.shape[0], img.shape[1], 1))
                img_rgba = np.concatenate([img, alpha], axis=2)
            else:
                img_rgba = img
            mosaic[yi*tile_h:(yi+1)*tile_h, xi*tile_w:(xi+1)*tile_w, :] = img_rgba
    return mosaic, (min_lng, max_lng, min_lat, max_lat)

# ---------------- Plotting ----------------

def _plot_window(df: pd.DataFrame,
                 date: str,
                 window_label: str,
                 extent: Tuple[float,float,float,float],
                 out_path: Path,
                 speed_threshold_kmh: float,
                 cmap_name: str,
                 tile_img: Optional[Any]) -> None:
    min_lng, max_lng, min_lat, max_lat = extent
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(min_lng, max_lng)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title(f'HCM Bus Congestion Hotspots\n{date} {window_label} (<= {speed_threshold_kmh:.0f} km/h)')
    if tile_img is not None:
        ax.imshow(tile_img, extent=(min_lng, max_lng, min_lat, max_lat), origin='upper')
    else:
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    if df is None or df.empty:
        ax.text(0.5,0.5,'No hotspots', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig); return
    cmap = plt.get_cmap(cmap_name)
    for _, row in df.iterrows():
        lat = float(row['center_lat']); lng = float(row['center_lng'])
        radius_m = float(row['radius_m'])
        slow_ratio = float(row['slow_ratio']) if not math.isnan(row['slow_ratio']) else 0.0
        slow_ratio_clipped = max(0.0, min(1.0, slow_ratio))
        lat_rad = math.radians(lat)
        deg_lat = radius_m / 111320.0
        denom = 111320.0 * math.cos(lat_rad)
        if abs(denom) < 1e-6: denom = 111320.0
        deg_lng = radius_m / denom
        color = cmap(slow_ratio_clipped)
        ell = Ellipse(xy=(lng, lat), width=2*deg_lng, height=2*deg_lat,
                      facecolor=color, edgecolor=color, alpha=0.55, linewidth=1.0)
        ax.add_patch(ell)
        if slow_ratio_clipped >= 0.7:
            ax.text(lng, lat, f"{int(row['n_points'])}", color='black', ha='center', va='center', fontsize=7, fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02); cbar.set_label('Slow Ratio')
    ax.text(0.01,0.01,'Ellipse size ~ radius\nLabel = slow points if ratio >= 0.7', transform=ax.transAxes,
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)

# ---------------- Generation ----------------

def generate_png_maps_tiles(date: str,
                             output_dir: Path,
                             speed_threshold_kmh: float = 12.0,
                             grid_size_m: float = 100.0,
                             sample_n: Optional[int] = None,
                             min_points_cell: int = 10,
                             min_points_cluster: int = 20,
                             cmap_name: str = 'RdYlGn_r',
                             zoom: int = 13,
                             windows: Optional[List[Tuple[int,int,int]]] = None) -> Tuple[List[Path], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = _compute_extent(date, sample_n)
    tile_img, _ = _build_tile_mosaic(extent, zoom)
    created: List[Path] = []
    table_rows: List[Dict[str, Any]] = []
    use_windows = windows if windows is not None else WINDOWS
    for hour, m_start, m_end in use_windows:
        hour_range = (hour, hour)
        df_hot = compute_congestion_hotspots(
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
            return_debug=False,
        )
        window_label = f"{hour:02d}:{m_start:02d}-{hour:02d}:{m_end:02d}"
        fname = f"hotspots_tiles_{date}_{hour:02d}{m_start:02d}-{hour:02d}{m_end:02d}.png"
        out_path = output_dir / fname
        _plot_window(df_hot, date, window_label, extent, out_path, speed_threshold_kmh, cmap_name, tile_img)
        created.append(out_path)
        if df_hot.empty:
            table_rows.append({
                'date': date,
                'window_start': f"{hour:02d}:{m_start:02d}",
                'window_end': f"{hour:02d}:{m_end:02d}",
                'cluster_id': None,
                'n_points': 0,
                'median_speed_kmh': None,
                'slow_ratio': None,
                'center_lat': None,
                'center_lng': None,
                'radius_m': None,
            })
        else:
            for _, r in df_hot.iterrows():
                table_rows.append({
                    'date': date,
                    'window_start': f"{hour:02d}:{m_start:02d}",
                    'window_end': f"{hour:02d}:{m_end:02d}",
                    'cluster_id': int(r['cluster_id']),
                    'n_points': int(r['n_points']),
                    'median_speed_kmh': float(r['median_speed_kmh']),
                    'slow_ratio': float(r['slow_ratio']),
                    'center_lat': float(r['center_lat']),
                    'center_lng': float(r['center_lng']),
                    'radius_m': float(r['radius_m']),
                })
    table_df = pd.DataFrame(table_rows)[['date','window_start','window_end','cluster_id','n_points','median_speed_kmh','slow_ratio','center_lat','center_lng','radius_m']]
    return created, table_df


def main():
    ap = argparse.ArgumentParser(description='Generate full-day quarter-hour congestion hotspot PNG maps with OSM tile background')
    ap.add_argument('date', type=str, help='Date YYYY-MM-DD')
    ap.add_argument('--out', type=str, default='quarter_hour_hotspots_png_tiles', help='Output directory for tile PNG maps')
    ap.add_argument('--table_csv', type=str, default=None, help='Optional path to write aggregated hotspot table CSV')
    ap.add_argument('--speed_threshold', type=float, default=12.0, help='Speed threshold (km/h) for slow points')
    ap.add_argument('--grid_size_m', type=float, default=100.0, help='Grid size meters')
    ap.add_argument('--sample_n', type=str, default='none', help="Rows per GPS file to sample: number or 'none' for full")
    ap.add_argument('--min_points_cell', type=int, default=10, help='Minimum slow points to keep cell')
    ap.add_argument('--min_points_cluster', type=int, default=20, help='Minimum total slow points for cluster')
    ap.add_argument('--cmap', type=str, default='RdYlGn_r', help='Matplotlib colormap name for slow ratio')
    ap.add_argument('--zoom', type=int, default=13, help='Tile zoom level (11-17 recommended)')
    ap.add_argument('--limit_hours', type=str, default='none', help='Optional hour range inclusive start-end like 7-9 to restrict windows')
    args = ap.parse_args()

    sample_n: Optional[int] = None if args.sample_n.lower() == 'none' else int(args.sample_n)
    if args.limit_hours.lower() != 'none':
        try:
            h0, h1 = [int(x) for x in args.limit_hours.split('-')]
            selected = [w for w in WINDOWS if w[0] >= h0 and w[0] <= h1]
            WINDOWS = selected
        except Exception:
            pass

    created, table_df = generate_png_maps_tiles(
        date=args.date,
        output_dir=Path(args.out),
        speed_threshold_kmh=args.speed_threshold,
        grid_size_m=args.grid_size_m,
        sample_n=sample_n,
        min_points_cell=args.min_points_cell,
        min_points_cluster=args.min_points_cluster,
        cmap_name=args.cmap,
        zoom=args.zoom,
    )

    print('Hotspot table (aggregated):')
    print(table_df.to_string(index=False))
    if args.table_csv:
        Path(args.table_csv).parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(args.table_csv, index=False)
        print(f'Hotspot table written to {args.table_csv}')
    print('Generated tile PNG maps:')
    for p in created:
        print(p)

if __name__ == '__main__':
    main()

