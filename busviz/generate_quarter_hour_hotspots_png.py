from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .congestion import compute_congestion_hotspots
from .data_loader import load_gps

# Quarter-hour windows between 07:00 and 09:00 (exclusive of 09:00 end minute)
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

# Fallback extent (HCMC approximate bounds) if GPS data unavailable
_FALLBACK_EXTENT: Tuple[float, float, float, float] = (106.60, 106.78, 10.70, 10.84)


def _compute_extent(date: str, sample_n: Optional[int]) -> Tuple[float, float, float, float]:
    """Compute plotting extent with small padding; fallback if no data."""
    gps = load_gps(start_date=date, end_date=date, n_rows=sample_n)
    if gps is None or gps.empty or not {'lat', 'lng'}.issubset(gps.columns):
        return _FALLBACK_EXTENT
    gps = gps.dropna(subset=['lat', 'lng'])
    if gps.empty:
        return _FALLBACK_EXTENT
    min_lng = float(gps['lng'].min()); max_lng = float(gps['lng'].max())
    min_lat = float(gps['lat'].min()); max_lat = float(gps['lat'].max())
    pad_lng = (max_lng - min_lng) * 0.05 if max_lng > min_lng else 0.01
    pad_lat = (max_lat - min_lat) * 0.05 if max_lat > min_lat else 0.01
    return (min_lng - pad_lng, max_lng + pad_lng, min_lat - pad_lat, max_lat + pad_lat)


def _plot_window_hotspots(df: pd.DataFrame,
                          date: str,
                          window_label: str,
                          extent: Tuple[float, float, float, float],
                          out_path: Path,
                          speed_threshold_kmh: float,
                          cmap_name: str = 'RdYlGn_r') -> None:
    """Render a single window's hotspots to a PNG file.

    df expected columns: cluster_id, n_points, median_speed_kmh, slow_ratio, center_lat, center_lng, radius_m.
    """
    min_lng, max_lng, min_lat, max_lat = extent
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(min_lng, max_lng)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'HCM Bus Congestion Hotspots\n{date} {window_label} (slow <= {speed_threshold_kmh:.0f} km/h)')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No hotspots', ha='center', va='center', fontsize=12, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return

    cmap = plt.get_cmap(cmap_name)
    # Draw each cluster as ellipse approximating its spatial span (radius_m)
    for _, row in df.iterrows():
        lat = float(row['center_lat']); lng = float(row['center_lng'])
        radius_m = float(row['radius_m'])
        slow_ratio = float(row['slow_ratio']) if not math.isnan(row['slow_ratio']) else 0.0
        slow_ratio_clipped = max(0.0, min(1.0, slow_ratio))
        lat_rad = math.radians(lat)
        # Degree conversions (rough)
        deg_lat = radius_m / 111320.0
        denom = 111320.0 * math.cos(lat_rad)
        if abs(denom) < 1e-6:
            denom = 111320.0
        deg_lng = radius_m / denom
        color = cmap(slow_ratio_clipped)
        ell = Ellipse(xy=(lng, lat), width=2 * deg_lng, height=2 * deg_lat,
                      facecolor=color, edgecolor=color, alpha=0.5, linewidth=1.2)
        ax.add_patch(ell)
        # Label n_points for highly congested clusters
        if slow_ratio_clipped >= 0.7:
            ax.text(lng, lat, f"{int(row['n_points'])}", color='black', ha='center', va='center', fontsize=7, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Slow Ratio (slow points / all points in cluster area)')

    # Legend-like annotation
    ax.text(0.01, 0.01,
            'Circle size ~ radius\nLabel = slow points if ratio >= 0.7',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def generate_png_maps(date: str,
                      output_dir: Path,
                      speed_threshold_kmh: float = 12.0,
                      grid_size_m: float = 100.0,
                      sample_n: Optional[int] = None,
                      min_points_cell: int = 10,
                      min_points_cluster: int = 20,
                      cmap_name: str = 'RdYlGn_r') -> Tuple[List[Path], pd.DataFrame]:
    """Generate PNG hotspot maps for each quarter-hour window.

    Returns (list_of_paths, aggregated_hotspot_table).
    Hotspot table columns: date, window_start, window_end, cluster_id, n_points, median_speed_kmh, slow_ratio, center_lat, center_lng, radius_m
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extent = _compute_extent(date, sample_n)
    table_rows: List[Dict[str, Any]] = []
    created: List[Path] = []
    for hour, m_start, m_end in WINDOWS:
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
        fname = f"hotspots_png_{date}_{hour:02d}{m_start:02d}-{hour:02d}{m_end:02d}.png"
        out_path = output_dir / fname
        _plot_window_hotspots(df_hot, date, window_label, extent, out_path, speed_threshold_kmh, cmap_name=cmap_name)
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
    ap = argparse.ArgumentParser(description='Generate 8 quarter-hour congestion hotspot PNG maps (07:00-09:00) and an aggregated hotspot table')
    ap.add_argument('date', type=str, help='Date YYYY-MM-DD')
    ap.add_argument('--out', type=str, default='quarter_hour_hotspots_png', help='Output directory for PNG maps')
    ap.add_argument('--table_csv', type=str, default=None, help='Optional path to write aggregated hotspot table CSV')
    ap.add_argument('--speed_threshold', type=float, default=12.0, help='Speed threshold (km/h) for slow points')
    ap.add_argument('--grid_size_m', type=float, default=100.0, help='Grid size meters')
    ap.add_argument('--sample_n', type=str, default='none', help="Rows per GPS file to sample: number or 'none' for full")
    ap.add_argument('--min_points_cell', type=int, default=10, help='Minimum slow points to keep cell')
    ap.add_argument('--min_points_cluster', type=int, default=20, help='Minimum total slow points for cluster')
    ap.add_argument('--cmap', type=str, default='RdYlGn_r', help='Matplotlib colormap name for slow ratio')
    args = ap.parse_args()

    sample_n: Optional[int] = None if args.sample_n.lower() == 'none' else int(args.sample_n)

    created, table_df = generate_png_maps(
        date=args.date,
        output_dir=Path(args.out),
        speed_threshold_kmh=args.speed_threshold,
        grid_size_m=args.grid_size_m,
        sample_n=sample_n,
        min_points_cell=args.min_points_cell,
        min_points_cluster=args.min_points_cluster,
        cmap_name=args.cmap,
    )

    # Print hotspot table
    print('Hotspot table (aggregated):')
    print(table_df.to_string(index=False))

    if args.table_csv:
        Path(args.table_csv).parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(args.table_csv, index=False)
        print(f'Hotspot table written to {args.table_csv}')

    # Warning if sample_n used and windows empty
    if sample_n is not None and (table_df['n_points'] == 0).all():
        print('WARNING: All windows empty with a row limit. Likely the limited sample did not include the 07-09 time range. Re-run with --sample_n none.')

    print('Generated PNG maps:')
    for p in created:
        print(p)

if __name__ == '__main__':
    main()
