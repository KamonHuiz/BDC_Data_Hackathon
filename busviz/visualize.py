from __future__ import annotations
import folium
from typing import Optional, Iterable, List
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from .data_loader import load_stops, center_of_stops, load_gps, list_route_ids, load_route_by_id
from .congestion import compute_congestion_hotspots  # added import
import folium
from branca.element import Figure, JavascriptLink

class FixedCenterMap(folium.Map):
    """A Folium Map that NEVER auto recenters to layers."""
    def render(self, **kwargs):
        super().render(**kwargs)

        # Remove fitBounds logic that auto-centers the map
        self.get_root().html.add_child(folium.Element("""
        <script>
            // Override Leaflet fitBounds with a no-op function
            L.Map.include({
                fitBounds: function () { return this; }
            });
        </script>
        """))

# ----------------------------- Folium maps -----------------------------

def route_stops_map(route_id: str, outbound: bool = True) -> folium.Map:
    stops = load_stops(route_id, outbound=outbound)
    if stops.empty:
        raise ValueError("No stops loaded")
    lat, lng = center_of_stops(stops)
    m = folium.Map(location=[lat, lng], zoom_start=13, tiles="CartoDB Positron")
    for _, r in stops.iterrows():
        folium.CircleMarker(
            location=[r["Lat"], r["Lng"]],
            radius=5,
            popup=f"{r['Name']} (StopId: {r['StopId']})",
            tooltip=r["Name"],
            color="#1f77b4",
            fill=True,
        ).add_to(m)
    folium.PolyLine([[r["Lat"], r["Lng"]] for _, r in stops.iterrows()], color="blue", weight=3, opacity=0.8).add_to(m)
    return m

# New: congestion hotspots map

def congestion_hotspots_map(start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            hour_range: Optional[tuple[int, int]] = None,
                            speed_threshold_kmh: float = 10.0,
                            grid_size_m: float = 80.0,
                            sample_n: Optional[int] = 50000,
                            min_points_cell: int = 15,
                            min_points_cluster: int = 30,
                            minute_start: Optional[int] = None,
                            minute_end: Optional[int] = None) -> folium.Map:
    """Render congestion hotspots as circles sized by extent and colored by slow ratio.

    Supports minute_start/minute_end to restrict within the selected hour_range.
    """
    hotspots = compute_congestion_hotspots(
        start_date=start_date,
        end_date=end_date,
        hour_range=hour_range,
        speed_threshold_kmh=speed_threshold_kmh,
        min_points_cell=min_points_cell,
        min_points_cluster=min_points_cluster,
        grid_size_m=grid_size_m,
        sample_n=sample_n,
        vehicle_ids=None,
        minute_start=minute_start,
        minute_end=minute_end,
        return_debug=False,
    )
    if hotspots.empty:
        gps_sample = load_gps(start_date=start_date, end_date=end_date, n_rows=1000)
        if gps_sample.empty:
            m = FixedCenterMap(location=[10.77, 106.68], zoom_start=12, tiles="CartoDB Positron")
            folium.Marker([10.77, 106.68], popup="No congestion data").add_to(m)
            return m
        lat0 = gps_sample["lat"].mean(); lng0 = gps_sample["lng"].mean()
        # Force fixed center at District 1 (HCMC center)
        CENTER_D1 = [10.7769, 106.7009]  # UBND TP.HCM / Nguyễn Huệ

        m = FixedCenterMap(
            location=CENTER_D1,
            zoom_start=13,  # overall city view
            tiles="CartoDB Positron"
        )
        folium.Marker([lat0, lng0], popup="No hotspots detected").add_to(m)
        return m
    lat0 = hotspots["center_lat"].mean(); lng0 = hotspots["center_lng"].mean()
    m = FixedCenterMap(location=[lat0, lng0], zoom_start=13, tiles="CartoDB Positron")
    def ratio_color(r: float) -> str:
        r = max(0.0, min(1.0, r))
        g = int((1 - r) * 150)
        red = int(200 * r)
        return f"#{red:02x}{g:02x}00"
    for _, row in hotspots.iterrows():
        color = ratio_color(row.get("slow_ratio", 0.0))
        radius = max(40.0, row.get("radius_m", 60.0))
        popup_html = (
            f"<b>Cluster {int(row['cluster_id'])}</b><br>"
            f"Slow points: {int(row['n_points'])}<br>"
            f"Median speed: {row['median_speed_kmh']:.1f} km/h<br>"
            f"Slow ratio: {row['slow_ratio']:.2f}<br>"
            f"Speed IQR: {row['p25_speed_kmh']:.1f}-{row['p75_speed_kmh']:.1f} km/h"
        )
        folium.Circle(
            location=[row["center_lat"], row["center_lng"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Slow ratio {row['slow_ratio']:.2f}",
        ).add_to(m)
    legend_html = '''\
     <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background: white; padding:8px 12px; border:1px solid #ccc; font-size:12px; border-radius:4px; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
        <b>Congestion Slow Ratio</b><br>
        <span style="display:inline-block;width:12px;height:12px;background:#00aa00;margin-right:4px;"></span>Low (<=0.3)<br>
        <span style="display:inline-block;width:12px;height:12px;background:#996600;margin-right:4px;"></span>Medium (~0.5)<br>
        <span style="display:inline-block;width:12px;height:12px;background:#cc0000;margin-right:4px;"></span>High (>=0.8)
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Add time/date box ---
    time_label = start_date
    if hour_range:
        time_label += f" | Giờ: {hour_range[0]:02d}:00–{hour_range[1]:02d}:59"
    if minute_start is not None and minute_end is not None:
        time_label += f" | Phút: {minute_start:02d}–{minute_end:02d}"

    time_html = f"""
     <div style="
       position: fixed;
       bottom: 30px;
       right: 30px;
       z-index:9999;
       background: white;
       padding: 8px 12px;
       border:1px solid #ccc;
       font-size: 18px;
       border-radius:4px;
       box-shadow: 0 2px 4px rgba(0,0,0,0.3);
     ">
       {time_label}
     </div>
    """

    m.get_root().html.add_child(folium.Element(time_html))
    # FORCE RECENTER WITH YOUR ORIGINAL VALUES
    recenter_js = f"""
    <script>
    var map = {m.get_name()};
    map.setView([{lat0}, {lng0}], 13);
    </script>
    """
    m.get_root().html.add_child(folium.Element(recenter_js))
    # --- Force center (Method 2)
    m.location = [lat0, lng0]
    m.options["center"] = [lat0, lng0]
    m.options["zoom"] = 13
    recenter_js = f"""
    <script>
    var map = {m.get_name()};
    map.once('load', function() {{
        map.setView([{lat0}, {lng0}], 13);
    }});

    // Prevent auto-shifts when layers add
    map.on('layeradd', function() {{
        map.setView([{lat0}, {lng0}], 13);
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(recenter_js))

    return m

# ----------------------------- Plotly time series -----------------------------

def vehicle_speed_timeseries(start_date: Optional[str]=None, end_date: Optional[str]=None, vehicle: Optional[str]=None, n_rows: Optional[int]=None):
    vehicles = [vehicle] if vehicle else None
    df = load_gps(start_date=start_date, end_date=end_date, vehicles=vehicles, n_rows=n_rows)
    if df.empty:
        return px.line(title="No data")
    fig = px.line(df, x="datetime", y="speed", color="anonymized_vehicle", title="Vehicle speed over time")
    return fig

# ----------------------------- Seaborn density heatmap -----------------------------

def gps_density_heatmap(start_date: Optional[str]=None, end_date: Optional[str]=None, vehicles: Optional[Iterable[str]]=None, n_rows: Optional[int]=10000):
    df = load_gps(start_date=start_date, end_date=end_date, vehicles=vehicles, n_rows=n_rows)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,'No data', ha='center')
        return fig
    fig, ax = plt.subplots(figsize=(6,5))
    sns.kdeplot(data=df, x="lng", y="lat", fill=True, cmap="viridis", thresh=0.05, ax=ax)
    ax.set_title("GPS point density (sample)")
    return fig

# ----------------------------- All routes map -----------------------------

def all_routes_map(route_ids: Optional[List[str]] = None, outbound: bool = True, max_routes: Optional[int] = None) -> folium.Map:
    """Render multiple bus routes with route numbers labeled on the map.

    Parameters:
        route_ids: list of route directory IDs; defaults to all discovered.
        outbound: True uses stops_by_var, False uses rev_stops_by_var.
        max_routes: limit count for performance.
    """
    if route_ids is None:
        route_ids = list_route_ids()
    if max_routes is not None:
        route_ids = route_ids[:max_routes]
    centers = []
    routes_meta = {}
    for rid in route_ids:
        try:
            meta_df = load_route_by_id(rid)
            routes_meta[rid] = meta_df.iloc[0].to_dict() if not meta_df.empty else {"RouteNo": rid}
            stops_df = load_stops(rid, outbound=outbound)
            if stops_df.empty:
                continue
            lat, lng = center_of_stops(stops_df)
            centers.append((lat, lng))
        except Exception:
            continue
    if not centers:
        raise ValueError("No route centers could be computed")
    avg_lat = sum(c[0] for c in centers) / len(centers)
    avg_lng = sum(c[1] for c in centers) / len(centers)
    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=11, tiles="CartoDB Positron")
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for idx, rid in enumerate(route_ids):
        try:
            stops_df = load_stops(rid, outbound=outbound)
            if stops_df.empty:
                continue
            meta = routes_meta.get(rid, {})
            route_no = meta.get("RouteNo", rid)
            color = meta.get("Color") if isinstance(meta.get("Color"), str) and str(meta.get("Color")).startswith('#') else palette[idx % len(palette)]
            coords = [[r["Lat"], r["Lng"]] for _, r in stops_df.iterrows()]
            folium.PolyLine(coords, color=color, weight=3, opacity=0.7, tooltip=f"Route {route_no}").add_to(m)
            first = stops_df.iloc[0]
            folium.map.Marker(
                [first["Lat"], first["Lng"]],
                icon=folium.DivIcon(html=f"<div style='font-size:10px; font-weight:bold; color:{color}; background:rgba(255,255,255,0.7); padding:2px 4px; border-radius:3px;'>{route_no}</div>")
            ).add_to(m)
        except Exception:
            continue
    return m
