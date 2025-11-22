from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path for `import busviz.*` when Streamlit runs from the script directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from busviz.data_loader import list_route_ids, load_stops, load_vars_by_route, available_gps_dates, load_route_by_id
from busviz.visualize import route_stops_map, vehicle_speed_timeseries, gps_density_heatmap, all_routes_map
# Reload gps_matching so newly added functions are available even if Streamlit cached the module
import importlib
from busviz import gps_matching as _gpsm
_gpsm = importlib.reload(_gpsm)
build_and_match_route_day = _gpsm.build_and_match_route_day
map_trips_to_schedule = _gpsm.map_trips_to_schedule
infer_vehicle_routes = getattr(_gpsm, "infer_vehicle_routes", None)
from busviz.timetable_loader import load_route_timetable
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="HCMC Bus Data Explorer", layout="wide")

st.title("HCMC Bus Data Explorer")

# Sidebar selections
route_ids = list_route_ids()
route_id = st.sidebar.selectbox("Route ID", route_ids)

meta_df = load_route_by_id(route_id)
route_no = str(meta_df.iloc[0]["RouteNo"]) if not meta_df.empty and "RouteNo" in meta_df.columns else route_id

direction = st.sidebar.radio("Direction", ["Outbound", "Inbound"], index=0)
outbound = direction == "Outbound"
show_all = st.sidebar.checkbox("Show aggregated all-routes map")
max_routes = st.sidebar.slider("Max routes (for aggregated map)", min_value=5, max_value=len(route_ids), value=min(20, len(route_ids))) if show_all else None

vars_df = load_vars_by_route(route_id)
st.subheader(f"Route {route_id} variants")
st.dataframe(vars_df)

stops_df = load_stops(route_id, outbound=outbound)
st.subheader(f"Stops ({'Outbound' if outbound else 'Inbound'}) - total {len(stops_df)}")
st.dataframe(stops_df.head(50))

# Map(s)
if show_all:
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.markdown("### Selected Route Map")
        try:
            m_route = route_stops_map(route_id, outbound=outbound)
            st_folium(m_route, width=700, height=500)
        except Exception as e:
            st.warning(f"Route map could not be generated: {e}")
    with col_map2:
        st.markdown("### Aggregated Routes Map")
        try:
            m_all = all_routes_map(outbound=outbound, max_routes=max_routes)
            st_folium(m_all, width=700, height=500)
        except Exception as e:
            st.warning(f"Aggregated map could not be generated: {e}")
else:
    st.markdown("### Selected Route Map")
    try:
        m_route = route_stops_map(route_id, outbound=outbound)
        st_folium(m_route, width=700, height=500)
    except Exception as e:
        st.warning(f"Route map could not be generated: {e}")

# GPS data section
st.markdown("### Raw GPS Data")
all_dates = available_gps_dates()
col1, col2 = st.columns(2)
start_date = col1.selectbox("Start date", [None] + all_dates, index=0)
end_date = col2.selectbox("End date", [None] + all_dates, index=0)
vehicle_id = st.text_input("Filter vehicle (anonymized_vehicle)") or None
sample_rows = st.slider("Rows per file to sample", min_value=1000, max_value=50000, value=5000, step=1000)

st.markdown("#### Vehicle Speed")
speed_fig = vehicle_speed_timeseries(start_date=start_date, end_date=end_date, vehicle=vehicle_id, n_rows=sample_rows)
st.plotly_chart(speed_fig, use_container_width=True)

st.markdown("#### GPS Density Heatmap (sample)")
heatmap_fig = gps_density_heatmap(start_date=start_date, end_date=end_date, vehicles=[vehicle_id] if vehicle_id else None, n_rows=sample_rows)
st.pyplot(heatmap_fig)

# Timetable matching
st.markdown("### Timetable matching (beta)")
match_date = st.selectbox("Select a date to build actual trips", all_dates, index=0 if all_dates else None)
if match_date:
    with st.spinner("Building actual trips from GPS and matching to timetable..."):
        res = build_and_match_route_day(route_id, outbound=outbound, date=match_date, n_rows_per_file=50000)
        trips = res.get("trips", [])
        st.write(f"Constructed trips: {len(trips)}")
        sched_df = load_route_timetable(route_no)
        if sched_df.empty:
            st.info("No timetable parsed for this route. Ensure Excel format contains start times.")
        else:
            matches = map_trips_to_schedule(trips, sched_df, start_time_col="StartTime", route_no=route_no, max_diff_min=15)
            st.write(f"Matched trips: {len(matches)}")
            st.dataframe(matches)
            if len(matches) > 0:
                st.caption("diff_min = |actual start - scheduled start| in minutes")

# GPS -> Route inference
st.markdown("### Map GPS to route number (per vehicle)")
inf_date = st.selectbox("Date for inference", all_dates, index=0 if all_dates else None, key="infer_date")
veh_filter = st.text_input("Optional vehicle filter (comma-separated anonymized_vehicle IDs)")
direction_pick = st.selectbox("Direction filter", ["Both", "Outbound", "Inbound"], index=0)
if st.button("Infer vehicle routes", type="primary"):
    if not inf_date:
        st.warning("Please select a date.")
    elif infer_vehicle_routes is None:
        st.error("Route inference function not available. Please refresh or restart the app.")
    else:
        outbound_opt = None if direction_pick == "Both" else (direction_pick == "Outbound")
        with st.spinner("Inferring routes from GPS proximity to stops..."):
            vehicles = [v.strip() for v in veh_filter.split(',')] if veh_filter.strip() else None
            infer_df = infer_vehicle_routes(start_date=inf_date, end_date=inf_date, outbound=outbound_opt, vehicles=vehicles, n_rows_per_file=50000)
            if infer_df.empty:
                st.info("No nearby-stop hits found for the selection.")
            else:
                st.dataframe(infer_df.sort_values(["anonymized_vehicle","hit_ratio"], ascending=[True, False]))
                st.caption("hit_ratio = fraction of GPS points near stops of the inferred route; Outbound indicates direction of those stops.")

st.caption("Data sampled for performance; adjust slider for more rows.")
