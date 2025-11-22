from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
import pandas as pd
from typing import Optional, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "Bus_route_data"
ROUTES_ROOT = DATA_ROOT / "HCMC_bus_routes"
GPS_ROOT = DATA_ROOT / "raw_GPS"

# ----------------------------- Route metadata loaders -----------------------------

@lru_cache(maxsize=128)
def list_route_ids() -> List[str]:
    return sorted([p.name for p in ROUTES_ROOT.iterdir() if p.is_dir()])

@lru_cache(maxsize=256)
def load_route_by_id(route_id: str) -> pd.DataFrame:
    path = ROUTES_ROOT / route_id / "route_by_id.csv"
    return _read_csv(path)

@lru_cache(maxsize=256)
def load_vars_by_route(route_id: str) -> pd.DataFrame:
    path = ROUTES_ROOT / route_id / "vars_by_route.csv"
    return _read_csv(path)

@lru_cache(maxsize=256)
def load_stops(route_id: str, outbound: bool = True) -> pd.DataFrame:
    fname = "stops_by_var.csv" if outbound else "rev_stops_by_var.csv"
    path = ROUTES_ROOT / route_id / fname
    return _read_csv(path)

@lru_cache(maxsize=32)
def load_all_stops(outbound: Optional[bool] = None) -> pd.DataFrame:
    """Load stops across all routes.

    Parameters:
        outbound: True -> only outbound stops_by_var; False -> only inbound rev_stops_by_var; None -> both with Outbound flag
    Returns columns: StopId, Code, Name, ..., Lng, Lat, RouteId, RouteNo, Outbound
    """
    rows = []
    for rid in list_route_ids():
        try:
            meta = load_route_by_id(rid)
            route_no = str(meta.iloc[0]["RouteNo"]) if not meta.empty and "RouteNo" in meta.columns else rid
            if outbound is None or outbound is True:
                df_o = load_stops(rid, outbound=True).copy()
                if not df_o.empty:
                    df_o["RouteId"] = rid
                    df_o["RouteNo"] = route_no
                    df_o["Outbound"] = True
                    rows.append(df_o)
            if outbound is None or outbound is False:
                df_i = load_stops(rid, outbound=False).copy()
                if not df_i.empty:
                    df_i["RouteId"] = rid
                    df_i["RouteNo"] = route_no
                    df_i["Outbound"] = False
                    rows.append(df_i)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    return all_df

# ----------------------------- Raw GPS loader -----------------------------

def available_gps_dates() -> List[str]:
    return sorted([f.name.split("_")[-1].split(".")[0] for f in GPS_ROOT.glob("anonymized_raw_*.csv")])


def _select_gps_files(start_date: Optional[str], end_date: Optional[str]) -> List[Path]:
    files = sorted(GPS_ROOT.glob("anonymized_raw_*.csv"))
    if start_date is None and end_date is None:
        return files
    def date_of(p: Path) -> str:
        return p.name.split("_")[-1].split(".")[0]
    selected = []
    for p in files:
        d = date_of(p)
        if (start_date is None or d >= start_date) and (end_date is None or d <= end_date):
            selected.append(p)
    return selected


def load_gps(start_date: Optional[str] = None, end_date: Optional[str] = None, vehicles: Optional[Iterable[str]] = None, n_rows: Optional[int] = None) -> pd.DataFrame:
    """Load GPS data filtered by date range and vehicles.

    Parameters:
        start_date/end_date: inclusive bounds in format YYYY-MM-DD
        vehicles: list of anonymized_vehicle ids to include (None => all)
        n_rows: if provided, limit rows per file (useful for sampling)
    """
    files = _select_gps_files(start_date, end_date)
    dfs = []
    for f in files:
        try:
            # Read in chunks for memory efficiency
            if n_rows is not None:
                df = pd.read_csv(f, encoding="utf-8", nrows=n_rows, parse_dates=["datetime"])
            else:
                df = pd.read_csv(f, encoding="utf-8", parse_dates=["datetime"])
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue
        if vehicles is not None:
            df = df[df["anonymized_vehicle"].isin(list(vehicles))]
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["datetime","lng","lat","speed","door_up","door_down","anonymized_vehicle","anonymized_driver"])
    data = pd.concat(dfs, ignore_index=True)
    data.sort_values("datetime", inplace=True)
    return data

# ----------------------------- Helpers -----------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8")


def stops_to_geojson(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["Lng"], row["Lat"]]},
            "properties": {k: row[k] for k in df.columns if k not in ("Lng","Lat")}
        })
    return {"type": "FeatureCollection", "features": features}


def center_of_stops(df: pd.DataFrame) -> Tuple[float,float]:
    return float(df["Lat"].mean()), float(df["Lng"].mean())
