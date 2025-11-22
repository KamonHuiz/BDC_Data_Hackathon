from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TT_ROOT = ROOT / "Bus_route_data" / "Timetable"

# ----------------------------- File discovery -----------------------------

def list_timetable_files() -> List[Path]:
    return sorted(list(TT_ROOT.glob("*.xls")) + list(TT_ROOT.glob("*.xlsx")))


def route_from_filename(path: Path) -> Optional[str]:
    """Extract route number from filename prefix like '01_2025-N.xls' -> '01'"""
    m = re.match(r"^(\d+)_", path.name)
    return m.group(1) if m else None

# ----------------------------- Parsing helpers -----------------------------

def _excel_engine_for(path: Path) -> Optional[str]:
    suf = path.suffix.lower()
    if suf == ".xlsx":
        return "openpyxl"
    if suf == ".xls":
        return "xlrd"
    return None


def _read_first_nonempty_sheet(path: Path) -> pd.DataFrame:
    # Try reading the first sheet that yields at least 1 non-empty row
    engine = _excel_engine_for(path)
    xl = pd.ExcelFile(path, engine=engine) if engine else pd.ExcelFile(path)
    for name in xl.sheet_names:
        try:
            df = xl.parse(name)
            # Drop completely empty columns
            df = df.dropna(axis=1, how='all')
            # Keep if any non-empty rows
            if len(df.dropna(how='all')) > 0:
                return df
        except Exception:
            continue
    # Fallback empty frame
    return pd.DataFrame()


def _find_time_column(df: pd.DataFrame) -> Optional[str]:
    # Candidate names (case-insensitive, Vietnamese + English variants)
    candidates = [
        'starttime','start time','start','departure','depart','giờ','gio','khởi hành','khoi hanh','bat dau'
    ]
    cols = [c for c in df.columns if isinstance(c, str)]
    lower_map = {c.lower().strip(): c for c in cols}
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    # Heuristic: choose first column whose dtype parses to many valid times
    best_col, best_valid = None, -1
    for c in cols:
        s = pd.to_datetime(df[c], errors='coerce')
        valid = s.notna().sum()
        if valid > best_valid:
            best_col, best_valid = c, valid
    if best_valid > 0:
        return best_col
    return None

# ----------------------------- Public loaders -----------------------------

def load_timetable_file(path: Path, route_no_hint: Optional[str] = None) -> pd.DataFrame:
    """Load one timetable Excel into a normalized DataFrame.

    Output columns:
      - RouteNo (str)
      - StartTime (datetime64[ns])
      - plus all original columns for reference
    """
    raw = _read_first_nonempty_sheet(path)
    if raw.empty:
        return pd.DataFrame(columns=['RouteNo','StartTime'])
    # Determine RouteNo
    route_no = route_no_hint or route_from_filename(path) or ''
    # Find time-like column
    time_col = _find_time_column(raw)
    df = raw.copy()
    if time_col is None:
        df['StartTime'] = pd.NaT
    else:
        df['StartTime'] = pd.to_datetime(df[time_col], errors='coerce')
    df['RouteNo'] = str(route_no)
    # Keep StartTime rows only
    df = df[df['StartTime'].notna()].copy()
    df.sort_values('StartTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_route_timetable(route_no: str) -> pd.DataFrame:
    """Load all timetable files matching a route number (with/without zero-padding)."""
    files = list_timetable_files()
    rn = str(route_no)
    rn2 = rn.zfill(2)
    selected = [p for p in files if p.name.startswith(rn+"_") or p.name.startswith(rn2+"_")]
    parts = [load_timetable_file(p, route_no_hint=rn) for p in selected]
    if not parts:
        return pd.DataFrame(columns=['RouteNo','StartTime'])
    out = pd.concat(parts, ignore_index=True)
    out.sort_values('StartTime', inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
