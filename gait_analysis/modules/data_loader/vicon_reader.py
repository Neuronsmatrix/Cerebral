"""Read Vicon XLSX into the unified pose DataFrame (documented contract)."""
import numpy as np
import pandas as pd
from openpyxl import load_workbook


def _read_rows(path: str) -> list[list]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    return [list(r) for r in ws.iter_rows(values_only=True)]


def _find_axis_header(rows: list[list]) -> int:
    """Index of the row whose cells are X/Y/Z axis labels."""
    for i, row in enumerate(rows):
        cells = [str(c).strip().upper() if c is not None else "" for c in row]
        non_empty = [c for c in cells if c]
        if non_empty and all(c in ("X", "Y", "Z") for c in non_empty):
            return i
    raise ValueError("Could not locate the X/Y/Z axis header row in Vicon XLSX")


def load_vicon_xlsx(filepath: str) -> pd.DataFrame:
    """Auto-detect the header, parse marker columns, convert mm->m if needed.

    Marker names live one row above the X/Y/Z axis row, spanning three columns
    each (NAME, "", "").
    mm->m conversion fires when median(|coord|) > 10, assuming capture volumes
    smaller than ~10 m (true for clinical labs).
    """
    rows = _read_rows(filepath)
    axis_i = _find_axis_header(rows)
    name_row = rows[axis_i - 1]
    axis_row = rows[axis_i]

    names: list[str] = []
    last = None
    for c in name_row:
        if c is not None and str(c).strip():
            last = str(c).strip()
        names.append(last)

    columns: list[str] = []
    for name, axis in zip(names, axis_row):
        if axis is None or name is None:
            columns.append(None)
            continue
        columns.append(f"{name}_{str(axis).strip().lower()}")

    data_rows = rows[axis_i + 1:]
    records = []
    for r in data_rows:
        if all(c is None for c in r):
            continue
        rec = {}
        for col, val in zip(columns, r):
            if col is None or val is None:
                continue
            rec[col] = float(val)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records)

    coord = df.to_numpy()
    if np.nanmedian(np.abs(coord)) > 10.0:
        df = df / 1000.0

    df.insert(0, "frame", range(len(df)))
    return df


def map_vicon_to_caliscope(vicon_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename ``MARKER_{x,y,z}`` columns to ``canonical_{x,y,z}`` via ``mapping``."""
    rename = {}
    for marker, landmark in mapping.items():
        for axis in ("x", "y", "z"):
            rename[f"{marker}_{axis}"] = f"{landmark}_{axis}"
    return vicon_df.rename(columns=rename)
