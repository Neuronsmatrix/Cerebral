import pytest
from openpyxl import Workbook

from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope


def _write_vicon_xlsx(path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Trajectories", "100"])           # junk row 1
    ws.append(["LKNE", "", "", "RKNE", "", ""])   # marker-name row
    ws.append(["X", "Y", "Z", "X", "Y", "Z"])     # axis row
    ws.append([100.0, 200.0, 300.0, -100.0, 200.0, 300.0])  # mm
    ws.append([110.0, 210.0, 310.0, -110.0, 210.0, 310.0])
    wb.save(path)


def test_load_vicon_finds_header_and_converts_mm_to_m(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_vicon_xlsx(p)
    df = load_vicon_xlsx(str(p))
    assert "LKNE_x" in df.columns and "RKNE_z" in df.columns
    assert len(df) == 2
    assert df["LKNE_x"].iloc[0] == pytest.approx(0.1, abs=1e-6)


def test_map_vicon_to_caliscope_renames_markers(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_vicon_xlsx(p)
    df = load_vicon_xlsx(str(p))
    mapping = {"LKNE": "left_knee", "RKNE": "right_knee"}
    out = map_vicon_to_caliscope(df, mapping)
    assert "left_knee_x" in out.columns and "right_knee_z" in out.columns
    assert "LKNE_x" not in out.columns
