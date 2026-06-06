import pytest
from openpyxl import Workbook

from modules.data_loader.vicon_reader import load_vicon_xlsx, map_vicon_to_caliscope


def _write_real_schema_xlsx(path):
    """Mirror the real Vicon export: subject-prefixed names, Frame/Sub Frame,
    a duplicate | ... | block, a Trajectory Count column, units row, mm data."""
    wb = Workbook()
    ws = wb.active
    ws.append(["Trajectories"])                                   # row0
    ws.append(["100"])                                            # row1
    # row2: marker-name row. Names span 3 cols each (NAME,"",""); Frame/SubFrame blank.
    ws.append([None, None,
               "Subj:LKNE", "", "", "Subj:RKNE", "", "",
               "| Subj:LKNE |", "", "", "Trajectory Count"])      # dup block + count
    ws.append(["Frame", "Sub Frame", "X", "Y", "Z", "X", "Y", "Z",
               "X", "Y", "Z", ""])                                # row3 axis row
    ws.append([None, None, "mm", "mm", "mm", "mm", "mm", "mm",
               "mm", "mm", "mm", ""])                             # row4 units
    ws.append([225, 0, 100.0, 200.0, 300.0, -100.0, 200.0, 300.0,
               100.0, 200.0, 300.0, 1])                           # row5 data (mm)
    ws.append([226, 0, 110.0, 210.0, 310.0, -110.0, 210.0, 310.0,
               110.0, 210.0, 310.0, 1])
    wb.save(path)


def test_load_vicon_real_schema_parses_colon_names_and_mm(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_real_schema_xlsx(p)
    df = load_vicon_xlsx(str(p), vicon_fps=100.0)
    # subject prefix stripped; clean marker columns present
    assert "LKNE_x" in df.columns and "RKNE_z" in df.columns
    # the | ... | duplicate block must NOT collide with the clean LKNE columns
    assert "LKNE_x" in df.columns and df["LKNE_x"].notna().all()
    # mm -> m
    assert df["LKNE_x"].iloc[0] == pytest.approx(0.1, abs=1e-6)
    assert len(df) == 2


def test_load_vicon_adds_timestamp_from_frame_and_fps(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_real_schema_xlsx(p)
    df = load_vicon_xlsx(str(p), vicon_fps=100.0)
    assert "timestamp" in df.columns
    # first frame zero-based; second frame is 1/100 s later
    assert df["timestamp"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df["timestamp"].iloc[1] == pytest.approx(0.01, abs=1e-9)


def test_map_vicon_to_caliscope_renames(tmp_path):
    p = tmp_path / "vicon.xlsx"
    _write_real_schema_xlsx(p)
    df = load_vicon_xlsx(str(p), vicon_fps=100.0)
    out = map_vicon_to_caliscope(df, {"LKNE": "left_knee", "RKNE": "right_knee"})
    assert "left_knee_x" in out.columns and "right_knee_z" in out.columns
    assert "LKNE_x" not in out.columns
