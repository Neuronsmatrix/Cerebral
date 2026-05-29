import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GAIT = REPO / "gait_analysis"
SESSION = GAIT / "data" / "caliscope_project" / "recordings" / "p1_3"

pytestmark = pytest.mark.skipif(not SESSION.exists(),
                                reason="real caliscope data not present")


def test_analyze_writes_gait_results(tmp_path):
    out = tmp_path / "p1_3_results.json"
    cmd = [sys.executable, str(GAIT / "cli.py"), "analyze",
           "--session", str(SESSION), "--model", "SIMPLE_HOLISTIC",
           "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GAIT))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["session_id"] == "p1_3"
    assert data["model"] == "SIMPLE_HOLISTIC"
    assert data["fps"] > 0
    assert "gait_events" in data and "spatiotemporal" in data
    assert "left_HS" in data["gait_events"]
    if data["joint_angles_mean"].get("left_knee"):
        assert len(data["joint_angles_mean"]["left_knee"]) == 101
