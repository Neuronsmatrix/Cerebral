import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")


def fixture_results_df():
    """A tiny processed-style df + results pair for GUI smoke tests."""
    n = 20
    df = pd.DataFrame({"timestamp": np.linspace(0, 1, n)})
    for name, base in (("left_hip", 0.9), ("left_knee", 0.5), ("left_ankle", 0.1)):
        df[f"{name}_x"] = np.zeros(n)
        df[f"{name}_y"] = np.zeros(n)
        df[f"{name}_z"] = base + np.linspace(0, 0.1, n)
    results = {
        "fps": 19.0, "n_frames": n,
        "spatiotemporal": {"cadence_steps_per_min": 110.0, "speed_m_per_s": 1.2},
        "joint_angles_mean": {"left_knee": list(np.linspace(0, 60, 101))},
        "joint_angles_std": {"left_knee": [2.0] * 101},
    }
    return results, df


def test_worker_error_path_emits_error():
    from gui.worker import PipelineWorker
    worker = PipelineWorker("/nonexistent/session/folder", "SIMPLE_HOLISTIC", {})
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors          # bad folder -> error signal, no crash, no exception escapes


def test_plot_widget_renders_angles(qtbot):
    from gui.widgets.plot_widget import PlotWidget
    w = PlotWidget()
    qtbot.addWidget(w)
    results, _ = fixture_results_df()
    w.render_angles(results["joint_angles_mean"], results["joint_angles_std"])
    assert w.current_figure.axes                     # axes exist after render
    assert any(ax.lines for ax in w.current_figure.axes)   # at least one curve drawn
