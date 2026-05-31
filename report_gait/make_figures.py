"""Generate authentic report figures from the project's own code on real p1 data.

Run with the gait_analysis venv:
    .venv/bin/python ../report_gait/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/grivin/Workspace/Cerebral/gait_analysis")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from modules.data_loader.caliscope_reader import load_caliscope_session
from modules.visualization import skeleton_3d as sk
from modules.visualization.angle_plots import plot_joint_angles
from modules.visualization.export import export_figure_png
from modules.visualization.video_overlay import draw_overlay, frame_marks, load_xy
from pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent  # report_gait/
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
GAIT = Path("/home/grivin/Workspace/Cerebral/gait_analysis")
REC = GAIT / "data" / "caliscope_project" / "recordings"
MODEL = "SIMPLE_HOLISTIC"
SESSION = "p1_3"

with open(GAIT / "settings.yaml") as fh:
    CFG = yaml.safe_load(fh)


def fig_angles() -> dict:
    """Joint-angle curves (hip/knee/ankle, L/R, +/-1 STD) via the project code."""
    df = load_caliscope_session(REC / SESSION, model=MODEL)
    results, _ = run_pipeline(df, CFG, model=MODEL, session_id=SESSION)
    fig = plot_joint_angles(results["joint_angles_mean"], results["joint_angles_std"])
    fig.suptitle(f"{SESSION} / {MODEL} — суставные углы по циклу походки (0–100%)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    export_figure_png(fig, FIG / "fig_angles.png")
    print("fig_angles.png  frames=%s fps=%.3f" % (results["n_frames"], results["fps"]))
    return results


def fig_skeleton() -> None:
    """3D skeleton snapshot from the processed dataframe (project geometry core).

    Framed tightly around the single pose (not the whole-recording bbox) so the
    skeleton fills the figure — the report shows one pose, the GUI shows the walk.
    """
    df = load_caliscope_session(REC / SESSION, model=MODEL)
    _, dfp = run_pipeline(df, CFG, model=MODEL, session_id=SESSION)
    names = sk.SKELETON_LANDMARKS
    n = len(dfp)
    # prefer a full-joint frame near the middle of the recording
    full = [i for i in range(n) if len(sk.frame_points(dfp, i, names=names)) == len(names)]
    best_idx = min(full, key=lambda i: abs(i - n // 2)) if full else n // 2
    pts = sk.frame_points(dfp, best_idx, names=names)
    segs = sk.segment_lines(pts)
    P = np.array(list(pts.values()), float)
    c = P.mean(axis=0)
    half = float(np.max(np.abs(P - c))) * 1.15 or 0.5   # cube half-extent around pose

    fig = plt.figure(figsize=(5.0, 6.0), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    for (p0, p1) in segs:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="#114643", linewidth=2.6, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color="#068176", s=55,
               depthshade=False, zorder=2)
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_zlim(c[2] - half, c[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (м)", labelpad=2); ax.set_ylabel("Y (м)", labelpad=2)
    ax.set_zlabel("Z — вертикаль (м)", labelpad=2)
    ax.tick_params(labelsize=7, pad=1)
    ax.locator_params(nbins=4)
    ax.set_title(f"{SESSION}: 3D-скелет (кадр {best_idx}, {len(pts)}/12 суставов)")
    ax.view_init(elev=10, azim=-60)
    fig.tight_layout()
    fig.savefig(FIG / "fig_skeleton.png", dpi=120)
    print("fig_skeleton.png  frame=%d joints=%d half=%.3f" % (best_idx, len(pts), half))


def fig_overlay() -> None:
    """A real marked-video still: skeleton drawn on a raw camera frame."""
    xy = load_xy(REC / SESSION, MODEL)
    port = 1
    video = REC / SESSION / f"port_{port}.mp4"
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 131
    best_idx, best_cnt = 0, -1
    for fi in range(total):
        cnt = len(frame_marks(xy, port, fi))
        if cnt > best_cnt:
            best_cnt, best_idx = cnt, fi
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("fig_overlay.png  FAILED to read frame")
        return
    marks = frame_marks(xy, port, best_idx)
    draw_overlay(frame, marks)
    h, w = frame.shape[:2]
    xs = [p[0] for p in marks.values()]
    ys = [p[1] for p in marks.values()]
    pad = 120
    x0 = max(0, min(xs) - pad); x1 = min(w, max(xs) + pad)
    y0 = max(0, min(ys) - pad); y1 = min(h, max(ys) + pad)
    crop = frame[y0:y1, x0:x1]
    cv2.imwrite(str(FIG / "fig_overlay.png"), crop)
    print("fig_overlay.png  port=%d frame=%d marks=%d crop=%dx%d"
          % (port, best_idx, best_cnt, crop.shape[1], crop.shape[0]))


def fig_cv() -> None:
    """Level-B reproducibility CV bar chart from results/reproducibility.json."""
    with open(GAIT / "results" / "reproducibility.json") as fh:
        data = json.load(fh)
    cv = data["cv_percent"]
    labels_ru = {
        "cadence_steps_per_min": "Темп\n(шаг/мин)",
        "speed_m_per_s": "Скорость\n(м/с)",
        "stride_length_m": "Длина\nцикла (м)",
        "step_length_m": "Длина\nшага (м)",
        "step_width_m": "Ширина\nшага (м)",
        "stance_pct": "Опора\n(%)",
        "swing_pct": "Перенос\n(%)",
    }
    keys = [k for k in labels_ru if cv.get(k) is not None]
    vals = [cv[k] for k in keys]
    colors = ["#3f7a3a" if v < 10 else "#a98b1f" if v < 15 else "#8a3a2e" for v in vals]
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=120)
    bars = ax.bar([labels_ru[k] for k in keys], vals, color=colors)
    ax.axhline(10, color="#3f7a3a", ls="--", lw=1.2, label="ТЗ §11.2: CV < 10%")
    ax.axhline(15, color="#8a3a2e", ls="--", lw=1.2, label="Критерий №6: CV < 15%")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Коэффициент вариации, %")
    ax.set_title("Воспроизводимость параметров между сессиями p1_1…p1_5 (SIMPLE_HOLISTIC)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cv.png", dpi=120)
    print("fig_cv.png  vals=%s" % {k: cv[k] for k in keys})


def _box(ax, x, y, w, h, text, fc, ec="#114643", fs=9, tc="#15282f"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.03",
                 linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=12, color="#068176", lw=1.4, zorder=1))


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 2.5), dpi=120)
    ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")
    stages = [
        ("caliscope\nxyz_*_labelled.csv", "#e6e1d4"),
        ("fill_gaps\n(≤10 кадров)", "#dce7ee"),
        ("Butterworth\n6 Гц, zero-phase", "#dce7ee"),
        ("detect_gait_events\nHS / TO", "#dce7ee"),
        ("calc_joint_angles\nhip/knee/ankle", "#dce7ee"),
        ("normalize\n101 точка", "#dce7ee"),
        ("spatiotemporal\nтемп/скорость…", "#dce7ee"),
        ("gait_results.json\n+ графики/CSV/XLSX", "#e8efdd"),
    ]
    n = len(stages); w, gap = 10.6, 1.4; x = 1.0; y = 9
    for i, (txt, fc) in enumerate(stages):
        _box(ax, x, y, w, 8, txt, fc, fs=7.2)
        if i < n - 1:
            _arrow(ax, x + w, y + 4, x + w + gap, y + 4)
        x += w + gap
    ax.set_title("Конвейер обработки (pipeline.run_pipeline)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pipeline.png", dpi=120)
    print("fig_pipeline.png")


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 6.2), dpi=120)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    done = "#e8efdd"; core = "#dce7ee"; gated = "#f4dad3"; io = "#e6e1d4"
    _box(ax, 4, 86, 92, 10,
         "Точки входа:  cli.py (analyze · reproducibility · produce-videos)   |   app.py → PyQt6 GUI",
         io, fs=8.6)
    _box(ax, 4, 70, 44, 12,
         "gui/ (PyQt6)\nMainWindow · QThread-воркеры\nвкладки «Анализ» / «Визуализация»", done, fs=8.2)
    _box(ax, 52, 70, 44, 12,
         "Видео-оверлей\nvideo_overlay + overlay_worker\nport_N_marked.mp4 (×N камер)", done, fs=8.2)
    _box(ax, 4, 50, 28, 14,
         "data_loader/\ncaliscope_reader\nconfig_reader · landmarks\nsynchronizer · vicon_reader", done, fs=7.6)
    _box(ax, 36, 50, 28, 14,
         "kinematics/\nfilters · gait_events\njoint_angles · normalizer\nspatiotemporal", done, fs=7.6)
    _box(ax, 68, 50, 28, 14,
         "visualization/\nangle_plots · skeleton_3d\nexport · video_overlay", done, fs=7.6)
    _box(ax, 20, 34, 60, 8,
         "pipeline.run_pipeline()  — общий конвейер для CLI и GUI", core, fs=9)
    _box(ax, 4, 16, 44, 12,
         "modules/comparison/  — НЕ РЕАЛИЗОВАНО (Этап 3)\nalignment (Umeyama) · metrics (RMSE/MAE/ICC)\nreport · CLI compare",
         gated, ec="#8a3a2e", fs=7.6)
    _box(ax, 52, 16, 44, 12,
         "Сравнение с Vicon — ЗАБЛОКИРОВАНО\nтребуются: Vicon XLSX + VvsC.py\nуровни валидации A и C",
         gated, ec="#8a3a2e", fs=7.6)
    _box(ax, 4, 2, 92, 9,
         "Данные / артефакты:  caliscope_project (xyz/xy CSV, port_*.mp4) · results/*.json · PNG · CSV/XLSX · marked.mp4",
         io, fs=8.2)
    for (x0, y0, x1, y1) in [
        (50, 86, 50, 82), (26, 70, 18, 64), (74, 70, 82, 64),
        (50, 50, 50, 42), (50, 34, 26, 28), (50, 34, 74, 28), (50, 16, 50, 11),
    ]:
        _arrow(ax, x0, y0, x1, y1)
    ax.set_title("Архитектура: реализованные модули и заблокированный контур Vicon (красным)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_architecture.png", dpi=120)
    print("fig_architecture.png")


if __name__ == "__main__":
    res = fig_angles()
    fig_skeleton()
    fig_overlay()
    fig_cv()
    fig_pipeline()
    fig_architecture()
    with open(ROOT / "p1_3_results_for_report.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("DONE")
    print("P1_3_SPATIOTEMPORAL", json.dumps(res["spatiotemporal"], ensure_ascii=False))
    print("P1_3_EVENTS", {k: len(v) for k, v in res["gait_events"].items()})
