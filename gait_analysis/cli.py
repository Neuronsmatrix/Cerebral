"""Command-line entry point for the gait analysis pipeline."""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml

from modules.data_loader.caliscope_reader import load_caliscope_session
from modules.kinematics.filters import butterworth_filter, fill_gaps
from modules.kinematics.gait_events import detect_gait_events
from modules.kinematics.joint_angles import calc_joint_angles_timeseries
from modules.kinematics.normalizer import get_mean_std_cycle, normalize_gait_cycle
from modules.kinematics.spatiotemporal import calc_spatiotemporal

HERE = Path(__file__).resolve().parent


def _load_settings() -> dict:
    with open(HERE / "settings.yaml") as f:
        return yaml.safe_load(f)


def _filter_coords(df, cutoff_hz, order, fps):
    """Filter coordinate columns. Skip columns that still contain NaN after
    gap-filling (filtfilt would propagate NaN), or that are too short for the
    zero-phase filter (filtfilt needs len > 3*(order+1))."""
    out = df.copy()
    min_len = 3 * (order + 1)
    for c in out.columns:
        if not c.endswith(("_x", "_y", "_z")):
            continue
        arr = out[c].to_numpy()
        if np.isnan(arr).any() or len(arr) <= min_len:
            continue
        out[c] = butterworth_filter(arr, cutoff_hz=cutoff_hz, fs=fps, order=order)
    return out


def analyze(session_dir: str, model: str, out_path: str) -> dict:
    cfg = _load_settings()
    proc = cfg["processing"]
    gcfg = cfg["gait_events"]
    scfg = cfg.get("spatiotemporal", {})

    df = load_caliscope_session(session_dir, model=model)
    fps = df.attrs["fps"]

    df = fill_gaps(df, max_gap_frames=proc["max_gap_frames"])
    df = _filter_coords(df, proc["filter_cutoff_hz"], proc["filter_order"], fps)

    events = detect_gait_events(
        df, fps=fps, method=gcfg["method"], heel=gcfg["heel_landmark"],
        toe=gcfg["toe_landmark"], vertical=gcfg["vertical_axis"],
        min_stride_sec=proc["min_stride_duration_sec"],
        cutoff_hz=proc["filter_cutoff_hz"],
    )
    df = calc_joint_angles_timeseries(df)
    spatiotemporal = calc_spatiotemporal(df, events, fps=fps,
                                         vertical=gcfg["vertical_axis"],
                                         max_stride_m=scfg.get("max_stride_m", 1.5),
                                         max_step_m=scfg.get("max_step_m", 1.0))

    angles_mean, angles_std = {}, {}
    for side in ("left", "right"):
        for joint in ("hip", "knee", "ankle"):
            col = f"{side}_{joint}_angle"
            if col not in df.columns or not events.get(f"{side}_HS"):
                continue
            mat = normalize_gait_cycle(df[col].to_numpy(), events, side=side)
            mean, std = get_mean_std_cycle(mat)
            if mat.shape[0] > 0:
                angles_mean[f"{side}_{joint}"] = np.nan_to_num(mean, nan=0.0).round(3).tolist()
                angles_std[f"{side}_{joint}"] = np.nan_to_num(std, nan=0.0).round(3).tolist()

    results = {
        "session_id": Path(session_dir).name,
        "model": model,
        "fps": round(float(fps), 3),
        "n_frames": int(len(df)),
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gait_events": {k: list(map(int, v)) for k, v in events.items()},
        "spatiotemporal": {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                               else v if isinstance(v, int)
                               else round(float(v), 3))
                           for k, v in spatiotemporal.items()},
        "joint_angles_mean": angles_mean,
        "joint_angles_std": angles_std,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    return results


def reproducibility(recordings_dir: str, sessions: list[str], model: str,
                    out_path: str) -> dict:
    per_session = {}
    params = ["cadence_steps_per_min", "speed_m_per_s", "stride_length_m",
              "step_length_m", "step_width_m", "stance_pct", "swing_pct"]
    collected = {p: [] for p in params}

    for sess in sessions:
        sess_dir = str(Path(recordings_dir) / sess)
        tmp_out = str(Path(out_path).parent / f"_{sess}_results.json")
        res = analyze(sess_dir, model, tmp_out)
        st = res["spatiotemporal"]
        n_cycles = max(len(res["gait_events"].get("left_HS", [])) - 1, 0)
        per_session[sess] = {**st, "n_left_cycles": n_cycles}
        for p in params:
            v = st.get(p)
            if v is not None:
                collected[p].append(v)

    cv_percent = {}
    for p, vals in collected.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) >= 2 and np.mean(arr) != 0:
            cv_percent[p] = round(float(np.std(arr) / np.mean(arr) * 100.0), 2)
        else:
            cv_percent[p] = None

    out = {"model": model, "sessions": sessions,
           "per_session": per_session, "cv_percent": cv_percent}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


def main():
    default_model = _load_settings()["processing"]["default_model"]
    p = argparse.ArgumentParser(description="Gait analysis CLI")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Run the full pipeline on one session")
    a.add_argument("--session", required=True)
    a.add_argument("--model", default=default_model)
    a.add_argument("--out", required=True)

    r = sub.add_parser("reproducibility", help="Level-B CV across sessions")
    r.add_argument("--recordings", required=True)
    r.add_argument("--sessions", required=True, help="comma-separated session names")
    r.add_argument("--model", default=default_model)
    r.add_argument("--out", required=True)

    args = p.parse_args()
    if args.command == "analyze":
        res = analyze(args.session, args.model, args.out)
        print(f"Wrote {args.out}: {res['n_frames']} frames @ {res['fps']} fps, "
              f"{len(res['gait_events'].get('left_HS', []))} left HS")
    elif args.command == "reproducibility":
        res = reproducibility(args.recordings, args.sessions.split(","),
                              args.model, args.out)
        print("CV% by parameter:")
        for p, v in res["cv_percent"].items():
            print(f"  {p}: {v}")


if __name__ == "__main__":
    main()
