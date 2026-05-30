"""Command-line entry point for the gait analysis pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from modules.data_loader.caliscope_reader import load_caliscope_session
from pipeline import run_pipeline

HERE = Path(__file__).resolve().parent


def _load_settings() -> dict:
    with open(HERE / "settings.yaml") as f:
        return yaml.safe_load(f)


def analyze(session_dir: str, model: str, out_path: str) -> dict:
    cfg = _load_settings()
    df = load_caliscope_session(session_dir, model=model)
    results, _ = run_pipeline(df, cfg, model=model, session_id=Path(session_dir).name)
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
