# Phase 2 Manual-Visual Checklist (real GL render)

Run on a machine with a display + working OpenGL:

    cd gait_analysis
    .venv/bin/python app.py

1. **Анализ tab:** click "Choose session folder…", pick `data/caliscope_project/recordings/p1_3`.
   - Run button enables.
2. Click "▶ Run analysis".
   - Progress bar fills to 100%; quality line shows `frames … · fps … · …s` (and any >5% NaN warning).
   - Results table fills with cadence / speed / stride / step / stance / swing.
3. Click "Export CSV" and "Export XLSX" → files open in a spreadsheet, values match the table.
4. **Визуализация tab:**
   - 3D skeleton renders; drag to orbit (turntable camera).
   - Drag the frame slider → skeleton pose updates.
   - Click ▶ → skeleton animates; change speed to 0.5× / 0.25× → playback slows; ⏸ stops.
   - Angle plots show knee/hip/ankle curves with the ±STD corridor.
5. Click "Export PNG" → saved figure matches the on-screen plots.
6. Trigger an error: pick a folder with no labelled CSV → a warning dialog appears, the app does
   not crash, and the Run button re-enables.

## Module 3 — Vicon comparison (Phase 3)

- [ ] `python app.py` → «Сравнение» tab: pick a caliscope session + a Vicon XLSX → Run.
- [ ] Per-joint table shows knee + ankle rows (hip absent — Vicon has no shoulder marker).
- [ ] Overlay plot: caliscope (blue) and Vicon (red) gait-cycle curves are drawn per joint.
- [ ] Verdict summary line reflects knee ICC + worst verdict.
- [ ] `cli.py validate-vicon --recordings data/caliscope_project/recordings --vicon-dir
      data/Vicon_10_series --out results/vicon_validation` writes the Level-A/-C CSVs.
- [ ] `test_vvsc_parity` passes (ported Umeyama reproduces VvsC's similarity transform exactly).

**Known real-data caveat (see Phase 3 closing report):** current Level-A agreement is poor
(knee ICC ≈ 0, RMSE ~15–19°). Cause is data-prep, not the apparatus: the 136 s Vicon trials
pool ~92 mixed-motion cycles (sync jump + walking + idle) against ~3–4 clean caliscope walking
cycles, and the caliscope↔Vicon pairing is an unverified order-assumption. Meaningful Level-A
numbers require cropping Vicon to a steady-walking window, phase-aligning gait cycles, and a
confirmed synchronous pairing.
