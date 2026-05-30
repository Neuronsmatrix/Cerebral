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
