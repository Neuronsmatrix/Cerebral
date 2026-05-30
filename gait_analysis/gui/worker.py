"""Background worker: load a session + run the pipeline off the GUI thread."""
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from modules.data_loader.caliscope_reader import load_caliscope_session
from pipeline import run_pipeline


class PipelineWorker(QObject):
    progress = pyqtSignal(float, str)        # (fraction 0..1, stage label)
    finished = pyqtSignal(dict, object)      # (results, df_processed)
    error = pyqtSignal(str)

    def __init__(self, folder, model, cfg):
        super().__init__()
        self._folder = folder
        self._model = model
        self._cfg = cfg

    def run(self):
        try:
            self.progress.emit(0.0, "Loading session")
            df = load_caliscope_session(self._folder, model=self._model)
            results, df_out = run_pipeline(
                df, self._cfg, model=self._model,
                session_id=Path(self._folder).name,
                progress_cb=lambda frac, stage: self.progress.emit(frac, stage),
            )
            self.finished.emit(results, df_out)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, must never crash
            self.error.emit(str(exc))
