"""Main window: two-tab shell wiring AnalyzePanel -> VizPanel."""
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from gui.panels.analyze_panel import AnalyzePanel
from gui.panels.viz_panel import VizPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis")
        self.resize(1000, 750)
        self.analyze = AnalyzePanel()
        self.viz = VizPanel()
        tabs = QTabWidget()
        tabs.addTab(self.analyze, "Анализ")
        tabs.addTab(self.viz, "Визуализация")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
        self.analyze.analysis_done.connect(self.viz.set_data)
        self.analyze.analysis_done.connect(
            lambda *_: self.statusBar().showMessage("Analysis complete"))
