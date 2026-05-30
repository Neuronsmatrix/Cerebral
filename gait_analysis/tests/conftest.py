"""Force a headless Qt platform before pytest-qt creates the QApplication."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
