import os
import sys

# Headless Qt.  Set before any test creates a QApplication.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make the repository root importable when pytest is invoked from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
