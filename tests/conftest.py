import os
import sys

# Make the repository root importable when pytest is invoked from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
