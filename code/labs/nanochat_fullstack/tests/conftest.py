"""Test configuration for nanochat_fullstack tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

for path in (PROJECT_ROOT, REPO_ROOT):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
