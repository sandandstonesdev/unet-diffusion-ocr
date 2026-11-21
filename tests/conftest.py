"""Widoczność pakietów repo (configs, data, model, pipeline) dla pytesta —
niezależnie od cwd. Styl jak w crnn-ocr/tests/conftest.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
