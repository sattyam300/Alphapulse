"""Ensure the project root is on sys.path so alphapulse is importable."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
