#!/usr/bin/env python3
"""Invoke the fixed cross-family table runner without importing a model on --help."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aperture_table.cli import main
if __name__=='__main__':main()
