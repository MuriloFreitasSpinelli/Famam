#!/usr/bin/env python
"""
Run the Famam Music Generation GUI.

Usage:
    python run_gui.py
"""

import sys

if __name__ == "__main__":
    from src.gui.generation_GUI import main
    sys.exit(main())
