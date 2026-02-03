#!/usr/bin/env python
"""
Run the Famam Music Generation CLI.

Usage:
    python run_cli.py             # Show menu to choose CLI
    python run_cli.py experiment  # Run experiment CLI (dataset & training)
    python run_cli.py generate    # Run generation CLI (music generation)
"""

import sys

if __name__ == "__main__":
    from src.cli.__main__ import main
    sys.exit(main())
