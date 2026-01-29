"""
Entry point for running src_v4.client as a module.

Usage:
    python -m src_v4.client <command> [options]
    python -m src_v4.client interactive
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
