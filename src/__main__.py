"""
Entry point for running src_v4 as a module.

Usage:
    python -m src_v4 <command> [options]
    python -m src_v4 interactive

See 'python -m src_v4 --help' for available commands.
"""

import sys
from .client.cli import main

if __name__ == "__main__":
    sys.exit(main())
