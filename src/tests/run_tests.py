#!/usr/bin/env python
"""
Run all tests for src_v4.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py test_core    # Run specific test file
    python run_tests.py -k "vocab"   # Run tests matching pattern
"""

import sys
import subprocess
from pathlib import Path


def main():
    # Change to project root directory
    project_root = Path(__file__).parent.parent.parent
    tests_dir = Path(__file__).parent

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", str(tests_dir)]

    # Add any additional arguments
    cmd.extend(sys.argv[1:])

    # Run pytest
    result = subprocess.run(cmd, cwd=str(project_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
